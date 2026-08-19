"""Main RWEQ computation entry point."""

from __future__ import annotations

import os
import logging
import time
from dataclasses import dataclass

import numpy as np

from pyrweq._types import RasterInput, FactorArray, RasterioProfile, is_dask_array
from pyrweq.factors.weather import calc_weather_factor
from pyrweq.factors.erodibility import calc_erodibility
from pyrweq.factors.crust import calc_crust_factor
from pyrweq.factors.roughness import calc_roughness_simple
from pyrweq.factors.vegetation import calc_vegetation
from pyrweq.erosion import calc_sl
from pyrweq.io import read_raster, read_raster_lazy, write_raster

logger = logging.getLogger(__name__)


@dataclass
class RWEQResult:
    """Result container for RWEQ computation."""
    sl: FactorArray
    s: FactorArray
    qmax: FactorArray
    wf: FactorArray
    ef: FactorArray
    scf: FactorArray
    k_prime: FactorArray
    c: FactorArray
    profile: RasterioProfile | None


@dataclass
class YearlyRWEQResult:
    """Result container for multi-period (e.g. monthly) RWEQ runs.

    Attributes
    ----------
    sl : FactorArray
        Total erosion over all periods (sum of per-period SL).
    months : list[RWEQResult]
        Per-period results, in input order.
    """
    sl: FactorArray
    months: list[RWEQResult]

    FACTOR_NAMES = ("wf", "ef", "scf", "k_prime", "c", "s", "qmax")

    def __getattr__(self, name: str) -> FactorArray:
        """Factor access: return the period-mean of the requested factor."""
        if name in self.FACTOR_NAMES:
            vals = [getattr(m, name) for m in self.months]
            return np.nanmean(np.stack(vals), axis=0)
        raise AttributeError(f"YearlyRWEQResult has no attribute {name!r}")


def compute_rweq(
    wind_speed: RasterInput,
    precip: RasterInput,
    temp: RasterInput,
    elevation: RasterInput,
    potential_et: RasterInput,
    snow_depth: RasterInput,
    sand_content: RasterInput,
    silt_content: RasterInput,
    clay_content: RasterInput,
    organic_matter: RasterInput,
    ndvi: RasterInput,
    calcium_carbonate: RasterInput | None = None,
    land_use: RasterInput | None = None,
    slope: RasterInput | None = None,
    wind_freq: FactorArray | None = None,
    threshold_speed: float = 5.0,
    downwind_distance: float = 50.0,
    veg_method: str = "simplified",
    input_10m: bool = True,
    nd: float = 15.0,
    n_obs: float | None = None,
    output_dir: str | None = None,
    n_workers: int | None = None,
    backend: str = "numpy",
    chunks: tuple[int, int] | str = "auto",
    masked: bool = True,
) -> RWEQResult:
    """Compute RWEQ wind erosion量.

    Parameters
    ----------
    wind_speed, precip, temp, elevation, potential_et, snow_depth :
        Input arrays or GeoTIFF paths. Wind speed is in m/s, either a 2D
        field or a 3D array (k, rows, cols) of wind speed observations
        (RWEQ wind-speed classes; see calc_weather_factor).
    wind_freq : array or None
        Optional per-pixel occurrence frequencies with the same shape as a
        3D wind_speed; must sum to 1 along axis 0. Arrays only (no paths).
    sand_content, silt_content, clay_content, organic_matter :
        Soil texture arrays (%).
    ndvi : array or str
        NDVI values.
    calcium_carbonate : array or str or None
        CaCO3 content (%). Default 0.
    land_use : array or str or None
        Land use codes (10/20/30/40/50/60). Needed for veg_method="typed".
    slope : array or str or None
        Slope in degrees. If None, derived as 0 (flat).
    threshold_speed : float
        Threshold wind speed (m/s), default 5.0.
    downwind_distance : float
        Downwind distance z (m), default 50.
    veg_method : str
        "simplified", "typed", or "full_cog".
    input_10m : bool
        If True, convert wind speed from 10m to 2m.
    nd : float
        Days in the calculation period. RWEQ natively uses a half-month
        period (nd=15, default); monthly runs should pass nd=30. Affects WF
        and the soil-moisture factor SW.
    n_obs : float or None
        Wind speed observations in the period. None (default) uses nd.
    output_dir : str or None
        If provided, write all intermediate and final rasters.
    n_workers : int or None
        Number of threads for factor computation. Default min(5, cpu_count).
        1 disables parallelism (sequential, same as before).
    backend : str
        "numpy" (default) or "dask". When "numpy" and dask arrays are detected,
        a warning is logged. When "dask", requires dask[array].
    chunks : (rows, cols) or "auto"
        Only used with backend="dask" and GeoTIFF path inputs: chunk size for
        lazy reading (pixels). "auto" uses each file's native block size.
        Array inputs are never rechunked.
    masked : bool
        If True (default), nodata values in input rasters are converted to
        NaN so invalid cells are excluded from results instead of treated as
        real data. NaN cells propagate through all factors and are written
        back as nodata in output rasters.

    Returns
    -------
    RWEQResult
    """
    t_start = time.time()
    base_profile = None

    def _load_with_profile(v, lazy: bool = False):
        nonlocal base_profile
        if isinstance(v, str):
            if lazy:
                arr, prof = read_raster_lazy(v, chunks=chunks, masked=masked)
            else:
                arr, prof = read_raster(v, masked=masked)
            if base_profile is None:
                base_profile = prof
            return arr
        return v

    lazy = backend == "dask"
    wind = _load_with_profile(wind_speed, lazy)
    shape = wind.shape[-2:]  # spatial shape (wind may be 3D)
    pr = _load_with_profile(precip, lazy)
    tmp = _load_with_profile(temp, lazy)
    ele = _load_with_profile(elevation, lazy)
    pet = _load_with_profile(potential_et, lazy)
    snow = _load_with_profile(snow_depth, lazy)
    sa = _load_with_profile(sand_content, lazy)
    si = _load_with_profile(silt_content, lazy)
    cl = _load_with_profile(clay_content, lazy)
    om = _load_with_profile(organic_matter, lazy)
    ndvi_arr = _load_with_profile(ndvi, lazy)

    caco3 = _load_with_profile(calcium_carbonate, lazy) if calcium_carbonate is not None else None
    lu = _load_with_profile(land_use, lazy) if land_use is not None else None
    slp = _load_with_profile(slope, lazy) if slope is not None else np.zeros(shape, dtype=np.float32)

    if base_profile is None:
        # All inputs were arrays; create a minimal placeholder profile
        base_profile = {
            "driver": "GTiff", "dtype": "float32", "width": shape[1],
            "height": shape[0], "count": 1, "crs": None,
            "transform": None, "nodata": -9999.0,
        }

    shape_ok = (wind.ndim == 2) or (wind.ndim == 3)
    if not shape_ok:
        raise ValueError(f"wind_speed must be 2D or 3D, got ndim={wind.ndim}")
    logger.info(
        "compute_rweq start: shape=%s backend=%s n_workers=%s veg_method=%s threshold=%s z=%s",
        shape, backend, n_workers, veg_method, threshold_speed, downwind_distance,
    )

    # --- dask backend detection & setup ---
    has_dask = any(is_dask_array(x) for x in [wind, wind_freq, pr, tmp, ele, pet, snow, sa, si, cl, om, ndvi_arr] if x is not None)

    if backend == "dask":
        try:
            import dask.array as da  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='dask' requires dask[array]. Install it with: pip install dask[array]"
            ) from e
        if not all(is_dask_array(x) for x in [wind, wind_freq, pr, tmp, ele, pet, snow, sa, si, cl, om, ndvi_arr] if x is not None):
            raise TypeError("dask backend requires all array inputs to be dask.array.Array")
    elif has_dask:
        logger.warning(
            "dask array detected but backend='numpy'; pass backend='dask' for lazy computing"
        )

    # --- factor computation (parallel via ThreadPoolExecutor) ---
    from concurrent.futures import ThreadPoolExecutor

    if n_workers is None:
        n_workers = min(5, os.cpu_count() or 1)

    factor_tasks = {
        "weather": lambda: calc_weather_factor(
            wind, pr, tmp, ele, pet, snow,
            wind_freq=wind_freq,
            threshold_speed=threshold_speed, input_10m=input_10m,
            nd=nd, n_obs=n_obs,
        ),
        "erodibility": lambda: calc_erodibility(sa, si, cl, om, caco3),
        "crust": lambda: calc_crust_factor(cl, om),
        "roughness": lambda: calc_roughness_simple(slp),
        "vegetation": lambda: calc_vegetation(ndvi_arr, method=veg_method, land_use=lu),
    }

    if n_workers <= 1:
        wf = factor_tasks["weather"]()
        logger.info("factor done: weather in %.2fs", time.time() - t_start)
        ef = factor_tasks["erodibility"]()
        logger.info("factor done: erodibility in %.2fs", time.time() - t_start)
        scf = factor_tasks["crust"]()
        logger.info("factor done: crust in %.2fs", time.time() - t_start)
        k_prime = factor_tasks["roughness"]()
        logger.info("factor done: roughness in %.2fs", time.time() - t_start)
        c = factor_tasks["vegetation"]()
        logger.info("factor done: vegetation in %.2fs", time.time() - t_start)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {name: ex.submit(fn) for name, fn in factor_tasks.items()}
            wf = futs["weather"].result()
            logger.info("factor done: weather in %.2fs", time.time() - t_start)
            ef = futs["erodibility"].result()
            logger.info("factor done: erodibility in %.2fs", time.time() - t_start)
            scf = futs["crust"].result()
            logger.info("factor done: crust in %.2fs", time.time() - t_start)
            k_prime = futs["roughness"].result()
            logger.info("factor done: roughness in %.2fs", time.time() - t_start)
            c = futs["vegetation"].result()
            logger.info("factor done: vegetation in %.2fs", time.time() - t_start)

    t_factors = time.time()
    sl, s, qmax = calc_sl(wf, ef, scf, k_prime, c, z=downwind_distance)
    logger.info("calc_sl done in %.2fs", time.time() - t_factors)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        p = base_profile
        for name, arr in [("wf", wf), ("ef", ef), ("scf", scf), ("k_prime", k_prime),
                          ("c", c), ("s", s), ("qmax", qmax), ("sl", sl)]:
            write_raster(arr, p, os.path.join(output_dir, f"{name}.tif"))
            logger.info("raster written: %s.tif", name)

    elapsed = time.time() - t_start
    logger.info(
        "compute_rweq done: sl.mean=%.4f sl.max=%.4f elapsed=%.2fs",
        float(np.nanmean(sl)), float(np.nanmax(sl)), elapsed,
    )

    return RWEQResult(sl=sl, s=s, qmax=qmax, wf=wf, ef=ef, scf=scf, k_prime=k_prime, c=c, profile=base_profile)


def compute_rweq_yearly(
    monthly_inputs: list[dict],
    period_days: float | None = None,
    output_dir: str | None = None,
    n_workers: int | None = None,
    backend: str = "numpy",
    chunks: tuple[int, int] | str = "auto",
    masked: bool = True,
    **factor_kwargs,
) -> YearlyRWEQResult:
    """Compute multi-period (typically 12 monthly) RWEQ erosion totals.

    Each period is computed with :func:`compute_rweq` and the per-period
    erosion amounts are summed into an annual (or multi-period) total.
    This follows the standard RWEQ workflow of computing WF per month and
    summing monthly erosion.

    Parameters
    ----------
    monthly_inputs : list of dict
        One dict per period, each with the same keys as :func:`compute_rweq`
        (wind_speed, precip, ..., ndvi, optional calcium_carbonate/land_use/
        slope). Keys may be GeoTIFF paths or arrays; mixing is allowed but
        shapes must match. All periods must use the same input kind (all
        paths or all arrays). A period dict may carry its own ``nd``/
        ``n_obs`` keys for per-period control.
    period_days : float or None
        Days per period, passed as ``nd`` to each period's computation
        unless the user supplied ``nd`` explicitly (in ``factor_kwargs`` or
        in a period dict). If None (default), it is inferred as
        365.25 / len(monthly_inputs) (uniform periods; 12 periods -> ~30.4).
        The RWEQ-native half-month value would be 15.
    output_dir : str or None
        If provided, write the total erosion raster (sl_yearly.tif) and the
        period-mean factors.
    n_workers, backend, chunks, masked :
        Passed through to :func:`compute_rweq` for each period.
    **factor_kwargs :
        Passed through to :func:`compute_rweq` (threshold_speed,
        downwind_distance, veg_method, input_10m, ...).

    Returns
    -------
    YearlyRWEQResult
        sl is the summed total; ``.months`` holds per-period RWEQResult;
        factor attributes (``.wf``, ``.ef``, ...) expose period means.
    """
    if not monthly_inputs:
        raise ValueError("monthly_inputs must contain at least one period dict")

    overlap = set(factor_kwargs) & {k for m in monthly_inputs for k in m}
    if overlap:
        raise ValueError(
            f"factor_kwargs {sorted(overlap)} collide with keys in period dicts; "
            "pass them per-period or globally, not both"
        )

    if "nd" not in factor_kwargs and all("nd" not in m for m in monthly_inputs):
        eff_nd = period_days if period_days is not None else 365.25 / len(monthly_inputs)
        if abs(eff_nd - 15.0) > 1e-9:
            logger.info(
                "yearly: nd inferred as %.2f days/period (%s periods); pass period_days or nd to override",
                eff_nd, len(monthly_inputs),
            )
    else:
        eff_nd = None  # explicit nd somewhere; do not inject

    t_start = time.time()
    months: list[RWEQResult] = []
    for i, inputs in enumerate(monthly_inputs):
        t0 = time.time()
        per_kwargs = dict(factor_kwargs)
        if eff_nd is not None and "nd" not in inputs:
            per_kwargs["nd"] = eff_nd
        res = compute_rweq(
            **inputs,
            n_workers=n_workers, backend=backend, chunks=chunks, masked=masked,
            **per_kwargs,
        )
        months.append(res)
        logger.info("period %d/%d done in %.2fs", i + 1, len(monthly_inputs), time.time() - t0)

    sl = np.sum(np.stack([m.sl for m in months]), axis=0)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        profile = months[0].profile
        if profile is not None:
            write_raster(sl, profile, os.path.join(output_dir, "sl_yearly.tif"))
            for name in YearlyRWEQResult.FACTOR_NAMES:
                write_raster(
                    np.nanmean(np.stack([getattr(m, name) for m in months]), axis=0),
                    profile, os.path.join(output_dir, f"{name}_mean.tif"),
                )

    logger.info(
        "compute_rweq_yearly done: %d periods, sl.sum=%.4f elapsed=%.2fs",
        len(monthly_inputs), float(np.nansum(sl)), time.time() - t_start,
    )
    return YearlyRWEQResult(sl=sl, months=months)
