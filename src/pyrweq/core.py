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
from pyrweq.io import read_raster, write_raster

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
    threshold_speed: float = 5.0,
    downwind_distance: float = 50.0,
    veg_method: str = "simplified",
    input_10m: bool = True,
    output_dir: str | None = None,
    n_workers: int | None = None,
    backend: str = "numpy",
) -> RWEQResult:
    """Compute RWEQ wind erosion量.

    Parameters
    ----------
    wind_speed, precip, temp, elevation, potential_et, snow_depth :
        Input arrays or GeoTIFF paths. Wind speed is in m/s.
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
    output_dir : str or None
        If provided, write all intermediate and final rasters.
    n_workers : int or None
        Number of threads for factor computation. Default min(5, cpu_count).
        1 disables parallelism (sequential, same as before).
    backend : str
        "numpy" (default) or "dask". When "numpy" and dask arrays are detected,
        a warning is logged. When "dask", requires dask[array].

    Returns
    -------
    RWEQResult
    """
    t_start = time.time()
    base_profile = None

    def _load_with_profile(v):
        nonlocal base_profile
        if isinstance(v, str):
            arr, prof = read_raster(v)
            if base_profile is None:
                base_profile = prof
            return arr
        return v

    wind = _load_with_profile(wind_speed)
    pr = _load_with_profile(precip)
    tmp = _load_with_profile(temp)
    ele = _load_with_profile(elevation)
    pet = _load_with_profile(potential_et)
    snow = _load_with_profile(snow_depth)
    sa = _load_with_profile(sand_content)
    si = _load_with_profile(silt_content)
    cl = _load_with_profile(clay_content)
    om = _load_with_profile(organic_matter)
    ndvi_arr = _load_with_profile(ndvi)

    caco3 = _load_with_profile(calcium_carbonate) if calcium_carbonate is not None else None
    lu = _load_with_profile(land_use) if land_use is not None else None
    slp = _load_with_profile(slope) if slope is not None else np.zeros_like(wind)

    if base_profile is None:
        # All inputs were arrays; create a minimal placeholder profile
        base_profile = {
            "driver": "GTiff", "dtype": "float32", "width": wind.shape[1],
            "height": wind.shape[0], "count": 1, "crs": None,
            "transform": None, "nodata": -9999.0,
        }

    shape = wind.shape
    logger.info(
        "compute_rweq start: shape=%s backend=%s n_workers=%s veg_method=%s threshold=%s z=%s",
        shape, backend, n_workers, veg_method, threshold_speed, downwind_distance,
    )

    # --- dask backend detection & setup ---
    has_dask = any(is_dask_array(x) for x in [wind, pr, tmp, ele, pet, snow, sa, si, cl, om, ndvi_arr] if x is not None)

    if backend == "dask":
        try:
            import dask.array as da  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='dask' requires dask[array]. Install with: pip install pyrweq[dask]"
            ) from e
        if not all(is_dask_array(x) for x in [wind, pr, tmp, ele, pet, snow, sa, si, cl, om, ndvi_arr] if x is not None):
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
            threshold_speed=threshold_speed, input_10m=input_10m,
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
