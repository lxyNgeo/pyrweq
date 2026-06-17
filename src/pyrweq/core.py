"""Main RWEQ computation entry point."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from pyrweq.factors.weather import calc_weather_factor
from pyrweq.factors.erodibility import calc_erodibility
from pyrweq.factors.crust import calc_crust_factor
from pyrweq.factors.roughness import calc_roughness_simple
from pyrweq.factors.vegetation import calc_vegetation
from pyrweq.erosion import calc_sl
from pyrweq.io import read_raster, write_raster


@dataclass
class RWEQResult:
    """Result container for RWEQ computation."""
    sl: np.ndarray
    s: np.ndarray
    qmax: np.ndarray
    wf: np.ndarray
    ef: np.ndarray
    scf: np.ndarray
    k_prime: np.ndarray
    c: np.ndarray
    profile: dict


def compute_rweq(
    wind_speed: np.ndarray | str,
    precip: np.ndarray | str,
    temp: np.ndarray | str,
    elevation: np.ndarray | str,
    potential_et: np.ndarray | str,
    snow_depth: np.ndarray | str,
    sand_content: np.ndarray | str,
    silt_content: np.ndarray | str,
    clay_content: np.ndarray | str,
    organic_matter: np.ndarray | str,
    ndvi: np.ndarray | str,
    calcium_carbonate: np.ndarray | str | None = None,
    land_use: np.ndarray | str | None = None,
    slope: np.ndarray | str | None = None,
    threshold_speed: float = 5.0,
    downwind_distance: float = 50.0,
    veg_method: str = "simplified",
    input_10m: bool = True,
    output_dir: str | None = None,
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

    Returns
    -------
    RWEQResult
    """
    def _load(v):
        if isinstance(v, str):
            arr, prof = read_raster(v)
            return arr, prof
        return v, None

    def _resolve(v, default_profile):
        if isinstance(v, str):
            arr, prof = read_raster(v)
            return arr
        return v

    base_profile = None

    def _load_with_profile(v):
        nonlocal base_profile
        if isinstance(v, str):
            arr, prof = read_raster(v)
            if base_profile is None:
                base_profile = prof
            return arr
        if base_profile is None:
            raise ValueError("First input must be a GeoTIFF path to establish georeference")
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

    wf = calc_weather_factor(wind, pr, tmp, ele, pet, snow,
                             threshold_speed=threshold_speed, input_10m=input_10m)
    ef = calc_erodibility(sa, si, cl, om, caco3)
    scf = calc_crust_factor(cl, om)
    k_prime = calc_roughness_simple(slp)
    c = calc_vegetation(ndvi_arr, method=veg_method, land_use=lu)

    sl, s, qmax = calc_sl(wf, ef, scf, k_prime, c, z=downwind_distance)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        p = base_profile
        write_raster(wf, p, os.path.join(output_dir, "wf.tif"))
        write_raster(ef, p, os.path.join(output_dir, "ef.tif"))
        write_raster(scf, p, os.path.join(output_dir, "scf.tif"))
        write_raster(k_prime, p, os.path.join(output_dir, "k_prime.tif"))
        write_raster(c, p, os.path.join(output_dir, "c.tif"))
        write_raster(s, p, os.path.join(output_dir, "s.tif"))
        write_raster(qmax, p, os.path.join(output_dir, "qmax.tif"))
        write_raster(sl, p, os.path.join(output_dir, "sl.tif"))

    return RWEQResult(sl=sl, s=s, qmax=qmax, wf=wf, ef=ef, scf=scf, k_prime=k_prime, c=c, profile=base_profile)
