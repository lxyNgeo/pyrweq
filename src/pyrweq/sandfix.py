"""Sand fixation (防风固沙量) calculation."""

from __future__ import annotations

import numpy as np

from pyrweq.core import RWEQResult, compute_rweq
from pyrweq.erosion import calc_sl


def compute_sandfix(
    wind_speed,
    precip,
    temp,
    elevation,
    potential_et,
    snow_depth,
    sand_content,
    silt_content,
    clay_content,
    organic_matter,
    ndvi,
    calcium_carbonate=None,
    land_use=None,
    slope=None,
    threshold_speed=5.0,
    downwind_distance=50.0,
    veg_method="simplified",
    input_10m=True,
) -> np.ndarray:
    """Compute sand fixation量 G = SL_potential - SL_actual.

    SL_potential is computed with C=1 (bare soil, no vegetation).
    SL_actual is computed with actual vegetation factor.

    Returns
    -------
    np.ndarray
        Sand fixation量 (kg/m^2).
    """
    actual = compute_rweq(
        wind_speed=wind_speed, precip=precip, temp=temp,
        elevation=elevation, potential_et=potential_et,
        snow_depth=snow_depth, sand_content=sand_content,
        silt_content=silt_content, clay_content=clay_content,
        organic_matter=organic_matter, ndvi=ndvi,
        calcium_carbonate=calcium_carbonate, land_use=land_use,
        slope=slope, threshold_speed=threshold_speed,
        downwind_distance=downwind_distance,
        veg_method=veg_method, input_10m=input_10m,
    )

    ones = np.ones_like(actual.sl)
    sl_pot, _, _ = calc_sl(actual.wf, actual.ef, actual.scf, actual.k_prime, ones, z=downwind_distance)

    return sl_pot - actual.sl
