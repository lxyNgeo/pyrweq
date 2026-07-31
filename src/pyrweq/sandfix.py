"""Sand fixation (防风固沙量) calculation."""

from __future__ import annotations

import logging

import numpy as np

from pyrweq._types import RasterInput
from pyrweq.core import compute_rweq
from pyrweq.erosion import calc_sl

logger = logging.getLogger(__name__)


def compute_sandfix(
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
) -> np.ndarray:
    """Compute sand fixation量 G = SL_potential - SL_actual.

    SL_potential is computed with C=None (bare soil).
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

    sl_pot, _, _ = calc_sl(actual.wf, actual.ef, actual.scf, actual.k_prime, c=None, z=downwind_distance)

    return sl_pot - actual.sl
