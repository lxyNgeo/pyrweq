"""Helper functions for RWEQ factor calculations."""

from __future__ import annotations

import logging

import numpy as np

from pyrweq._types import FactorArray

logger = logging.getLogger(__name__)

def wind_speed_2m(wind_speed_10m: FactorArray) -> FactorArray:
    """Convert 10m wind speed to 2m height using power law.

    U2 = U10 * (z2 / z10)^(1/7)
    """
    return wind_speed_10m * (2.0 / 10.0) ** (1.0 / 7.0)


def air_density(elevation_km: FactorArray, temp_kelvin: FactorArray) -> FactorArray:
    """Calculate air density from elevation and temperature.

    rho = 348 * (1.013 - 0.1183*EL + 0.0048*EL^2) / T
    """
    return 348.0 * (1.013 - 0.1183 * elevation_km + 0.0048 * elevation_km ** 2) / temp_kelvin


def soil_moisture_factor(
    pet: FactorArray,
    precip: FactorArray,
    irrigation: float = 0.0,
    rain_days: float = 1.0,
    nd: float = 15.0,
) -> FactorArray:
    """Calculate soil moisture factor SW.

    SW = [ETp - (R + I)] / Rd * Nd / ETp

    Cells with pet==0 (no evaporative demand, e.g. winter months) get SW=0
    instead of a divide-by-zero.
    """
    safe_pet = np.where(pet == 0, 1.0, pet)
    sw = (pet - (precip + irrigation)) / rain_days * nd / safe_pet
    return np.where(pet == 0, 0.0, np.clip(sw, 0.0, 1.0))


def snow_cover_factor(snow_depth: FactorArray, threshold_mm: float = 25.4) -> FactorArray:
    """Calculate snow cover factor SD.

    SD = 1 - P, where P is probability of snow depth > threshold.
    Input is snow depth array; P is computed as fraction of cells exceeding threshold.
    For per-pixel computation with time series, pass the probability directly.
    """
    # astype keeps the input dtype (np.where with python scalars would promote to float64)
    p = (snow_depth > threshold_mm).astype(snow_depth.dtype)
    return 1.0 - p
