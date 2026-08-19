from __future__ import annotations

import logging

import numpy as np

from pyrweq._types import FactorArray
from pyrweq.factors.helpers import air_density, soil_moisture_factor, snow_cover_factor, wind_speed_2m

logger = logging.getLogger(__name__)


def calc_weather_factor(
    wind_speed: FactorArray,
    precip: FactorArray,
    temp: FactorArray,
    elevation: FactorArray,
    potential_et: FactorArray,
    snow_depth: FactorArray,
    threshold_speed: float = 5.0,
    nd: float = 15.0,
    n_obs: float | None = None,
    g: float = 9.8,
    input_10m: bool = True,
) -> FactorArray:
    """Calculate weather factor WF.

    WF = sum_i[ Ui * (Ui - Ut)^2 * Nd * rho / (N * g) ] * SW * SD

    Parameters
    ----------
    wind_speed : np.ndarray
        Wind speed (m/s). Default expects 10m height; set input_10m=False for 2m.
    precip : np.ndarray
        Precipitation (mm) for the period.
    temp : np.ndarray
        Temperature in Celsius.
    elevation : np.ndarray
        Elevation in km.
    potential_et : np.ndarray
        Potential evapotranspiration (mm).
    snow_depth : np.ndarray
        Snow depth (mm).
    threshold_speed : float
        Threshold wind speed for erosion (m/s), default 5.0.
    nd : float
        Number of days in the calculation period. RWEQ natively uses a
        half-month period (nd=15); monthly runs should pass nd=30 (or
        365.25/12). Also drives the soil-moisture factor SW.
    n_obs : float or None
        Number of wind speed observations in the period. None (default)
        uses nd (one observation per day).
    g : float
        Gravitational acceleration (m/s^2), default 9.8.
    input_10m : bool
        If True, convert wind speed from 10m to 2m height.

    Returns
    -------
    np.ndarray
        Weather factor WF (kg/m).
    """
    u2 = wind_speed_2m(wind_speed) if input_10m else wind_speed.copy()
    if n_obs is None:
        n_obs = nd

    frac_above = float(np.mean(u2 >= threshold_speed))
    if frac_above > 0.90:
        logger.warning(
            "%.1f%% of wind cells exceed threshold=%s m/s (erosion aggressive); check wind unit",
            frac_above * 100, threshold_speed,
        )

    u2 = np.where(u2 < threshold_speed, 0.0, u2)

    temp_k = temp + 273.15
    rho = air_density(elevation, temp_k)

    wf_wind = u2 * (u2 - threshold_speed) ** 2 * nd * rho / (n_obs * g)

    sw = soil_moisture_factor(potential_et, precip, nd=nd)
    sd = snow_cover_factor(snow_depth)

    return wf_wind * sw * sd
