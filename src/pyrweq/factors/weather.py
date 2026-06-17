"""Weather factor (WF) calculation for RWEQ."""

import numpy as np

from pyrweq.factors.helpers import air_density, soil_moisture_factor, snow_cover_factor, wind_speed_2m


def calc_weather_factor(
    wind_speed: np.ndarray,
    precip: np.ndarray,
    temp: np.ndarray,
    elevation: np.ndarray,
    potential_et: np.ndarray,
    snow_depth: np.ndarray,
    threshold_speed: float = 5.0,
    nd: float = 15.0,
    n_obs: float = 15.0,
    g: float = 9.8,
    input_10m: bool = True,
) -> np.ndarray:
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
        Number of days in the calculation period, default 15 (half-month).
    n_obs : float
        Number of wind speed observations, default 15.
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

    u2 = np.where(u2 < threshold_speed, 0.0, u2)

    temp_k = temp + 273.15
    rho = air_density(elevation, temp_k)

    wf_wind = u2 * (u2 - threshold_speed) ** 2 * nd * rho / (n_obs * g)

    sw = soil_moisture_factor(potential_et, precip, nd=nd)
    sd = snow_cover_factor(snow_depth)

    return wf_wind * sw * sd
