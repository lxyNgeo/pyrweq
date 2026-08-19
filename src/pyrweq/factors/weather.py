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
    wind_freq: FactorArray | None = None,
    threshold_speed: float = 5.0,
    nd: float = 15.0,
    n_obs: float | None = None,
    g: float = 9.8,
    input_10m: bool = True,
) -> FactorArray:
    """Calculate weather factor WF.

    WF = sum_i[ f_i * U_i * (U_i - Ut)^2 * Nd * rho / g ] * SW * SD

    Wind speed input modes
    ----------------------
    1. Single field (2D array): the classic point-value form. ``u`` is one
       representative wind speed per pixel and the sum collapses to
       u * (u - Ut)^2 * nd * rho / (n_obs * g). Here n_obs acts as the
       inverse occurrence frequency of u: n_obs=1 means u already
       aggregates the whole period; n_obs=nd (default) treats u as a
       per-observation (e.g. daily) representative speed.
    2. Observation series (3D array, shape (k, rows, cols)): each of the k
       layers is one wind speed observation (or speed-bin representative).
       Equal weights are assumed, i.e. f_i = 1/k. ``n_obs`` is ignored.
    3. Frequency distribution (3D array + ``wind_freq`` with the same
       shape): ``wind_freq[i]`` is the occurrence frequency of
       ``wind_speed[i]`` per pixel and must sum to 1 along axis 0.
       ``n_obs`` is ignored.

    Modes 2/3 follow the RWEQ original formulation (Fryear et al. 1998)
    where WF sums over wind speed classes. Because v*(v-Ut)^2 is convex in
    v, feeding the period-mean wind speed (mode 1) systematically
    underestimates WF relative to the true distribution — prefer mode 2
    (e.g. daily wind speeds) whenever the data is available.

    Parameters
    ----------
    wind_speed : FactorArray
        Wind speed (m/s), 2D or 3D (k, rows, cols). Default expects 10m
        height; set input_10m=False for 2m. Values below threshold_speed
        contribute zero.
    wind_freq : FactorArray or None
        Optional per-pixel occurrence frequencies, same shape as a 3D
        wind_speed; must sum to 1 along axis 0.
    precip : FactorArray
        Precipitation (mm) for the period.
    temp : FactorArray
        Temperature in Celsius.
    elevation : FactorArray
        Elevation in km.
    potential_et : FactorArray
        Potential evapotranspiration (mm).
    snow_depth : FactorArray
        Snow depth (mm).
    threshold_speed : float
        Threshold wind speed for erosion (m/s), default 5.0.
    nd : float
        Number of days in the calculation period. RWEQ natively uses a
        half-month period (nd=15); monthly runs should pass nd=30 (or
        365.25/12). Also drives the soil-moisture factor SW.
    n_obs : float or None
        Number of wind speed observations in the period. None (default)
        uses nd (one observation per day). Ignored for 3D wind_speed input
        (frequencies carry the normalization).
    g : float
        Gravitational acceleration (m/s^2), default 9.8.
    input_10m : bool
        If True, convert wind speed from 10m to 2m height.

    Returns
    -------
    FactorArray
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

    if u2.ndim == 3:
        # wind speed distribution over k observations/bins (axis 0)
        term = u2 * (u2 - threshold_speed) ** 2  # zero where u2 was clipped
        if wind_freq is not None:
            if np.shape(wind_freq) != np.shape(u2):
                raise ValueError(
                    f"wind_freq shape {np.shape(wind_freq)} does not match "
                    f"3D wind_speed {np.shape(u2)}"
                )
            _warn_unnormalized_freq(wind_freq)
            wf_wind = np.sum(term * wind_freq, axis=0) * nd * rho / g
        else:
            wf_wind = np.sum(term, axis=0) / u2.shape[0] * nd * rho / g
    else:
        if wind_freq is not None:
            raise ValueError(
                "wind_freq requires 3D wind_speed (k, rows, cols); "
                "got 2D wind_speed"
            )
        wf_wind = u2 * (u2 - threshold_speed) ** 2 * nd * rho / (n_obs * g)

    sw = soil_moisture_factor(potential_et, precip, nd=nd)
    sd = snow_cover_factor(snow_depth)

    return wf_wind * sw * sd


def _warn_unnormalized_freq(wind_freq) -> None:
    """Warn (numpy inputs only) if per-pixel frequencies do not sum to 1."""
    if type(wind_freq).__module__.startswith("dask.array"):
        return  # checking would force computation
    totals = np.sum(wind_freq, axis=0)
    if not np.allclose(totals, 1.0, atol=1e-6):
        logger.warning(
            "wind_freq sums to [%.3f, %.3f] per cell (expected 1.0); "
            "WF is scaled accordingly — normalize if unintended",
            float(np.nanmin(totals)), float(np.nanmax(totals)),
        )
