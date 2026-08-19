"""Model validation against observed values.

Compares RWEQ predictions (erosion modulus, sand fixation, ...) with
field measurements (erosion pins, ^137Cs, sediment traps, station
observations). Works on paired arrays; use :func:`sample_points` to
extract predictions at measurement locations first.
"""

from __future__ import annotations

import logging

import numpy as np

from pyrweq._types import FactorArray

logger = logging.getLogger(__name__)


def validate(observed: FactorArray, predicted: FactorArray) -> dict:
    """Validation metrics for paired observed / predicted values.

    NaN cells (nodata or missing observations) are dropped pairwise.

    Parameters
    ----------
    observed, predicted : array
        Paired values in the same unit (e.g. t/(km^2 * a)). Rasters must
        share a shape; 1D arrays of station values also work.

    Returns
    -------
    dict with keys:
        n : int — number of valid pairs
        r2 : float — coefficient of determination
        rmse : float — root mean squared error
        mae : float — mean absolute error
        bias : float — mean error (predicted - observed); negative means
            the model underestimates
        nash_sutcliffe : float — Nash–Sutcliffe efficiency (1 = perfect,
            0 = as good as the observed mean, <0 = worse)
    """
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if obs.shape != pred.shape:
        raise ValueError(f"shape mismatch: observed {obs.shape} vs predicted {pred.shape}")

    mask = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[mask], pred[mask]
    n = int(o.size)
    if n < 2:
        raise ValueError(f"need at least 2 valid pairs, got {n}")

    err = p - o
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    result = {
        "n": n,
        "r2": r2,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "nash_sutcliffe": nse,
    }
    logger.info(
        "validate: n=%d r2=%.4f rmse=%.4f bias=%.4f nse=%.4f",
        n, result["r2"], result["rmse"], result["bias"], result["nash_sutcliffe"],
    )
    if abs(result["bias"]) > 0.5 * np.mean(np.abs(o)) and np.mean(np.abs(o)) > 0:
        logger.warning(
            "large bias (%.3f) relative to mean observed magnitude (%.3f); "
            "check units and input parameters",
            result["bias"], float(np.mean(np.abs(o))),
        )
    return result


def sample_points(raster_path: str, xs, ys) -> np.ndarray:
    """Sample raster values at point coordinates.

    Thin convenience wrapper around ``rasterio.sample`` for extracting
    predictions at measurement stations before :func:`validate`.

    Parameters
    ----------
    raster_path : str
        Path of the prediction raster (any CRS; coordinates below must be
        in the same CRS).
    xs, ys : sequences of float
        Point coordinates (x = easting/longitude, y = northing/latitude).

    Returns
    -------
    np.ndarray
        1D array of sampled values; NaN where the raster has nodata.
    """
    import rasterio

    coords = list(zip(xs, ys))
    with rasterio.open(raster_path) as src:
        values = np.array([v[0] for v in src.sample(coords)], dtype=np.float64)
    nodata = src.nodata
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    return values


__all__ = ["validate", "sample_points"]
