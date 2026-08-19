"""Wind erosion intensity classification."""

from __future__ import annotations

import logging

import numpy as np

from pyrweq.io import read_raster, write_raster
from pyrweq._types import RasterInput, RasterioProfile
from pyrweq.units import g_per_m_to_t_per_km2

logger = logging.getLogger(__name__)


CHINA_STANDARD = [
    (0, 200, 1),
    (200, 2500, 2),
    (2500, 5000, 3),
    (5000, 8000, 4),
    (8000, 15000, 5),
    (15000, np.inf, 6),
]

CHINA_LABELS = {
    1: "微度",
    2: "轻度",
    3: "中度",
    4: "强烈",
    5: "极强烈",
    6: "剧烈",
}


def classify_erosion(
    sl_raster: RasterInput,
    profile: RasterioProfile | None = None,
    scheme: str = "china_standard",
    cell_size: float | tuple[float, float] | None = None,
    output_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Classify wind erosion量 into intensity classes.

    The SL190-2007 thresholds are expressed in t/(km^2 * a). Two input
    conventions are supported:

    - ``cell_size=None`` (default): ``sl_raster`` already holds the
      erosion modulus in t/(km^2 * a) (or equivalently kg/m^2).
    - ``cell_size=<metres>``: ``sl_raster`` holds RWEQ-native SL in g/m
      (the direct output of ``calc_sl`` / ``compute_rweq``) and is
      converted to a modulus before classification.

    Parameters
    ----------
    sl_raster : str or np.ndarray
        Wind erosion量 raster path or array (see conventions above).
    profile : dict or None
        Rasterio profile. Required if sl_raster is ndarray and output_path is set.
    scheme : str
        Classification scheme. Currently only "china_standard" (SL190-2007).
    cell_size : float or (xres, yres) or None
        Grid cell edge length in metres. Provide it when passing native
        SL (g/m) so the thresholds apply to a proper modulus.
    output_path : str or None
        If provided, write classified raster.

    Returns
    -------
    classified : np.ndarray
        Integer class codes.
    labels : dict
        Mapping of class code to label string.
    """
    if isinstance(sl_raster, str):
        arr, profile = read_raster(sl_raster)
    else:
        arr = sl_raster

    if cell_size is not None:
        arr = g_per_m_to_t_per_km2(arr, cell_size)
        logger.info("classified input converted from g/m to t/(km^2*a) with cell_size=%s", cell_size)

    thresholds = CHINA_STANDARD
    classified = np.zeros_like(arr, dtype=np.int32)

    for low, high, code in thresholds:
        mask = (arr >= low) & (arr < high)
        classified[mask] = code

    if output_path and profile is not None:
        write_raster(classified, profile, output_path, dtype="int32", nodata=0)

    return classified, CHINA_LABELS
