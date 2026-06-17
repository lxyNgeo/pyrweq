"""Wind erosion intensity classification."""

from __future__ import annotations

import numpy as np

from pyrweq.io import read_raster, write_raster


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
    sl_raster: str | np.ndarray,
    profile: dict | None = None,
    scheme: str = "china_standard",
    output_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Classify wind erosion量 into intensity classes.

    Parameters
    ----------
    sl_raster : str or np.ndarray
        Wind erosion量 raster path or array (t/km^2/a or kg/m^2).
    profile : dict or None
        Rasterio profile. Required if sl_raster is ndarray and output_path is set.
    scheme : str
        Classification scheme. Currently only "china_standard" (SL190-2007).
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

    thresholds = CHINA_STANDARD
    classified = np.zeros_like(arr, dtype=np.int32)

    for low, high, code in thresholds:
        mask = (arr >= low) & (arr < high)
        classified[mask] = code

    if arr.max() >= thresholds[-1][0]:
        classified[arr >= thresholds[-1][0]] = thresholds[-1][2]

    if output_path and profile is not None:
        out_profile = profile.copy()
        out_profile.update(dtype="int32", nodata=0, count=1, driver="GTiff", compress="lzw")
        with __import__("rasterio").open(output_path, "w", **out_profile) as dst:
            dst.write(classified, 1)

    return classified, CHINA_LABELS
