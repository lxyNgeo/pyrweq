"""Zonal statistics for RWEQ results."""

from __future__ import annotations

import csv
import logging

import numpy as np

from pyrweq.io import read_raster
from pyrweq._types import RasterInput, ZonalStatsRow

logger = logging.getLogger(__name__)

def zonal_stats(
    sl_raster: RasterInput,
    zone_raster: RasterInput,
    output_csv: str | None = None,
) -> list[ZonalStatsRow]:
    """Compute zonal statistics of erosion量 by zone.

    Parameters
    ----------
    sl_raster : str or np.ndarray
        Wind erosion量 raster.
    zone_raster : str or np.ndarray
        Zone classification raster (e.g. land use codes, admin codes).
    output_csv : str or None
        If provided, write statistics to CSV.

    Returns
    -------
    list of dict
        Each dict: {"zone": int, "count": int, "mean": float, "sum": float,
                     "min": float, "max": float, "std": float}.
    """
    if isinstance(sl_raster, str):
        sl_arr, _ = read_raster(sl_raster)
    else:
        sl_arr = sl_raster

    if isinstance(zone_raster, str):
        zone_arr, _ = read_raster(zone_raster)
    else:
        zone_arr = zone_raster

    zones = np.unique(zone_arr)
    zones = zones[zones != 0]

    results = []
    for zone in zones:
        mask = zone_arr == zone
        vals = sl_arr[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        results.append({
            "zone": int(zone),
            "count": int(mask.sum()),
            "mean": float(np.mean(vals)),
            "sum": float(np.sum(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "std": float(np.std(vals)),
        })

    if output_csv:
        if not results:
            return results
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ZonalStatsRow.__annotations__.keys()))
            writer.writeheader()
            writer.writerows(results)

    return results
