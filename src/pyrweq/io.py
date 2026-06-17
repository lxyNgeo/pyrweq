"""GeoTIFF I/O utilities for pyrweq."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_bounds


def read_raster(path: str) -> tuple[np.ndarray, dict]:
    """Read a GeoTIFF and return (data, profile).

    Returns
    -------
    data : np.ndarray
        2D array of raster values.
    profile : dict
        Rasterio profile dict (for writing output with same georeference).
    """
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def write_raster(
    data: np.ndarray,
    profile: dict,
    path: str,
    dtype: str | None = None,
    nodata: float | None = None,
) -> None:
    """Write a 2D array as GeoTIFF.

    Parameters
    ----------
    data : np.ndarray
        2D array to write.
    profile : dict
        Rasterio profile (copied from an input raster).
    path : str
        Output file path.
    dtype : str or None
        Override data type. If None, uses float32.
    nodata : float or None
        NoData value. If None, uses -9999.
    """
    if dtype is None:
        dtype = "float32"
    if nodata is None:
        nodata = -9999.0

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype(dtype), 1)


def ensure_same_shape(*arrays: np.ndarray) -> None:
    """Raise ValueError if arrays have different shapes."""
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) > 1:
        raise ValueError(f"Shape mismatch: {shapes}")


def load_inputs(paths: dict[str, str]) -> tuple[dict[str, np.ndarray], dict]:
    """Load multiple rasters, validate shapes, return arrays and base profile.

    Parameters
    ----------
    paths : dict
        Mapping of name -> GeoTIFF path, e.g. {"wind_speed": "wind.tif", ...}.

    Returns
    -------
    data : dict
        Mapping of name -> 2D np.ndarray.
    profile : dict
        Profile from the first loaded raster (for output georeference).
    """
    data = {}
    profile = None
    for name, path in paths.items():
        arr, prof = read_raster(path)
        data[name] = arr
        if profile is None:
            profile = prof
    ensure_same_shape(*data.values())
    return data, profile
