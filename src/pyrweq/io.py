"""GeoTIFF I/O utilities for pyrweq."""

from __future__ import annotations

import logging

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from pyrweq._types import RasterioProfile

logger = logging.getLogger(__name__)


def read_raster(path: str, masked: bool = True) -> tuple[np.ndarray, RasterioProfile]:
    """Read a GeoTIFF and return (data, profile).

    Parameters
    ----------
    path : str
        GeoTIFF path.
    masked : bool
        If True (default), convert the file's nodata values to NaN so they
        propagate through factor math instead of being treated as real data.
        Integer rasters are promoted to float when masked.

    Returns
    -------
    data : np.ndarray
        2D array of raster values (NaN where nodata if masked).
    profile : dict
        Rasterio profile dict (for writing output with same georeference).
    """
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    if masked and profile.get("nodata") is not None:
        data = np.where(data == profile["nodata"], np.nan, data)
    return data, profile


def read_raster_lazy(
    path: str,
    chunks: tuple[int, int] | str = "auto",
    masked: bool = True,
) -> tuple["da.Array", RasterioProfile]:
    """Read a GeoTIFF lazily as a dask array without loading it into RAM.

    Each rasterio block (or the given ``chunks``) is read on demand via
    ``dask.delayed`` window reads, so memory use stays proportional to one
    chunk instead of the full raster.

    Parameters
    ----------
    path : str
        GeoTIFF path.
    chunks : (rows, cols) or "auto"
        Chunk size in pixels. "auto" uses the file's native block size.
    masked : bool
        If True (default), convert nodata values to NaN (lazy, per chunk).

    Returns
    -------
    arr : dask.array.Array
        Lazy 2D array (eager only when computed).
    profile : dict
        Rasterio profile (same as read_raster).

    Notes
    -----
    Requires ``dask[array]``.
    """
    try:
        import dask.array as da
        from dask import delayed
    except ImportError as e:
        raise ImportError(
            "read_raster_lazy requires dask[array]. Install with: pip install pyrweq[dask]"
        ) from e

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        dtype = profile["dtype"]
        nodata = profile.get("nodata")
        native = src.block_shapes[0]  # (rows, cols) per block

    if masked and nodata is not None and np.dtype(dtype).kind in "iu":
        # NaN masking promotes integer arrays to float
        dtype = "float64"

    if chunks == "auto":
        block_rows, block_cols = native
    else:
        block_rows, block_cols = chunks
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError(f"chunks must be positive, got {chunks}")

    @delayed
    def _read_window(row0: int, col0: int, rows: int, cols: int) -> np.ndarray:
        from rasterio.windows import Window

        with rasterio.open(path) as src:
            data = src.read(1, window=Window(col0, row0, cols, rows))
        if masked and nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        return data

    grid = []
    for row0 in range(0, height, block_rows):
        row_blocks = []
        for col0 in range(0, width, block_cols):
            h = min(block_rows, height - row0)
            w = min(block_cols, width - col0)
            arr = da.from_delayed(
                _read_window(row0, col0, h, w), shape=(h, w), dtype=dtype
            )
            row_blocks.append(arr)
        grid.append(row_blocks)

    return da.block(grid), profile


def write_raster(
    data: np.ndarray,
    profile: RasterioProfile,
    path: str,
    dtype: str | None = None,
    nodata: float | None = None,
    nan_to_nodata: bool = True,
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
    nan_to_nodata : bool
        If True (default), replace NaN cells with the nodata value so invalid
        cells are marked properly in the output (NaN cannot round-trip
        through all GDAL formats).
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
    if nan_to_nodata:
        data = np.where(np.isnan(data), nodata, data)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype(dtype), 1)


def ensure_same_shape(*arrays: np.ndarray) -> None:
    """Raise ValueError if arrays have different shapes."""
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) > 1:
        raise ValueError(f"Shape mismatch: {shapes}")


# rasterio resampling methods exposed by align_inputs
_RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}

# inputs that are categorical and must never be interpolated
_CATEGORICAL_KEYS = {"land_use", "landuse"}


def _resolve_resampling(name: str, resampling: str | dict[str, str]) -> Resampling:
    """Pick the rasterio Resampling for input `name` from user spec."""
    if isinstance(resampling, dict):
        spec = resampling.get(name, "bilinear")
    elif resampling == "auto":
        spec = "nearest" if name in _CATEGORICAL_KEYS else "bilinear"
    else:
        spec = resampling
    if spec not in _RESAMPLING_METHODS:
        raise ValueError(
            f"unknown resampling {spec!r} for input {name!r}; "
            f"choose from {sorted(_RESAMPLING_METHODS)}"
        )
    return _RESAMPLING_METHODS[spec]


def _grids_match(a: RasterioProfile, b: RasterioProfile) -> bool:
    """True if two profiles share CRS, transform and shape."""
    if a.get("crs") != b.get("crs"):
        return False
    if (a.get("width"), a.get("height")) != (b.get("width"), b.get("height")):
        return False
    ta, tb = a.get("transform"), b.get("transform")
    if ta is None or tb is None:
        return ta is tb
    return np.allclose(tuple(ta)[:6], tuple(tb)[:6])


def align_inputs(
    paths: dict[str, str],
    reference: str | None = None,
    resampling: str | dict[str, str] = "auto",
) -> tuple[dict[str, np.ndarray], RasterioProfile]:
    """Load rasters onto one common grid (CRS + transform + shape).

    Real studies combine rasters from different sources — meteorology at
    0.25 deg, soil at 1 km, NDVI at 250 m — and every RWEQ input must share
    one grid before computation. This function warps all inputs onto the
    grid of a reference raster using ``rasterio.warp.reproject``.

    Parameters
    ----------
    paths : dict
        Mapping of input name -> GeoTIFF path, using the same keys as
        ``compute_rweq`` (wind_speed, precip, ..., land_use, ...).
    reference : str or None
        Path of the reference raster whose grid is the target. Default:
        the first raster in ``paths`` (dict order).
    resampling : "auto" or str or dict
n        Resampling method per input. "auto" (default) uses nearest for
        categorical inputs (land_use) and bilinear for everything else;
        a string applies one method globally; a dict maps input names to
        methods and falls back to bilinear for unlisted names.
        Methods: nearest, bilinear, cubic, average.

    Returns
    -------
    data : dict
        Mapping of input name -> 2D float array on the reference grid,
        NaN where nodata / outside the source footprint.
    profile : dict
        The reference profile (float32, nodata NaN).

    Notes
    -----
    All outputs are float32. Integer codes (land_use) survive nearest
    resampling exactly; categorical inputs keep their code set.
    """
    if not paths:
        raise ValueError("paths must not be empty")
    if reference is None:
        reference = next(iter(paths.values()))

    with rasterio.open(reference) as ref_src:
        ref_profile = ref_src.profile.copy()
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_height, ref_width = ref_src.height, ref_src.width

    if ref_crs is None:
        raise ValueError(
            f"reference raster {reference!r} has no CRS; cannot align inputs"
        )

    out_profile = ref_profile.copy()
    out_profile.update(dtype="float32", nodata=np.nan)

    data: dict[str, np.ndarray] = {}
    n_warped = 0
    for name, path in paths.items():
        with rasterio.open(path) as src:
            src_profile = src.profile.copy()
        if _grids_match(src_profile, ref_profile):
            arr, _ = read_raster(path)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            data[name] = arr
            continue

        method = _resolve_resampling(name, resampling)
        dst = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_nodata=np.nan,
                resampling=method,
            )
        data[name] = dst
        n_warped += 1
        logger.info(
            "aligned %s (%s) onto reference grid with %s",
            name, path, method.name,
        )

    if n_warped:
        logger.info("align_inputs: warped %d/%d inputs onto %s", n_warped, len(paths), reference)
    else:
        logger.info("align_inputs: all %d inputs already on the reference grid", len(paths))
    return data, out_profile


def load_inputs(paths: dict[str, str]) -> tuple[dict[str, np.ndarray], RasterioProfile]:
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
