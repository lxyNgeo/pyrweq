from __future__ import annotations
from typing import TYPE_CHECKING, Union, TypedDict
from numpy.typing import ArrayLike, DTypeLike

if TYPE_CHECKING:
    import dask.array as da  # 字符串前向引用，避免硬依赖 dask

# 顶层 API 接受的输入：array / GeoTIFF 路径 / (dask Array)
RasterInput = Union[ArrayLike, str, "da.Array"]

# 因子函数体接受的已加载数组
FactorArray = Union[ArrayLike, "da.Array"]

class RasterioProfile(TypedDict, total=False):
    """
    Rasterio profile. total=False 因为不是所有键都必备(alt栅格可能无 nodata)。
    额外键仍允许，因为 TypedDict 在运行时就是 dict。
    """
    driver: str           # "GTiff"
    dtype: str            # "float32" / "int32" / ...
    width: int
    height: int
    count: int
    crs: object           # CRS 对象或 EPSG 字符串
    transform: object     # Affine 对象或 6-tuple
    nodata: float | None
    blockxsize: int
    blockysize: int
    compress: str
    tiled: bool

class ZonalStatsRow(TypedDict):
    """Zonal statistics per zone."""
    zone: int
    count: int
    mean: float
    sum: float
    min: float
    max: float
    std: float


def is_dask_array(arr) -> bool:
    """Detect whether arr is a dask Array without a hard dask dependency."""
    return type(arr).__module__.startswith("dask.array") and hasattr(arr, "compute")


__all__ = ["RasterInput", "FactorArray", "RasterioProfile", "ZonalStatsRow", "is_dask_array"]
