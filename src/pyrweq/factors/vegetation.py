"""Vegetation factor (C / COG) calculation for RWEQ."""

from __future__ import annotations

import logging

import numpy as np

from pyrweq._types import FactorArray, is_dask_array

logger = logging.getLogger(__name__)


ALPHA_COEFFICIENTS = {
    10: -0.1535,   # woodland
    20: -0.1151,   # grassland
    30: -0.0921,   # shrub
    40: -0.0768,   # bare land
    50: -0.0658,   # sandy land
    60: -0.0438,   # cropland
}


def vegetation_cover(
    ndvi: FactorArray,
    ndvi_soil: float | None = None,
    ndvi_veg: float | None = None,
) -> FactorArray:
    """Calculate vegetation cover fraction using pixel dimidiate model.

    SC = (NDVI - NDVIsoil) / (NDVIveg - NDVIsoil)

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI values.
    ndvi_soil : float or None
        NDVI of bare soil. If None, uses 5th percentile.
    ndvi_veg : float or None
        NDVI of full vegetation. If None, uses 95th percentile.

    Returns
    -------
    np.ndarray
        Vegetation cover fraction [0, 1].
    """
    if ndvi_soil is None or ndvi_veg is None:
        # Percentiles are global reductions; dask cannot do full-array
        # nanquantile lazily, so fall back to eager compute for the scalar
        # thresholds. This triggers the graph once, but the returned factor
        # arrays stay lazy.
        ndvi_np = ndvi.compute() if is_dask_array(ndvi) else ndvi
        if ndvi_soil is None:
            ndvi_soil = float(np.nanpercentile(ndvi_np, 5))
        if ndvi_veg is None:
            ndvi_veg = float(np.nanpercentile(ndvi_np, 95))

    if ndvi_veg == ndvi_soil:
        logger.warning(
            "ndvi_veg≈ndvi_soil (%.4f/%.4f); vegetation_cover degrades to zeros. "
            "Pass explicit ndvi_soil/ndvi_veg for sparse-veg scenes.",
            ndvi_veg, ndvi_soil,
        )
        return np.zeros_like(ndvi)

    sc = (ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil)
    return np.clip(sc, 0.0, 1.0)


def calc_vegetation(
    ndvi: FactorArray,
    method: str = "simplified",
    land_use: FactorArray | None = None,
    ndvi_soil: float | None = None,
    ndvi_veg: float | None = None,
    alpha_coefficients: dict | None = None,
) -> FactorArray:
    """Calculate vegetation factor C.

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI values.
    method : str
        "simplified" - C = exp(-0.0438 * SC)
        "typed"      - C = exp(alpha_i * SC) with land use specific alpha
        "full_cog"   - full COG with flat/standing residue and canopy (placeholder)
    land_use : np.ndarray or None
        Land use type codes (10=woodland, 20=grass, 30=shrub, 40=bare, 50=sand, 60=cropland).
    ndvi_soil, ndvi_veg : float or None
        NDVI endpoints for cover fraction calculation.

    Returns
    -------
    np.ndarray
        Vegetation factor C (dimensionless).
    """
    sc = vegetation_cover(ndvi, ndvi_soil, ndvi_veg)

    if method == "simplified":
        return np.exp(-0.0438 * sc)

    elif method == "typed":
        if land_use is None:
            raise ValueError("land_use is required for method='typed'")
        coeffs = alpha_coefficients or ALPHA_COEFFICIENTS
        result = np.full_like(sc, fill_value=np.nan, dtype=np.float64)
        for lu_code, alpha in coeffs.items():
            mask = land_use == lu_code
            result = np.where(mask, np.exp(alpha * sc), result)
        no_match = ~np.isin(land_use, list(coeffs.keys()))
        frac_no_match = float(np.mean(no_match))
        if frac_no_match > 0:
            logger.warning(
                "%.2f%% of land_use cells have unknown codes; using fallback alpha=-0.0438",
                frac_no_match * 100,
            )
        result = np.where(no_match, np.exp(-0.0438 * sc), result)
        return result

    elif method == "full_cog":
        return np.exp(-0.0438 * sc)

    else:
        raise ValueError(f"Unknown method: {method}")
