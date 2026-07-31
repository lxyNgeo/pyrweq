"""Soil erodibility factor (EF) calculation for RWEQ."""

from __future__ import annotations

import logging

import numpy as np

from pyrweq._types import FactorArray

logger = logging.getLogger(__name__)

def calc_erodibility(
    sand: FactorArray,
    silt: FactorArray,
    clay: FactorArray,
    organic_matter: FactorArray,
    calcium_carbonate: FactorArray | None = None,
) -> FactorArray:
    """Calculate soil erodibility factor EF.

    EF = (29.09 + 0.31*Sa + 0.17*Si + 0.33*Sa/Cl - 2.59*OM - 0.95*CaCO3) / 100

    Parameters
    ----------
    sand, silt, clay : np.ndarray
        Sand, silt, clay content (%).
    organic_matter : np.ndarray
        Organic matter content (%). If you have organic carbon, multiply by 1.724.
    calcium_carbonate : np.ndarray or None
        CaCO3 content (%). If None, assumed 0.

    Returns
    -------
    np.ndarray
        Soil erodibility factor EF (dimensionless).
    """
    if calcium_carbonate is None:
        calcium_carbonate = np.zeros_like(sand)

    clay_safe = np.where(clay == 0, 1e-6, clay)

    frac_clay_zero = float(np.mean(clay == 0))
    if frac_clay_zero > 0:
        logger.info("clay==0 cells: %.2f%%; using eps=1e-6 to avoid div-by-zero", frac_clay_zero * 100)

    ef = (29.09 + 0.31 * sand + 0.17 * silt + 0.33 * sand / clay_safe
          - 2.59 * organic_matter - 0.95 * calcium_carbonate) / 100.0

    return np.clip(ef, 0.0, 1.0)
