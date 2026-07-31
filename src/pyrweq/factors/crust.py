"""Soil crust factor (SCF) calculation for RWEQ."""

from __future__ import annotations

import logging

import numpy as np

from pyrweq._types import FactorArray

logger = logging.getLogger(__name__)

def calc_crust_factor(clay: FactorArray, organic_matter: FactorArray) -> FactorArray:
    """Calculate soil crust factor SCF.

    SCF = 1 / (1 + 0.0066 * Cl^2 + 0.021 * OM^2)

    Parameters
    ----------
    clay : np.ndarray
        Clay content (%).
    organic_matter : np.ndarray
        Organic matter content (%).

    Returns
    -------
    np.ndarray
        Soil crust factor SCF (dimensionless).
    """
    return 1.0 / (1.0 + 0.0066 * clay ** 2 + 0.021 * organic_matter ** 2)
