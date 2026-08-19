"""pyrweq - Python implementation of the Revised Wind Erosion Equation (RWEQ)."""

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.3.0"

from pyrweq.core import compute_rweq, compute_rweq_yearly
from pyrweq.sandfix import compute_sandfix
from pyrweq.classify import classify_erosion
from pyrweq.units import (
    g_per_m_to_t_per_km2,
    t_per_km2_to_g_per_m,
    cell_size_from_profile,
)
from pyrweq.io import align_inputs
from pyrweq.validate import validate, sample_points
from pyrweq.sensitivity import oat_sensitivity

__all__ = [
    "compute_rweq",
    "compute_rweq_yearly",
    "compute_sandfix",
    "classify_erosion",
    "g_per_m_to_t_per_km2",
    "t_per_km2_to_g_per_m",
    "cell_size_from_profile",
    "align_inputs",
    "validate",
    "sample_points",
    "oat_sensitivity",
]
