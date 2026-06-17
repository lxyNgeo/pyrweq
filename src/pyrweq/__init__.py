"""pyrweq - Python implementation of the Revised Wind Erosion Equation (RWEQ)."""

__version__ = "0.1.0"

from pyrweq.core import compute_rweq
from pyrweq.sandfix import compute_sandfix
from pyrweq.classify import classify_erosion

__all__ = ["compute_rweq", "compute_sandfix", "classify_erosion"]
