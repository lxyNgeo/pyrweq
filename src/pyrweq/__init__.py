"""pyrweq - Python implementation of the Revised Wind Erosion Equation (RWEQ)."""

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.3.0"

from pyrweq.core import compute_rweq, compute_rweq_yearly
from pyrweq.sandfix import compute_sandfix
from pyrweq.classify import classify_erosion

__all__ = ["compute_rweq", "compute_rweq_yearly", "compute_sandfix", "classify_erosion"]
