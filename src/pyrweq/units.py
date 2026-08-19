"""Unit conversions between RWEQ-native and reporting units.

RWEQ (Fryear et al. 1998) natively reports soil loss SL in g/m: the mass
eroded from a strip of unit width (1 m) perpendicular to the prevailing
wind over the full downwind field length. China's SL190-2007 standard and
most publications report an areal erosion modulus in t/(km^2 * a).

Conversion between the two needs the grid cell size. Under the standard
assumptions (near-square cells, wind direction isotropy over the period)
the strip of unit width corresponds to one cell along-wind, so:

    modulus [t/(km^2 * a)] = SL [g/m] / cell_size [m]

Numerically 1 g/m^2 == 1 t/km^2 (1000 kg / 10^6 m^2), so dividing g/m by
the cell edge in metres directly yields t/(km^2 * a). For non-square
cells pass ``(xres, yres)`` and the mean is used.

Assumptions
-----------
- one period per year: for multi-period (e.g. monthly) runs the summed
  annual SL is the correct input; the per-period modulus would carry the
  same conversion.
- cells are near-square and wind direction is isotropic within the
  period; strongly anisotropic winds or strongly elongated cells make the
  1-D SL-to-areal mapping approximate. Validate against plot
  measurements where possible.
"""

from __future__ import annotations

import logging

import numpy as np

from pyrweq._types import FactorArray, RasterioProfile

logger = logging.getLogger(__name__)

# 1 g/m^2 == 1 t/km^2 exactly
_G_PER_M2_PER_T_PER_KM2 = 1.0


def _cell_size_value(cell_size: float | tuple[float, float]) -> float:
    """Normalise a scalar or (xres, yres) cell size to one positive float."""
    if isinstance(cell_size, (tuple, list)):
        if len(cell_size) != 2:
            raise ValueError(f"cell_size tuple must be (xres, yres), got {cell_size!r}")
        xres, yres = float(cell_size[0]), float(cell_size[1])
        if xres <= 0 or yres <= 0:
            raise ValueError(f"cell size must be positive, got ({xres}, {yres})")
        if abs(xres - yres) / max(xres, yres) > 0.05:
            logger.warning(
                "non-square cells (%.6g x %.6g); using the mean %.6g — "
                "verify the SL-to-modulus conversion for your wind regime",
                xres, yres, (xres + yres) / 2.0,
            )
        return (xres + yres) / 2.0
    size = float(cell_size)
    if size <= 0:
        raise ValueError(f"cell size must be positive, got {size}")
    return size


def g_per_m_to_t_per_km2(
    sl: FactorArray, cell_size: float | tuple[float, float]
) -> FactorArray:
    """Convert RWEQ soil loss SL (g/m) to an erosion modulus (t/(km^2 * a)).

    Parameters
    ----------
    sl : array
        Soil loss in g/m (calc_sl output; sand fixation differences share
        the unit).
    cell_size : float or (xres, yres)
        Grid cell edge length in metres.

    Returns
    -------
    array
        Erosion modulus in t/(km^2 * a); numerically equal to g/m^2.
    """
    size = _cell_size_value(cell_size)
    return sl / size


def t_per_km2_to_g_per_m(
    modulus: FactorArray, cell_size: float | tuple[float, float]
) -> FactorArray:
    """Convert an erosion modulus (t/(km^2 * a)) back to RWEQ SL (g/m)."""
    size = _cell_size_value(cell_size)
    return modulus * size


def cell_size_from_profile(profile: RasterioProfile) -> float:
    """Extract the (mean) cell size in metres from a rasterio profile.

    Only meaningful for projected CRS (metre units). Returns 1.0 for
    placeholder profiles without a transform so that conversions become
    no-ops rather than crashes.
    """
    transform = profile.get("transform") if profile else None
    if transform is None:
        logger.warning("profile has no transform; cell size unknown, returning 1.0")
        return 1.0
    try:
        xres = abs(transform.a)
        yres = abs(transform.e)
    except AttributeError:  # plain 6-tuple
        xres = abs(transform[0])
        yres = abs(transform[4])
    return _cell_size_value((xres, yres))


__all__ = [
    "g_per_m_to_t_per_km2",
    "t_per_km2_to_g_per_m",
    "cell_size_from_profile",
]
