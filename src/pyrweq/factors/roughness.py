"""Surface roughness factor (K') calculation for RWEQ."""

import numpy as np


def calc_roughness_simple(slope_deg: np.ndarray) -> np.ndarray:
    """Simplified roughness factor using slope.

    K' = cos(alpha)

    Parameters
    ----------
    slope_deg : np.ndarray
        Slope angle in degrees.

    Returns
    -------
    np.ndarray
        Roughness factor K'.
    """
    return np.cos(np.radians(slope_deg))


def calc_roughness_full(
    slope_deg: np.ndarray,
    elevation: np.ndarray,
    terrain_level: str = "gentle",
    random_roughness: float = 0.0,
) -> np.ndarray:
    """Full roughness factor using Smith-Carson equation.

    K' = exp(1.86*Kr - 2.41*Kr^0.934 - 0.127*Crr)
    Kr = 0.2 * (dH)^2 / L

    Parameters
    ----------
    slope_deg : np.ndarray
        Slope angle in degrees.
    elevation : np.ndarray
        Elevation in meters.
    terrain_level : str
        Terrain level: "micro", "gentle", "moderate", "mountain", "alpine".
    random_roughness : float
        Random roughness Crr (cm), default 0.

    Returns
    -------
    np.ndarray
        Roughness factor K'.
    """
    L_map = {
        "micro": 5.0,
        "gentle": 5.0,
        "moderate": 10.0,
        "mountain": 10.0,
        "alpine": 50.0,
    }
    L = L_map.get(terrain_level, 5.0)

    tan_alpha = np.tan(np.radians(slope_deg))
    delta_h = tan_alpha * L * 1000.0
    kr = 0.2 * delta_h ** 2 / L

    kr = np.where(kr == 0, 1e-6, kr)

    k_prime = np.exp(1.86 * kr - 2.41 * kr ** 0.934 - 0.127 * random_roughness)
    return k_prime
