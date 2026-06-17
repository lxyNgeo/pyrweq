"""Wind erosion量 (SL) calculation for RWEQ."""

import numpy as np


def calc_sl(
    wf: np.ndarray,
    ef: np.ndarray,
    scf: np.ndarray,
    k_prime: np.ndarray,
    c: np.ndarray,
    z: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate wind erosion量 SL and intermediate values.

    SL = (2z / s^2) * Qmax * exp(-(z/s)^2)
    s  = 150.71 * (WF * EF * SCF * K' * C)^(-0.3711)
    Qmax = 109.8 * (WF * EF * SCF * K' * C)

    Parameters
    ----------
    wf, ef, scf, k_prime, c : np.ndarray
        RWEQ factor arrays.
    z : float
        Downwind distance (m), default 50.

    Returns
    -------
    sl : np.ndarray
        Wind erosion量 (same units as input factors produce).
    s : np.ndarray
        Critical field length (m).
    qmax : np.ndarray
        Maximum transport capacity.
    """
    product = wf * ef * scf * k_prime * c
    product = np.where(product <= 0, 1e-10, product)

    s = 150.71 * product ** (-0.3711)
    qmax = 109.8 * product

    ratio = z / s
    sl = (2.0 * z / s ** 2) * qmax * np.exp(-(ratio ** 2))

    return sl, s, qmax
