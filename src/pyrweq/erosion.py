"""Wind erosion量 (SL) calculation for RWEQ."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calc_sl(
    wf: np.ndarray,
    ef: np.ndarray,
    scf: np.ndarray,
    k_prime: np.ndarray,
    c: np.ndarray | None = None,
    z: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate wind erosion量 SL and intermediate values.

    SL = (2z / s^2) * Qmax * exp(-(z/s)^2)
    s  = 150.71 * (WF * EF * SCF * K' * C)^(-0.3711)
    Qmax = 109.8 * (WF * EF * SCF * K' * C)

    Parameters
    ----------
    wf, ef, scf, k_prime : np.ndarray
        RWEQ factor arrays.
    c : np.ndarray or None
        Vegetation factor. If None, treated as bare soil (C=1).
    z : float
        Downwind distance (m), default 50.

    Returns
    -------
    sl : np.ndarray
        Wind erosion量 in RWEQ-native g/m (mass eroded per metre of
        field width). Convert to t/(km^2 * a) with
        ``pyrweq.units.g_per_m_to_t_per_km2`` before classifying.
    s : np.ndarray
        Critical field length (m).
    qmax : np.ndarray
        Maximum transport capacity (g/m).
    """
    if c is None:
        product = wf * ef * scf * k_prime
    else:
        product = wf * ef * scf * k_prime * c
    product = np.where(product <= 0, 1e-10, product)

    frac_zero = float(np.mean(product <= 1e-9))
    if frac_zero > 0.01:
        logger.warning(
            "%.2f%% of cells had product<=0; clipped to 1e-10. Inspect WF/EF/SCF/K'/C inputs.",
            frac_zero * 100,
        )

    s = 150.71 * product ** (-0.3711)
    qmax = 109.8 * product

    ratio = z / s
    sl = (2.0 * z / s ** 2) * qmax * np.exp(-(ratio ** 2))

    return sl, s, qmax
