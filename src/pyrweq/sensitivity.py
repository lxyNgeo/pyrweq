"""One-at-a-time (OAT) parameter sensitivity analysis.

Reviewers of wind-erosion modelling papers routinely ask how sensitive
the results are to parameter choices (threshold wind speed, downwind
distance, period length, ...). This module re-runs a user-supplied
computation while perturbing one parameter at a time and reports the
relative response of a scalar summary of the result.

The design is deliberately generic: ``fn`` is any callable that accepts
keyword arguments and returns an array (e.g. a closure over
``compute_rweq`` returning ``result.sl``). This keeps the sensitivity
machinery independent of the RWEQ call signature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

# relative perturbations applied when no explicit values are given
DEFAULT_DELTAS = (-0.20, -0.10, 0.10, 0.20)


def oat_sensitivity(
    fn: Callable[..., np.ndarray],
    params: dict[str, float],
    values: dict[str, list] | None = None,
    deltas: tuple = DEFAULT_DELTAS,
    summary: str = "mean",
    base_kwargs: dict | None = None,
) -> dict:
    """One-at-a-time sensitivity of a scalar summary to each parameter.

    Parameters
    ----------
    fn : callable
        ``fn(**kwargs) -> array``; receives ``base_kwargs`` merged with the
        perturbed parameter, e.g.
        ``lambda **kw: compute_rweq(**inputs, **kw).sl``.
    params : dict
        Base values per parameter, e.g.
        ``{"threshold_speed": 5.0, "downwind_distance": 50.0}``.
    values : dict or None
        Explicit values to test per parameter (overrides ``deltas``).
        Keys must be a subset of ``params``.
    deltas : tuple of float
        Relative perturbations of the base value used when no explicit
        ``values`` are given (default -20, -10, +10, +20 percent).
    summary : str
        How to reduce the returned array to a scalar: "mean", "sum",
        "max" or "nanmean" (default "mean"; NaN cells are always ignored).
    base_kwargs : dict or None
        Extra fixed keyword arguments passed to ``fn`` on every call.

    Returns
    -------
    dict
        ``{param: {"base": s0, "values": [...], "summary": [...],
        "elasticity": {...}}}`` where ``elasticity`` maps each tested
        value to ``(s - s0) / s0 / ((v - v0) / v0)`` — the percent change
        of the summary per percent change of the parameter.
    """
    if not params:
        raise ValueError("params must contain at least one parameter")
    base_kwargs = dict(base_kwargs or {})
    summaries = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax, "nanmean": np.nanmean}
    if summary not in summaries:
        raise ValueError(f"unknown summary {summary!r}; choose from {sorted(summaries)}")
    reduce_fn = summaries[summary]

    values = values or {}
    unknown = set(values) - set(params)
    if unknown:
        raise ValueError(f"values keys {sorted(unknown)} are not in params")

    def _run(**overrides) -> float:
        kwargs = {**base_kwargs, **params, **overrides}
        arr = fn(**kwargs)
        return float(reduce_fn(np.asarray(arr, dtype=np.float64)))

    results: dict = {}
    base_all = _run()
    for name, base in params.items():
        tested = values.get(name) or [base * (1.0 + d) for d in deltas]
        if base == 0:
            if name not in values:
                raise ValueError(
                    f"parameter {name!r} has base value 0; relative deltas are "
                    "undefined — pass explicit values"
                )
            tested = values[name]
        s_values = [_run(**{name: v}) for v in tested]
        elasticity = {}
        for v, s in zip(tested, s_values):
            dv = (v - base) / base if base != 0 else float("nan")
            elasticity[v] = (s - base_all) / base_all / dv if dv != 0 and base_all != 0 else float("nan")
        results[name] = {
            "base": base_all,
            "values": list(tested),
            "summary": s_values,
            "elasticity": elasticity,
        }
        logger.info(
            "OAT %s: base=%.6g, summary range [%.6g, %.6g]",
            name, base_all, min(s_values), max(s_values),
        )
    return results


__all__ = ["oat_sensitivity"]
