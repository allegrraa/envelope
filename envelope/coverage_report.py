"""Honest coverage statistics.

* Clopper-Pearson (exact binomial) intervals for coverage measured on a held-out set.
* The finite-sample law of split-conformal coverage: conditional on the calibration set,
  coverage ~ Beta(n + 1 - l, l) with l = floor((n + 1) * alpha) (exchangeable, continuous scores).
* A small-sample ("PAC"-style) correction: the largest alpha' <= alpha such that
  P(coverage >= 1 - alpha) >= 1 - delta under that Beta law.
"""
from __future__ import annotations

import math

from scipy.stats import beta as beta_dist


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval for k successes out of n trials."""
    if n <= 0:
        return float("nan"), float("nan")
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def beta_params(n: int, alpha: float) -> tuple[int, int]:
    """Beta(a, b) parameters of split-conformal coverage for n calibration points.

    l = 0 means the conformal quantile is +inf (coverage 1); returned as (n + 1, 0).
    """
    l = math.floor((n + 1) * alpha)
    return n + 1 - l, l


def conformal_coverage_law(n: int, alpha: float, lower_q: float = 0.05) -> dict[str, float]:
    """Mean and lower quantile of split-conformal coverage over calibration draws."""
    a, b = beta_params(n, alpha)
    if b == 0:
        return {"n": n, "alpha": alpha, "l": 0, "mean": 1.0, "p05": 1.0, "prob_at_least_target": 1.0}
    return {
        "n": n,
        "alpha": alpha,
        "l": b,
        "mean": a / (a + b),
        "p05": float(beta_dist.ppf(lower_q, a, b)),
        "prob_at_least_target": float(beta_dist.sf(1 - alpha, a, b)),
    }


def corrected_alpha(n: int, alpha: float, delta: float = 0.1) -> float:
    """Largest alpha' = l'/(n+1) <= alpha with P(Beta(n+1-l', l') >= 1 - alpha) >= 1 - delta.

    Returns 0.0 when no finite quantile achieves it (n too small): intervals become infinite.
    """
    l_max = math.floor((n + 1) * alpha)
    for l in range(l_max, 0, -1):
        if beta_dist.sf(1 - alpha, n + 1 - l, l) >= 1 - delta:
            return l / (n + 1)
    return 0.0
