"""Split conformal prediction, implemented from scratch.

Three variants, all using the finite-sample quantile at level ceil((n+1)(1-alpha))/n:

* :class:`SplitConformal`      score |y - yhat|,               interval yhat +/- q
* :class:`AdaptiveConformal`   score |y - yhat| / s(x),        interval yhat +/- q * s(x)
  where s(x) is a small model of |residual| fitted on a disjoint half of the calibration set
* :class:`NormalizedConformal` score |y - yhat| / (sigma(x)+beta), interval yhat +/- q * (sigma(x)+beta)
  where sigma(x) comes from weight-perturbation fragility (white-box models only)

The marginal coverage guarantee P(y in interval) >= 1 - alpha holds only when calibration and
query points are exchangeable. It does NOT hold under distribution shift (e.g. extrapolation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: the k-th smallest score with k = ceil((n+1)(1-alpha)).

    Returns ``inf`` when k > n (too few calibration points for the requested alpha).
    """
    scores = np.sort(np.asarray(scores, dtype=float).ravel())
    n = len(scores)
    if n == 0:
        raise ValueError("no calibration scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil((n + 1) * (1.0 - alpha) - 1e-9)  # guard: alpha may be an exact multiple of 1/(n+1)
    if k > n:
        return float("inf")
    return float(scores[k - 1])


@dataclass
class Intervals:
    """Prediction intervals."""

    lower: np.ndarray
    upper: np.ndarray

    @property
    def half_width(self) -> np.ndarray:
        return (self.upper - self.lower) / 2.0

    def covers(self, y: np.ndarray) -> np.ndarray:
        return (y >= self.lower) & (y <= self.upper)


class SplitConformal:
    """Standard split conformal with absolute-residual scores (constant width)."""

    name = "split"

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.q: float | None = None

    def fit(self, y_cal: np.ndarray, yhat_cal: np.ndarray) -> "SplitConformal":
        self.q = conformal_quantile(np.abs(np.asarray(y_cal) - np.asarray(yhat_cal)), self.alpha)
        return self

    def predict(self, yhat: np.ndarray) -> Intervals:
        assert self.q is not None, "call fit first"
        yhat = np.asarray(yhat, dtype=float)
        return Intervals(yhat - self.q, yhat + self.q)


class AdaptiveConformal:
    """Locally adaptive split conformal.

    The calibration set is split in two: half A fits a shallow gradient-boosting model of
    |residual| (the local scale s(x)); half B computes normalised scores |r| / s(x) and the
    conformal quantile. Using disjoint halves keeps the finite-sample guarantee intact.
    """

    name = "adaptive"

    def __init__(self, alpha: float = 0.1, seed: int = 0, fit_fraction: float = 0.5) -> None:
        self.alpha = alpha
        self.seed = seed
        self.fit_fraction = fit_fraction
        self.q: float | None = None

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray, yhat_cal: np.ndarray) -> "AdaptiveConformal":
        X_cal = np.asarray(X_cal, dtype=float)
        abs_res = np.abs(np.asarray(y_cal) - np.asarray(yhat_cal))
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(abs_res))
        n_a = int(len(perm) * self.fit_fraction)
        a, b = perm[:n_a], perm[n_a:]
        self.scaler = StandardScaler().fit(X_cal[a])
        self.scale_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=self.seed
        ).fit(self.scaler.transform(X_cal[a]), abs_res[a])
        self.floor = 0.05 * float(abs_res[a].mean()) + 1e-12
        self.q = conformal_quantile(abs_res[b] / self.scale(X_cal[b]), self.alpha)
        return self

    def scale(self, X: np.ndarray) -> np.ndarray:
        """Local residual scale s(x), floored to avoid division by ~0."""
        return np.maximum(self.scale_model.predict(self.scaler.transform(np.asarray(X, dtype=float))), self.floor)

    def predict(self, X: np.ndarray, yhat: np.ndarray) -> Intervals:
        assert self.q is not None, "call fit first"
        w = self.q * self.scale(X)
        yhat = np.asarray(yhat, dtype=float)
        return Intervals(yhat - w, yhat + w)


class NormalizedConformal:
    """Fragility-normalised split conformal: score |y - yhat| / (sigma(x) + beta)."""

    name = "fragility"

    def __init__(self, alpha: float = 0.1, beta: float | None = None) -> None:
        self.alpha = alpha
        self.beta = beta
        self.q: float | None = None

    def fit(self, y_cal: np.ndarray, yhat_cal: np.ndarray, sigma_cal: np.ndarray) -> "NormalizedConformal":
        sigma_cal = np.asarray(sigma_cal, dtype=float)
        if self.beta is None:
            # default stabiliser: 10% of the median calibration sigma
            self.beta = 0.1 * float(np.median(sigma_cal)) + 1e-12
        scores = np.abs(np.asarray(y_cal) - np.asarray(yhat_cal)) / (sigma_cal + self.beta)
        self.q = conformal_quantile(scores, self.alpha)
        return self

    def predict(self, yhat: np.ndarray, sigma: np.ndarray) -> Intervals:
        assert self.q is not None and self.beta is not None, "call fit first"
        w = self.q * (np.asarray(sigma, dtype=float) + self.beta)
        yhat = np.asarray(yhat, dtype=float)
        return Intervals(yhat - w, yhat + w)


def quantile_bin_edges(scores: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Interior edges (n_bins - 1 values) at equally spaced quantiles of ``scores``."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.quantile(np.asarray(scores, dtype=float), qs)


class GroupConditionalConformal:
    """Group-conditional (Mondrian) split conformal over bins of a group score (here the envelope score).

    Each bin gets its own conformal quantile of |y - yhat| from the calibration points that fall in it;
    bins with fewer than ``min_count`` calibration points fall back to the global quantile (and then
    carry no bin-conditional guarantee). The bin-wise guarantee also assumes exchangeability within bins.
    """

    name = "group"

    def __init__(self, alpha: float = 0.1, min_count: int = 30) -> None:
        self.alpha = alpha
        self.min_count = min_count

    def fit(self, y_cal: np.ndarray, yhat_cal: np.ndarray, group_cal: np.ndarray, edges: np.ndarray) -> "GroupConditionalConformal":
        self.edges = np.asarray(edges, dtype=float)
        res = np.abs(np.asarray(y_cal) - np.asarray(yhat_cal))
        bins = np.digitize(group_cal, self.edges)
        self.global_q = conformal_quantile(res, self.alpha)
        self.bins: list[dict] = []
        for b in range(len(self.edges) + 1):
            r = res[bins == b]
            use_bin = len(r) >= self.min_count
            self.bins.append({
                "bin": b,
                "n_cal": int(len(r)),
                "q": conformal_quantile(r, self.alpha) if use_bin else self.global_q,
                "source": "bin" if use_bin else "global fallback",
            })
        return self

    def assign(self, group: np.ndarray) -> np.ndarray:
        return np.digitize(np.asarray(group, dtype=float), self.edges)

    def predict(self, yhat: np.ndarray, group: np.ndarray) -> Intervals:
        q = np.array([self.bins[b]["q"] for b in self.assign(group)])
        yhat = np.asarray(yhat, dtype=float)
        return Intervals(yhat - q, yhat + q)
