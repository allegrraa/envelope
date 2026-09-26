"""Applicability domain ("validity envelope") via kNN distance in standardised feature space.

Two ways to flag a query as outside:

* percentile: score = distance / (percentile of leave-one-out training distances) > 1;
* conformal p-value: with calibration kNN distances c_1..c_n (calibration points are
  exchangeable with in-distribution queries), p(x) = (1 + #{c_i >= d(x)}) / (n + 1).
  Flag if p < beta. For in-distribution queries P(p <= beta) <= beta. For a batch of queries,
  Benjamini-Hochberg controls the false discovery rate of the OUTSIDE flags at level beta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class EnvelopeResult:
    """Per-query envelope output.

    ``distance``: mean distance to the k nearest training points (standardised units).
    ``score``: distance / threshold; score <= 1 means inside the envelope.
    """

    distance: np.ndarray
    score: np.ndarray
    inside: np.ndarray
    pvalue: np.ndarray | None = None


class ApplicabilityDomain:
    """kNN applicability domain.

    The threshold is the ``percentile``-th percentile of leave-one-out kNN distances of the
    training points to the rest of the training set. By construction roughly
    (100 - percentile)% of in-distribution points fall just outside, so the envelope is a
    screening flag, not a hard boundary.
    """

    def __init__(self, k: int = 5, percentile: float = 95.0) -> None:
        self.k = k
        self.percentile = percentile

    def fit(self, X_train: np.ndarray) -> "ApplicabilityDomain":
        X_train = np.asarray(X_train, dtype=float)
        if len(X_train) <= self.k:
            raise ValueError("need more training points than k")
        self.scaler = StandardScaler().fit(X_train)
        Z = self.scaler.transform(X_train)
        self.nn = NearestNeighbors(n_neighbors=self.k + 1).fit(Z)
        dist, _ = self.nn.kneighbors(Z)
        self.loo_distances = dist[:, 1:].mean(axis=1)  # drop self-match (distance 0)
        self.threshold = float(np.percentile(self.loo_distances, self.percentile))
        return self

    def calibrate(self, X_cal: np.ndarray) -> "ApplicabilityDomain":
        """Store kNN distances of calibration points (not used for training) for conformal p-values."""
        self.cal_distances = np.sort(self.distance(X_cal))
        return self

    def pvalues(self, X: np.ndarray) -> np.ndarray:
        """Conformal p-value (1 + #{cal >= d}) / (n + 1) of each query's kNN distance."""
        if not hasattr(self, "cal_distances"):
            raise RuntimeError("call calibrate(X_cal) first")
        d = self.distance(X)
        n = len(self.cal_distances)
        n_ge = n - np.searchsorted(self.cal_distances, d, side="left")
        return (1.0 + n_ge) / (n + 1.0)

    def distance(self, X: np.ndarray) -> np.ndarray:
        """Mean distance to the k nearest training points (standardised space)."""
        Z = self.scaler.transform(np.asarray(X, dtype=float))
        dist, _ = self.nn.kneighbors(Z, n_neighbors=self.k)
        return dist.mean(axis=1)

    def evaluate(self, X: np.ndarray) -> EnvelopeResult:
        d = self.distance(X)
        score = d / self.threshold
        p = self.pvalues(X) if hasattr(self, "cal_distances") else None
        return EnvelopeResult(distance=d, score=score, inside=score <= 1.0, pvalue=p)


def benjamini_hochberg(pvalues: np.ndarray, level: float = 0.05) -> np.ndarray:
    """Boolean mask of rejections (flagged OUTSIDE) under the Benjamini-Hochberg procedure."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    passed = p[order] <= level * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        k = int(np.max(np.nonzero(passed)[0]))
        reject[order[: k + 1]] = True
    return reject
