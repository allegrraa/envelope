"""Evaluation metrics: accuracy, coverage, flag quality, tilted/tail/worst-region risk."""
from __future__ import annotations

import numpy as np

from .term import tilted_risk_np


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of targets inside [lower, upper] (NaN for empty input)."""
    y = np.asarray(y)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((y >= lower) & (y <= upper)))


def mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    if len(lower) == 0:
        return float("nan")
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def flag_precision_recall(flagged: np.ndarray, abs_err: np.ndarray, err_tol: float) -> dict[str, float]:
    """How well a flag (e.g. outside-envelope) catches large errors (|error| > err_tol)."""
    flagged = np.asarray(flagged, dtype=bool)
    large = np.asarray(abs_err) > err_tol
    tp = int(np.sum(flagged & large))
    precision = tp / flagged.sum() if flagged.sum() else float("nan")
    recall = tp / large.sum() if large.sum() else float("nan")
    return {"precision": float(precision), "recall": float(recall), "n_large_errors": int(large.sum()), "n_flagged": int(flagged.sum())}


def tilted_risk(abs_err: np.ndarray, t: float) -> float:
    """Tilted risk of absolute errors (target units): (1/t) log mean exp(t |e|); mean at t=0."""
    return tilted_risk_np(np.abs(abs_err), t)


def tail_risk(abs_err: np.ndarray, fraction: float = 0.1) -> float:
    """Mean of the worst ``fraction`` of absolute errors."""
    e = np.sort(np.abs(np.asarray(abs_err, dtype=float)))[::-1]
    k = max(1, int(np.ceil(fraction * len(e))))
    return float(e[:k].mean())


def worst_region_error(
    X: np.ndarray, y: np.ndarray, yhat: np.ndarray, bins: int = 4, min_count: int = 5
) -> dict[str, object]:
    """Split each input dimension into ``bins`` equal-width bins over the data range and return
    the maximum per-cell RMSE (cells with fewer than ``min_count`` points are ignored)."""
    X = np.asarray(X, dtype=float)
    err2 = (np.asarray(y) - np.asarray(yhat)) ** 2
    lo, hi = X.min(0), X.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    idx = np.clip(((X - lo) / span * bins).astype(int), 0, bins - 1)
    keys = [tuple(r) for r in idx]
    cells: dict[tuple, list[float]] = {}
    for k, e in zip(keys, err2):
        cells.setdefault(k, []).append(e)
    best_key, best = None, -1.0
    for k, errs in cells.items():
        if len(errs) < min_count:
            continue
        val = float(np.sqrt(np.mean(errs)))
        if val > best:
            best_key, best = k, val
    if best_key is None:
        return {"rmse": float("nan"), "cell": None, "bounds": None}
    width = span / bins
    bounds = [(float(lo[d] + best_key[d] * width[d]), float(lo[d] + (best_key[d] + 1) * width[d])) for d in range(X.shape[1])]
    return {"rmse": best, "cell": best_key, "bounds": bounds, "n": len(cells[best_key])}
