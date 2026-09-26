"""Assumption check: can a classifier tell calibration inputs from query inputs?

If calibration and query inputs are exchangeable, no classifier should separate them better
than chance (held-out AUC about 0.5). We pick logistic regression or a small random forest by
5-fold cross-validated AUC on a training split, report AUC on a held-out split, and a
permutation p-value: the held-out labels are permuted ``n_perm`` times with the fitted
classifier held fixed (a valid test conditional on the train/test split).

This is a diagnostic, not a proof: OK does not certify exchangeability, it only means this
test found no evidence against it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AUC_CUTOFF = 0.6
P_CUTOFF = 0.05


def shift_check(
    X_cal: np.ndarray,
    X_query: np.ndarray,
    seed: int = 0,
    n_perm: int = 200,
    test_size: float = 0.3,
    auc_cutoff: float = AUC_CUTOFF,
    p_cutoff: float = P_CUTOFF,
) -> dict[str, Any]:
    """Classifier two-sample test between calibration and query inputs.

    Returns a dict with ``auc``, ``p_value``, ``model``, ``cv_auc`` (per candidate), ``status``
    (``OK`` if auc < auc_cutoff and p > p_cutoff, else ``WARN``) and a one-line ``message``.
    """
    X_cal = np.asarray(X_cal, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    if len(X_cal) < 10 or len(X_query) < 10:
        return {"status": "SKIP", "auc": float("nan"), "p_value": float("nan"), "model": None, "cv_auc": {},
                "n_perm": n_perm, "message": "Too few points for the shift check (need >= 10 per group)."}
    X = np.vstack([X_cal, X_query])
    y = np.r_[np.zeros(len(X_cal)), np.ones(len(X_query))]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    candidates = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random forest": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=seed),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = {name: float(cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()) for name, m in candidates.items()}
    best = max(cv_auc, key=cv_auc.get)
    model = candidates[best].fit(X_tr, y_tr)
    scores = model.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, scores))
    rng = np.random.default_rng(seed)
    perm = np.array([roc_auc_score(rng.permutation(y_te), scores) for _ in range(n_perm)])
    p = float((1 + np.sum(perm >= auc)) / (n_perm + 1))
    status = "OK" if (auc < auc_cutoff and p > p_cutoff) else "WARN"
    if status == "OK":
        msg = (f"Calibration and query inputs look alike (held-out AUC {auc:.2f}, permutation p = {p:.3f}); "
               "no evidence against the exchangeability the coverage guarantee needs.")
    else:
        msg = (f"A {best} separates calibration from query inputs (held-out AUC {auc:.2f}, permutation p = {p:.3f}). "
               "The coverage guarantee may not apply to these queries.")
    return {"status": status, "auc": auc, "p_value": p, "model": best, "cv_auc": cv_auc, "n_perm": n_perm,
            "auc_cutoff": auc_cutoff, "p_cutoff": p_cutoff, "message": msg}
