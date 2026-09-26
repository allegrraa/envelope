"""Intake of a company's own files: detect / map columns to the pipeline's canonical names.

Three tables, as exported from any model (the model itself is never uploaded):

* training: the model's training inputs (only used to define the validity envelope)
* calibration: held-out inputs with the true value and the model's prediction
* query: the inputs you want assessed, with the model's prediction (true value optional)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_GUESSES = ("y_true", "y", "target", "actual", "true", "label", "observed")
PRED_GUESSES = ("y_pred", "pred", "prediction", "predicted", "yhat", "y_hat", "forecast")


@dataclass
class ColumnMapping:
    features: list[str]
    target: str            # true-value column in the calibration table
    prediction: str        # prediction column in calibration and query tables
    query_target: str | None = None  # optional true-value column in the query table


def _numeric(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _guess(cols: list[str], names: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def guess_mapping(train: pd.DataFrame, cal: pd.DataFrame, query: pd.DataFrame) -> ColumnMapping:
    """Best-effort guess; the UI lets the user correct every field."""
    cal_num = _numeric(cal)
    target = _guess(cal_num, TARGET_GUESSES) or (cal_num[-2] if len(cal_num) >= 2 else "")
    prediction = _guess(cal_num, PRED_GUESSES) or (cal_num[-1] if cal_num else "")
    q_target = _guess(_numeric(query), TARGET_GUESSES)
    exclude = {target, prediction, q_target} | set(TARGET_GUESSES) | set(PRED_GUESSES)
    common = [c for c in _numeric(train) if c in cal.columns and c in query.columns and c.lower() not in exclude and c not in exclude]
    return ColumnMapping(features=common, target=target, prediction=prediction, query_target=q_target)


def apply_mapping(
    train: pd.DataFrame, cal: pd.DataFrame, query: pd.DataFrame, m: ColumnMapping
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the three tables with canonical column names (features, y_true, y_pred).

    Raises ValueError with a user-readable message when the mapping is unusable.
    """
    problems = []
    if not m.features:
        problems.append("choose at least one input (feature) column")
    for name, df, cols in (
        ("training", train, m.features),
        ("calibration", cal, m.features + [m.target, m.prediction]),
        ("query", query, m.features + [m.prediction] + ([m.query_target] if m.query_target else [])),
    ):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            problems.append(f"{name} file is missing column(s) {missing}")
    if m.target == m.prediction:
        problems.append("the true-value and prediction columns must be different")
    if m.target in m.features or m.prediction in m.features:
        problems.append("true-value / prediction columns cannot also be inputs")
    if problems:
        raise ValueError("; ".join(problems))
    tr = train[m.features].copy()
    ca = cal[m.features].copy()
    ca["y_true"], ca["y_pred"] = cal[m.target].to_numpy(), cal[m.prediction].to_numpy()
    qu = query[m.features].copy()
    qu["y_pred"] = query[m.prediction].to_numpy()
    if m.query_target:
        qu["y_true"] = query[m.query_target].to_numpy()
    return tr, ca, qu


def suggest_tolerance(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """A starting value for the acceptable uncertainty: 1.25x the (1 - alpha) quantile of calibration
    absolute errors, rounded to 2 significant figures. Users should replace it with what their
    application can actually tolerate."""
    err = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return 1.0
    v = 1.25 * float(np.quantile(err, 1 - alpha))
    if v <= 0:
        return 1e-6
    return float(f"{v:.2g}")
