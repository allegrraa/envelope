"""Synthetic "expensive simulator", demo data splits, CSV loading and verification checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

FEATURES: list[str] = ["x1", "x2", "x3"]
TRAIN_RANGE: tuple[float, float] = (0.0, 5.0)
TEST_MAX: float = 8.0
REGIME_THRESHOLD: float = 5.0
RESERVED_COLUMNS = {"y", "y_true", "y_pred"}


def simulator(X: np.ndarray) -> np.ndarray:
    """Noise-free "expensive simulator" response.

    Smooth and nonlinear in all three inputs. A stiffening term
    ``0.9 * max(0, x1 - 5)^2 * (1 + 0.15 * x2)`` only activates for x1 > 5, i.e. the
    physics changes regime outside the region the surrogate is trained on.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    base = 2.0 * np.sin(0.9 * x1) + 0.4 * x2 + 0.3 * x1 * np.cos(0.6 * x3) + 0.1 * x2 * x3
    stiffening = 0.9 * np.maximum(0.0, x1 - REGIME_THRESHOLD) ** 2 * (1.0 + 0.15 * x2)
    return base + stiffening


def noise_sd(X: np.ndarray) -> np.ndarray:
    """Heteroscedastic observation noise (grows with x2)."""
    X = np.atleast_2d(X)
    return 0.1 + 0.04 * X[:, 1]


def observe(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noisy simulator observations."""
    return simulator(X) + rng.normal(0.0, 1.0, len(X)) * noise_sd(X)


@dataclass
class DemoData:
    """All splits of the synthetic demo problem."""

    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_ref: np.ndarray
    y_ref: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _sample(rng: np.random.Generator, n: int, x1_max: float) -> np.ndarray:
    lo, hi = TRAIN_RANGE
    X = rng.uniform(lo, hi, size=(n, 3))
    X[:, 0] = rng.uniform(lo, x1_max, size=n)
    return X


def make_demo_data(
    seed: int = 0, n_train: int = 800, n_cal: int = 500, n_ref: int = 300, n_test: int = 800
) -> DemoData:
    """Generate train / calibration / reference (all inside [0,5]^3) and a test set with x1 up to 8."""
    rng = np.random.default_rng(seed)
    hi = TRAIN_RANGE[1]
    X_train, X_cal, X_ref = (_sample(rng, n, hi) for n in (n_train, n_cal, n_ref))
    X_test = _sample(rng, n_test, TEST_MAX)
    return DemoData(
        feature_names=list(FEATURES),
        X_train=X_train,
        y_train=observe(X_train, rng),
        X_cal=X_cal,
        y_cal=observe(X_cal, rng),
        X_ref=X_ref,
        y_ref=observe(X_ref, rng),
        X_test=X_test,
        y_test=observe(X_test, rng),
    )


def slice_grid(n: int = 200, x_max: float = TEST_MAX, fixed: float = 2.5) -> np.ndarray:
    """1-D slice through input space: x1 in [0, x_max], x2 = x3 = ``fixed``."""
    X = np.full((n, 3), fixed)
    X[:, 0] = np.linspace(0.0, x_max, n)
    return X


def corrupt_labels(
    y: np.ndarray, fraction: float, rng: np.random.Generator, magnitude: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return a copy of ``y`` where ``fraction`` of labels get a large random offset, plus the mask."""
    y = np.asarray(y, dtype=float).copy()
    n_bad = int(round(fraction * len(y)))
    idx = rng.choice(len(y), size=n_bad, replace=False)
    mag = magnitude if magnitude is not None else 4.0 * float(np.std(y))
    y[idx] += rng.choice([-1.0, 1.0], size=n_bad) * mag * rng.uniform(0.75, 1.25, size=n_bad)
    mask = np.zeros(len(y), dtype=bool)
    mask[idx] = True
    return y, mask


# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single verification check. ``status`` is PASS, WARN, FAIL or SKIP."""

    name: str
    status: str
    detail: str


def check_schema(frames: dict[str, pd.DataFrame], required: dict[str, Iterable[str]]) -> list[CheckResult]:
    """Required columns are present and numeric."""
    out: list[CheckResult] = []
    for name, cols in required.items():
        df = frames[name]
        cols = list(cols)
        missing = [c for c in cols if c not in df.columns]
        non_numeric = [c for c in cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
        if missing or non_numeric:
            out.append(CheckResult(f"schema[{name}]", "FAIL", f"missing={missing}, non-numeric={non_numeric}"))
        else:
            out.append(CheckResult(f"schema[{name}]", "PASS", f"{len(df)} rows; columns {cols} present and numeric"))
    return out


def check_missing(frames: dict[str, pd.DataFrame], required: dict[str, Iterable[str]]) -> list[CheckResult]:
    """No NaN / infinite values in the required columns."""
    out: list[CheckResult] = []
    for name, cols in required.items():
        df = frames[name]
        cols = [c for c in cols if c in df.columns]
        vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        n_nan = int(np.isnan(vals).sum())
        n_inf = int(np.isinf(vals).sum())
        status = "PASS" if n_nan == 0 and n_inf == 0 else "FAIL"
        out.append(CheckResult(f"nan/inf[{name}]", status, f"NaN={n_nan}, inf={n_inf}"))
    return out


def check_units(
    frames: dict[str, pd.DataFrame],
    bounds: dict[str, list[float]],
    units: dict[str, str],
    target_columns: Iterable[str] = ("y", "y_true", "y_pred"),
) -> list[CheckResult]:
    """Unit/plausibility checks.

    * Values lie within declared physical bounds (target columns use the ``y`` bound).
    * ``y_pred`` and ``y_true`` are on a comparable scale (a ratio of spreads far from 1
      is a classic symptom of a unit mismatch, e.g. Pa vs MPa).
    """
    out: list[CheckResult] = []
    target_columns = set(target_columns)
    for name, df in frames.items():
        for col in df.columns:
            key = "y" if col in target_columns else col
            if key not in bounds:
                continue
            lo, hi = bounds[key]
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            n_out = int(((vals < lo) | (vals > hi)).sum())
            unit = units.get(key, "?")
            status = "PASS" if n_out == 0 else "WARN"
            out.append(
                CheckResult(f"bounds[{name}.{col}]", status, f"{n_out} values outside [{lo}, {hi}] {unit}")
            )
    for name, df in frames.items():
        if {"y_true", "y_pred"} <= set(df.columns):
            s_true = float(np.nanstd(df["y_true"]))
            s_pred = float(np.nanstd(df["y_pred"]))
            ratio = s_pred / s_true if s_true > 0 else np.inf
            status = "PASS" if 0.2 <= ratio <= 5.0 else "FAIL"
            out.append(
                CheckResult(f"unit-scale[{name}]", status, f"std(y_pred)/std(y_true) = {ratio:.3g} (expect ~1)")
            )
    if not out:
        out.append(CheckResult("units", "SKIP", "no units/bounds declared for these columns"))
    return out


def run_verification(
    frames: dict[str, pd.DataFrame],
    required: dict[str, Iterable[str]],
    bounds: dict[str, list[float]] | None = None,
    units: dict[str, str] | None = None,
) -> list[CheckResult]:
    """Run schema, missing-value and unit checks."""
    required = {k: list(v) for k, v in required.items()}
    return (
        check_schema(frames, required)
        + check_missing(frames, required)
        + check_units(frames, bounds or {}, units or {})
    )


# ---------------------------------------------------------------------------
# Upload mode
# ---------------------------------------------------------------------------


@dataclass
class ExternalData:
    """Outputs of an external (black-box) model, loaded from three CSVs."""

    feature_names: list[str]
    X_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    yhat_cal: np.ndarray
    X_query: np.ndarray
    yhat_query: np.ndarray
    y_query: np.ndarray | None
    checks: list[CheckResult] = field(default_factory=list)
    n_dropped: int = 0


def load_external(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bounds: dict[str, list[float]] | None = None,
    units: dict[str, str] | None = None,
) -> ExternalData:
    """Parse the three upload CSVs.

    * training: feature columns (any ``y``/``y_true``/``y_pred`` columns are ignored)
    * calibration: features + ``y_true`` + ``y_pred``
    * query: features + ``y_pred`` (optional ``y_true`` enables empirical evaluation)

    Rows with NaN in required columns are dropped (and reported as a failed check).
    Raises ``ValueError`` when the schema is unusable.
    """
    features = [c for c in train_df.columns if c not in RESERVED_COLUMNS]
    if not features:
        raise ValueError("training CSV has no feature columns")
    frames = {"training": train_df, "calibration": cal_df, "query": query_df}
    required = {
        "training": features,
        "calibration": features + ["y_true", "y_pred"],
        "query": features + ["y_pred"],
    }
    checks = run_verification(frames, required, bounds, units)
    schema_fail = [c for c in checks if c.name.startswith("schema") and c.status == "FAIL"]
    if schema_fail:
        raise ValueError("; ".join(f"{c.name}: {c.detail}" for c in schema_fail))

    n_dropped = 0

    def clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        nonlocal n_dropped
        df2 = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
        n_dropped += len(df) - len(df2)
        return df2

    tr = clean(train_df, features)
    ca = clean(cal_df, required["calibration"])
    qcols = required["query"] + (["y_true"] if "y_true" in query_df.columns else [])
    qu = clean(query_df, qcols)
    return ExternalData(
        feature_names=features,
        X_train=tr[features].to_numpy(float),
        X_cal=ca[features].to_numpy(float),
        y_cal=ca["y_true"].to_numpy(float),
        yhat_cal=ca["y_pred"].to_numpy(float),
        X_query=qu[features].to_numpy(float),
        yhat_query=qu["y_pred"].to_numpy(float),
        y_query=qu["y_true"].to_numpy(float) if "y_true" in qu.columns else None,
        checks=checks,
        n_dropped=n_dropped,
    )
