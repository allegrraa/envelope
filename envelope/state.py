"""Save / load the calibrated state used by the API.

The state file is a single ``.npz`` holding the raw arrays (training features, calibration
features / labels / predictions) plus a JSON ``meta`` string (config, feature names, method,
envelope threshold, gate cutoffs, report path). On load the engine is refitted
deterministically and the envelope threshold is checked against the saved value.
No pickles: the file is loaded with ``allow_pickle=False``.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gate import explain_decision

STATE_VERSION = 1


@dataclass
class CalibratedState:
    """A loaded, ready-to-query calibration state."""

    cfg: dict[str, Any]
    feature_names: list[str]
    method: str
    engine: Any  # EnvelopeEngine (imported lazily to avoid a cycle)
    report_path: Path | None
    created: str
    notes: list[str]

    @property
    def alpha(self) -> float:
        return float(self.engine.alpha)

    def assumptions(self) -> str:
        from .pipeline import METHOD_LABELS

        return (
            f"{METHOD_LABELS[self.method]} at alpha={self.alpha:g}: the interval contains the true value with "
            f"probability >= {1 - self.alpha:.0%} only on average over queries exchangeable with the calibration "
            "data; this does not hold outside the validity envelope. y_pred must come from the same model that "
            "produced the calibration predictions. Envelope threshold and gate cutoffs are illustrative defaults. "
            "Decision support only, not a certified guarantee."
        )

    def assess(self, features: list[float], y_pred: float) -> dict[str, Any]:
        """Assess a single query."""
        X = np.asarray(features, dtype=float).reshape(1, -1)
        row = self.engine.evaluate(X, np.array([float(y_pred)]), method=self.method, feature_names=self.feature_names).iloc[0]
        score, half = float(row["env_score"]), float(row["half_width"])
        return {
            "y_pred": float(y_pred),
            "interval_low": float(row["lower"]),
            "interval_high": float(row["upper"]),
            "in_envelope": bool(row["inside"]),
            "envelope_score": score,
            "decision": str(row["decision"]),
            "reason": explain_decision(score, half, self.engine.gate_cfg),
            "alpha": self.alpha,
            "assumptions": self.assumptions(),
        }


def save_state(
    path: str | Path,
    *,
    cfg: dict[str, Any],
    feature_names: list[str],
    method: str,
    X_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    threshold: float,
    report_path: str | Path | None = None,
) -> Path:
    """Write the calibrated state to ``path`` (.npz). ``report_path`` is stored relative to it when possible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rp = None
    if report_path is not None:
        report_path = Path(report_path)
        try:
            rp = str(report_path.resolve().relative_to(path.parent.resolve()))
        except ValueError:
            rp = str(report_path.resolve())
    meta = {
        "version": STATE_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "cfg": cfg,
        "feature_names": list(feature_names),
        "method": method,
        "alpha": float(cfg["alpha"]),
        "envelope_threshold": float(threshold),
        "gate": {k: float(v) for k, v in cfg["gate"].items()},
        "report_path": rp,
    }
    np.savez(
        path,
        meta=np.array(json.dumps(meta)),
        X_train=np.asarray(X_train, float),
        X_cal=np.asarray(X_cal, float),
        y_cal=np.asarray(y_cal, float),
        yhat_cal=np.asarray(yhat_cal, float),
    )
    return path


def load_state(path: str | Path) -> CalibratedState:
    """Load and refit a state saved by :func:`save_state`. Raises FileNotFoundError / ValueError."""
    from .pipeline import EnvelopeEngine

    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in ("X_train", "X_cal", "y_cal", "yhat_cal")}
    if meta.get("version") != STATE_VERSION:
        raise ValueError(f"unsupported state version {meta.get('version')}")
    notes: list[str] = []
    method = meta["method"]
    if method == "fragility":
        # the API only receives y_pred (no model weights), so sigma(x) cannot be computed
        notes.append("Saved method 'fragility' needs model weights; API uses split conformal instead.")
        method = "split"
    engine = EnvelopeEngine(meta["cfg"]).fit(arrays["X_train"], arrays["X_cal"], arrays["y_cal"], arrays["yhat_cal"])
    if not np.isclose(engine.ad.threshold, meta["envelope_threshold"], rtol=1e-6):
        raise ValueError("refitted envelope threshold does not match the saved state")
    rp = meta.get("report_path")
    report_path = None if rp is None else (Path(rp) if Path(rp).is_absolute() else path.parent / rp)
    return CalibratedState(
        cfg=meta["cfg"],
        feature_names=meta["feature_names"],
        method=method,
        engine=engine,
        report_path=report_path,
        created=meta["created"],
        notes=notes,
    )
