"""Per-query gate decision: TRUST / RERUN_FULL_SIMULATION / ESCALATE_TO_HUMAN."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Decision(str, Enum):
    TRUST = "TRUST"
    RERUN = "RERUN_FULL_SIMULATION"
    ESCALATE = "ESCALATE_TO_HUMAN"


@dataclass
class GateConfig:
    """Gate cutoffs.

    tolerance: max interval half-width (target units) for TRUST.
    escalate_score: envelope score above which a query is "far outside" -> ESCALATE.
    """

    tolerance: float = 0.6
    escalate_score: float = 1.5

    @classmethod
    def from_config(cls, cfg: dict) -> "GateConfig":
        g = cfg.get("gate", {})
        return cls(tolerance=float(g.get("tolerance", 0.6)), escalate_score=float(g.get("escalate_score", 1.5)))


def gate_decision(score: float, half_width: float, cfg: GateConfig, inside: bool | None = None) -> Decision:
    """Decide for one query. ``inside`` overrides the default ``score <= 1`` membership test
    (e.g. when a conformal p-value flag defines the envelope); ESCALATE always uses the score.

    * score <= 1 (inside) and half_width <= tolerance       -> TRUST
    * score <= 1 (inside) and half_width  > tolerance       -> RERUN_FULL_SIMULATION
    * 1 < score <= escalate_score (slightly outside)        -> RERUN_FULL_SIMULATION
    * score > escalate_score (far outside)                  -> ESCALATE_TO_HUMAN
    """
    if not np.isfinite(score) or score > cfg.escalate_score:
        return Decision.ESCALATE
    if not (score <= 1.0 if inside is None else bool(inside)):
        return Decision.RERUN
    if np.isfinite(half_width) and half_width <= cfg.tolerance:
        return Decision.TRUST
    return Decision.RERUN


def explain_decision(score: float, half_width: float, cfg: GateConfig, inside: bool | None = None) -> str:
    """Human-readable reason for the decision :func:`gate_decision` returns."""
    d = gate_decision(score, half_width, cfg, inside)
    if d is Decision.ESCALATE:
        return (
            f"Far outside the validity envelope (score {score:.2f} > escalate cutoff {cfg.escalate_score:g}); "
            "the surrogate is extrapolating and its interval is not trustworthy."
        )
    if d is Decision.RERUN and not (score <= 1.0 if inside is None else bool(inside)):
        return (
            f"Slightly outside the validity envelope (score {score:.2f}, below the escalate cutoff {cfg.escalate_score:g}); "
            "run the full simulation for this point."
        )
    if d is Decision.TRUST:
        return (
            f"Inside the validity envelope (score {score:.2f} <= 1) and interval half-width "
            f"{half_width:.3g} <= tolerance {cfg.tolerance:g}."
        )
    return (
        f"Inside the validity envelope (score {score:.2f} <= 1) but interval half-width "
        f"{half_width:.3g} > tolerance {cfg.tolerance:g}; too uncertain to trust."
    )


def gate_decisions(
    scores: np.ndarray, half_widths: np.ndarray, cfg: GateConfig, inside: np.ndarray | None = None
) -> np.ndarray:
    """Vectorised :func:`gate_decision`; returns an array of decision strings."""
    ins = [None] * len(scores) if inside is None else list(inside)
    return np.array(
        [gate_decision(float(s), float(h), cfg, i).value for s, h, i in zip(scores, half_widths, ins)], dtype=object
    )
