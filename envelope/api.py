"""FastAPI service over a saved calibration state.

Run: uvicorn envelope.api:app --port 8000
State file: $ENVELOPE_STATE (default ./out/state.npz, created by ``python run_demo.py``).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .state import CalibratedState, load_state

DEFAULT_STATE_PATH = Path(os.environ.get("ENVELOPE_STATE", "out/state.npz"))


class AssessRequest(BaseModel):
    features: list[float] = Field(..., min_length=1, description="Feature values, in the training column order")
    y_pred: float = Field(..., description="The surrogate's prediction for these features")


class AssessResponse(BaseModel):
    y_pred: float
    interval_low: float
    interval_high: float
    in_envelope: bool
    envelope_score: float
    decision: Literal["TRUST", "RERUN_FULL_SIMULATION", "ESCALATE_TO_HUMAN"]
    reason: str
    alpha: float
    assumptions: str


def create_app(state_path: str | Path | None = None) -> FastAPI:
    """Build the app. The state is loaded lazily on first use, so the app starts even without it."""
    path = Path(state_path) if state_path else DEFAULT_STATE_PATH
    app = FastAPI(title="envelope", version="0.3.0", description="Validity-envelope gate for surrogate predictions.")
    cache: dict[str, CalibratedState] = {}

    def get_state() -> CalibratedState:
        if "state" not in cache:
            try:
                cache["state"] = load_state(path)
            except FileNotFoundError:
                raise HTTPException(503, f"state file {path} not found; run `python run_demo.py` first")
            except (ValueError, KeyError) as exc:
                raise HTTPException(503, f"state file {path} is invalid: {exc}")
        return cache["state"]

    @app.get("/health")
    def health() -> dict:
        s = get_state()
        return {
            "status": "ok",
            "state_file": str(path),
            "created": s.created,
            "features": s.feature_names,
            "method": s.method,
            "alpha": s.alpha,
            "envelope_threshold": s.engine.ad.threshold,
            "gate": {"tolerance": s.engine.gate_cfg.tolerance, "escalate_score": s.engine.gate_cfg.escalate_score},
            "notes": s.notes,
        }

    @app.post("/assess", response_model=AssessResponse)
    def assess(req: AssessRequest) -> dict:
        s = get_state()
        if len(req.features) != len(s.feature_names):
            raise HTTPException(
                422, f"expected {len(s.feature_names)} features {s.feature_names}, got {len(req.features)}"
            )
        if not all(math.isfinite(v) for v in [*req.features, req.y_pred]):
            raise HTTPException(422, "features and y_pred must be finite numbers")
        return s.assess(req.features, req.y_pred)

    @app.get("/report", response_class=PlainTextResponse)
    def report() -> PlainTextResponse:
        s = get_state()
        if s.report_path is None or not s.report_path.exists():
            raise HTTPException(404, "no credibility report found; run `python run_demo.py`")
        # read on every request so the latest report on disk is returned
        return PlainTextResponse(s.report_path.read_text(), media_type="text/markdown; charset=utf-8")

    return app


app = create_app()
