"""Shared, cached resources for the Streamlit modes."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from envelope.config import load_config
from envelope.data import DemoData, make_demo_data
from envelope.surrogate import TorchMLPSurrogate, load_or_train_mlp

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "out" / "surrogate_mlp.pt"
BASE_CFG = load_config()


@st.cache_resource(show_spinner="Loading the surrogate model (first load only)...")
def model_and_data() -> tuple[DemoData, TorchMLPSurrogate]:
    """Demo data + MLP surrogate (saved weights are reused when they match settings and data)."""
    seed = int(BASE_CFG["seed"])
    data = make_demo_data(seed)
    return data, load_or_train_mlp(BASE_CFG, data.X_train, data.y_train, str(MODEL_PATH), seed=seed)


@st.cache_resource(show_spinner=False)
def shift_result() -> dict:
    """Calibration-vs-query shift check; depends only on the data, so computed once."""
    from envelope.shift_check import shift_check

    data, _ = model_and_data()
    return shift_check(data.X_cal, data.X_test, seed=int(BASE_CFG["seed"]))


@st.cache_data(show_spinner=False)
def sample_frames():
    """Sample files in the same shape a company would upload (synthetic demo data + MLP predictions)."""
    import pandas as pd

    data, model = model_and_data()
    f = data.feature_names
    train = pd.DataFrame(data.X_train, columns=f)
    cal = pd.DataFrame(data.X_cal, columns=f).assign(y_true=data.y_cal, y_pred=model.predict(data.X_cal))
    query = pd.DataFrame(data.X_test, columns=f).assign(y_pred=model.predict(data.X_test), y_true=data.y_test)
    return train, cal, query


SITE_CSS = """
<style>
.block-container {padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1240px;}
.hero h1 {font-size: 2.7rem; line-height: 1.15; font-weight: 800; letter-spacing: -0.01em; margin: 0.4rem 0 0.6rem; color: #0b0b0b;}
.hero p {font-size: 1.2rem; color: #3d3c3a; max-width: 760px; margin: 0 0 1.2rem;}
.eyebrow {font-size: 0.8rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #1f66c1;}
.home-meta {font-size: 0.84rem; color: #52514e; margin: 0.3rem 0 1.4rem;}
.decision-preview {background: linear-gradient(145deg, #f5f8fc, #faf9f6); border: 1px solid #dfe5ed; border-radius: 20px; padding: 24px; margin: 0.4rem 0 1.3rem;}
.preview-intro {font-size: 1.2rem; font-weight: 700; margin: 10px 0 16px; color: #172b46;}
.preview-row {display: flex; gap: 12px; align-items: flex-start; background: #fff; border: 1px solid #e4e8ed; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;}
.preview-row strong {font-size: 0.95rem; color: #172b46;}
.preview-row p {font-size: 0.85rem; line-height: 1.5; color: #52514e; margin: 3px 0 0;}
.preview-icon {display: flex; align-items: center; justify-content: center; width: 30px; height: 30px; flex-shrink: 0; border-radius: 9px;}
@media (max-width: 640px) {.hero h1 {font-size: 2.1rem;} .decision-preview {padding: 18px;}}
.sec {font-size: 1.45rem; font-weight: 800; color: #0b0b0b; margin: 2.2rem 0 0.8rem;}
.grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px;}
.scard {border: 1px solid rgba(11,11,11,0.10); border-radius: 12px; padding: 16px 18px; background: #fff;}
.scard .h {font-weight: 700; font-size: 1.02rem; color: #0b0b0b; margin-bottom: 4px;}
.scard .b {font-size: 0.92rem; color: #3d3c3a; line-height: 1.45;}
.scard .n {display: inline-flex; width: 26px; height: 26px; border-radius: 50%; background: #1f66c1; color: #fff;
           align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; margin-bottom: 8px;}
.metric {border: 1px solid rgba(11,11,11,0.10); border-radius: 12px; padding: 12px 16px; background: #fff; height: 100%;}
.metric .k {font-size: 0.82rem; color: #52514e;}
.metric .v {font-size: 1.7rem; font-weight: 800; color: #0b0b0b; font-variant-numeric: tabular-nums;}
.metric .s {font-size: 0.8rem; color: #52514e;}
.pill {font-size: 0.78rem; font-weight: 700; padding: 2px 10px; border-radius: 999px; display: inline-block; vertical-align: middle;}
.ok {background: #e6f5e6; color: #006300;} .warn {background: #fdf3dc; color: #7a5200;} .bad {background: #fae3e3; color: #a12828;}
.verdict {border-radius: 12px; padding: 16px 18px; margin: 0.4rem 0 1rem;}
.verdict .t {font-weight: 800; font-size: 1.3rem;}
.verdict ul {margin: 8px 0 0 0; padding-left: 20px; font-size: 0.93rem;}
.note {font-size: 0.85rem; color: #52514e;}
.footer {margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e4e2dc; font-size: 0.8rem; color: #898781;}
</style>
"""

FOOTER = (
    '<div class="footer">envelope · decision-support draft, not regulatory advice or a compliance claim. '
    "Assessments are session-based; download results to keep them. "
    "Coverage guarantees require exchangeable calibration and query data.</div>"
)
