"""Assess your model: upload files -> check columns -> settings -> results and exports."""
from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from envelope.config import load_config
from envelope.data import load_external
from envelope.gate import Decision, GateConfig, explain_decision
from envelope.intake import ColumnMapping, apply_mapping, guess_mapping, suggest_tolerance
from envelope.pipeline import METHOD_LABELS, run_pipeline
from envelope.plots import plot_coverage_by_bin, plot_error_vs_score, plot_query_intervals
from envelope.report import assess_adequacy
from envelope.state import save_state
from ui_common import FOOTER, SITE_CSS, sample_frames

STRICTNESS = {"Loose": 99, "Relaxed": 97, "Standard": 95, "Strict": 90, "Very strict": 85}
LABELS = {Decision.TRUST.value: "Trust", Decision.RERUN.value: "Re-run simulation", Decision.ESCALATE.value: "Ask an expert"}
ss = st.session_state

st.markdown(SITE_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><div class="eyebrow">Assess your model</div><h1>Check your model\'s predictions</h1>'
    "<p>Bring numeric inputs and predictions from your model. Get uncertainty ranges, "
    "action recommendations, and the evidence behind them.</p></div>",
    unsafe_allow_html=True,
)


def _read(f) -> pd.DataFrame | None:
    if f is None:
        return None
    try:
        return pd.read_csv(f)
    except Exception as exc:  # noqa: BLE001 - show any parse problem to the user
        st.error(f"Could not read {getattr(f, 'name', 'file')}: {exc}")
        return None


# ------------------------------------------------------------------ 1. data
st.markdown('<div class="sec">1. Your data</div>', unsafe_allow_html=True)
source = st.radio("Data source", ["Upload my files", "Use sample data"], horizontal=True, label_visibility="collapsed")
st.caption(
    "Queries are the new cases you want checked. Each row contains input values and your model's "
    "prediction; add the true outcome if you know it to measure accuracy and coverage."
)
s_train, s_cal, s_query = sample_frames()
with st.expander("What files do I need?"):
    st.markdown(
        "| File | Contains | Used for |\n|---|---|---|\n"
        "| **Training inputs** | The input columns your model was trained on | Mapping where your data covers the input space |\n"
        "| **Calibration set** | Held-out inputs, the **true value**, and your model's **prediction** | Calibrating the uncertainty ranges |\n"
        "| **Queries** | Inputs and your model's **prediction** (true value optional) | The predictions to assess |\n\n"
        "The calibration set must not have been used for training. Column names are detected automatically, "
        "and you can change them in step 2. Templates:"
    )
    t1, t2, t3 = st.columns(3)
    t1.download_button("training_inputs.csv", s_train.head(50).to_csv(index=False), "training_inputs.csv", "text/csv")
    t2.download_button("calibration.csv", s_cal.head(50).to_csv(index=False), "calibration.csv", "text/csv")
    t3.download_button("queries.csv", s_query.head(50).to_csv(index=False), "queries.csv", "text/csv")

if source == "Upload my files":
    u1, u2, u3 = st.columns(3)
    f_train = u1.file_uploader("Training inputs (CSV)", type="csv")
    f_cal = u2.file_uploader("Calibration set (CSV)", type="csv")
    f_query = u3.file_uploader("Queries to assess (CSV)", type="csv")
    train, cal, query = _read(f_train), _read(f_cal), _read(f_query)
    sig_src = "|".join(f"{f.name}:{f.size}" for f in (f_train, f_cal, f_query) if f is not None)
else:
    train, cal, query = s_train, s_cal, s_query
    sig_src = "sample"
    st.caption("Sample data: a synthetic simulator with 3 inputs; the model was trained on inputs in [0, 5] "
               "and the queries go up to 8.")

if train is None or cal is None or query is None:
    st.info("Upload all three files to continue, or choose **Use sample data**.")
    st.markdown(FOOTER, unsafe_allow_html=True)
    st.stop()

sig = hashlib.md5(sig_src.encode()).hexdigest()[:8]
st.markdown(
    f'<div class="note">Loaded {len(train):,} training rows, {len(cal):,} calibration rows, {len(query):,} queries.</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ 2. columns
st.markdown('<div class="sec">2. Columns</div>', unsafe_allow_html=True)
guess = guess_mapping(train, cal, query)
num = lambda df: [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]  # noqa: E731
common = [c for c in num(train) if c in cal.columns and c in query.columns]
c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1])
features = c1.multiselect("Input columns (features)", common, default=[c for c in guess.features if c in common],
                          key=f"feat_{sig}")
cal_cols = num(cal)
target = c2.selectbox("True value (calibration)", cal_cols,
                      index=cal_cols.index(guess.target) if guess.target in cal_cols else 0, key=f"tgt_{sig}")
pred = c3.selectbox("Prediction", cal_cols,
                    index=cal_cols.index(guess.prediction) if guess.prediction in cal_cols else 0, key=f"pred_{sig}")
q_opts = ["(none)"] + num(query)
q_target = c4.selectbox("True value (queries, optional)", q_opts,
                        index=q_opts.index(guess.query_target) if guess.query_target in q_opts else 0, key=f"qt_{sig}")
mapping = ColumnMapping(features=features, target=target, prediction=pred,
                        query_target=None if q_target == "(none)" else q_target)
try:
    tr, ca, qu = apply_mapping(train, cal, query, mapping)
except ValueError as exc:
    st.error(f"Column problem: {exc}.")
    st.markdown(FOOTER, unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------ 3. settings
st.markdown('<div class="sec">3. Settings</div>', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
confidence = s1.slider("Confidence level", 80, 99, 90, 1, format="%d%%",
                       help="How often the likely range should contain the true value.")
strictness = s2.select_slider("Envelope strictness", list(STRICTNESS), value="Standard",
                              help="How close to the training data a query must be to count as familiar.")
risk = s3.select_slider("Model risk level", ["low", "medium", "high"], value="medium", format_func=str.capitalize,
                        help="Sets the report's adequacy rules (illustrative defaults), not the per-prediction gate cutoffs.")
alpha = round(1 - confidence / 100, 4)
tol_default = suggest_tolerance(ca["y_true"].to_numpy(), ca["y_pred"].to_numpy(), alpha)
tolerance = s4.number_input("Acceptable error (±, target units)", min_value=0.0, value=tol_default, format="%.4g",
                            key=f"tol_{sig}_{confidence}",
                            help="Predictions whose likely range is wider than this are not trusted. The default is "
                                 "1.25× the selected-confidence quantile of absolute calibration errors. "
                                 "Set it to what your application tolerates.")
m1, m2 = st.columns([1, 2])
model_name = m1.text_input("Model name", "my surrogate model")
context = m2.text_area("Context of use (goes into the report)", height=80,
                       placeholder="What decisions will these predictions support, and in which operating range?")

params = dict(sig=sig, features=tuple(features), target=target, pred=pred, q_target=q_target, confidence=confidence,
              strictness=strictness, risk=risk, tolerance=float(tolerance), model_name=model_name, context=context)
run = st.button("Run assessment", type="primary")

if run:
    with st.spinner("Calibrating and checking your predictions..."):
        try:
            cfg = load_config(overrides={
                "alpha": alpha,
                "envelope": {"percentile": STRICTNESS[strictness]},
                "gate": {"tolerance": float(tolerance)},
                "error_tolerance": float(tolerance),
            })
            cfg["bounds"], cfg["units"] = {}, {}  # demo bounds/units do not apply to your data
            ext = load_external(tr, ca, qu, {}, {})
            if len(ext.X_cal) < 20:
                raise ValueError(f"need at least 20 complete calibration rows (got {len(ext.X_cal)})")
            res = run_pipeline(
                feature_names=ext.feature_names, X_train=ext.X_train, X_cal=ext.X_cal, y_cal=ext.y_cal,
                yhat_cal=ext.yhat_cal, X_query=ext.X_query, yhat_query=ext.yhat_query, y_query=ext.y_query,
                cfg=cfg, risk_level=risk, context_of_use=context, checks=ext.checks,
                model_name=model_name or "model", mode="upload",
            )
            with tempfile.TemporaryDirectory() as d:
                p = save_state(Path(d) / "envelope_state.npz", cfg=cfg, feature_names=ext.feature_names,
                               method=res.method, X_train=ext.X_train, X_cal=ext.X_cal, y_cal=ext.y_cal,
                               yhat_cal=ext.yhat_cal, threshold=res.engine.ad.threshold)
                state_bytes = p.read_bytes()
            ss.assess = {"res": res, "cfg": cfg, "params": params, "state": state_bytes, "dropped": ext.n_dropped}
        except ValueError as exc:
            ss.pop("assess", None)
            st.error(f"Could not run the assessment: {exc}")

out = ss.get("assess")
if not out:
    st.markdown(FOOTER, unsafe_allow_html=True)
    st.stop()
if out["params"] != params:
    st.warning("Settings or data changed since this result. Press **Run assessment** to update it.")

# ------------------------------------------------------------------ results
res, cfg = out["res"], out["cfg"]
df, s = res.queries.copy(), res.summary
risk_r = out["params"]["risk"]
st.markdown('<div class="sec">Results</div>', unsafe_allow_html=True)
st.caption(
    "The verdict below assesses the submitted query set against illustrative report rules. "
    "Individual predictions can still qualify for Trust when the overall verdict is INSUFFICIENT. "
    "Simulation re-runs and expert reviews are recommendations."
)
verdict, gaps, passed = assess_adequacy(res.report_inputs, cfg)
ok = verdict == "ADEQUATE"
st.markdown(
    f'<div class="verdict {"ok" if ok else "bad"}"><div class="t">{"✓" if ok else "✕"} {verdict} at {risk_r} risk</div>'
    + ("<ul>" + "".join(f"<li>{g}</li>" for g in gaps) + "</ul>" if gaps else
       "<div>All illustrative rules for this risk level are met. A reviewer still decides adequacy.</div>")
    + "</div>",
    unsafe_allow_html=True,
)
n = len(df)
cnt = s["gate_counts"]
sh = s["shift"]
row = res.method_table.set_index("method").loc[res.method]
labelled = "y_true" in df
k = st.columns(5)
cards = [
    ("Predictions assessed", f"{n:,}", f"{out['dropped']} incomplete rows dropped" if out["dropped"] else "all rows complete"),
    ("● Trust", f"{cnt['TRUST'] / n:.0%}", f"{cnt['TRUST']:,} predictions"),
    ("■ Re-run simulation", f"{cnt['RERUN_FULL_SIMULATION'] / n:.0%}", f"{cnt['RERUN_FULL_SIMULATION']:,} predictions"),
    ("▲ Ask an expert", f"{cnt['ESCALATE_TO_HUMAN'] / n:.0%}", f"{cnt['ESCALATE_TO_HUMAN']:,} predictions"),
]
if labelled and not math.isnan(row["coverage_inside"]):
    cards.append(("Coverage inside envelope", f"{row['coverage_inside']:.1%}",
                  f"target {1 - res.engine.alpha:.0%}; 95% CI {row['ci_low_inside']:.1%}–{row['ci_high_inside']:.1%}"))
else:
    cards.append(("Coverage (estimated)", f"{row['coverage_overall']:.1%}",
                  f"target {1 - res.engine.alpha:.0%}; from calibration splits, in-distribution only"))
for col, (kk, v, sub) in zip(k, cards):
    col.markdown(f'<div class="metric"><div class="k">{kk}</div><div class="v">{v}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)
sh_cls = {"OK": "ok", "WARN": "warn"}.get(sh["status"], "warn")
st.markdown(
    f'<div style="margin:0.9rem 0 0.4rem"><b>Guarantee validity</b> <span class="pill {sh_cls}">{sh["status"]}</span> '
    f'<span class="note">{sh["message"]} A WARN means the coverage guarantee may not apply.</span></div>',
    unsafe_allow_html=True,
)

tab_dec, tab_cov, tab_env, tab_chk, tab_exp = st.tabs(["Decisions", "Coverage", "Envelope", "Data checks", "Report & export"])
gcfg = GateConfig.from_config(cfg)
df["decision_label"] = df["decision"].map(LABELS)
df["reason"] = [explain_decision(sc, hw, gcfg, ins if res.engine.flag_mode != "percentile" else None)
                for sc, hw, ins in zip(df["env_score"], df["half_width"], df["inside"])]
feat = res.report_inputs.envelope["features"]
cols = feat + ["y_pred", "lower", "upper", "decision_label", "reason", "env_score", "env_pvalue"] + (["y_true", "abs_err"] if labelled else [])
table = df[cols].rename(columns={"y_pred": "prediction", "lower": "range_low", "upper": "range_high",
                                 "decision_label": "decision", "env_score": "envelope_score", "env_pvalue": "envelope_p_value",
                                 "y_true": "true_value", "abs_err": "abs_error"})

with tab_dec:
    pick = st.multiselect("Show", list(LABELS.values()), default=list(LABELS.values()))
    view = table[table["decision"].isin(pick)]
    st.dataframe(view.round(4), hide_index=True, width="stretch", height=420)
    st.download_button("Download decisions (CSV)", table.to_csv(index=False), "envelope_decisions.csv", "text/csv")

with tab_cov:
    mt = res.method_table.copy()
    show = pd.DataFrame({
        "method": mt["method"].map(METHOD_LABELS),
        "coverage (all)": [f"{c:.1%}" if not math.isnan(c) else "n/a" for c in mt["coverage_overall"]],
        "inside envelope": [f"{c:.1%} [{lo:.1%}, {hi:.1%}]" if not math.isnan(c) else "n/a"
                            for c, lo, hi in zip(mt["coverage_inside"], mt["ci_low_inside"], mt["ci_high_inside"])],
        "outside envelope": [f"{c:.1%} [{lo:.1%}, {hi:.1%}]" if not math.isnan(c) else "n/a"
                             for c, lo, hi in zip(mt["coverage_outside"], mt["ci_low_outside"], mt["ci_high_outside"])],
        "mean width": mt["width_overall"].round(4),
    })
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption(f"Measured on: {res.coverage_source}. Brackets: 95% Clopper-Pearson intervals.")
    cs = s["coverage_stats"]
    law = cs["law_nominal"]
    st.markdown(
        f"With your {cs['n_cal']:,} calibration points, split-conformal coverage averages {law['mean']:.1%} over "
        f"calibration draws and is at least {law['p05']:.1%} in 95% of them. To make coverage ≥ {1 - res.engine.alpha:.0%} "
        f"hold with 90% probability, use alpha' = {cs['alpha_corrected']:.4f}."
    )
    if labelled:
        st.pyplot(plot_coverage_by_bin(s["group_table"], res.engine.alpha), clear_figure=True)
    else:
        st.dataframe(s["group_table"][["bin", "score_low", "score_high", "n_query", "n_cal", "quantile_source", "half_width"]]
                     .round(3), hide_index=True)

with tab_env:
    e = res.report_inputs.envelope
    st.markdown(
        f"**{s['pct_outside']:.1f}%** of your queries are outside the validity envelope, and **{s['pct_far_outside']:.1f}%** "
        f"are far outside. A query is outside when its average distance to the {e['k']} nearest training points "
        f"exceeds the {e['percentile']:g}th percentile of distances within your training data."
    )
    a, b = st.columns(2)
    if labelled:
        a.pyplot(plot_error_vs_score(df, gcfg.escalate_score, float(cfg["error_tolerance"])), clear_figure=True)
        pr = s["flag_pr"]
        a.caption(f"The outside-envelope flag catches {pr['recall']:.0%} of errors larger than {cfg['error_tolerance']:g}; "
                  f"{pr['precision']:.0%} of flagged queries had such an error.")
    fx = b.selectbox("Plot predictions against", feat)
    b.pyplot(plot_query_intervals(df, fx), clear_figure=True)
    st.dataframe(pd.DataFrame({
        "outside flag": ["distance percentile", f"conformal p-value < {e['pvalue_beta']:g}", "Benjamini-Hochberg (batch)"],
        "% of queries": [s["pct_flag_percentile"], s["pct_flag_pvalue"], s["pct_flag_bh"]],
    }).round(1), hide_index=True)

with tab_chk:
    st.dataframe(pd.DataFrame([{"check": c.name, "status": c.status, "detail": c.detail} for c in res.report_inputs.checks]),
                 hide_index=True, width="stretch")
    if out["dropped"]:
        st.warning(f"{out['dropped']} rows with missing or non-numeric values were dropped.")

with tab_exp:
    e1, e2, e3 = st.columns(3)
    e1.download_button("Credibility report (.md)", res.report_md, "credibility_report.md", "text/markdown", width="stretch")
    e2.download_button("Decisions (.csv)", table.to_csv(index=False), "envelope_decisions.csv", "text/csv", width="stretch")
    e3.download_button("API state (.npz)", out["state"], "envelope_state.npz", "application/octet-stream", width="stretch")
    st.caption("The API state lets you check new predictions one at a time from your own systems. See the API page.")
    with st.expander("Preview report"):
        st.markdown("\n".join("##" + ln if ln.startswith("#") else ln for ln in res.report_md.splitlines()))

st.markdown(FOOTER, unsafe_allow_html=True)
