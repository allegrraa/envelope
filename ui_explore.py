"""Explore mode: interactive single-screen UI (settings, point checker, scorecard, report)."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from envelope.config import load_config
from envelope.data import TEST_MAX, TRAIN_RANGE, make_demo_data, simulator
from envelope.gate import Decision
from envelope.pipeline import DemoRun, run_demo
from envelope.report import DISCLAIMER, assess_adequacy
from ui_common import BASE_CFG, model_and_data, shift_result

CONTEXT = (
    "Screening surrogate for a structural-response simulator (illustrative). Predictions prioritise which "
    "design points get a full simulation run; they do not replace the simulator for final sign-off. "
    "Intended operating region: all inputs in [0, 5]."
)
STRICTNESS = {"Loose": 99, "Relaxed": 97, "Standard": 95, "Strict": 90, "Very strict": 85}
RISK_RULES = {
    "low": "coverage within 10 pts of target, at most 30% unfamiliar queries",
    "medium": "coverage within 5 pts of target, at most 15% unfamiliar queries",
    "high": "coverage within 2 pts of target, at most 5% unfamiliar queries",
}
# Status colours always come with a shape + a word, never colour alone.
DECISIONS = {
    Decision.TRUST.value: {"label": "Trust", "color": "#0ca30c", "shape": "circle", "icon": "●",
                           "tint": "#e6f5e6", "ink": "#006300", "action": "Use the prediction."},
    Decision.RERUN.value: {"label": "Re-run simulation", "color": "#fab219", "shape": "square", "icon": "■",
                           "tint": "#fdf3dc", "ink": "#7a5200", "action": "Run the full simulation for this point."},
    Decision.ESCALATE.value: {"label": "Ask an expert", "color": "#d03b3b", "shape": "triangle-up", "icon": "▲",
                              "tint": "#fae3e3", "ink": "#a12828", "action": "Don't use this prediction. Have an expert review it."},
}
INK, INK_2, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6"

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1440px;}
h1.env-title {font-size: 1.7rem; font-weight: 700; margin: 0; padding: 0; color: #0b0b0b;}
p.env-sub {color: #52514e; font-size: 0.98rem; margin: 0.2rem 0 0.9rem; max-width: 980px;}
.steps {display: flex; gap: 12px; margin-bottom: 1.3rem; flex-wrap: wrap;}
.step {flex: 1 1 220px; background: #f3f2ee; border-radius: 10px; padding: 10px 14px; font-size: 0.87rem; color: #52514e;}
.step b {color: #0b0b0b;}
.num {display: inline-flex; width: 21px; height: 21px; border-radius: 50%; background: #2a78d6; color: #fff;
      align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; margin-right: 7px;}
.panel-title {font-size: 0.8rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #52514e; margin-bottom: 0.5rem;}
.hint {font-size: 0.8rem; color: #52514e; margin: -0.6rem 0 1rem;}
.legend {display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.82rem; color: #52514e; margin: 0 0 4px 4px;}
.legend span {display: inline-flex; align-items: center; gap: 6px;}
.ln {display: inline-block; width: 18px; border-top: 2.5px solid;}
.box {display: inline-block; width: 14px; height: 12px; background: #e1e0d9; border-radius: 2px;}
.decision {border-radius: 10px; padding: 12px 14px; margin: 4px 0 10px;}
.decision .t {font-size: 1.3rem; font-weight: 700;}
.decision .a {font-size: 0.88rem; margin-top: 2px;}
.kv {display: flex; justify-content: space-between; gap: 8px; font-size: 0.88rem; padding: 4px 0; border-bottom: 1px solid #efeee9;}
.kv .k {color: #52514e;} .kv .v {font-weight: 600; color: #0b0b0b; font-variant-numeric: tabular-nums; text-align: right;}
.reason {font-size: 0.88rem; color: #0b0b0b; margin: 10px 0 6px;}
.meter {position: relative; height: 10px; border-radius: 5px; margin: 6px 0 2px;
        background: linear-gradient(90deg, #cfeccf 0 50%, #fbe7b8 50% 75%, #f4c7c7 75% 100%);}
.meter .tick {position: absolute; top: -4px; width: 3px; height: 18px; background: #0b0b0b; border-radius: 2px;}
.meter-lbl {display: flex; justify-content: space-between; font-size: 0.72rem; color: #898781;}
.card {border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 12px 14px; background: #fff; min-height: 150px;}
.card .k {font-size: 0.82rem; color: #52514e;}
.card .v {font-size: 1.75rem; font-weight: 700; color: #0b0b0b; line-height: 1.25; font-variant-numeric: tabular-nums;}
.card .s {font-size: 0.8rem; color: #52514e; margin-top: 2px;}
.pill {font-size: 0.74rem; font-weight: 700; padding: 2px 9px; border-radius: 999px; display: inline-block; vertical-align: middle; margin-left: 6px;}
.ok {background: #e6f5e6; color: #006300;} .warn {background: #fdf3dc; color: #7a5200;} .bad {background: #fae3e3; color: #a12828;}
.verdict {border-radius: 10px; padding: 14px 16px; margin: 0.3rem 0 0.8rem;}
.verdict .t {font-weight: 700; font-size: 1.05rem;}
.verdict ul {margin: 6px 0 0 0; padding-left: 18px; font-size: 0.87rem;}
.disclaimer {font-size: 0.76rem; color: #898781; margin-top: 1.4rem;}
</style>
"""

TOL = float(BASE_CFG["gate"]["tolerance"])
ERR_TOL = float(BASE_CFG["error_tolerance"])


# ------------------------------------------------------------------ computation (cached)
@st.cache_resource(max_entries=256, show_spinner=False)
def settings_run(confidence: int, percentile: int, risk: str) -> DemoRun:
    data, model = model_and_data()
    cfg = load_config(overrides={"alpha": round(1 - confidence / 100, 4), "envelope": {"percentile": percentile}})
    return run_demo(cfg, risk_level=risk, context_of_use=CONTEXT, data=data, model=model, shift=shift_result())


def evaluate(run: DemoRun, X: np.ndarray) -> pd.DataFrame:
    """Envelope + interval + gate for arbitrary points, plus the (demo-only) true simulator value."""
    df = run.result.engine.evaluate(X, run.model.predict(X), method=run.result.method, feature_names=run.data.feature_names)
    df["f_true"] = simulator(X)
    df["label"] = df["decision"].map(lambda d: DECISIONS[d]["label"])
    return df


@st.cache_data(max_entries=512, show_spinner=False)
def slice_frame(confidence: int, percentile: int, risk: str, x2: float, x3: float) -> pd.DataFrame:
    X = np.column_stack([np.linspace(0, TEST_MAX, 161), np.full(161, x2), np.full(161, x3)])
    return evaluate(settings_run(confidence, percentile, risk), X)


def plain_reason(row: pd.Series) -> str:
    s, hw = float(row["env_score"]), float(row["half_width"])
    if row["decision"] == Decision.ESCALATE.value:
        return (f"This point is far from anything the model learned from ({s:.1f}× the familiarity limit). "
                "The model is extrapolating, so neither the prediction nor its range can be relied on.")
    if s > 1:
        return (f"This point is just outside the familiar region ({s:.2f}× the limit). "
                "The prediction may still be close, but it's not safe to assume so.")
    if row["decision"] == Decision.TRUST.value:
        return (f"This point is close to the training data, and the uncertainty (±{hw:.2f}) "
                f"is within the tolerance (±{TOL:g}).")
    return f"The point is familiar, but the uncertainty (±{hw:.2f}) is wider than the tolerance (±{TOL:g})."


def status(gap: float) -> tuple[str, str]:
    if gap >= -0.02:
        return "ok", "✓ On target"
    if gap >= -0.05:
        return "warn", "! Slightly low"
    return "bad", "✕ Below target"


def card(title: str, value: str, sub: str, pill: tuple[str, str] | None = None) -> str:
    p = f'<span class="pill {pill[0]}">{pill[1]}</span>' if pill else ""
    return f'<div class="card"><div class="k">{title}</div><div class="v">{value}{p}</div><div class="s">{sub}</div></div>'



def render() -> None:
    """Render the Explore mode."""
    st.markdown(CSS, unsafe_allow_html=True)
    # ------------------------------------------------------------------ header
    st.markdown('<h1 class="env-title">Can I trust the surrogate here?</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="env-sub">A fast surrogate model stands in for an expensive simulator. It was trained only on inputs '
        'between 0 and 5, but the physics changes above x1 = 5. This tool tells you, point by point, when to use the '
        'surrogate, when to re-run the simulator, and when to ask an expert.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="steps">'
        '<div class="step"><span class="num">1</span><b>Set how cautious to be.</b> Confidence, strictness and risk level on the left.</div>'
        '<div class="step"><span class="num">2</span><b>Check a design point.</b> Move the inputs on the right and read the verdict.</div>'
        '<div class="step"><span class="num">3</span><b>Review the evidence.</b> Scorecard below, then generate the credibility report.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1.05, 2.7, 1.35], gap="large")

    # ------------------------------------------------------------------ 1. settings
    with left:
        st.markdown('<div class="panel-title"><span class="num">1</span>Settings</div>', unsafe_allow_html=True)
        confidence = st.slider("Confidence level", 80, 99, 90, 1, format="%d%%")
        st.markdown(f'<div class="hint">Ranges should contain the real value {confidence}% of the time.</div>', unsafe_allow_html=True)
        strictness = st.select_slider("Envelope strictness", list(STRICTNESS), value="Standard")
        st.markdown('<div class="hint">How close to the training data a point must be to count as familiar. '
                    'Stricter sends more points to re-run.</div>', unsafe_allow_html=True)
        risk = st.select_slider("Model risk level", ["low", "medium", "high"], value="medium", format_func=str.capitalize)
        st.markdown(f'<div class="hint">What\'s at stake if the model is wrong. {risk.capitalize()} risk requires '
                    f'{RISK_RULES[risk]}.</div>', unsafe_allow_html=True)

    run = settings_run(confidence, STRICTNESS[strictness], risk)
    res, df = run.result, run.result.queries

    # ------------------------------------------------------------------ 2. point checker
    with right:
        st.markdown('<div class="panel-title"><span class="num">2</span>Check a design point</div>', unsafe_allow_html=True)
        x1 = st.slider("x1 (physics changes above 5)", 0.0, TEST_MAX, 3.0, 0.1)
        c2, c3 = st.columns(2)
        x2 = c2.slider("x2", 0.0, 5.0, 2.5, 0.5)
        x3 = c3.slider("x3", 0.0, 5.0, 2.5, 0.5)
        pt = evaluate(run, np.array([[x1, x2, x3]])).iloc[0]
        d = DECISIONS[pt["decision"]]
        st.markdown(
            f'<div class="decision" style="background:{d["tint"]}; color:{d["ink"]}">'
            f'<div class="t">{d["icon"]} {d["label"]}</div><div class="a">{d["action"]}</div></div>',
            unsafe_allow_html=True,
        )
        hit = pt["lower"] <= pt["f_true"] <= pt["upper"]
        st.markdown(
            f'<div class="kv"><span class="k">Surrogate prediction</span><span class="v">{pt["y_pred"]:.2f}</span></div>'
            f'<div class="kv"><span class="k">Likely range ({confidence}%)</span><span class="v">{pt["lower"]:.2f} to {pt["upper"]:.2f}</span></div>'
            f'<div class="kv"><span class="k">Real value (demo only)</span><span class="v">{pt["f_true"]:.2f} '
            f'{"✓ in range" if hit else "✕ outside range"}</span></div>'
            f'<div class="reason">{plain_reason(pt)}</div>',
            unsafe_allow_html=True,
        )
        esc = float(res.engine.gate_cfg.escalate_score)
        s = float(pt["env_score"])
        # piecewise scale so the edge (score 1) sits at 50% and the escalate cutoff at 75%
        pos = 50 * s if s <= 1 else (50 + 25 * (s - 1) / (esc - 1) if s <= esc else min(75 + 25 * (s - esc) / esc, 100))
        st.markdown(
            '<div style="font-size:0.8rem;color:#52514e;margin-top:8px">Distance from training data</div>'
            f'<div class="meter"><div class="tick" style="left:calc({pos:.1f}% - 1px)"></div></div>'
            '<div class="meter-lbl"><span>familiar</span><span>edge</span><span>far outside</span></div>',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------ chart
    with center:
        sl = slice_frame(confidence, STRICTNESS[strictness], risk, x2, x3)
        st.markdown(
            '<div class="legend">'
            '<span><i class="box"></i>training range</span>'
            f'<span><i class="ln" style="border-color:{INK}"></i>real simulator</span>'
            f'<span><i class="ln" style="border-color:{BLUE}"></i>surrogate ± likely range</span>'
            + "".join(f'<span><b style="color:{v["color"]}">{v["icon"]}</b>{v["label"]}</span>' for v in DECISIONS.values())
            + '<span><b>◯</b>your point</span></div>',
            unsafe_allow_html=True,
        )
        lo, hi = TRAIN_RANGE
        ymin = float(min(sl["lower"].min(), sl["f_true"].min())) - 0.5
        ymax = float(max(sl["upper"].max(), sl["f_true"].max())) + 0.5
        xscale = alt.Scale(domain=[0, TEST_MAX], nice=False)
        yscale = alt.Scale(domain=[ymin, ymax], nice=False)
        xs = alt.X("x1:Q", title=f"x1   (slice through your point: x2 = {x2:g}, x3 = {x3:g})", scale=xscale,
                   axis=alt.Axis(values=list(range(int(TEST_MAX) + 1)), format="d"))
        ys = alt.Y("y:Q", title="response", scale=yscale)
        region = alt.Chart(pd.DataFrame({"a": [lo], "b": [hi]})).mark_rect(color=GRID, opacity=0.6).encode(
            x=alt.X("a:Q", scale=xscale), x2="b:Q")
        region_lbl = alt.Chart(pd.DataFrame({"x": [lo + 0.1], "t": ["model trained here"]})).mark_text(
            color=MUTED, fontSize=11, baseline="top", align="left", dy=6).encode(x=alt.X("x:Q", scale=xscale), y=alt.value(0), text="t:N")
        band = alt.Chart(sl).mark_area(color=BLUE, opacity=0.13).encode(
            x=xs, y=alt.Y("lower:Q", scale=yscale), y2="upper:Q")
        truth = alt.Chart(sl.assign(y=sl["f_true"])).mark_line(color=INK, strokeWidth=2.2).encode(x=xs, y=ys)
        pred = alt.Chart(sl.assign(y=sl["y_pred"])).mark_line(color=BLUE, strokeWidth=2).encode(x=xs, y=ys)
        pts = sl.iloc[::5].assign(y=lambda f: f["y_pred"])
        tooltip = [
            alt.Tooltip("x1:Q", format=".2f"), alt.Tooltip("label:N", title="decision"),
            alt.Tooltip("y_pred:Q", title="surrogate", format=".2f"),
            alt.Tooltip("lower:Q", title="range low", format=".2f"), alt.Tooltip("upper:Q", title="range high", format=".2f"),
            alt.Tooltip("f_true:Q", title="real simulator", format=".2f"),
            alt.Tooltip("env_score:Q", title="distance (1 = edge)", format=".2f"),
        ]
        bars = alt.Chart(pts).mark_rule(color=INK_2, opacity=0.55).encode(
            x=xs, y=alt.Y("lower:Q", scale=yscale), y2="upper:Q")
        dots = alt.Chart(pts).mark_point(filled=True, size=70, opacity=1, stroke="white", strokeWidth=1).encode(
            x=xs, y=ys,
            color=alt.Color("decision:N", scale=alt.Scale(domain=list(DECISIONS), range=[v["color"] for v in DECISIONS.values()]), legend=None),
            shape=alt.Shape("decision:N", scale=alt.Scale(domain=list(DECISIONS), range=[v["shape"] for v in DECISIONS.values()]), legend=None),
            tooltip=tooltip,
        )
        me = pd.DataFrame([{**pt.to_dict(), "y": pt["y_pred"]}])
        me_rule = alt.Chart(me).mark_rule(color=INK, strokeDash=[4, 3], opacity=0.6).encode(x=xs)
        me_dot = alt.Chart(me).mark_point(size=320, stroke=INK, strokeWidth=2.5, filled=False).encode(x=xs, y=ys, tooltip=tooltip)
        chart = (
            alt.layer(region, region_lbl, band, truth, pred, bars, dots, me_rule, me_dot)
            .properties(height=440)
            .configure_view(stroke=None)
            .configure_axis(labelColor=INK_2, titleColor=INK_2, gridColor="#efeee9", domainColor="#c3c2b7", tickColor="#c3c2b7",
                            labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Hover any point for details. Past x1 = 5 the real simulator bends upward, but the surrogate doesn't "
                   "know that and its range stays just as narrow. That's why the gate is needed.")

    # ------------------------------------------------------------------ 3. scorecard + report
    st.markdown('<div class="panel-title" style="margin-top:0.8rem"><span class="num">3</span>'
                f'How the gate performs on {len(df)} test points</div>', unsafe_allow_html=True)
    target = confidence / 100
    row = res.method_table.set_index("method").loc[res.method]
    trusted = df["decision"] == Decision.TRUST.value
    false_trust = float((df.loc[trusted, "abs_err"] > ERR_TOL).mean()) if trusted.any() else 0.0
    n_esc = int((df["decision"] == Decision.ESCALATE.value).sum())
    ft_pill = ("ok", "✓ Safe") if false_trust < 0.01 else (("warn", "! Watch") if false_trust < 0.05 else ("bad", "✕ Unsafe"))
    k1, k2, k3, k4 = st.columns(4)
    cs = res.summary["coverage_stats"]
    law = cs["law_nominal"]
    k1.markdown(card("Range coverage on familiar points", f"{row['coverage_inside']:.1%}",
                     f"95% CI {row['ci_low_inside']:.1%} to {row['ci_high_inside']:.1%} (target {target:.0%}). "
                     f"With n = {cs['n_cal']} calibration points, expect {law['mean']:.1%} on average and at least "
                     f"{law['p05']:.1%} in 95% of calibration draws. Small-sample alpha' = {cs['alpha_corrected']:.3f}.",
                     status(row["coverage_inside"] - target)),
                unsafe_allow_html=True)
    k2.markdown(card("Range coverage on all points", f"{row['coverage_overall']:.1%}",
                     f"95% CI {row['ci_low_overall']:.1%} to {row['ci_high_overall']:.1%} (target {target:.0%}). "
                     "It collapses on unfamiliar points, which the gate catches.",
                     status(row["coverage_overall"] - target)), unsafe_allow_html=True)
    k3.markdown(card("Sent to an expert", f"{n_esc / len(df):.1%}",
                     f"{n_esc} of {len(df)} points were far outside the training data."), unsafe_allow_html=True)
    k4.markdown(card("False trust", f"{false_trust:.1%}",
                     f"Share of trusted points that were off by more than {ERR_TOL:g}.", ft_pill), unsafe_allow_html=True)

    st.write("")
    sh = res.summary["shift"]
    sh_cls = {"OK": "ok", "WARN": "warn"}.get(sh["status"], "warn")
    sh_icon = {"OK": "✓", "WARN": "!"}.get(sh["status"], "?")
    st.markdown(
        f'<div style="font-size:0.9rem;margin:0.2rem 0 0.6rem"><b>Guarantee validity</b>'
        f'<span class="pill {sh_cls}">{sh_icon} {sh["status"]}</span> '
        f'<span style="color:#52514e">{sh["message"]} A WARN means the coverage guarantee may not apply.</span></div>',
        unsafe_allow_html=True,
    )
    verdict, gaps, _ = assess_adequacy(res.report_inputs, BASE_CFG)
    ok = verdict == "ADEQUATE"
    short = [g.split(" -> ")[0] for g in gaps]
    st.markdown(
        f'<div class="verdict {"ok" if ok else "bad"}"><div class="t">{"✓" if ok else "✕"} Credibility verdict at '
        f'{risk} risk: {verdict}</div>'
        + ("" if ok else "<ul>" + "".join(f"<li>{g}</li>" for g in short) + "</ul>")
        + "</div>",
        unsafe_allow_html=True,
    )
    b1, b2, _ = st.columns([1.2, 1.2, 4])
    if b1.button("Generate full report", type="primary", width="stretch"):
        st.session_state["show_report"] = True
    b2.download_button("Download report (.md)", res.report_md, "credibility_report.md", "text/markdown", width="stretch")
    if st.session_state.get("show_report"):
        with st.expander("Credibility report (matches the current settings)", expanded=True):
            # demote headings so the report sits comfortably inside the expander
            st.markdown("\n".join("##" + l if l.startswith("#") else l for l in res.report_md.splitlines()))

    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
