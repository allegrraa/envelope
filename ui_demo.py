"""Demo Mode: a scripted five-stage walkthrough designed for a 60-second screen recording."""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from envelope.demo_story import (
    FINAL_DISCLAIMER,
    FINAL_GUARANTEE,
    STAGES,
    SYNTHETIC_LABEL,
    Story,
    build_story,
    caption,
    stage_figure,
)
from ui_common import BASE_CFG, model_and_data, shift_result

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1500px;}
html, body, [class*="st-"] {font-size: 18px;}
.topbar {display: flex; align-items: center; gap: 16px; flex-wrap: wrap;}
.stage {font-size: 1.5rem; font-weight: 800; color: #0b0b0b;}
.stage .of {color: #6b6a66; font-weight: 600; margin-right: 8px;}
.synth {font-size: 0.95rem; font-weight: 700; color: #7a5200; background: #fdf3dc; border-radius: 999px; padding: 4px 14px;}
.dots {display: flex; gap: 8px; margin: 8px 0 2px;}
.dot {height: 8px; flex: 1; border-radius: 4px; background: #e4e2dc;}
.dot.on {background: #1f66c1;}
.cap {font-size: 1.7rem; font-weight: 600; line-height: 1.35; color: #0b0b0b; margin: 14px 0 12px;}
.tiles {display: flex; gap: 16px; margin-top: 6px;}
.tile {flex: 1; border-radius: 12px; padding: 14px 18px; border: 2px solid;}
.tile .n {font-size: 2.6rem; font-weight: 800; line-height: 1.1;}
.tile .l {font-size: 1.05rem; font-weight: 700;}
.panel {border-radius: 14px; padding: 22px 26px; background: #f6f5f1; height: 100%;}
.panel .t {font-size: 1.5rem; font-weight: 800; color: #0b0b0b;}
.panel .big {font-size: 4.2rem; font-weight: 800; line-height: 1.1; margin: 6px 0;}
.panel .s {font-size: 1.1rem; color: #3d3c3a;}
table.cov {width: 100%; border-collapse: collapse; font-size: 1.15rem; margin-top: 18px;}
table.cov th, table.cov td {text-align: left; padding: 10px 12px; border-bottom: 1px solid #dcdad2;}
table.cov th {color: #3d3c3a; font-weight: 700;}
.verdict {border-radius: 14px; padding: 22px 26px; margin: 8px 0 12px;}
.verdict .v {font-size: 2.6rem; font-weight: 800;}
.verdict ul {font-size: 1.2rem; margin: 10px 0 0 0;}
.final {font-size: 1.35rem; font-weight: 700; color: #0b0b0b; margin-top: 18px;}
.final2 {font-size: 1.05rem; color: #3d3c3a;}
div.stButton > button {font-size: 1.1rem; font-weight: 700; padding: 0.6rem 1.2rem;}
</style>
"""

# Right Arrow advances to the next stage (ignored while typing in an input or moving a slider).
KEYS_JS = """
<script>
const doc = window.parent.document;
if (!doc.__envelopeDemoKeys) {
  doc.__envelopeDemoKeys = true;
  doc.addEventListener('keydown', (e) => {
    const t = e.target;
    if (e.key !== 'ArrowRight' || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName) || t.getAttribute('role') === 'slider') return;
    const btn = Array.from(doc.querySelectorAll('button')).find(b => b.innerText.trim().startsWith('Next'));
    if (btn && !btn.disabled) { e.preventDefault(); btn.click(); }
  });
}
</script>
"""

DECISION_TILE = {
    "TRUST": ("TRUST", "#0ca30c", "#006300", "#e6f5e6", "●"),
    "RERUN_FULL_SIMULATION": ("RERUN", "#fab219", "#7a5200", "#fdf3dc", "■"),
    "ESCALATE_TO_HUMAN": ("ESCALATE", "#d03b3b", "#a12828", "#fae3e3", "▲"),
}


@st.cache_resource(show_spinner=False)
def story_for(risk: str) -> Story:
    data, model = model_and_data()
    return build_story(BASE_CFG, risk=risk, data=data, model=model, shift=shift_result())


def _next() -> None:
    st.session_state.demo_stage = min(st.session_state.demo_stage + 1, 5)


def _reset() -> None:
    st.session_state.demo_stage = 1
    st.session_state.demo_report = False
    st.session_state.demo_risk = "high"


def _show_report() -> None:
    st.session_state.demo_report = True


def render() -> None:
    ss = st.session_state
    ss.setdefault("demo_stage", 1)
    ss.setdefault("demo_report", False)
    ss.setdefault("demo_risk", "high")
    st.markdown(CSS, unsafe_allow_html=True)
    # our own constant script (trusted); st.iframe replaces the deprecated components.v1.html
    if hasattr(st, "iframe"):
        st.iframe(KEYS_JS, height=1)
    else:
        components.html(KEYS_JS, height=0)

    if ss.demo_risk not in ("low", "medium", "high"):
        ss.demo_risk = "high"
    story = story_for(ss.demo_risk)
    n = story.numbers
    stage = int(ss.demo_stage)

    # --- top bar: progress, label, navigation (fixed position across stages for a clean recording)
    left, right = st.columns([5, 2], vertical_alignment="center")
    left.markdown(
        f'<div class="topbar"><span class="stage"><span class="of">Stage {stage}/5</span>{STAGES[stage]}</span>'
        f'<span class="synth">{SYNTHETIC_LABEL}</span></div>'
        '<div class="dots">' + "".join(f'<div class="dot{" on" if i <= stage else ""}"></div>' for i in range(1, 6)) + "</div>",
        unsafe_allow_html=True,
    )
    b1, b2 = right.columns([1, 2])
    b1.button("Reset", on_click=_reset, width="stretch")
    b2.button("Next →", on_click=_next, type="primary", disabled=stage >= 5, width="stretch")

    st.markdown(f'<div class="cap">{caption(stage, n)}</div>', unsafe_allow_html=True)

    if stage <= 3:
        st.pyplot(stage_figure(stage, story), clear_figure=True)
        if stage == 3:
            st.markdown(
                '<div class="tiles">' + "".join(
                    f'<div class="tile" style="border-color:{c};background:{bg};color:{ink}">'
                    f'<div class="l">{icon} {label}</div><div class="n">{n["gate_counts"][d]}</div>'
                    f'<div style="font-size:0.95rem">of {n["n_test"]} test queries</div></div>'
                    for d, (label, c, ink, bg, icon) in DECISION_TILE.items()
                ) + "</div>",
                unsafe_allow_html=True,
            )
    elif stage == 4:
        sb = n["scoreboard"]
        p1, p2 = st.columns(2, gap="large")
        for col, title, pct, cnt, color in (
            (p1, "Trust everything", sb["trust_everything_pct"], sb["trust_everything_n"], "#a12828"),
            (p2, "With Envelope", sb["with_envelope_pct"], sb["with_envelope_n"], "#006300"),
        ):
            col.markdown(
                f'<div class="panel"><div class="t">{title}</div><div class="big" style="color:{color}">{pct:.1%}</div>'
                f'<div class="s">{cnt} of {sb["n_out_of_range"]} out-of-range answers were off by more than '
                f'{n["error_tolerance"]:g} and trusted anyway.</div></div>',
                unsafe_allow_html=True,
            )
        ct = n["coverage_table"]
        rows = "".join(
            f"<tr><td>{label}</td><td>{n['target_coverage']:.0%}</td><td><b>{ct[k]['coverage']:.1%}</b></td>"
            f"<td>{ct[k]['ci_low']:.1%} to {ct[k]['ci_high']:.1%}</td><td>{ct[k]['n']}</td></tr>"
            for k, label in (("inside", "Inside envelope"), ("outside", "Outside envelope"))
        )
        st.markdown(
            '<table class="cov"><tr><th>Interval coverage</th><th>Target</th><th>Actual</th>'
            f'<th>95% CI (Clopper-Pearson)</th><th>Queries</th></tr>{rows}</table>',
            unsafe_allow_html=True,
        )
    else:
        c1, c2, _ = st.columns([2, 1.3, 3], vertical_alignment="bottom")
        risks = ["low", "medium", "high"]
        # not bound to demo_risk via key: keyed widgets that are absent on stages 1-4 lose their state
        choice = c1.radio("Model risk level", risks, index=risks.index(ss.demo_risk), format_func=str.capitalize,
                          horizontal=True, key=f"demo_risk_radio_{ss.demo_risk}")
        if choice != ss.demo_risk:
            ss.demo_risk = choice
            st.rerun()
        c2.button("Generate report", type="primary", on_click=_show_report, width="stretch")
        if ss.demo_report:
            ok = n["verdict"] == "ADEQUATE"
            gaps = "".join(f"<li>{g}</li>" for g in n["gaps"][:3])
            st.markdown(
                f'<div class="verdict" style="background:{"#e6f5e6" if ok else "#fae3e3"};color:{"#006300" if ok else "#a12828"}">'
                f'<div class="v">{"✓" if ok else "✕"} {n["verdict"]} at {n["risk_level"]} risk</div>'
                + (f"<ul>{gaps}</ul>" if gaps else "") + "</div>",
                unsafe_allow_html=True,
            )
            st.download_button("Download full report (.md)", story.run.result.report_md,
                               f"credibility_report_{n['risk_level']}.md", "text/markdown")
        st.markdown(f'<div class="final">{FINAL_GUARANTEE}</div><div class="final2">{FINAL_DISCLAIMER}</div>',
                    unsafe_allow_html=True)
