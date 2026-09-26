"""Home page."""
import streamlit as st

from ui_common import FOOTER, SITE_CSS

st.markdown(SITE_CSS, unsafe_allow_html=True)
intro, preview = st.columns([1.4, 1], gap="large")
with intro:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Prediction checks for simulation teams</div>'
        "<h1>Know when to trust your model's predictions.</h1>"
        "<p>A fast model can give a confident answer in unfamiliar territory. "
        "Check each prediction against your training data and measured errors, "
        "then decide where simulation time and expert attention are needed.</p></div>"
        '<div class="home-meta">Numeric predictions · CSV assessment · API integration</div>',
        unsafe_allow_html=True,
    )
with preview:
    st.markdown(
        '<div class="decision-preview"><div class="eyebrow">From prediction to next step</div>'
        '<div class="preview-intro">Two checks. Three recommendations.</div>'
        '<div class="preview-row"><span class="preview-icon ok">●</span><div><strong>Trust</strong>'
        '<p>Familiar input, uncertainty within your tolerance.</p></div></div>'
        '<div class="preview-row"><span class="preview-icon warn">■</span><div><strong>Re-run simulation</strong>'
        '<p>Near the boundary, or too uncertain to accept.</p></div></div>'
        '<div class="preview-row"><span class="preview-icon bad">▲</span><div><strong>Ask an expert</strong>'
        '<p>Far from the data the model learned from.</p></div></div>'
        '<div class="note">Each result includes a range, a score, and a reason.</div></div>',
        unsafe_allow_html=True,
    )
c1, c2, c3 = st.columns([1, 1, 1])
if c1.button("Assess your model →", type="primary", width="stretch"):
    st.switch_page("site/assess.py")
if c2.button("See an interactive example", width="stretch"):
    st.switch_page("site/example.py")
if c3.button("Take the guided tour →", width="stretch"):
    st.switch_page("site/tour.py")
st.caption("Try synthetic sample data on the assessment page, or bring your own CSVs. No model upload needed.")

st.markdown('<div class="sec">How it works</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="grid">'
    '<div class="scard"><div class="n">1</div><div class="h">Upload three files</div><div class="b">Your training inputs, '
    "a held-out calibration set with true values and predictions, and the predictions you want checked. "
    "Use numeric inputs and one numeric output; the model stays in your own system.</div></div>"
    '<div class="scard"><div class="n">2</div><div class="h">We calibrate and map the envelope</div><div class="b">'
    "Conformal prediction turns your calibration errors into uncertainty ranges. A distance check maps where your "
    "training data actually covers the input space.</div></div>"
    '<div class="scard"><div class="n">3</div><div class="h">Get a decision per prediction</div><div class="b">'
    "Recommendations to trust, re-run or escalate, each with a plain-language reason. Download the decisions, the credibility report, "
    "and a state file for the API.</div></div>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="sec">What you get</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="grid">'
    '<div class="scard"><div class="h">Per-prediction decisions</div><div class="b">A likely range for every prediction '
    "and a gate decision you can wire into your workflow.</div></div>"
    '<div class="scard"><div class="h">Honest coverage</div><div class="b">Measured coverage with exact confidence '
    "intervals, the expected finite-sample behaviour, and coverage by region.</div></div>"
    '<div class="scard"><div class="h">Assumption checks</div><div class="b">A test for whether your queries look like '
    "your calibration data, so you know when the guarantee may not apply.</div></div>"
    '<div class="scard"><div class="h">Credibility report</div><div class="b">Context of use, verification, validation, '
    "uncertainty, applicability and an adequacy verdict for your risk level, as Markdown.</div></div>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="sec">Understand the evidence</div>', unsafe_allow_html=True)
st.markdown(
    "**Input familiarity** asks whether the model has seen similar conditions. "
    "**Prediction uncertainty** uses held-out errors to put a range around its answer. "
    "Envelope combines both checks with your acceptance thresholds."
)
with st.expander("See the mathematics behind a decision"):
    familiarity, uncertainty, decisions = st.tabs(["1. Familiarity", "2. Uncertainty", "3. Decision"])
    with familiarity:
        st.markdown(
            "Inputs are standardised using the training data. We average the distances to the "
            "**5 nearest training examples**, then divide by a threshold derived from distances "
            "within the training set (the **95th percentile** by default)."
        )
        st.latex(r"s(x) = \frac{\frac{1}{k}\sum_{j=1}^{k}\lVert z(x)-z(x_{(j)})\rVert_2}{T}")
        st.markdown(
            "A score of **1** is the boundary. Smaller scores mean more familiar inputs. "
            "This measures distance, not the probability that a prediction is correct."
        )
    with uncertainty:
        st.markdown(
            "The default is **split conformal prediction**. On held-out calibration examples, "
            "compute absolute errors and select their finite-sample-adjusted quantile."
        )
        st.latex(r"r_i = |y_i-\hat y_i|,\quad m=\lceil(n+1)(1-\alpha)\rceil,\quad q=r_{(m)}")
        st.latex(r"C(x)=[\hat y(x)-q,\;\hat y(x)+q]")
        st.markdown(
            "Here, **n** is the calibration sample size, errors are sorted from smallest to largest, "
            "and **α = 0.10** targets 90% marginal coverage. If m exceeds n, the range is unbounded. "
            "Coverage relies on exchangeable calibration and query examples; it is not a "
            "90% correctness probability for each individual prediction."
        )
    with decisions:
        st.markdown(
            "| Default rule | Recommendation |\n|---|---|\n"
            "| Score ≤ 1 and interval half-width ≤ your tolerance | Trust |\n"
            "| Score ≤ 1 but interval too wide | Re-run simulation |\n"
            "| 1 < score ≤ 1.5 | Re-run simulation |\n"
            "| Score > 1.5 | Ask an expert |\n\n"
            "The gate applies explicit rules. **Model risk level** selects the report's adequacy "
            "rules; it does not change these per-prediction cutoffs."
        )
    st.caption(
        "Additional evidence includes measured coverage with exact binomial confidence intervals, "
        "a classifier-based distribution-shift check, coverage by region, and tail-error metrics. "
        "A shift warning is a diagnostic; it does not automatically override the gate."
    )

st.markdown('<div class="sec">Use the results in context</div>', unsafe_allow_html=True)
st.markdown(
    "- Coverage guarantees require exchangeable calibration and query data. Input similarity alone "
    "does not establish that assumption.\n"
    "- Re-runs and expert reviews are recommendations; envelope does not execute or assign them.\n"
    "- Adequacy thresholds are illustrative defaults. Your reviewers decide what is adequate for your context of use.\n"
    "- The report is a decision-support draft, not regulatory advice or a compliance claim."
)
st.markdown(FOOTER, unsafe_allow_html=True)
