"""API page: how to integrate envelope into your own systems."""
import streamlit as st

from ui_common import FOOTER, SITE_CSS

st.markdown(SITE_CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">Integrate</div><h1>API</h1>'
            "<p>Check predictions from your own pipeline, one request per prediction.</p></div>", unsafe_allow_html=True)
st.markdown("""
**1. Calibrate.** On **Assess your model**, run an assessment on your data, then download
**API state (.npz)** from the *Report & export* tab. The file holds your calibration set, the envelope
threshold and the gate settings, as plain arrays and JSON, with no pickles.

**2. Serve.**
```bash
pip install fastapi uvicorn
ENVELOPE_STATE=/path/to/envelope_state.npz uvicorn envelope.api:app --port 8000
```

**3. Call it.**
```bash
curl -s -X POST localhost:8000/assess -H 'Content-Type: application/json' \\
  -d '{"features": [2.5, 2.5, 2.5], "y_pred": 3.2}'
```
```json
{"y_pred": 3.2, "interval_low": 2.84, "interval_high": 3.56, "in_envelope": true, "envelope_score": 0.73,
 "decision": "TRUST", "reason": "Inside the validity envelope ...", "alpha": 0.1, "assumptions": "..."}
```

| Endpoint | Returns |
|---|---|
| `GET /health` | Loaded state, features (in order), method, alpha, envelope threshold, gate cutoffs |
| `POST /assess` | Likely range, envelope score, decision and reason for one prediction |
| `GET /report` | The latest credibility report (Markdown), if the state file points to one |
| `GET /docs` | Interactive OpenAPI documentation |

`features` must be in the same order as the input columns you chose when calibrating.
`y_pred` must come from the same model that produced the calibration predictions.
""")
st.markdown(FOOTER, unsafe_allow_html=True)
