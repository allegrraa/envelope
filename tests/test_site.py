import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from envelope.config import load_config
from envelope.data import load_external
from envelope.intake import ColumnMapping, apply_mapping, guess_mapping, suggest_tolerance
from envelope.pipeline import run_pipeline

APP = __file__.rsplit("/tests/", 1)[0] + "/app.py"


def company_files(demo_data, mlp):
    """The demo data exported with a company's own column names."""
    d = demo_data
    names = ["temperature", "load", "speed"]
    train = pd.DataFrame(d.X_train, columns=names).assign(batch_id="A")  # extra non-numeric column
    cal = pd.DataFrame(d.X_cal, columns=names).assign(measured=d.y_cal, model_output=mlp.predict(d.X_cal))
    query = pd.DataFrame(d.X_test, columns=names).assign(model_output=mlp.predict(d.X_test))
    return train, cal, query


def test_guess_mapping_company_names(demo_data, mlp):
    train, cal, query = company_files(demo_data, mlp)
    cal = cal.rename(columns={"measured": "actual", "model_output": "prediction"})
    query = query.rename(columns={"model_output": "prediction"})
    m = guess_mapping(train, cal, query)
    assert m.features == ["temperature", "load", "speed"]
    assert (m.target, m.prediction, m.query_target) == ("actual", "prediction", None)


def test_apply_mapping_and_pipeline(demo_data, mlp):
    train, cal, query = company_files(demo_data, mlp)
    m = ColumnMapping(features=["temperature", "load", "speed"], target="measured", prediction="model_output")
    tr, ca, qu = apply_mapping(train, cal, query, m)
    assert list(ca.columns) == ["temperature", "load", "speed", "y_true", "y_pred"]
    cfg = load_config()
    cfg["bounds"], cfg["units"] = {}, {}
    ext = load_external(tr, ca, qu, {}, {})
    res = run_pipeline(feature_names=ext.feature_names, X_train=ext.X_train, X_cal=ext.X_cal, y_cal=ext.y_cal,
                       yhat_cal=ext.yhat_cal, X_query=ext.X_query, yhat_query=ext.yhat_query, cfg=cfg)
    assert res.summary["gate_counts"]["ESCALATE_TO_HUMAN"] > 0
    assert "temperature" in res.report_md


def test_apply_mapping_errors(demo_data, mlp):
    train, cal, query = company_files(demo_data, mlp)
    with pytest.raises(ValueError, match="missing"):
        apply_mapping(train, cal, query, ColumnMapping(["temperature", "pressure"], "measured", "model_output"))
    with pytest.raises(ValueError, match="different"):
        apply_mapping(train, cal, query, ColumnMapping(["temperature"], "measured", "measured"))
    with pytest.raises(ValueError, match="at least one"):
        apply_mapping(train, cal, query, ColumnMapping([], "measured", "model_output"))


def test_suggest_tolerance():
    y = np.zeros(1000)
    yhat = np.linspace(-1, 1, 1000)
    assert suggest_tolerance(y, yhat, 0.1) == 1.1  # 1.25 * 0.9 = 1.125, rounded to 2 significant figures


def test_all_pages_load():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert any("Know when to trust" in m.value for m in at.markdown)
    for page in ("site/assess.py", "site/example.py", "site/tour.py", "site/api_docs.py"):
        at.switch_page(page).run()
        assert not at.exception, page


def test_assess_with_sample_data():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.switch_page("site/assess.py").run()
    assert any("Upload all three files" in i.value for i in at.info)  # waits for files by default
    at.radio[0].set_value("Use sample data").run()
    next(b for b in at.button if b.label == "Run assessment").click().run()
    assert not at.exception and not at.error
    assert any("INSUFFICIENT at medium risk" in m.value or "ADEQUATE at medium risk" in m.value for m in at.markdown)
    assert any("Guarantee validity" in m.value for m in at.markdown)
    labels = [b.label for b in at.get("download_button")]
    assert {"Credibility report (.md)", "Decisions (.csv)", "API state (.npz)"} <= set(labels)


def test_downloaded_state_works_with_api(tmp_path, demo_data, mlp):
    """The 'API state' a company downloads after an assessment loads in the API with their column names."""
    from fastapi.testclient import TestClient

    from envelope.api import create_app
    from envelope.state import save_state

    train, cal, query = company_files(demo_data, mlp)
    tr, ca, qu = apply_mapping(train, cal, query, ColumnMapping(["temperature", "load", "speed"], "measured", "model_output"))
    cfg = load_config()
    cfg["bounds"], cfg["units"] = {}, {}
    ext = load_external(tr, ca, qu, {}, {})
    res = run_pipeline(feature_names=ext.feature_names, X_train=ext.X_train, X_cal=ext.X_cal, y_cal=ext.y_cal,
                       yhat_cal=ext.yhat_cal, X_query=ext.X_query, yhat_query=ext.yhat_query, cfg=cfg)
    path = save_state(tmp_path / "envelope_state.npz", cfg=cfg, feature_names=ext.feature_names, method=res.method,
                      X_train=ext.X_train, X_cal=ext.X_cal, y_cal=ext.y_cal, yhat_cal=ext.yhat_cal,
                      threshold=res.engine.ad.threshold)
    client = TestClient(create_app(path))
    assert client.get("/health").json()["features"] == ["temperature", "load", "speed"]
    assert client.post("/assess", json={"features": [8.0, 2.5, 2.5], "y_pred": 0.4}).json()["decision"] == "ESCALATE_TO_HUMAN"
    assert client.get("/report").status_code == 404  # no report bundled with a website-downloaded state
