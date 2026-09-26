import pytest
from fastapi.testclient import TestClient

from envelope.api import create_app
from envelope.pipeline import run_demo
from envelope.state import load_state, save_state


@pytest.fixture(scope="module")
def state_file(tmp_path_factory, cfg, demo_data, mlp):
    out = tmp_path_factory.mktemp("state")
    run = run_demo(cfg, data=demo_data, model=mlp, context_of_use="API test")
    (out / "report.md").write_text(run.result.report_md)
    return save_state(
        out / "state.npz",
        cfg=cfg,
        feature_names=demo_data.feature_names,
        method=run.result.method,
        X_train=demo_data.X_train,
        X_cal=demo_data.X_cal,
        y_cal=demo_data.y_cal,
        yhat_cal=mlp.predict(demo_data.X_cal),
        threshold=run.result.engine.ad.threshold,
        report_path=out / "report.md",
    )


@pytest.fixture(scope="module")
def client(state_file):
    return TestClient(create_app(state_file))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["features"] == ["x1", "x2", "x3"]
    assert body["alpha"] == pytest.approx(0.1)


def test_assess_inside_trust(client, mlp):
    x = [2.5, 2.5, 2.5]
    r = client.post("/assess", json={"features": x, "y_pred": float(mlp.predict([x])[0])})
    assert r.status_code == 200
    b = r.json()
    assert set(b) == {"y_pred", "interval_low", "interval_high", "in_envelope", "envelope_score",
                      "decision", "reason", "alpha", "assumptions"}
    assert b["in_envelope"] is True and b["decision"] == "TRUST"
    assert b["interval_low"] < b["y_pred"] < b["interval_high"]
    assert "exchangeable" in b["assumptions"]


def test_assess_far_outside_escalates(client):
    r = client.post("/assess", json={"features": [8.0, 2.5, 2.5], "y_pred": 0.4})
    b = r.json()
    assert b["in_envelope"] is False and b["envelope_score"] > 1.5
    assert b["decision"] == "ESCALATE_TO_HUMAN"
    assert "outside" in b["reason"]


def test_assess_slightly_outside_reruns(client):
    b = client.post("/assess", json={"features": [5.5, 2.5, 2.5], "y_pred": 0.0}).json()
    assert 1.0 < b["envelope_score"] <= 1.5
    assert b["decision"] == "RERUN_FULL_SIMULATION"


@pytest.mark.parametrize("payload", [
    {"features": [1.0, 2.0], "y_pred": 1.0},           # wrong length
    {"features": [], "y_pred": 1.0},                   # empty
    {"features": [1.0, 2.0, 3.0]},                     # missing y_pred
    {"features": ["a", 2.0, 3.0], "y_pred": 1.0},      # non-numeric
])
def test_assess_validation(client, payload):
    assert client.post("/assess", json=payload).status_code == 422


def test_report(client):
    r = client.get("/report")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert "# Credibility Evidence Report" in r.text and "API test" in r.text


def test_matches_pipeline(state_file, cfg, demo_data, mlp):
    """API assessments equal the batch pipeline's decisions."""
    s = load_state(state_file)
    run = run_demo(cfg, data=demo_data, model=mlp)
    df = run.result.queries.head(25)
    for _, row in df.iterrows():
        a = s.assess([row.x1, row.x2, row.x3], row.y_pred)
        assert a["decision"] == row.decision
        assert a["interval_low"] == pytest.approx(row.lower)


def test_missing_state_returns_503(tmp_path):
    c = TestClient(create_app(tmp_path / "nope.npz"))
    assert c.get("/health").status_code == 503
    assert c.post("/assess", json={"features": [1, 2, 3], "y_pred": 0}).status_code == 503
