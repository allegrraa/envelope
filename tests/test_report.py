import pandas as pd

from envelope.data import load_external
from envelope.pipeline import run_demo, run_pipeline
from envelope.report import DISCLAIMER


def test_report_demo_white_box(cfg, demo_data, mlp):
    run = run_demo(cfg, use_fragility=True, risk_level="high", context_of_use="Test context", data=demo_data, model=mlp)
    md = run.result.report_md
    for section in [
        "Context of Use", "Model Risk Level", "Verification", "Validation", "Uncertainty Quantification",
        "Applicability", "Adequacy Verdict", "Sensitivity to parameter perturbation", "Tail and tilted risk",
    ]:
        assert section in md, section
    assert "Test context" in md
    assert "Mean sigma(x)" in md
    assert "not a certified robustness bound" in md
    assert "illustrative design choice" in md
    assert "t = **3**" in md  # high risk -> t=3
    assert "INSUFFICIENT" in md  # test set goes to x1 = 8
    assert DISCLAIMER in md
    assert set(run.result.method_table["method"]) == {"split", "adaptive", "fragility"}


def test_report_black_box_upload(cfg, demo_data):
    from envelope.surrogate import GBRSurrogate

    d = demo_data
    gbr = GBRSurrogate().fit(d.X_train, d.y_train)
    f = d.feature_names
    tr = pd.DataFrame(d.X_train, columns=f)
    ca = pd.DataFrame(d.X_cal, columns=f).assign(y_true=d.y_cal, y_pred=gbr.predict(d.X_cal))
    qu = pd.DataFrame(d.X_ref, columns=f).assign(y_pred=gbr.predict(d.X_ref))  # unlabelled, in range
    ext = load_external(tr, ca, qu, cfg["bounds"], cfg["units"])
    res = run_pipeline(
        feature_names=ext.feature_names, X_train=ext.X_train, X_cal=ext.X_cal, y_cal=ext.y_cal,
        yhat_cal=ext.yhat_cal, X_query=ext.X_query, yhat_query=ext.yhat_query, y_query=ext.y_query,
        cfg=cfg, risk_level="low", context_of_use="", use_fragility=True, checks=ext.checks,
    )
    md = res.report_md
    assert res.method == "split"  # fragility request falls back for black-box
    assert "Not available" in md and "black-box" in md
    assert "Tail and tilted risk" in md
    assert "calibration set" in md
    assert not res.fragility_available


def test_upload_schema_error():
    import pytest

    tr = pd.DataFrame({"a": [1.0, 2.0]})
    ca = pd.DataFrame({"a": [1.0], "y_true": [1.0]})  # missing y_pred
    qu = pd.DataFrame({"a": [1.0], "y_pred": [1.0]})
    with pytest.raises(ValueError):
        load_external(tr, ca, qu)


def test_group_table_in_report(cfg, demo_data, mlp):
    run = run_demo(cfg, data=demo_data, model=mlp)
    gt, gi = run.result.summary["group_table"], run.result.summary["group_info"]
    assert len(gt) == 4 and gt["n_query"].sum() == len(demo_data.X_test)
    assert gt["coverage_group"].notna().all()
    assert gi["worst"] is not None and gi["worst"]["coverage_group"] == gt["coverage_group"].min()
    md = run.result.report_md
    assert "Coverage by envelope-score bin" in md and "Worst bin" in md
    assert "Guarantee validity" in md and "Clopper-Pearson" in md
