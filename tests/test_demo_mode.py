import json
import re

import pytest
from streamlit.testing.v1 import AppTest

from envelope.config import load_config
from envelope.demo_story import FINAL_DISCLAIMER, FINAL_GUARANTEE, build_story, caption, demo_script, numbers_json

APP = __file__.rsplit("/tests/", 1)[0] + "/app.py"


@pytest.fixture(scope="module")
def full_cfg():
    return load_config()  # the real demo configuration (full training epochs)


def test_determinism_two_runs_identical(full_cfg):
    a = numbers_json(build_story(full_cfg))
    b = numbers_json(build_story(full_cfg))  # retrains the surrogate from scratch
    assert a == b
    json.loads(a)


def test_simulator_design_sanity(full_cfg, demo_data):
    """The fixed simulator design (regime change past x1 = 5) makes out-of-range error >= 3x in-range error."""
    n = build_story(full_cfg, data=demo_data).numbers
    assert n["rmse_out_of_range"] >= 3 * n["rmse_in_range"]


def test_script_and_captions_filled(full_cfg, demo_data):
    n = build_story(full_cfg, data=demo_data).numbers
    md = demo_script(n)
    for mark in ("0-8s", "8-20s", "20-35s", "35-47s", "47-56s", "56-60s"):
        assert f"## {mark}" in md
    assert not re.search(r"\{|\}|TODO|TBD|XXX|\?\?|<[a-z_]+>|nan", md), "unfilled placeholder in demo_script.md"
    assert FINAL_GUARANTEE in md and FINAL_DISCLAIMER in md
    assert "Confident, but wrong outside its training range" in caption(2, n)
    for s in range(1, 6):
        assert "{" not in caption(s, n)


def _stage(at: AppTest) -> str:
    return next(m.value for m in at.markdown if "Stage " in m.value and "/5" in m.value)


def test_navigation_and_reset():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.switch_page("site/tour.py").run()
    assert not at.exception
    assert "Stage 1/5" in _stage(at)
    assert any("Synthetic demo data" in m.value for m in at.markdown)
    nxt = lambda: next(b for b in at.button if b.label.startswith("Next"))
    for k in range(2, 6):
        nxt().click().run()
        assert not at.exception
        assert f"Stage {k}/5" in _stage(at)
    assert nxt().disabled
    assert any(FINAL_GUARANTEE in m.value for m in at.markdown)
    assert any(FINAL_DISCLAIMER in m.value for m in at.markdown)
    next(b for b in at.button if b.label == "Generate report").click().run()
    assert any("at high risk" in m.value and ("INSUFFICIENT" in m.value or "ADEQUATE" in m.value) for m in at.markdown)
    next(r for r in at.radio if r.label == "Model risk level").set_value("Medium").run()  # displayed label
    assert any("at medium risk" in m.value for m in at.markdown)
    next(b for b in at.button if b.label == "Generate report").click().run()
    assert any("at medium risk" in m.value for m in at.markdown)  # risk survives the rerun
    next(b for b in at.button if b.label == "Reset").click().run()
    assert "Stage 1/5" in _stage(at)


def test_explore_mode_still_works():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.switch_page("site/example.py").run()
    assert not at.exception
    assert any("Can I trust the surrogate here?" in m.value for m in at.markdown)
