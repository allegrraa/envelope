import numpy as np

from envelope.gate import Decision, GateConfig, gate_decision, gate_decisions

CFG = GateConfig(tolerance=0.5, escalate_score=1.5)


def test_trust_inside_and_narrow():
    assert gate_decision(0.8, 0.3, CFG) is Decision.TRUST
    assert gate_decision(1.0, 0.5, CFG) is Decision.TRUST  # boundaries inclusive


def test_rerun_inside_but_wide():
    assert gate_decision(0.8, 0.9, CFG) is Decision.RERUN
    assert gate_decision(0.8, float("inf"), CFG) is Decision.RERUN


def test_rerun_slightly_outside():
    assert gate_decision(1.2, 0.1, CFG) is Decision.RERUN
    assert gate_decision(1.5, 0.1, CFG) is Decision.RERUN


def test_escalate_far_outside():
    assert gate_decision(1.51, 0.1, CFG) is Decision.ESCALATE
    assert gate_decision(float("nan"), 0.1, CFG) is Decision.ESCALATE


def test_vectorised_and_config():
    out = gate_decisions(np.array([0.5, 0.5, 1.2, 3.0]), np.array([0.1, 1.0, 0.1, 0.1]), CFG)
    assert list(out) == ["TRUST", "RERUN_FULL_SIMULATION", "RERUN_FULL_SIMULATION", "ESCALATE_TO_HUMAN"]
    g = GateConfig.from_config({"gate": {"tolerance": 2.0, "escalate_score": 3.0}})
    assert gate_decision(2.5, 1.5, g) is Decision.RERUN


def test_inside_override():
    # p-value flag says outside although score <= 1 -> RERUN; score still drives ESCALATE
    assert gate_decision(0.9, 0.1, CFG, inside=False) is Decision.RERUN
    assert gate_decision(1.2, 0.1, CFG, inside=True) is Decision.TRUST
    assert gate_decision(2.0, 0.1, CFG, inside=True) is Decision.ESCALATE
