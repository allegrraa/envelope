import numpy as np

from envelope.shift_check import shift_check


def test_iid_is_ok():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 5, size=(900, 3))
    res = shift_check(X[:400], X[400:], seed=0)
    assert res["status"] == "OK", res
    assert res["auc"] < 0.6 and res["p_value"] > 0.05


def test_clear_shift_warns(demo_data):
    # demo query set extends x1 to 8: clearly shifted relative to calibration inputs in [0, 5]
    res = shift_check(demo_data.X_cal, demo_data.X_test, seed=0)
    assert res["status"] == "WARN"
    assert res["auc"] > 0.7 and res["p_value"] <= 0.05
    assert res["model"] in {"logistic regression", "random forest"}


def test_p_value_bounds_and_skip():
    rng = np.random.default_rng(1)
    res = shift_check(rng.normal(size=(200, 2)), rng.normal(size=(200, 2)) + 3, n_perm=50)
    assert res["p_value"] == 1 / 51  # perfectly separated -> smallest attainable p
    assert shift_check(np.zeros((3, 2)), np.zeros((3, 2)))["status"] == "SKIP"
