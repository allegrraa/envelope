import numpy as np
import torch

from envelope.metrics import tail_risk, worst_region_error
from envelope.term import tilted_loss, tilted_risk_np

L = torch.tensor([0.1, 0.5, 1.0, 2.0, 4.0])


def test_t_zero_is_mean():
    assert torch.isclose(tilted_loss(L, 0.0), L.mean())
    assert torch.isclose(tilted_loss(L, 1e-4), L.mean(), atol=1e-3)
    assert torch.isclose(tilted_loss(L, -1e-4), L.mean(), atol=1e-3)


def test_large_positive_t_approaches_max():
    assert abs(float(tilted_loss(L, 200.0)) - 4.0) < 0.02


def test_large_negative_t_approaches_min():
    assert abs(float(tilted_loss(L, -200.0)) - 0.1) < 0.02


def test_monotone_in_t_and_stable():
    vals = [float(tilted_loss(L, t)) for t in (-5, -1, 0, 1, 5)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))
    big = torch.tensor([1000.0, 1.0])
    assert torch.isfinite(tilted_loss(big, 50.0))  # no overflow thanks to logsumexp


def test_numpy_matches_torch():
    for t in (-3.0, 0.0, 0.5, 3.0):
        assert np.isclose(tilted_risk_np(L.numpy(), t), float(tilted_loss(L, t)), atol=1e-5)


def test_tail_and_worst_region():
    e = np.arange(1, 11, dtype=float)
    assert tail_risk(e, 0.1) == 10.0
    assert tail_risk(e, 0.2) == 9.5
    X = np.column_stack([np.linspace(0, 1, 100), np.zeros(100)])
    y = np.zeros(100)
    yhat = np.where(X[:, 0] > 0.75, 2.0, 0.0)
    wr = worst_region_error(X, y, yhat, bins=4, min_count=5)
    assert np.isclose(wr["rmse"], 2.0)
