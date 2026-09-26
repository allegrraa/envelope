import pytest

from envelope.fragility import BlackBoxModelError, perturbation_sigma
from envelope.surrogate import GBRSurrogate


def test_sigma_larger_outside(demo_data, mlp):
    X = demo_data.X_test
    sigma = perturbation_sigma(mlp, X, K=30, tau=0.02, seed=0)
    inside = X[:, 0] <= 5.0
    assert (sigma >= 0).all()
    assert sigma[~inside].mean() > 1.1 * sigma[inside].mean()


def test_deterministic_with_seed(demo_data, mlp):
    a = perturbation_sigma(mlp, demo_data.X_ref[:20], seed=3)
    b = perturbation_sigma(mlp, demo_data.X_ref[:20], seed=3)
    assert (a == b).all()


def test_black_box_rejected(demo_data):
    gbr = GBRSurrogate().fit(demo_data.X_train[:100], demo_data.y_train[:100])
    with pytest.raises(BlackBoxModelError):
        perturbation_sigma(gbr, demo_data.X_ref[:5])
