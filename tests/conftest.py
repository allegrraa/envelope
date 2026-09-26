"""Shared fixtures."""
import numpy as np
import pytest

from envelope.config import load_config
from envelope.data import make_demo_data
from envelope.surrogate import TorchMLPSurrogate


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides={"surrogate": {"mlp_epochs": 600}})


@pytest.fixture(scope="session")
def demo_data():
    return make_demo_data(seed=0)


@pytest.fixture(scope="session")
def mlp(demo_data):
    return TorchMLPSurrogate(epochs=600, seed=0).fit(demo_data.X_train, demo_data.y_train)


def iid_problem(seed: int, n_cal: int = 1000, n_test: int = 4000):
    """i.i.d. heteroscedastic regression with a fixed (imperfect) predictor."""
    rng = np.random.default_rng(seed)
    n = n_cal + n_test
    X = rng.uniform(-2, 2, size=(n, 2))
    f = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1]
    y = f + rng.normal(0, 0.1 + 0.3 * np.abs(X[:, 0]), n)
    yhat = f + 0.05 * X[:, 1] ** 2  # slightly biased model
    return X[:n_cal], y[:n_cal], yhat[:n_cal], X[n_cal:], y[n_cal:], yhat[n_cal:]
