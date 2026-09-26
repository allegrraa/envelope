import numpy as np
import pytest

from envelope.conformal import conformal_quantile
from envelope.coverage_report import beta_params, clopper_pearson, conformal_coverage_law, corrected_alpha


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(45, 50)
    assert lo == pytest.approx(0.7819, abs=1e-3) and hi == pytest.approx(0.9667, abs=1e-3)
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0


def _mc_coverage(n: int, alpha: float, reps: int = 20000, seed: int = 0) -> np.ndarray:
    """Coverage of split conformal over calibration draws, with U(0,1) scores (coverage = CDF(q) = q)."""
    rng = np.random.default_rng(seed)
    scores = np.sort(rng.uniform(size=(reps, n)), axis=1)
    k = int(np.ceil((n + 1) * (1 - alpha) - 1e-9))
    return scores[:, k - 1]


@pytest.mark.parametrize("n,alpha", [(50, 0.1), (200, 0.1), (500, 0.05)])
def test_beta_law_matches_monte_carlo(n, alpha):
    cov = _mc_coverage(n, alpha)
    law = conformal_coverage_law(n, alpha)
    assert cov.mean() == pytest.approx(law["mean"], abs=0.002)
    assert np.quantile(cov, 0.05) == pytest.approx(law["p05"], abs=0.004)
    assert (cov >= 1 - alpha).mean() == pytest.approx(law["prob_at_least_target"], abs=0.015)


def test_beta_params_consistent_with_quantile():
    # the k-th smallest of n uniform scores is Beta(k, n + 1 - k) and k = n + 1 - l
    n, alpha = 99, 0.1
    a, b = beta_params(n, alpha)
    assert a == int(np.ceil((n + 1) * (1 - alpha)))
    assert conformal_quantile(np.arange(1, n + 1, dtype=float), alpha) == a


@pytest.mark.parametrize("n", [100, 500])
def test_corrected_alpha_achieves_pac(n):
    alpha, delta = 0.1, 0.1
    a2 = corrected_alpha(n, alpha, delta)
    assert 0 < a2 <= alpha
    cov = _mc_coverage(n, a2, seed=1)
    assert (cov >= 1 - alpha).mean() >= 1 - delta - 0.01
    # the next-larger alpha' would violate the requirement (it is the largest feasible one)
    a3 = a2 + 1 / (n + 1)
    assert conformal_coverage_law(n, a3)["prob_at_least_target"] < 1 - delta or a3 > alpha


def test_corrected_alpha_tiny_n():
    assert corrected_alpha(5, 0.1, 0.1) == 0.0
