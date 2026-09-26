import numpy as np
import pytest

from envelope.conformal import AdaptiveConformal, NormalizedConformal, SplitConformal, conformal_quantile
from tests.conftest import iid_problem

ALPHA = 0.1
TOL = 0.03


def test_quantile_level():
    scores = np.arange(1, 101, dtype=float)  # n = 100
    # k = ceil(101 * 0.9) = 91 -> 91st smallest
    assert conformal_quantile(scores, 0.1) == 91.0
    assert conformal_quantile(np.arange(5.0), 0.01) == float("inf")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_coverage_iid(seed):
    Xc, yc, hc, Xt, yt, ht = iid_problem(seed)
    iv = SplitConformal(ALPHA).fit(yc, hc).predict(ht)
    assert iv.covers(yt).mean() >= 1 - ALPHA - TOL


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adaptive_coverage_iid(seed):
    Xc, yc, hc, Xt, yt, ht = iid_problem(seed)
    ac = AdaptiveConformal(ALPHA, seed=seed).fit(Xc, yc, hc)
    iv = ac.predict(Xt, ht)
    assert iv.covers(yt).mean() >= 1 - ALPHA - TOL
    # adaptive intervals should actually vary with the noise level
    assert iv.half_width.std() > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_normalized_coverage_iid(seed):
    Xc, yc, hc, Xt, yt, ht = iid_problem(seed)
    sigma = lambda X: 0.05 + 0.1 * np.abs(X[:, 0])  # any fixed function of x keeps validity
    nc = NormalizedConformal(ALPHA).fit(yc, hc, sigma(Xc))
    iv = nc.predict(ht, sigma(Xt))
    assert iv.covers(yt).mean() >= 1 - ALPHA - TOL


def test_normalized_coverage_with_fragility_sigma(demo_data, mlp):
    """Real fragility sigma on in-range (i.i.d.) demo data."""
    from envelope.fragility import perturbation_sigma

    d = demo_data
    nc = NormalizedConformal(ALPHA).fit(d.y_cal, mlp.predict(d.X_cal), perturbation_sigma(mlp, d.X_cal))
    iv = nc.predict(mlp.predict(d.X_ref), perturbation_sigma(mlp, d.X_ref))
    assert iv.covers(d.y_ref).mean() >= 1 - ALPHA - 0.05  # n_ref = 300 -> looser tolerance


def test_group_conditional_iid_per_bin():
    from envelope.conformal import GroupConditionalConformal, quantile_bin_edges

    Xc, yc, hc, Xt, yt, ht = iid_problem(0, n_cal=2000, n_test=8000)
    g_cal, g_test = np.abs(Xc[:, 0]), np.abs(Xt[:, 0])  # group by |x0| (noise grows with it)
    edges = quantile_bin_edges(g_test, 4)
    gc = GroupConditionalConformal(ALPHA, min_count=30).fit(yc, hc, g_cal, edges)
    iv = gc.predict(ht, g_test)
    bins = gc.assign(g_test)
    cov = [iv.covers(yt)[bins == b].mean() for b in range(4)]
    assert all(c >= 1 - ALPHA - 0.03 for c in cov), cov
    assert all(b["source"] == "bin" for b in gc.bins)
    # heteroscedastic noise -> widths increase across bins
    assert gc.bins[0]["q"] < gc.bins[-1]["q"]


def test_group_conditional_fallback():
    from envelope.conformal import GroupConditionalConformal

    rng = np.random.default_rng(0)
    y, yhat = rng.normal(size=200), np.zeros(200)
    group = np.r_[np.zeros(190), np.ones(10)]  # bin 1 has only 10 calibration points
    gc = GroupConditionalConformal(ALPHA, min_count=30).fit(y, yhat, group, edges=np.array([0.5]))
    assert gc.bins[0]["source"] == "bin"
    assert gc.bins[1]["source"] == "global fallback" and gc.bins[1]["q"] == gc.global_q
