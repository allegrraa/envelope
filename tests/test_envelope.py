import numpy as np

from envelope.envelope import ApplicabilityDomain


def test_out_of_range_points_flagged(demo_data):
    ad = ApplicabilityDomain(k=5, percentile=95).fit(demo_data.X_train)
    far = np.array([[8.0, 2.5, 2.5], [7.5, 1.0, 4.0], [2.5, 9.0, 2.5], [-3.0, -3.0, -3.0]])
    res = ad.evaluate(far)
    assert not res.inside.any()
    assert (res.score > 1.5).all()


def test_in_range_mostly_inside(demo_data):
    ad = ApplicabilityDomain(k=5, percentile=95).fit(demo_data.X_train)
    center = np.array([[2.5, 2.5, 2.5], [1.5, 3.0, 2.0], [3.5, 2.0, 3.0]])
    assert ad.evaluate(center).inside.all()
    # held-out in-range points: roughly the percentile falls inside
    frac_inside = ad.evaluate(demo_data.X_cal).inside.mean()
    assert 0.85 <= frac_inside <= 1.0


def test_score_monotone_in_extrapolation(demo_data):
    ad = ApplicabilityDomain().fit(demo_data.X_train)
    X = np.column_stack([np.linspace(5, 8, 10), np.full(10, 2.5), np.full(10, 2.5)])
    assert np.all(np.diff(ad.evaluate(X).score) > 0)


def _fresh(seed, n):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 5, size=(n, 3))


def test_pvalue_flag_rate_controlled():
    from envelope.envelope import ApplicabilityDomain

    rates = []
    for seed in range(5):
        ad = ApplicabilityDomain().fit(_fresh(seed, 600)).calibrate(_fresh(100 + seed, 400))
        p = ad.pvalues(_fresh(200 + seed, 3000))
        assert ((p > 0) & (p <= 1)).all()
        rates.append((p < 0.05).mean())
    assert np.mean(rates) <= 0.05 + 0.01
    assert max(rates) <= 0.05 + 0.03


def test_bh_controls_and_detects():
    from envelope.envelope import ApplicabilityDomain, benjamini_hochberg

    ad = ApplicabilityDomain().fit(_fresh(0, 600)).calibrate(_fresh(1, 400))
    p_null = ad.pvalues(_fresh(2, 1000))
    assert benjamini_hochberg(p_null, 0.05).mean() <= 0.01  # all-null batch: almost nothing flagged
    far = np.column_stack([np.linspace(7, 9, 50), np.full(50, 2.5), np.full(50, 2.5)])
    mix = np.vstack([_fresh(3, 950), far])
    flags = benjamini_hochberg(ad.pvalues(mix), 0.05)
    assert flags[-50:].all()           # every far point flagged
    assert flags[:950].mean() <= 0.05  # few in-distribution points flagged


def test_bh_matches_definition():
    from envelope.envelope import benjamini_hochberg

    p = np.array([0.001, 0.02, 0.03, 0.2, 0.5])
    # thresholds 0.05*k/5 = .01,.02,.03,.04,.05 -> largest k with p_(k) <= thr is 3
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, True, False, False]
