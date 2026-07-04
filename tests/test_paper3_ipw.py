"""IPW ATE + assumption diagnostics (balance, positivity, negative control)."""

import numpy as np

from paper3_cardiocausal.causal_validation import (
    ipw_ate,
    negative_control_test,
    positivity_check,
    stabilized_weights,
    standardized_mean_difference,
)


def test_ipw_recovers_effect_in_randomized_case():
    # Randomized treatment (propensity 0.5) with a known additive effect of 2.0.
    rng = np.random.default_rng(0)
    n = 4000
    t = rng.integers(0, 2, size=n)
    y = 1.0 + 2.0 * t + rng.normal(0, 0.5, size=n)  # true ATE = 2.0
    e = np.full(n, 0.5)
    assert abs(ipw_ate(t, y, e) - 2.0) < 0.1


def test_ipw_adjusts_for_confounding():
    # Confounder x drives both treatment and outcome; naive diff is biased, IPW
    # with the true propensity recovers the (zero) causal effect.
    rng = np.random.default_rng(1)
    n = 8000
    x = rng.normal(0, 1, size=n)
    e = 1 / (1 + np.exp(-x))                 # propensity depends on x
    t = (rng.uniform(size=n) < e).astype(float)
    y = x + rng.normal(0, 0.5, size=n)       # outcome depends on x, NOT on t
    naive = y[t == 1].mean() - y[t == 0].mean()
    adjusted = ipw_ate(t, y, e)
    assert abs(naive) > 0.3                   # confounded
    assert abs(adjusted) < 0.15               # de-confounded


def test_stabilized_weights_positive():
    t = np.array([1, 0, 1, 0])
    e = np.array([0.5, 0.5, 0.8, 0.2])
    w = stabilized_weights(t, e)
    assert np.all(w > 0)


def test_smd_zero_when_balanced():
    cov = np.array([1.0, 2.0, 1.0, 2.0])
    t = np.array([1, 1, 0, 0])               # identical means per arm
    assert abs(standardized_mean_difference(cov, t)) < 1e-9


def test_smd_detects_imbalance():
    # Treated centred at 5, controls at 1, each arm with unit-ish variance.
    cov = np.array([4.0, 6.0, 0.0, 2.0])
    t = np.array([1, 1, 0, 0])
    assert abs(standardized_mean_difference(cov, t)) > 1.0


def test_positivity_flags_extreme_propensity():
    e = np.array([0.01, 0.5, 0.5, 0.99])
    out = positivity_check(e, eps=0.05)
    assert out["violation_fraction"] == 0.5
    assert out["min_propensity"] == 0.01


def test_negative_control_passes_when_no_effect():
    rng = np.random.default_rng(2)
    n = 4000
    t = rng.integers(0, 2, size=n)
    nco = rng.normal(0, 1, size=n)           # unrelated to treatment
    e = np.full(n, 0.5)
    res = negative_control_test(t, nco, e, tol=0.1)
    assert res["passes"]
    assert abs(res["nco_effect"]) < 0.1
