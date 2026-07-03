"""Significance tooling: t-CI, bootstrap CI, paired tests, McNemar — edge cases."""

from functools import partial

import numpy as np

from paper1_echofusenet.training.metrics import accuracy_score, macro_f1_score
from paper1_echofusenet.training.stats import (
    bootstrap_metric_ci,
    mcnemar_test,
    mean_confidence_interval,
    paired_ttest,
    wilcoxon_test,
)


def test_mean_ci_contains_mean_and_is_ordered():
    vals = [0.90, 0.91, 0.89, 0.92, 0.88]
    iv = mean_confidence_interval(vals, confidence=0.95)
    assert iv.low <= iv.point <= iv.high
    assert abs(iv.point - np.mean(vals)) < 1e-12
    assert iv.high > iv.low  # non-degenerate spread


def test_mean_ci_degenerate_cases():
    # Single value -> interval collapses to the point.
    one = mean_confidence_interval([0.9])
    assert one.low == one.point == one.high == 0.9
    # Identical values -> zero-width interval, no NaN.
    same = mean_confidence_interval([0.8, 0.8, 0.8])
    assert same.low == same.high == same.point
    assert same.point == np.mean([0.8, 0.8, 0.8])


def test_bootstrap_ci_perfect_prediction_is_tight():
    y = np.array([0, 1, 2, 3, 4] * 20)
    iv = bootstrap_metric_ci(y, y.copy(), accuracy_score, n_boot=200, seed=0)
    assert iv.point == 1.0
    assert iv.low == 1.0 and iv.high == 1.0  # no variance when always correct


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=300)
    y_pred = y_true.copy()
    flip = rng.random(300) < 0.2  # ~20% errors
    y_pred[flip] = (y_pred[flip] + 1) % 3
    mf1 = partial(macro_f1_score, n_classes=3)
    iv = bootstrap_metric_ci(y_true, y_pred, mf1, n_boot=500, seed=1)
    assert iv.low <= iv.point <= iv.high
    assert 0.0 < iv.low < iv.high < 1.0


def test_paired_tests_on_identical_scores_not_significant():
    a = [0.9, 0.91, 0.89]
    assert paired_ttest(a, list(a)).pvalue == 1.0
    assert wilcoxon_test(a, list(a)).pvalue == 1.0


def test_paired_ttest_detects_consistent_gap():
    a = [0.80, 0.81, 0.79, 0.82, 0.80]
    b = [0.85, 0.87, 0.83, 0.88, 0.86]  # consistently higher, with varied gap
    res = paired_ttest(a, b)
    assert res.significant(alpha=0.05)
    assert res.pvalue < 0.05


def test_mcnemar_symmetric_vs_lopsided():
    n = 100
    # Symmetric disagreement -> not significant.
    a = np.zeros(n, dtype=bool)
    b = np.zeros(n, dtype=bool)
    a[:10] = True   # a right, b wrong on 10
    b[10:20] = True  # b right, a wrong on 10
    assert not mcnemar_test(a, b).significant()

    # Lopsided: a right & b wrong on 20, reverse on 1 -> significant.
    a2 = np.zeros(n, dtype=bool)
    b2 = np.zeros(n, dtype=bool)
    a2[:20] = True
    b2[20:21] = True
    assert mcnemar_test(a2, b2).significant()


def test_mcnemar_no_disagreement_is_pvalue_one():
    mask = np.array([True, False, True, True])
    assert mcnemar_test(mask, mask.copy()).pvalue == 1.0
