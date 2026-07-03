"""Statistical-significance tooling for EchoFuseNet evaluation (Day 11).

Two complementary sources of uncertainty are handled:

* **Across-fold spread** — with k-fold CV (see ``crossval``) each fold yields one
  score; ``mean_confidence_interval`` turns those k scores into a mean and a
  t-based confidence interval (small-sample correct — Student-t, not normal).
* **Within-test-set sampling noise** — for a single fixed test fold (e.g. DS2),
  ``bootstrap_metric_ci`` resamples the *beats* with replacement to put a CI
  around any metric, without retraining.

For *comparing* two systems (e.g. a fusion model vs. a single-branch ablation)
there are paired tests: ``paired_ttest`` / ``wilcoxon_test`` over per-fold scores,
and ``mcnemar_test`` over per-sample correctness on one shared test set.

SciPy (already a project dependency) provides the distributions; everything
degrades gracefully on the small-sample / all-tied edge cases that arise with a
handful of folds instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class Interval:
    """A point estimate with a (lower, upper) confidence interval."""

    point: float
    low: float
    high: float
    confidence: float

    def __str__(self) -> str:
        pct = int(round(self.confidence * 100))
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}] ({pct}% CI)"


@dataclass
class TestResult:
    """A significance-test statistic + p-value with the test's name."""

    name: str
    statistic: float
    pvalue: float

    def significant(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha

    def __str__(self) -> str:
        return f"{self.name}: stat={self.statistic:.4f}, p={self.pvalue:.4g}"


def mean_confidence_interval(
    values: Sequence[float], confidence: float = 0.95
) -> Interval:
    """Mean and Student-t confidence interval of a small sample (e.g. k folds).

    With fewer than two values a CI cannot be formed, so the interval collapses
    to the point estimate.
    """
    a = np.asarray(values, dtype=np.float64).ravel()
    n = a.size
    if n == 0:
        return Interval(0.0, 0.0, 0.0, confidence)
    mean = float(a.mean())
    if n < 2 or np.ptp(a) == 0:  # single value / all folds identical
        return Interval(mean, mean, mean, confidence)
    sem = float(stats.sem(a))    # sample std / sqrt(n), ddof=1
    if sem == 0.0:
        return Interval(mean, mean, mean, confidence)
    half = sem * float(stats.t.ppf((1 + confidence) / 2, n - 1))
    return Interval(mean, mean - half, mean + half, confidence)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: MetricFn,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile-bootstrap CI for a metric over one fixed test set.

    Resamples ``(y_true, y_pred)`` index pairs with replacement ``n_boot`` times
    and reports the empirical percentile interval. ``metric_fn`` is any
    ``(y_true, y_pred) -> float`` callable (see ``metrics.accuracy_score`` /
    ``metrics.macro_f1_score``).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    n = y_true.size
    point = float(metric_fn(y_true, y_pred))
    if n == 0:
        return Interval(point, point, point, confidence)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1 - confidence) / 2
    low = float(np.percentile(boot, 100 * alpha))
    high = float(np.percentile(boot, 100 * (1 - alpha)))
    return Interval(point, low, high, confidence)


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Paired two-sided t-test over matched per-fold scores of two systems."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("paired_ttest needs equal-length score vectors")
    if a.size < 2 or np.allclose(a, b):
        return TestResult("paired t-test", 0.0, 1.0)  # no detectable difference
    res = stats.ttest_rel(a, b)
    return TestResult("paired t-test", float(res.statistic), float(res.pvalue))


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Wilcoxon signed-rank test (non-parametric paired) over per-fold scores."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("wilcoxon_test needs equal-length score vectors")
    # All-zero differences make Wilcoxon undefined -> report "not significant".
    if a.size < 1 or np.allclose(a, b):
        return TestResult("wilcoxon", 0.0, 1.0)
    res = stats.wilcoxon(a, b)
    return TestResult("wilcoxon", float(res.statistic), float(res.pvalue))


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> TestResult:
    """Exact McNemar test comparing two classifiers on one shared test set.

    ``correct_a`` / ``correct_b`` are per-sample boolean correctness masks for
    the two models over the *same* samples. Uses the exact binomial p-value on
    the discordant pairs (the count where exactly one model is right), which is
    the right choice when discordances are few.
    """
    a = np.asarray(correct_a, dtype=bool).ravel()
    b = np.asarray(correct_b, dtype=bool).ravel()
    if a.shape != b.shape:
        raise ValueError("mcnemar_test needs equal-length correctness masks")
    b01 = int(np.sum(a & ~b))  # a right, b wrong
    b10 = int(np.sum(~a & b))  # a wrong, b right
    n_disc = b01 + b10
    if n_disc == 0:
        return TestResult("mcnemar", 0.0, 1.0)
    smaller = min(b01, b10)
    pvalue = float(
        stats.binomtest(smaller, n_disc, 0.5, alternative="two-sided").pvalue
    )
    return TestResult("mcnemar", float(smaller), pvalue)
