"""Discrimination + calibration metrics for binary risk (numpy/scipy only).

Deliberately dependency-light (no sklearn): each metric is a small, verifiable
function so the whole suite is unit-testable now, before any real risk model is
fit. All take 1-D arrays of ground-truth labels ``y`` (0/1) and predicted risk
probabilities ``p`` in [0, 1].
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _as_arrays(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y and p must match shape; got {y.shape} vs {p.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    return y, p


def auroc(y, p) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney U) statistic.

    Handles ties correctly through average ranks. Undefined when only one class
    is present -> returns ``nan``.
    """
    y, p = _as_arrays(y, p)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    sum_pos_ranks = ranks[y == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y, p) -> float:
    """Average precision (area under the precision-recall curve, AP form).

    ``AP = Σ_n (R_n - R_{n-1}) · P_n`` over thresholds set at each score, sorted
    descending. Equivalent to sklearn's ``average_precision_score``.
    """
    y, p = _as_arrays(y, p)
    total_pos = (y == 1).sum()
    if total_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")  # stable; highest score first
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / total_pos
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_recall) * precision))


def brier_score(y, p) -> float:
    """Mean squared error between predicted risk and outcome (lower = better)."""
    y, p = _as_arrays(y, p)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Expected Calibration Error over equal-width probability bins.

    ``ECE = Σ_b (n_b / N) · |acc_b - conf_b|`` — the average gap between predicted
    confidence and observed frequency, weighted by bin population. Lower = better
    calibrated.
    """
    y, p = _as_arrays(y, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index per sample; clip so p == 1.0 lands in the last bin.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    n = y.size
    for b in range(n_bins):
        in_bin = idx == b
        n_b = in_bin.sum()
        if n_b == 0:
            continue
        acc_b = y[in_bin].mean()
        conf_b = p[in_bin].mean()
        ece += (n_b / n) * abs(acc_b - conf_b)
    return float(ece)
