"""Paper 3 evaluation metrics: AUROC, average precision, Brier, ECE."""

import math

import numpy as np
import pytest

from paper3_cardiocausal.evaluation import (
    auroc,
    average_precision,
    brier_score,
    expected_calibration_error,
)


def test_auroc_perfect_and_reversed():
    y = [0, 0, 1, 1]
    assert auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auroc_known_value():
    # Matches sklearn.roc_auc_score for this canonical example.
    assert abs(auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.75) < 1e-9


def test_auroc_handles_ties():
    # All scores equal -> every pair is a tie -> AUROC 0.5.
    assert abs(auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9


def test_auroc_single_class_is_nan():
    assert math.isnan(auroc([1, 1, 1], [0.2, 0.5, 0.9]))


def test_average_precision_known_value():
    # Matches sklearn.average_precision_score for this canonical example.
    assert abs(average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.8333333) < 1e-6


def test_average_precision_perfect():
    assert abs(average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-9


def test_brier_score():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert abs(brier_score([1, 0], [0.5, 0.5]) - 0.25) < 1e-9


def test_ece_perfectly_calibrated_is_zero():
    # Predictions equal to outcomes -> zero calibration gap.
    y = [0, 0, 1, 1]
    assert expected_calibration_error(y, [0.0, 0.0, 1.0, 1.0], n_bins=10) == 0.0


def test_ece_detects_miscalibration():
    # Confident-but-wrong predictions -> large ECE (near 1).
    y = [0, 0, 0, 0]
    ece = expected_calibration_error(y, [0.99, 0.98, 0.97, 0.99], n_bins=10)
    assert ece > 0.9


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        auroc([0, 1], [0.5])
