"""Metrics: confusion matrix, per-class P/R/F1, macro-F1, division guards."""

import numpy as np

from paper1_echofusenet.training.metrics import (
    classification_report,
    confusion_matrix,
    format_report,
)


def test_confusion_matrix_counts():
    y_true = np.array([0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    expected = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
    assert np.array_equal(cm, expected)
    assert cm.sum() == len(y_true)


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 2, 3, 4, 0, 1])
    report = classification_report(y, y.copy(), n_classes=5)
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert np.allclose(report.f1, 1.0)


def test_accuracy_matches_manual():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])  # 3/4 correct
    report = classification_report(y_true, y_pred, n_classes=2)
    assert report.accuracy == 0.75


def test_macro_f1_averages_only_supported_classes():
    # Class 2 never appears in y_true -> excluded from macro average.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    report = classification_report(y_true, y_pred, n_classes=3)
    assert report.support[2] == 0
    # Both present classes are perfect -> macro is 1.0, not dragged down by class 2.
    assert report.macro_f1 == 1.0


def test_division_guards_never_nan():
    # Model predicts only class 0 -> classes 1,2 have zero precision/f1, no NaN.
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 0, 0])
    report = classification_report(y_true, y_pred, n_classes=3)
    assert not np.isnan(report.precision).any()
    assert not np.isnan(report.f1).any()
    assert report.precision[1] == 0.0


def test_format_report_renders_all_classes():
    y = np.array([0, 1, 2, 3, 4])
    report = classification_report(y, y.copy(), n_classes=5)
    text = format_report(report, ("N", "S", "V", "F", "Q"))
    for name in ("N", "S", "V", "F", "Q"):
        assert name in text
    assert "macro-F1" in text
    assert "confusion" in text
