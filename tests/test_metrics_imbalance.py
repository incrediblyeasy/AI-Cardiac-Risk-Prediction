"""§2 extra metrics: MCC, Cohen's kappa, macro-P/R, per-class breakdown."""

import numpy as np

from paper1_echofusenet.training.metrics import (
    classification_report,
    cohen_kappa_score,
    mcc_score,
)


def test_perfect_prediction_mcc_and_kappa_one():
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    r = classification_report(y, y.copy(), n_classes=5)
    assert np.isclose(r.mcc, 1.0)
    assert np.isclose(r.cohen_kappa, 1.0)
    assert np.isclose(r.macro_precision, 1.0)
    assert np.isclose(r.macro_recall, 1.0)


def test_majority_only_classifier_mcc_near_zero():
    # 90% class 0, predict all 0: high accuracy but MCC ~ 0 (no real skill).
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros_like(y_true)
    r = classification_report(y_true, y_pred, n_classes=2)
    assert r.accuracy == 0.9
    assert abs(r.mcc) < 1e-9
    assert abs(r.cohen_kappa) < 1e-9


def test_scalar_metrics_keys():
    y = np.array([0, 1, 2, 0, 1])
    scalars = classification_report(y, y.copy(), n_classes=3).scalar_metrics()
    assert set(scalars) == {
        "accuracy", "macro_f1", "macro_precision", "macro_recall",
        "mcc", "cohen_kappa",
    }


def test_per_class_metrics_structure():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    r = classification_report(y_true, y_pred, n_classes=3)
    pc = r.per_class_metrics(("N", "S", "V"))
    assert set(pc) == {"N", "S", "V"}
    assert pc["V"]["support"] == 1
    assert set(pc["N"]) == {"precision", "recall", "f1", "support"}


def test_scalar_score_helpers_match_report():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 5, 200)
    yp = yt.copy()
    mask = rng.random(200) < 0.3
    yp[mask] = rng.integers(0, 5, int(mask.sum()))
    r = classification_report(yt, yp, 5)
    assert np.isclose(mcc_score(yt, yp, 5), r.mcc)
    assert np.isclose(cohen_kappa_score(yt, yp, 5), r.cohen_kappa)
