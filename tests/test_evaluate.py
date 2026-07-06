"""§2 evaluate.py: full evaluation report (scalars + CIs + per-class + CM)."""

import numpy as np

from paper1_echofusenet.training.evaluate import evaluation_report, format_evaluation


def test_evaluation_report_structure_with_ci():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 5, 300)
    yp = yt.copy()
    mask = rng.random(300) < 0.2
    yp[mask] = rng.integers(0, 5, int(mask.sum()))

    rep = evaluation_report(yt, yp, n_classes=5, n_boot=200, with_ci=True)
    assert set(rep["scalars"]) >= {"accuracy", "macro_f1", "mcc", "cohen_kappa"}
    assert set(rep["bootstrap_ci"]) == {"accuracy", "macro_f1", "mcc", "cohen_kappa"}
    for iv in rep["bootstrap_ci"].values():
        assert iv["low"] <= iv["point"] <= iv["high"]
    assert len(rep["per_class"]) == 5
    assert np.array(rep["confusion"]).shape == (5, 5)
    assert rep["n_samples"] == 300


def test_evaluation_report_without_ci_skips_bootstrap():
    y = np.array([0, 1, 2, 3, 4])
    rep = evaluation_report(y, y.copy(), n_classes=5, with_ci=False)
    assert "bootstrap_ci" not in rep
    assert rep["scalars"]["macro_f1"] == 1.0


def test_format_evaluation_renders():
    y = np.array([0, 1, 2, 3, 4, 0])
    rep = evaluation_report(y, y.copy(), n_classes=5, with_ci=False)
    text = format_evaluation(rep)
    assert "MCC" in text and "macro-F1" in text and "per-class" in text
