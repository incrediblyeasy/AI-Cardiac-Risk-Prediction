"""Risk-model evaluation: discrimination + calibration metrics.

Implements the discrimination/calibration half of the Paper 3 evaluation suite
(roadmap §4) with numpy/scipy only — no sklearn dependency. Bootstrap CIs,
decision-curve analysis, and the causal-effect metrics live alongside these once
the cohort exists; the causal sensitivity metrics are in ``causal_validation``.
"""

from .metrics import auroc, average_precision, brier_score, expected_calibration_error

__all__ = ["auroc", "average_precision", "brier_score", "expected_calibration_error"]
