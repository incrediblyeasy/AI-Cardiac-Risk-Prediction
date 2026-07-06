"""Evaluation utilities for EchoFuseNet (§5 significance, at the checklist path).

The statistical machinery already lives in ``training.stats`` (Wilcoxon,
McNemar, bootstrap CIs, paired t-test, k-fold t-intervals). This package is the
convenience layer the §5 checklist item points at (``evaluation/significance.py``)
— it re-exports those primitives from their expected location and adds a single
``compare_models`` call that runs the full paired-comparison battery at once.
"""

from .significance import (
    ComparisonReport,
    bootstrap_metric_ci,
    compare_models,
    mcnemar_test,
    mean_confidence_interval,
    paired_ttest,
    wilcoxon_test,
)

__all__ = [
    "ComparisonReport",
    "compare_models",
    "bootstrap_metric_ci",
    "mcnemar_test",
    "mean_confidence_interval",
    "paired_ttest",
    "wilcoxon_test",
]
