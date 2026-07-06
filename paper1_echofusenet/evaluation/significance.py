"""Statistical-significance convenience layer (§5, checklist path).

The §5 checklist item asks for Wilcoxon signed-rank, McNemar, and bootstrap CIs
and lists ``evaluation/significance.py`` as the target — noting the machinery
"likely already partially exists" in the roadmap. It does: ``training.stats``
implements all of it, correctly and with small-sample guards. Rather than fork a
second copy (which would drift), this module **re-exports** those functions from
the checklist's expected import path and adds one aggregator, ``compare_models``,
that runs the whole paired-comparison battery for two systems in a single call
and returns a serialisable report.

``compare_models`` combines:

* **per-fold paired tests** — Wilcoxon signed-rank + paired t-test over matched
  k-fold scores (needs the two models' per-fold score vectors); and
* **per-sample McNemar** — over the two models' correctness on one shared test
  fold (needs each model's ``(y_true, y_pred)`` on the *same* samples); plus
* **bootstrap CIs** on each model's headline metric on that shared fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Re-export the primitives from their implementation home (single source of truth).
from ..training.stats import (  # noqa: F401  (re-exported by design)
    Interval,
    TestResult,
    bootstrap_metric_ci,
    mcnemar_test,
    mean_confidence_interval,
    paired_ttest,
    wilcoxon_test,
)
from ..training.metrics import accuracy_score, macro_f1_score


@dataclass
class ComparisonReport:
    """Full paired comparison of model A vs model B."""

    name_a: str
    name_b: str
    wilcoxon: TestResult | None
    paired_t: TestResult | None
    mcnemar: TestResult | None
    ci_a: Interval | None
    ci_b: Interval | None

    def summary_dict(self) -> dict:
        def _t(r: TestResult | None) -> dict | None:
            return None if r is None else {
                "name": r.name, "statistic": r.statistic, "pvalue": r.pvalue,
                "significant_0.05": r.significant(),
            }

        def _iv(iv: Interval | None) -> dict | None:
            return None if iv is None else {
                "point": iv.point, "low": iv.low, "high": iv.high,
                "confidence": iv.confidence,
            }

        return {
            "models": {"a": self.name_a, "b": self.name_b},
            "wilcoxon": _t(self.wilcoxon),
            "paired_t": _t(self.paired_t),
            "mcnemar": _t(self.mcnemar),
            "bootstrap_ci": {"a": _iv(self.ci_a), "b": _iv(self.ci_b)},
        }

    def __str__(self) -> str:
        lines = [f"{self.name_a} vs {self.name_b}"]
        if self.wilcoxon is not None:
            lines.append(f"  per-fold  {self.wilcoxon}")
        if self.paired_t is not None:
            lines.append(f"  per-fold  {self.paired_t}")
        if self.mcnemar is not None:
            lines.append(f"  per-sample {self.mcnemar}")
        if self.ci_a is not None:
            lines.append(f"  {self.name_a} metric CI: {self.ci_a}")
        if self.ci_b is not None:
            lines.append(f"  {self.name_b} metric CI: {self.ci_b}")
        return "\n".join(lines)


def compare_models(
    name_a: str = "A",
    name_b: str = "B",
    fold_scores_a: Sequence[float] | None = None,
    fold_scores_b: Sequence[float] | None = None,
    preds_a: tuple[np.ndarray, np.ndarray] | None = None,
    preds_b: tuple[np.ndarray, np.ndarray] | None = None,
    metric: str = "macro_f1",
    n_classes: int = 5,
    n_boot: int = 2000,
    seed: int = 0,
) -> ComparisonReport:
    """Run the full paired-comparison battery for two models.

    Provide ``fold_scores_*`` (matched per-fold score vectors) to get the paired
    per-fold tests, and/or ``preds_*`` as ``(y_true, y_pred)`` on a *shared* test
    fold to get McNemar + per-model bootstrap CIs of ``metric``. Any part with
    missing inputs is simply left out of the report (its field is ``None``).
    """
    wilcoxon = paired_t = mcnemar = None
    ci_a = ci_b = None

    if fold_scores_a is not None and fold_scores_b is not None:
        wilcoxon = wilcoxon_test(fold_scores_a, fold_scores_b)
        paired_t = paired_ttest(fold_scores_a, fold_scores_b)

    if preds_a is not None and preds_b is not None:
        yt_a, yp_a = preds_a
        yt_b, yp_b = preds_b
        mcnemar = mcnemar_test(
            np.asarray(yt_a) == np.asarray(yp_a),
            np.asarray(yt_b) == np.asarray(yp_b),
        )
        metric_fn = (
            accuracy_score
            if metric == "accuracy"
            else (lambda t, p: macro_f1_score(t, p, n_classes))
        )
        ci_a = bootstrap_metric_ci(yt_a, yp_a, metric_fn, n_boot=n_boot, seed=seed)
        ci_b = bootstrap_metric_ci(yt_b, yp_b, metric_fn, n_boot=n_boot, seed=seed)

    return ComparisonReport(
        name_a=name_a, name_b=name_b,
        wilcoxon=wilcoxon, paired_t=paired_t, mcnemar=mcnemar,
        ci_a=ci_a, ci_b=ci_b,
    )
