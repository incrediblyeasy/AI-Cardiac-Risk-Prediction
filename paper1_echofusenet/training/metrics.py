"""Classification metrics for EchoFuseNet (Day 9).

Inter-patient MIT-BIH is heavily imbalanced (N dominates; S/F/Q are rare), so raw
accuracy is misleading — a majority-only classifier still scores >0.85. We
therefore track a **confusion matrix**, **per-class precision/recall/F1**, and
**macro-F1** (the headline number for imbalanced arrhythmia classification).

Everything is computed from a confusion matrix with plain NumPy — no scikit-learn
dependency — so metrics run anywhere the training loop runs. Division guards
return 0.0 for empty denominators (e.g. a class the model never predicts, or a
class absent from the test fold) rather than NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    """``(n_classes, n_classes)`` counts; rows = true, cols = predicted."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    # np.add.at scatters counts even with repeated (t, p) index pairs.
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ``num / den`` with 0.0 where ``den == 0`` (no NaN)."""
    out = np.zeros_like(num, dtype=np.float64)
    nonzero = den != 0
    out[nonzero] = num[nonzero] / den[nonzero]
    return out


@dataclass
class ClassificationReport:
    """Summary metrics derived from a confusion matrix."""

    confusion: np.ndarray          # (C, C) int counts
    support: np.ndarray            # (C,) true count per class
    precision: np.ndarray          # (C,)
    recall: np.ndarray             # (C,)
    f1: np.ndarray                 # (C,)
    accuracy: float                # overall
    macro_f1: float                # unweighted mean over *supported* classes

    def scalar_metrics(self) -> dict[str, float]:
        """Flat name->value dict of the scalar metrics (for logging)."""
        return {"accuracy": self.accuracy, "macro_f1": self.macro_f1}


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> ClassificationReport:
    """Full per-class + macro report from predictions.

    ``macro_f1`` averages F1 only over classes that actually appear in
    ``y_true`` (support > 0), which is the fair convention when a fold happens
    not to contain a class (e.g. Q is near-absent in DS2 once paced records are
    excluded).
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)  # column sums
    support = cm.sum(axis=1).astype(np.float64)    # row sums (true per class)

    precision = _safe_divide(tp, predicted)
    recall = _safe_divide(tp, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    total = cm.sum()
    accuracy = float(tp.sum() / total) if total else 0.0

    present = support > 0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0

    return ClassificationReport(
        confusion=cm,
        support=support.astype(np.int64),
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        macro_f1=macro_f1,
    )


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall accuracy — a thin ``(y_true, y_pred) -> float`` callable for
    bootstrap resampling (see ``training.stats``)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 as a scalar (support-aware; see ``classification_report``).

    Bind ``n_classes`` with ``functools.partial`` to get a bootstrap-ready
    ``(y_true, y_pred) -> float`` callable.
    """
    return classification_report(y_true, y_pred, n_classes).macro_f1


def format_report(
    report: ClassificationReport, class_names: tuple[str, ...]
) -> str:
    """Render a report as an aligned text table (for stdout / log files)."""
    lines = [
        f"{'class':>6} {'prec':>7} {'recall':>7} {'f1':>7} {'support':>8}",
        "-" * 40,
    ]
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:>6} {report.precision[i]:7.3f} {report.recall[i]:7.3f} "
            f"{report.f1[i]:7.3f} {report.support[i]:8d}"
        )
    lines.append("-" * 40)
    lines.append(
        f"{'acc':>6} {report.accuracy:7.3f}   "
        f"macro-F1 {report.macro_f1:7.3f}   "
        f"N {int(report.support.sum()):8d}"
    )
    lines.append("")
    lines.append("confusion (rows=true, cols=pred):")
    header = "       " + " ".join(f"{n:>6}" for n in class_names)
    lines.append(header)
    for i, name in enumerate(class_names):
        row = " ".join(f"{int(v):6d}" for v in report.confusion[i])
        lines.append(f"{name:>6} {row}")
    return "\n".join(lines)
