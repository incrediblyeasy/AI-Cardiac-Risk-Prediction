"""EchoFuseNet training (Day 9).

Config-driven, reproducible training for the assembled three-branch model:

* ``config``  — nested dataclass config with JSON (de)serialisation.
* ``metrics`` — confusion matrix, per-class P/R/F1, macro-F1 (NumPy only).
* ``train``   — the loss/optimizer/schedule/checkpoint/logging loop, plus the
  ``python -m paper1_echofusenet.training.train --config ...`` entrypoint.
"""

from .config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    TrainLoopConfig,
)
from .metrics import (
    ClassificationReport,
    accuracy_score,
    classification_report,
    confusion_matrix,
    format_report,
    macro_f1_score,
)
from .stats import (
    Interval,
    TestResult,
    bootstrap_metric_ci,
    mcnemar_test,
    mean_confidence_interval,
    paired_ttest,
    wilcoxon_test,
)
from .crossval import (
    CrossValReport,
    FoldResult,
    cross_validate,
    patient_kfold,
)
from .ablation import (
    MODALITY_SETS,
    AblationEntry,
    AblationReport,
    modality_key,
    run_ablation,
    run_ablation_from_config,
)
from .train import evaluate, run_from_config, train

__all__ = [
    "DataConfig",
    "ModelConfig",
    "OptimConfig",
    "TrainConfig",
    "TrainLoopConfig",
    "ClassificationReport",
    "accuracy_score",
    "classification_report",
    "confusion_matrix",
    "format_report",
    "macro_f1_score",
    "Interval",
    "TestResult",
    "bootstrap_metric_ci",
    "mcnemar_test",
    "mean_confidence_interval",
    "paired_ttest",
    "wilcoxon_test",
    "CrossValReport",
    "FoldResult",
    "cross_validate",
    "patient_kfold",
    "MODALITY_SETS",
    "AblationEntry",
    "AblationReport",
    "modality_key",
    "run_ablation",
    "run_ablation_from_config",
    "evaluate",
    "run_from_config",
    "train",
]
