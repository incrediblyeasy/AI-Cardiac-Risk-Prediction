"""Config system for EchoFuseNet training (Day 9).

Every training hyperparameter lives in a serialisable config object so nothing is
hardcoded in the training loop (project convention: *config-driven*). A run is
fully described by a single JSON file plus a seed, which makes experiments
reproducible and diff-able. JSON is used deliberately — it needs no extra
dependency and round-trips losslessly through the dataclasses below.

Layout
------
``TrainConfig`` nests four sub-configs mirroring the pipeline stages:

    data    -> how beats are loaded/split/oversampled (Day 6 DataLoaders)
    model   -> EchoFuseNet architecture knobs (Day 8)
    optim   -> optimizer + LR schedule
    train   -> loop control, loss, checkpointing, logging

Load with ``TrainConfig.from_file(path)``; dump the *resolved* config next to the
checkpoints with ``cfg.to_file(...)`` so every run records exactly what produced
it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Beat loading / split / balancing (feeds ``build_dataloaders``)."""

    batch_size: int = 32
    oversample: bool = True          # train fold only, applied post-split
    # Per-epoch class-balanced WeightedRandomSampler (data/sampler.py) as an
    # alternative to materialised oversampling. When True it supersedes
    # ``oversample`` on the train loader (they are mutually exclusive — the
    # sampler already rebalances, so double-balancing is avoided).
    use_balanced_sampler: bool = False
    normalize: bool = True           # put RP/GAF/MTF on a common [0, 1] scale
    num_workers: int = 0
    data_dir: str | None = None      # None -> library default (data/raw/mitdb)
    train_records: list[int] | None = None  # override DS1 list (fast tests)
    test_records: list[int] | None = None    # override DS2 list (fast tests)


@dataclass
class ModelConfig:
    """EchoFuseNet architecture (see ``models/echofusenet.py``)."""

    n_classes: int = 5
    widths: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 256])
    fusion_hidden: int = 128
    dropout: float = 0.3
    # Active modality subset (Day-12 ablation). Any subset of ("rp","gaf","mtf");
    # all three = the full EchoFuseNet.
    modalities: list[str] = field(default_factory=lambda: ["rp", "gaf", "mtf"])


@dataclass
class OptimConfig:
    """Optimizer and learning-rate schedule."""

    name: str = "adamw"              # adamw | adam | sgd
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9            # SGD only
    scheduler: str = "cosine"        # cosine | step | none
    step_size: int = 10              # StepLR only
    gamma: float = 0.1               # StepLR decay factor
    min_lr: float = 1e-5             # cosine floor (eta_min)


@dataclass
class LossConfig:
    """Class-imbalance loss recipe (see ``training/losses.py``).

    ``name`` selects one of the four recipes the §2 enhancement compares:
    ``ce`` (plain cross-entropy), ``weighted_ce`` (inverse-frequency weights),
    ``focal`` (Lin et al.), or ``class_balanced`` (Cui et al., effective-number
    weighting). ``gamma`` is the focal focusing parameter; ``beta`` the
    class-balanced effective-number hyperparameter; ``cb_base`` whether the
    class-balanced loss wraps ``ce`` or ``focal``.

    Backward-compatible default: ``ce``. For legacy configs that only set
    ``train.class_weighted_loss = true`` (and leave ``name = "ce"``), the trainer
    promotes the recipe to ``weighted_ce`` so old runs reproduce.
    """

    name: str = "ce"                 # ce | weighted_ce | focal | class_balanced
    gamma: float = 2.0               # focal focusing parameter
    beta: float = 0.999              # class-balanced effective-number beta
    cb_base: str = "focal"           # class_balanced base loss: ce | focal


@dataclass
class TrainLoopConfig:
    """Training-loop control, loss shaping, checkpointing, logging."""

    epochs: int = 30
    device: str = "auto"             # auto -> cuda if available else cpu
    seed: int = 0
    # Loss: class-imbalance handling. Oversampling already balances the train
    # fold, so class weighting defaults off; enable one or the other, not both.
    class_weighted_loss: bool = False
    label_smoothing: float = 0.0
    grad_clip: float | None = None   # max grad norm, or None to disable
    # --- §7 training engineering ------------------------------------------
    # Mixed precision (AMP): speeds up GPU training; a no-op on CPU (the edge
    # deployment target), where it is silently disabled.
    amp: bool = False
    # Early stopping: halt if ``checkpoint_metric`` has not improved for
    # ``early_stopping_patience`` epochs (0 disables). ``min_delta`` is the
    # smallest change counted as an improvement.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    # Exponential moving average of weights: a shadow copy updated each step;
    # evaluated/checkpointed alongside the raw model when enabled. Often a small
    # free accuracy/calibration gain at zero inference cost.
    ema: bool = False
    ema_decay: float = 0.999
    # Checkpoint averaging: average the last ``checkpoint_avg_last`` epoch
    # weight snapshots into ``averaged.pt`` at the end of training (0 disables).
    checkpoint_avg_last: int = 0
    # Checkpointing / logging.
    out_dir: str = "runs/echofusenet"
    checkpoint_metric: str = "macro_f1"  # best-model selection metric
    log_interval: int = 50           # batches between train-loss log lines
    tensorboard: bool = False        # optional; silently skipped if unavailable


@dataclass
class TrainConfig:
    """Top-level training config — the single source of truth for a run."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)

    # --- (de)serialisation ------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        """Build a config from a (possibly partial) nested dict.

        Unknown keys raise, so typos in a config file fail loudly instead of
        being silently ignored. Missing keys fall back to the dataclass default.
        """
        sub = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for name, subcls in _SUBCONFIGS.items():
            section = data.get(name, {})
            if not isinstance(section, dict):
                raise TypeError(f"config section '{name}' must be a mapping")
            _reject_unknown(name, section, subcls)
            kwargs[name] = subcls(**section)
        _reject_unknown("<root>", data, cls, allowed=set(_SUBCONFIGS))
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainConfig":
        """Load a config from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_file(self, path: str | Path) -> None:
        """Dump the resolved config as pretty JSON (create parent dirs)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")


_SUBCONFIGS: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "optim": OptimConfig,
    "loss": LossConfig,
    "train": TrainLoopConfig,
}


def _reject_unknown(
    section: str, provided: dict[str, Any], cls: type, allowed: set[str] | None = None
) -> None:
    known = allowed if allowed is not None else {f.name for f in fields(cls)}
    unknown = set(provided) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in config section '{section}': {sorted(unknown)}"
        )
