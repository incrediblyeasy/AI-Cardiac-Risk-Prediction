"""Config system for CausalEchoNet CVAE training.

Same design as Paper 1's ``training/config.py`` (nested dataclasses, lossless
JSON round-trip, unknown keys rejected loudly) so the two papers share one mental
model. A run is fully described by a JSON file + seed.

Sections
--------
    encoder -> where Paper 1's frozen checkpoint lives (the hard dependency gate)
    cvae    -> FeatureCVAE architecture knobs (must keep the whole system < 1M)
    optim   -> optimizer + LR
    train   -> loop control, β schedule, checkpointing
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class EncoderConfig:
    """Locates and validates the frozen Paper-1 encoder."""

    checkpoint: str | None = None    # path to Paper-1 best.pt; None -> must be set to run
    representation_dim: int | None = None  # None -> inferred from the loaded encoder


@dataclass
class CVAEConfig:
    """FeatureCVAE architecture (see ``cvae/model.py``)."""

    n_classes: int = 5
    latent_dim: int = 32
    hidden_dim: int = 128


@dataclass
class OptimConfig:
    """Optimizer and learning rate."""

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class CVAELoopConfig:
    """Training-loop control, β schedule, checkpointing."""

    epochs: int = 50
    batch_size: int = 128
    device: str = "auto"
    seed: int = 0
    beta: float = 1.0                # β-VAE KL weight
    beta_warmup_epochs: int = 0      # linearly ramp β from 0 over N epochs (0 = off)
    out_dir: str = "runs/causalechonet_cvae"
    checkpoint_metric: str = "validity"  # counterfactual-quality selection metric
    log_interval: int = 50


@dataclass
class CVAETrainConfig:
    """Top-level CVAE training config — single source of truth for a run."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    cvae: CVAEConfig = field(default_factory=CVAEConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: CVAELoopConfig = field(default_factory=CVAELoopConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CVAETrainConfig":
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
    def from_file(cls, path: str | Path) -> "CVAETrainConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")


_SUBCONFIGS: dict[str, type] = {
    "encoder": EncoderConfig,
    "cvae": CVAEConfig,
    "optim": OptimConfig,
    "train": CVAELoopConfig,
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
