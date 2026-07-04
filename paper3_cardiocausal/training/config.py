"""Config system for CardioCausal training.

Same nested-dataclass + lossless-JSON design as Papers 1-2 (unknown keys rejected
loudly), so all three papers share one config idiom. A run is fully described by a
JSON file + seed.

Sections
--------
    encoder -> frozen ECG encoder (Paper 1/2 checkpoint)
    fusion  -> ECG-repr + tabular -> shared latent (see ``fusion/model.py``)
    scm     -> risk head / SCM knobs
    optim   -> optimizer + LR
    train   -> loop control, checkpointing
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class EncoderConfig:
    checkpoint: str | None = None    # Paper 1/2 frozen ECG encoder checkpoint
    representation_dim: int | None = None


@dataclass
class FusionConfig:
    tabular_dim: int = 32            # number of tabular clinical features
    latent_dim: int = 128            # shared fused latent width


@dataclass
class SCMConfig:
    hidden_dim: int = 64
    dropout: float = 0.2
    max_horizon: int = 4             # bounded longitudinal rollout cap


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class LoopConfig:
    epochs: int = 50
    batch_size: int = 128
    device: str = "auto"
    seed: int = 0
    out_dir: str = "runs/cardiocausal"
    checkpoint_metric: str = "auroc"
    log_interval: int = 50


@dataclass
class CardioCausalConfig:
    """Top-level CardioCausal training config — single source of truth for a run."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    scm: SCMConfig = field(default_factory=SCMConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: LoopConfig = field(default_factory=LoopConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CardioCausalConfig":
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
    def from_file(cls, path: str | Path) -> "CardioCausalConfig":
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
    "fusion": FusionConfig,
    "scm": SCMConfig,
    "optim": OptimConfig,
    "train": LoopConfig,
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
