"""Baseline-vs-EchoFuseNet comparison table (§6 enhancement).

Produces the single table the paper needs: every baseline **and** EchoFuseNet,
each with accuracy, macro-F1, parameter count, exported size, and CPU latency —
all trained/measured on the *same* DS1/DS2 inter-patient split through the same
code path. The framing is deliberate (checklist §6): the story is "competitive
accuracy at a fraction of the size/latency", so param count and latency sit in
the table next to accuracy, not hidden.

Reuses ``training.train`` / ``training.evaluate`` / ``benchmark`` unchanged — the
baseline adapter (``BaselineClassifier``) keeps the EchoFuseNet ``forward`` API,
so nothing about the protocol differs between a baseline and EchoFuseNet. The
per-model ``train_and_eval`` step is injectable so tests can assemble the table
from cheap stand-ins instead of a full training run.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from ..benchmark import measure_latency_ms, measure_model_size_mb
from ..data.dataset import build_dataloaders
from ..models import EchoFuseNet, count_parameters
from ..training.config import TrainConfig
from ..training.evaluate import collect_predictions, evaluation_report
from ..training.train import build_model, resolve_device, set_seed, train
from .models import build_baseline


@dataclass
class BaselineRow:
    """One row of the comparison table."""

    name: str
    n_params: int
    size_mb: float
    latency_ms_median: float
    accuracy: float
    macro_f1: float
    mcc: float

    def as_dict(self) -> dict:
        return asdict(self)


def _measure(model: nn.Module, image_size: int, n_iter: int, num_threads: int | None) -> tuple[float, float, int]:
    """Return (size_mb, median_latency_ms, n_params) for a model."""
    size_mb = measure_model_size_mb(model)
    latency = measure_latency_ms(
        model, batch=1, image_size=image_size, n_iter=n_iter, num_threads=num_threads
    )
    return size_mb, latency.median, count_parameters(model)


def _default_train_and_eval(
    model: nn.Module, cfg: TrainConfig, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Train a model on the config's split and return DS2 (y_true, y_pred)."""
    set_seed(cfg.train.seed)
    train_loader, test_loader = build_dataloaders(
        batch_size=cfg.data.batch_size,
        oversample=cfg.data.oversample,
        use_balanced_sampler=cfg.data.use_balanced_sampler,
        normalize=cfg.data.normalize,
        seed=cfg.train.seed,
        num_workers=cfg.data.num_workers,
        data_dir=Path(cfg.data.data_dir) if cfg.data.data_dir else None,
        train_records=tuple(cfg.data.train_records) if cfg.data.train_records else None,
        test_records=tuple(cfg.data.test_records) if cfg.data.test_records else None,
    )
    train(model, train_loader, test_loader, cfg, device=device)
    return collect_predictions(model, test_loader, device)


def _row_for_model(
    name: str,
    model: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    train_and_eval: Callable,
    image_size: int,
    latency_iter: int,
    num_threads: int | None,
) -> BaselineRow:
    y_true, y_pred = train_and_eval(model, cfg, device)
    report = evaluation_report(
        y_true, y_pred, cfg.model.n_classes, with_ci=False
    )["scalars"]
    size_mb, latency_med, n_params = _measure(model, image_size, latency_iter, num_threads)
    return BaselineRow(
        name=name,
        n_params=n_params,
        size_mb=size_mb,
        latency_ms_median=latency_med,
        accuracy=report["accuracy"],
        macro_f1=report["macro_f1"],
        mcc=report["mcc"],
    )


def compare_baselines(
    cfg: TrainConfig,
    baseline_names: list[str],
    include_echofusenet: bool = True,
    out_dir: str | Path = "runs/baselines",
    image_size: int = 256,
    latency_iter: int = 50,
    num_threads: int | None = 1,
    device: torch.device | None = None,
    train_and_eval: Callable | None = None,
) -> list[BaselineRow]:
    """Train + measure every baseline (and EchoFuseNet) and tabulate them.

    ``train_and_eval(model, cfg, device) -> (y_true, y_pred)`` is injectable;
    the default trains on the config's DS1/DS2 split. Returns the list of rows
    (also written to ``comparison.json``) sorted by parameter count so the
    size/accuracy trade-off is legible at a glance.
    """
    device = device or resolve_device(cfg.train.device)
    train_and_eval = train_and_eval or _default_train_and_eval
    out_dir = Path(out_dir)

    rows: list[BaselineRow] = []

    if include_echofusenet:
        efn_cfg = copy.deepcopy(cfg)
        efn_cfg.train.out_dir = str(out_dir / "echofusenet")
        model = build_model(efn_cfg)
        rows.append(
            _row_for_model(
                "EchoFuseNet", model, efn_cfg, device, train_and_eval,
                image_size, latency_iter, num_threads,
            )
        )

    for name in baseline_names:
        bcfg = copy.deepcopy(cfg)
        bcfg.train.out_dir = str(out_dir / name)
        model = build_baseline(name, n_classes=cfg.model.n_classes)
        rows.append(
            _row_for_model(
                name, model, bcfg, device, train_and_eval,
                image_size, latency_iter, num_threads,
            )
        )

    rows.sort(key=lambda r: r.n_params)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump([r.as_dict() for r in rows], fh, indent=2)
        fh.write("\n")

    print(format_comparison(rows))
    return rows


def format_comparison(rows: list[BaselineRow]) -> str:
    """Render the comparison rows as an aligned table."""
    lines = [
        f"{'model':<16}{'params':>12}{'size(MB)':>10}{'lat(ms)':>9}"
        f"{'acc':>8}{'macroF1':>9}{'MCC':>8}",
        "-" * 72,
    ]
    for r in rows:
        lines.append(
            f"{r.name:<16}{r.n_params:>12,}{r.size_mb:>10.2f}"
            f"{r.latency_ms_median:>9.2f}{r.accuracy:>8.4f}"
            f"{r.macro_f1:>9.4f}{r.mcc:>8.4f}"
        )
    lines.append("-" * 72)
    lines.append(
        "Claim: competitive accuracy at a fraction of the parameters / size / "
        "latency - not raw-accuracy supremacy over far larger backbones."
    )
    return "\n".join(lines)


def main() -> None:
    import argparse

    from .models import BASELINE_ARCHITECTURES

    parser = argparse.ArgumentParser(
        description="EchoFuseNet vs standard-CNN baselines (§6 comparison table)."
    )
    parser.add_argument("--config", required=True, help="Base JSON training config.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["resnet18", "densenet121", "efficientnet_b0"],
        help=f"Architectures from {sorted(BASELINE_ARCHITECTURES)}.",
    )
    parser.add_argument("--out", default="runs/baselines", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument(
        "--no-echofusenet", action="store_true", help="Skip the EchoFuseNet row."
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    compare_baselines(
        cfg,
        args.baselines,
        include_echofusenet=not args.no_echofusenet,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
