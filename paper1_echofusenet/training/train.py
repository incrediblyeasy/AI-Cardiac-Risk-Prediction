"""Config-driven training loop for EchoFuseNet (Day 9).

Ties together the Day-6 multimodal DataLoaders, the Day-8 assembled model, and a
reproducible optimisation loop. Everything the loop needs comes from a
``TrainConfig`` (see ``config.py``) — there are no hardcoded hyperparameters —
so a run is fully reproducible from ``config.json`` + the seed it records.

Responsibilities
----------------
* **Loss** — cross-entropy, with optional class weighting and label smoothing.
* **Optimizer / schedule** — AdamW/Adam/SGD + cosine/step/none LR schedule.
* **Checkpointing** — best model (by ``checkpoint_metric``) and last model, each
  saving weights + optimizer + epoch + config, plus a ``config.json`` snapshot.
* **Metrics logging** — accuracy, per-class F1, macro-F1, confusion matrix each
  epoch, appended to ``history.jsonl`` and printed. TensorBoard is optional and
  silently skipped when the package is unavailable.

Usage
-----
    python -m paper1_echofusenet.training.train --config configs/echofusenet_default.json

The heavy lifting (``train``, ``evaluate``) takes loaders + a model as plain
arguments so it can be unit-tested without downloaded data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.aami import AAMI_CLASSES
from ..data.dataset import build_dataloaders
from ..models import EchoFuseNet, count_parameters
from .config import TrainConfig
from .metrics import ClassificationReport, classification_report, format_report


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python/NumPy/Torch RNGs for a deterministic run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    """Resolve ``"auto"`` to cuda-if-available, else honour the literal spec."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# --------------------------------------------------------------------------- #
# Builders (config -> objects)
# --------------------------------------------------------------------------- #
def build_model(cfg: TrainConfig) -> EchoFuseNet:
    return EchoFuseNet(
        n_classes=cfg.model.n_classes,
        widths=tuple(cfg.model.widths),
        fusion_hidden=cfg.model.fusion_hidden,
        dropout=cfg.model.dropout,
        modalities=tuple(cfg.model.modalities),
    )


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    o = cfg.optim
    name = o.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=o.weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=o.lr, weight_decay=o.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=o.lr,
            momentum=o.momentum,
            weight_decay=o.weight_decay,
            nesterov=o.momentum > 0,
        )
    raise ValueError(f"unknown optimizer '{o.name}'")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    o = cfg.optim
    sched = o.scheduler.lower()
    if sched == "none":
        return None
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs, eta_min=o.min_lr
        )
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=o.step_size, gamma=o.gamma
        )
    raise ValueError(f"unknown scheduler '{o.scheduler}'")


def compute_class_weights(
    loader: DataLoader, n_classes: int, device: torch.device
) -> torch.Tensor:
    """Inverse-frequency class weights from a loader's labels (normalised to
    mean 1). Used only when ``class_weighted_loss`` is set."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for *_, labels in loader:
        vals, c = np.unique(labels.numpy(), return_counts=True)
        counts[vals] += c
    counts = np.maximum(counts, 1)  # guard: no division by zero
    inv = counts.sum() / counts
    inv = inv / inv.mean()          # keep the loss on a comparable scale
    return torch.tensor(inv, dtype=torch.float32, device=device)


def build_loss(
    cfg: TrainConfig, train_loader: DataLoader, device: torch.device
) -> nn.Module:
    weight = None
    if cfg.train.class_weighted_loss:
        weight = compute_class_weights(train_loader, cfg.model.n_classes, device)
    return nn.CrossEntropyLoss(
        weight=weight, label_smoothing=cfg.train.label_smoothing
    )


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = None,
    log_interval: int = 0,
    epoch: int = 0,
) -> float:
    """Run one training epoch; return the mean per-sample loss."""
    model.train()
    running = 0.0
    n = 0
    for step, (rp, gaf, mtf, labels) in enumerate(loader):
        rp, gaf, mtf = rp.to(device), gaf.to(device), mtf.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(rp, gaf, mtf)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = labels.size(0)
        running += loss.item() * bs
        n += bs
        if log_interval and step % log_interval == 0:
            print(f"  epoch {epoch:3d} | step {step:5d} | loss {loss.item():.4f}")
    return running / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> ClassificationReport:
    """Predict over a loader and return the full classification report."""
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for rp, gaf, mtf, labels in loader:
        rp, gaf, mtf = rp.to(device), gaf.to(device), mtf.to(device)
        logits = model(rp, gaf, mtf)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(preds)
        all_true.append(labels.numpy())
    if not all_true:
        return classification_report(np.array([]), np.array([]), n_classes)
    return classification_report(
        np.concatenate(all_true), np.concatenate(all_pred), n_classes
    )


# --------------------------------------------------------------------------- #
# Checkpointing / logging
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    cfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metric": metric,
            "config": cfg.to_dict(),
        },
        path,
    )


def _open_tensorboard(out_dir: Path, enabled: bool):
    """Return a SummaryWriter if requested *and* available, else None."""
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("  [tensorboard requested but unavailable — skipping]")
        return None
    return SummaryWriter(log_dir=str(out_dir / "tb"))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> dict:
    """Full training run. Returns the best-epoch summary dict.

    Takes loaders + model directly so it is unit-testable without real data.
    Writes checkpoints, ``config.json``, and ``history.jsonl`` under
    ``cfg.train.out_dir``.
    """
    device = device or resolve_device(cfg.train.device)
    model = model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = build_loss(cfg, train_loader, device)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_file(out_dir / "config.json")  # reproducibility snapshot
    history_path = out_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")  # fresh run
    writer = _open_tensorboard(out_dir, cfg.train.tensorboard)

    metric_key = cfg.train.checkpoint_metric
    best_metric = -float("inf")
    best_summary: dict = {}

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip=cfg.train.grad_clip,
            log_interval=cfg.train.log_interval,
            epoch=epoch,
        )
        if scheduler is not None:
            scheduler.step()

        report = evaluate(model, test_loader, device, cfg.model.n_classes)
        scalars = report.scalar_metrics()
        lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": lr,
            **scalars,
            "per_class_f1": report.f1.tolist(),
        }
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        print(
            f"epoch {epoch:3d}/{cfg.train.epochs} | loss {train_loss:.4f} | "
            f"acc {scalars['accuracy']:.4f} | macro-F1 {scalars['macro_f1']:.4f} "
            f"| lr {lr:.2e}"
        )
        print(format_report(report, AAMI_CLASSES[: cfg.model.n_classes]))

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/lr", lr, epoch)
            for k, v in scalars.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        save_checkpoint(
            out_dir / "last.pt", model, optimizer, epoch, scalars.get(metric_key, 0.0), cfg
        )
        current = scalars.get(metric_key)
        if current is None:
            raise KeyError(
                f"checkpoint_metric '{metric_key}' not in metrics {list(scalars)}"
            )
        if current > best_metric:
            best_metric = current
            best_summary = {"epoch": epoch, **scalars}
            save_checkpoint(
                out_dir / "best.pt", model, optimizer, epoch, current, cfg
            )
            print(f"  -> new best {metric_key} = {current:.4f} (saved best.pt)")

    if writer is not None:
        writer.close()
    print(f"\nBest {metric_key}: {best_metric:.4f} @ epoch {best_summary.get('epoch')}")
    return best_summary


def run_from_config(cfg: TrainConfig) -> dict:
    """Build data + model from a config and run training end-to-end."""
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    data_kwargs = dict(
        batch_size=cfg.data.batch_size,
        oversample=cfg.data.oversample,
        normalize=cfg.data.normalize,
        seed=cfg.train.seed,
        num_workers=cfg.data.num_workers,
        train_records=tuple(cfg.data.train_records) if cfg.data.train_records else None,
        test_records=tuple(cfg.data.test_records) if cfg.data.test_records else None,
    )
    if cfg.data.data_dir:
        data_kwargs["data_dir"] = Path(cfg.data.data_dir)
    train_loader, test_loader = build_dataloaders(**data_kwargs)

    model = build_model(cfg)
    print(
        f"Model: EchoFuseNet | params {count_parameters(model):,} | device {device}"
    )
    return train(model, train_loader, test_loader, cfg, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EchoFuseNet (Day 9).")
    parser.add_argument(
        "--config", required=True, help="Path to a JSON training config."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override config train.epochs."
    )
    parser.add_argument(
        "--out-dir", default=None, help="Override config train.out_dir."
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir

    run_from_config(cfg)


if __name__ == "__main__":
    main()
