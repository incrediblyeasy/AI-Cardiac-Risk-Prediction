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
from .losses import build_loss_fn, class_counts_from_loader
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
    """Build the training loss from ``cfg.loss`` (§2 imbalance recipes).

    Recipe selection lives in ``cfg.loss.name`` (ce | weighted_ce | focal |
    class_balanced). For backward compatibility, legacy configs that only set
    ``train.class_weighted_loss = true`` (leaving ``loss.name = "ce"``) are
    promoted to ``weighted_ce`` so old runs reproduce exactly.
    """
    name = cfg.loss.name.lower()
    if name == "ce" and cfg.train.class_weighted_loss:
        name = "weighted_ce"

    if name == "ce":
        return nn.CrossEntropyLoss(
            label_smoothing=cfg.train.label_smoothing
        ).to(device)

    # Every other recipe needs train-fold class counts (never the test fold).
    counts = class_counts_from_loader(train_loader, cfg.model.n_classes)
    return build_loss_fn(
        name,
        cfg.model.n_classes,
        counts,
        gamma=cfg.loss.gamma,
        beta=cfg.loss.beta,
        cb_base=cfg.loss.cb_base,
        label_smoothing=cfg.train.label_smoothing,
        device=device,
    )


# --------------------------------------------------------------------------- #
# §7 training-engineering helpers
# --------------------------------------------------------------------------- #
class ModelEma:
    """Exponential moving average of model parameters (a shadow copy).

    Each optimiser step nudges the shadow weights toward the live weights by
    ``(1 - decay)``. The averaged weights are usually smoother and generalise a
    touch better at *zero* inference cost (the shadow replaces the live weights
    at deployment). ``store``/``restore`` let evaluation temporarily swap the EMA
    weights into the live model and put the training weights back afterwards.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:  # int buffers (e.g. BN num_batches_tracked): just track latest
                s.copy_(v.detach())

    def store(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self._backup:
            model.load_state_dict(self._backup, strict=True)
            self._backup = {}


def average_state_dicts(states: list[dict]) -> dict:
    """Elementwise mean of a list of ``state_dict``s (checkpoint averaging).

    Floating-point tensors are averaged; non-float buffers take the last value
    (averaging integer counters like BN's ``num_batches_tracked`` is meaningless).
    """
    if not states:
        raise ValueError("need at least one state_dict to average")
    avg: dict = {}
    for key in states[0]:
        tensors = [s[key] for s in states]
        if tensors[0].dtype.is_floating_point:
            stacked = torch.stack([t.float() for t in tensors], dim=0)
            avg[key] = stacked.mean(dim=0).to(tensors[0].dtype)
        else:
            avg[key] = tensors[-1].clone()
    return avg


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
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    ema: ModelEma | None = None,
) -> float:
    """Run one training epoch; return the mean per-sample loss.

    ``scaler`` enables mixed-precision (AMP) training when supplied (GPU only);
    ``ema`` maintains an exponential-moving-average shadow of the weights.
    """
    model.train()
    use_amp = scaler is not None
    running = 0.0
    n = 0
    for step, (rp, gaf, mtf, labels) in enumerate(loader):
        rp, gaf, mtf = rp.to(device), gaf.to(device), mtf.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(rp, gaf, mtf)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(rp, gaf, mtf)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

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

    # AMP is GPU-only; silently disable on CPU (the edge-deployment target).
    use_amp = cfg.train.amp and device.type == "cuda"
    if cfg.train.amp and not use_amp:
        print("  [AMP requested but device is not CUDA — running full precision]")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    ema = ModelEma(model, decay=cfg.train.ema_decay) if cfg.train.ema else None

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_file(out_dir / "config.json")  # reproducibility snapshot
    history_path = out_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")  # fresh run
    writer = _open_tensorboard(out_dir, cfg.train.tensorboard)

    metric_key = cfg.train.checkpoint_metric
    best_metric = -float("inf")
    best_summary: dict = {}
    epochs_since_improve = 0
    # Ring buffer of recent weight snapshots for checkpoint averaging.
    recent_states: list[dict] = []

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
            scaler=scaler,
            ema=ema,
        )
        if scheduler is not None:
            scheduler.step()

        # Evaluate on the EMA weights when enabled (they are what would ship),
        # temporarily swapping them into the model and restoring afterwards.
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        report = evaluate(model, test_loader, device, cfg.model.n_classes)
        if ema is not None:
            ema.restore(model)
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
            f"| MCC {scalars['mcc']:.4f} | lr {lr:.2e}"
        )
        print(format_report(report, AAMI_CLASSES[: cfg.model.n_classes]))

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/lr", lr, epoch)
            for k, v in scalars.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        # Snapshot weights for checkpoint averaging (EMA weights if enabled).
        if cfg.train.checkpoint_avg_last > 0:
            if ema is not None:
                snap = {k: v.detach().clone() for k, v in ema.shadow.items()}
            else:
                snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            recent_states.append(snap)
            recent_states = recent_states[-cfg.train.checkpoint_avg_last :]

        save_checkpoint(
            out_dir / "last.pt", model, optimizer, epoch, scalars.get(metric_key, 0.0), cfg
        )
        current = scalars.get(metric_key)
        if current is None:
            raise KeyError(
                f"checkpoint_metric '{metric_key}' not in metrics {list(scalars)}"
            )
        if current > best_metric + cfg.train.early_stopping_min_delta:
            best_metric = current
            best_summary = {"epoch": epoch, **scalars}
            epochs_since_improve = 0
            save_checkpoint(
                out_dir / "best.pt", model, optimizer, epoch, current, cfg
            )
            print(f"  -> new best {metric_key} = {current:.4f} (saved best.pt)")
        else:
            epochs_since_improve += 1

        # Early stopping: bail once the metric has plateaued for `patience` epochs.
        if (
            cfg.train.early_stopping_patience > 0
            and epochs_since_improve >= cfg.train.early_stopping_patience
        ):
            print(
                f"  early stopping at epoch {epoch}: no {metric_key} improvement "
                f"for {epochs_since_improve} epochs"
            )
            break

    # Checkpoint averaging: mean of the last N snapshots, evaluated + saved.
    if cfg.train.checkpoint_avg_last > 0 and recent_states:
        avg_state = average_state_dicts(recent_states)
        avg_model = build_model(cfg).to(device)
        avg_model.load_state_dict(avg_state)
        avg_report = evaluate(avg_model, test_loader, device, cfg.model.n_classes)
        torch.save(
            {"model_state": avg_state, "config": cfg.to_dict(),
             "n_averaged": len(recent_states)},
            out_dir / "averaged.pt",
        )
        best_summary["averaged"] = avg_report.scalar_metrics()
        print(
            f"  checkpoint-averaged {len(recent_states)} snapshots -> "
            f"macro-F1 {avg_report.macro_f1:.4f} (saved averaged.pt)"
        )

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
        use_balanced_sampler=cfg.data.use_balanced_sampler,
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
    summary = train(model, train_loader, test_loader, cfg, device=device)

    # §8: capture the full reproducibility record (git hash, seed, config,
    # final metrics) once, next to the checkpoints and in the shared ledger.
    try:
        from shared.utils import log_experiment

        log_experiment(
            name=Path(cfg.train.out_dir).name,
            seed=cfg.train.seed,
            config=cfg.to_dict(),
            metrics=summary,
            out_dir=cfg.train.out_dir,
            extra={"n_params": count_parameters(model)},
        )
    except Exception as exc:  # logging must never break a completed run
        print(f"  [experiment log skipped: {exc}]")
    return summary


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
