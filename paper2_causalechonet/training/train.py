"""CVAE training loop for CausalEchoNet.

Fully wired and config-driven, mirroring Paper 1's training package. The pieces:

* ``build_cvae`` / ``build_optimizer`` — config -> objects.
* ``train_one_epoch`` — one optimisation epoch over ``(representation, label)``
  pairs (pure representation space; unit-testable without any data).
* ``run_from_config`` — the end-to-end entry point: load the **frozen** Paper-1
  encoder from ``cfg.encoder.checkpoint``, encode the inter-patient DS1/DS2 folds
  into cached representations (``paper2_causalechonet.data``), train the CVAE, and
  checkpoint on counterfactual validity.

Runtime gate (not a code stub)
------------------------------
``run_from_config`` needs a **frozen Paper-1 encoder checkpoint** — it raises a
clear ``ValueError`` if ``cfg.encoder.checkpoint`` is unset. Per
``PROJECT_STATUS_AND_ROADMAP.md`` §2/§3 this checkpoint must be Paper 1's *locked*
model (do not train against a moving target); export it once the GPU headline run
is done, point the config at it, and this runs unchanged.

Usage
-----
    python -m paper2_causalechonet.training.train --config configs/causalechonet_cvae_smoke.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..cvae.model import FeatureCVAE, cvae_loss
from ..cvae.metrics import counterfactual_report
from ..data import build_representation_dataset
from ..encoder import load_frozen_encoder
from .config import CVAETrainConfig


def build_cvae(cfg: CVAETrainConfig, representation_dim: int) -> FeatureCVAE:
    """Construct the FeatureCVAE for a given representation width."""
    return FeatureCVAE(
        representation_dim=representation_dim,
        n_classes=cfg.cvae.n_classes,
        latent_dim=cfg.cvae.latent_dim,
        hidden_dim=cfg.cvae.hidden_dim,
    )


def build_optimizer(model: nn.Module, cfg: CVAETrainConfig) -> torch.optim.Optimizer:
    o = cfg.optim
    name = o.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=o.weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=o.lr, weight_decay=o.weight_decay)
    raise ValueError(f"unknown optimizer '{o.name}'")


def beta_at(cfg: CVAETrainConfig, epoch: int) -> float:
    """β for a given (1-indexed) epoch, with optional linear KL warm-up."""
    warm = cfg.train.beta_warmup_epochs
    if warm and epoch <= warm:
        return cfg.train.beta * epoch / warm
    return cfg.train.beta


def train_one_epoch(
    model: FeatureCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 1.0,
) -> dict[str, float]:
    """One CVAE epoch over ``(representation, label)`` batches.

    Reconstructs each representation conditioned on its own class. Returns mean
    total/recon/kl for the epoch. Operates purely in representation space, so it
    is unit-testable without the frozen encoder or any downloaded data.
    """
    model.train()
    sums = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    for x, cond in loader:
        x, cond = x.to(device), cond.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x, cond)
        losses = cvae_loss(x, x_hat, mu, logvar, beta=beta)
        losses["total"].backward()
        optimizer.step()
        bs = x.size(0)
        for k in sums:
            sums[k] += float(losses[k].item()) * bs
        n += bs
    n = max(n, 1)
    return {k: v / n for k, v in sums.items()}


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/Torch RNGs for a deterministic run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@torch.no_grad()
def evaluate_counterfactuals(
    model: FeatureCVAE,
    encoder,
    loader: DataLoader,
    n_classes: int,
    device: torch.device,
) -> dict[str, float]:
    """Mean validity/proximity/sparsity of A->B counterfactuals over a loader.

    For each batch of ``(representation, source_label)`` we generate a
    counterfactual toward a *shifted* target class (``(source + 1) mod C``) and
    score it against the frozen decision head — the CVAE's headline quality
    metrics (roadmap §3.3), used here as the checkpoint-selection signal.
    """
    model.eval()
    agg = {"validity": 0.0, "proximity": 0.0, "sparsity": 0.0}
    n = 0
    for x, src in loader:
        x, src = x.to(device), src.to(device)
        tgt = (src + 1) % n_classes
        x_cf = model.counterfactual(x, src, tgt)
        rep = counterfactual_report(x, x_cf, tgt, encoder.decision)
        bs = x.size(0)
        for k in agg:
            agg[k] += rep[k] * bs
        n += bs
    n = max(n, 1)
    return {k: v / n for k, v in agg.items()}


def run_from_config(cfg: CVAETrainConfig) -> dict:
    """End-to-end: load frozen encoder, cache representations, train the CVAE.

    Requires ``cfg.encoder.checkpoint`` to point at a **frozen Paper-1 encoder**
    (roadmap §2 gate) — raises ``ValueError`` if unset. Writes checkpoints,
    ``config.json`` and ``history.jsonl`` under ``cfg.train.out_dir``; returns the
    best-epoch summary (selected by ``cfg.train.checkpoint_metric``).
    """
    if not cfg.encoder.checkpoint:
        raise ValueError(
            "cfg.encoder.checkpoint is required: point it at a FROZEN Paper-1 "
            "encoder checkpoint (export it once Paper 1's headline run is locked — "
            "roadmap §2). Do not train against a moving Paper-1 target."
        )

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    encoder = load_frozen_encoder(cfg.encoder.checkpoint, map_location=device).to(device)

    # Encode the inter-patient folds once (the encoder is frozen -> cache-friendly).
    from paper1_echofusenet.data.dataset import build_dataloaders

    beat_train, beat_test = build_dataloaders(
        batch_size=cfg.train.batch_size, oversample=False, seed=cfg.train.seed
    )
    out_dir = Path(cfg.train.out_dir)
    train_ds = build_representation_dataset(
        encoder, beat_train, cache_path=out_dir / "repr_train.pt"
    )
    test_ds = build_representation_dataset(
        encoder, beat_test, cache_path=out_dir / "repr_test.pt"
    )
    rep_train = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    rep_test = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False)

    model = build_cvae(cfg, encoder.representation_dim).to(device)
    optimizer = build_optimizer(model, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_file(out_dir / "config.json")
    history_path = out_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")

    metric_key = cfg.train.checkpoint_metric
    best_metric = -float("inf")
    best_summary: dict = {}

    for epoch in range(1, cfg.train.epochs + 1):
        losses = train_one_epoch(
            model, rep_train, optimizer, device, beta=beta_at(cfg, epoch)
        )
        quality = evaluate_counterfactuals(
            model, encoder, rep_test, cfg.cvae.n_classes, device
        )
        record = {"epoch": epoch, **losses, **quality}
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"epoch {epoch:3d}/{cfg.train.epochs} | loss {losses['total']:.4f} | "
            f"validity {quality['validity']:.3f} | proximity {quality['proximity']:.3f}"
        )

        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "config": cfg.to_dict()},
            out_dir / "last.pt",
        )
        current = {**losses, **quality}.get(metric_key)
        if current is None:
            raise KeyError(f"checkpoint_metric '{metric_key}' not in {list(record)}")
        if current > best_metric:
            best_metric = current
            best_summary = {"epoch": epoch, **quality}
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "config": cfg.to_dict()},
                out_dir / "best.pt",
            )
    print(f"\nBest {metric_key}: {best_metric:.4f} @ epoch {best_summary.get('epoch')}")
    return best_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CausalEchoNet CVAE.")
    parser.add_argument("--config", required=True, help="Path to a JSON CVAE config.")
    args = parser.parse_args()
    cfg = CVAETrainConfig.from_file(args.config)
    run_from_config(cfg)


if __name__ == "__main__":
    main()
