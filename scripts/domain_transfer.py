"""PTB-XL domain-transfer probe (Day 15).

Runs the frozen DS1-trained EchoFuseNet on beats from PTB-XL **NORM** records
(normal ECGs) and reports how well it still calls them "N" under a new domain,
plus the full predicted-class distribution. PTB-XL has no per-beat AAMI truth, so
this is a normal-beat transfer probe, not a 5-class benchmark (see ``data/ptbxl``).

Prereqs:
    python -m paper1_echofusenet.data.ptbxl        # download PTB-XL (several GB)

Usage:
    python -m scripts.domain_transfer \
        --config configs/echofusenet_ds1ds2_baseline.json \
        --checkpoint runs/ds1ds2_baseline/best.pt --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper1_echofusenet.data.aami import AAMI_CLASSES
from paper1_echofusenet.data.dataset import MultimodalBeatDataset
from paper1_echofusenet.data.ptbxl import DEFAULT_PTBXL_DEST, load_ptbxl_norm_beats
from paper1_echofusenet.training.config import TrainConfig
from paper1_echofusenet.training.crossval import _collect_predictions
from paper1_echofusenet.training.stats import bootstrap_metric_ci
from paper1_echofusenet.training.metrics import accuracy_score
from paper1_echofusenet.training.train import build_model, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="PTB-XL domain-transfer probe (Day 15).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ptbxl-dir", default=str(DEFAULT_PTBXL_DEST))
    parser.add_argument("--limit", type=int, default=200, help="Max NORM records.")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    device = resolve_device(cfg.train.device)
    n_classes = cfg.model.n_classes

    print(f"Loading PTB-XL NORM beats (<= {args.limit} records, 100->256 resampled)...")
    beats = load_ptbxl_norm_beats(
        data_dir=Path(args.ptbxl_dir), limit=args.limit, normalize=cfg.data.normalize
    )
    print(f"  {len(beats):,} NORM beats (all expected class N)")
    loader = DataLoader(
        MultimodalBeatDataset(beats, normalize=cfg.data.normalize),
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    y_true, y_pred = _collect_predictions(model, loader, device)
    # All truth is N -> accuracy == N-recall under domain shift.
    n_recall = accuracy_score(y_true, y_pred)
    ci = bootstrap_metric_ci(y_true, y_pred, accuracy_score, n_boot=args.n_boot)

    counts = np.bincount(y_pred, minlength=n_classes)
    dist = {AAMI_CLASSES[i]: int(counts[i]) for i in range(n_classes)}

    print("\nPTB-XL domain transfer (NORM subset):")
    print(f"  beats            : {len(beats):,}")
    print(f"  N-recall         : {n_recall:.4f}   (CI {ci.low:.4f}-{ci.high:.4f})")
    print(f"  predicted dist   : {dist}")
    frac_n = counts[0] / max(counts.sum(), 1)
    print(f"  fraction -> N    : {frac_n:.4f}")

    out = Path(args.checkpoint).with_name("ptbxl_domain_transfer.json")
    out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "n_beats": len(beats),
                "n_recall": n_recall,
                "n_recall_ci": {"low": ci.low, "high": ci.high},
                "predicted_distribution": dist,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
