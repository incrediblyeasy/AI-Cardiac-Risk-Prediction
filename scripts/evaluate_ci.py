"""Bootstrap confidence intervals for a trained EchoFuseNet on DS2 (Day 11).

Loads a checkpoint (e.g. ``runs/ds1ds2_baseline/best.pt``), evaluates it on the
DS2 test fold, and puts a percentile-bootstrap 95% CI around accuracy and
macro-F1 by resampling the test beats — the statistical-significance complement
to Day 10's point estimates. No retraining involved.

Usage:
    python -m scripts.evaluate_ci --config configs/echofusenet_ds1ds2_baseline.json \
        --checkpoint runs/ds1ds2_baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch

from paper1_echofusenet.data.aami import AAMI_CLASSES
from paper1_echofusenet.data.dataset import build_dataloaders
from paper1_echofusenet.training.config import TrainConfig
from paper1_echofusenet.training.crossval import _collect_predictions
from paper1_echofusenet.training.metrics import (
    accuracy_score,
    classification_report,
    format_report,
    macro_f1_score,
)
from paper1_echofusenet.training.stats import bootstrap_metric_ci
from paper1_echofusenet.training.train import build_model, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CIs on DS2 (Day 11).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    device = resolve_device(cfg.train.device)
    n_classes = cfg.model.n_classes

    # DS2 test loader (natural distribution; never oversampled).
    data_kwargs = dict(
        batch_size=cfg.data.batch_size,
        oversample=False,
        normalize=cfg.data.normalize,
        seed=cfg.train.seed,
    )
    if cfg.data.data_dir:
        data_kwargs["data_dir"] = Path(cfg.data.data_dir)
    _, test_loader = build_dataloaders(**data_kwargs)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    y_true, y_pred = _collect_predictions(model, test_loader, device)
    report = classification_report(y_true, y_pred, n_classes)
    print("\nDS2 point estimates:")
    print(format_report(report, AAMI_CLASSES[:n_classes]))

    acc_ci = bootstrap_metric_ci(
        y_true, y_pred, accuracy_score, n_boot=args.n_boot, seed=args.seed
    )
    f1_ci = bootstrap_metric_ci(
        y_true,
        y_pred,
        partial(macro_f1_score, n_classes=n_classes),
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(f"\nBootstrap 95% CIs ({args.n_boot} resamples):")
    print(f"  accuracy : {acc_ci}")
    print(f"  macro-F1 : {f1_ci}")

    out = Path(args.checkpoint).with_name("ds2_bootstrap_ci.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "n_boot": args.n_boot,
                "accuracy": {"point": acc_ci.point, "low": acc_ci.low, "high": acc_ci.high},
                "macro_f1": {"point": f1_ci.point, "low": f1_ci.low, "high": f1_ci.high},
                "per_class_f1": report.f1.tolist(),
                "support": report.support.tolist(),
            },
            fh,
            indent=2,
        )
        fh.write("\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
