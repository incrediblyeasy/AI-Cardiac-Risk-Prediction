"""External validation of a trained EchoFuseNet on INCART (Day 14).

Loads a DS1-trained checkpoint and evaluates it — **no retraining** — on the
St. Petersburg INCART database via the same RP/GAF/MTF pipeline, then prints the
INCART metrics next to the DS2 baseline so the generalisation gap is explicit.

Prereqs:
    python -m paper1_echofusenet.data.incart            # download INCART (~large)

Usage:
    python -m scripts.external_validation \
        --config configs/echofusenet_ds1ds2_baseline.json \
        --checkpoint runs/ds1ds2_baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from paper1_echofusenet.data.aami import AAMI_CLASSES
from paper1_echofusenet.data.dataset import MultimodalBeatDataset
from paper1_echofusenet.data.incart import DEFAULT_INCART_DEST, load_incart
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


def _baseline_ds2(run_dir: Path, n_classes: int) -> dict | None:
    """Best DS2 numbers from a completed Day-10 run, if available."""
    ci = run_dir / "ds2_bootstrap_ci.json"
    if ci.exists():
        d = json.loads(ci.read_text())
        return {"accuracy": d["accuracy"]["point"], "macro_f1": d["macro_f1"]["point"]}
    hist = run_dir / "history.jsonl"
    if hist.exists():
        rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
        if rows:
            best = max(rows, key=lambda r: r.get("macro_f1", 0.0))
            return {"accuracy": best["accuracy"], "macro_f1": best["macro_f1"]}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="INCART external validation (Day 14).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--incart-dir", default=str(DEFAULT_INCART_DEST))
    parser.add_argument("--records", nargs="*", default=None, help="Subset of INCART records.")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    device = resolve_device(cfg.train.device)
    n_classes = cfg.model.n_classes

    print("Loading INCART beats (resampled 257->256 to match training)...")
    beats = load_incart(
        data_dir=Path(args.incart_dir),
        records=tuple(args.records) if args.records else None,
        normalize=cfg.data.normalize,
    )
    print(f"  {len(beats):,} INCART beats")
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
    report = classification_report(y_true, y_pred, n_classes)
    acc_ci = bootstrap_metric_ci(y_true, y_pred, accuracy_score, n_boot=args.n_boot)
    f1_ci = bootstrap_metric_ci(
        y_true, y_pred, lambda t, p: macro_f1_score(t, p, n_classes), n_boot=args.n_boot
    )

    print("\nINCART external validation:")
    print(format_report(report, AAMI_CLASSES[:n_classes]))
    print(f"\n  accuracy : {acc_ci}")
    print(f"  macro-F1 : {f1_ci}")

    # Side-by-side with the DS2 baseline.
    baseline = _baseline_ds2(Path(args.checkpoint).parent, n_classes)
    print("\n" + "=" * 44)
    print(f"{'metric':>10} {'DS2 (internal)':>16} {'INCART (external)':>18}")
    print("-" * 44)
    for key, label in (("accuracy", "accuracy"), ("macro_f1", "macro-F1")):
        ext = report.accuracy if key == "accuracy" else report.macro_f1
        base = f"{baseline[key]:.4f}" if baseline else "n/a"
        print(f"{label:>10} {base:>16} {ext:>18.4f}")
    print("=" * 44)

    out = Path(args.checkpoint).with_name("incart_external.json")
    out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "n_beats": len(beats),
                "accuracy": {"point": acc_ci.point, "low": acc_ci.low, "high": acc_ci.high},
                "macro_f1": {"point": f1_ci.point, "low": f1_ci.low, "high": f1_ci.high},
                "per_class_f1": report.f1.tolist(),
                "support": report.support.tolist(),
                "ds2_baseline": baseline,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
