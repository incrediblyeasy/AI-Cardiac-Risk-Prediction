"""Full evaluation report for a trained EchoFuseNet checkpoint (§2 enhancement).

The training loop prints per-epoch metrics, but the *paper* needs a single,
complete, reproducible evaluation of a chosen checkpoint on the DS2 test fold —
every metric the enhancement checklist calls for, each with a bootstrap
confidence interval so the numbers can be reported honestly.

Reported per run:

* accuracy, macro-F1, macro-precision, macro-recall, **MCC**, **Cohen's kappa**
  (the imbalance-robust headline set), each with a percentile bootstrap CI;
* per-class precision / recall / F1 / support (where the minority recipes are
  actually judged);
* the confusion matrix.

Everything is derived from a single ``(y_true, y_pred)`` pass, so adding a metric
is cheap. Results are printed as a table and written to ``eval_report.json`` next
to the checkpoint for downstream aggregation.

CLI:
    python -m paper1_echofusenet.training.evaluate --config configs/x.json \\
        --checkpoint runs/x/best.pt
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.aami import AAMI_CLASSES
from ..data.dataset import build_dataloaders
from .config import TrainConfig
from .metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    format_report,
    macro_f1_score,
    mcc_score,
)
from .stats import Interval, bootstrap_metric_ci
from .train import build_model, resolve_device


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)`` over a whole loader (test fold)."""
    model.eval()
    trues: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for rp, gaf, mtf, labels in loader:
        rp, gaf, mtf = rp.to(device), gaf.to(device), mtf.to(device)
        logits = model(rp, gaf, mtf)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(labels.numpy())
    if not trues:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(trues), np.concatenate(preds)


def evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    class_names: tuple[str, ...] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    with_ci: bool = True,
) -> dict:
    """Assemble the full metric dict (scalars + bootstrap CIs + per-class + CM).

    ``with_ci=False`` skips the (relatively expensive) bootstrap, e.g. when the
    caller only wants point estimates for a many-config comparison sweep.
    """
    class_names = class_names or AAMI_CLASSES[:n_classes]
    report = classification_report(y_true, y_pred, n_classes)

    out: dict = {
        "scalars": report.scalar_metrics(),
        "per_class": report.per_class_metrics(class_names),
        "confusion": report.confusion.tolist(),
        "n_samples": int(report.support.sum()),
    }

    if with_ci and y_true.size:
        metric_fns = {
            "accuracy": accuracy_score,
            "macro_f1": partial(macro_f1_score, n_classes=n_classes),
            "mcc": partial(mcc_score, n_classes=n_classes),
            "cohen_kappa": partial(cohen_kappa_score, n_classes=n_classes),
        }
        cis: dict[str, dict] = {}
        for name, fn in metric_fns.items():
            iv: Interval = bootstrap_metric_ci(
                y_true, y_pred, fn, n_boot=n_boot, seed=seed
            )
            cis[name] = {"point": iv.point, "low": iv.low, "high": iv.high,
                         "confidence": iv.confidence}
        out["bootstrap_ci"] = cis
    return out


def format_evaluation(report_dict: dict) -> str:
    """Render the evaluation dict as a human-readable block."""
    s = report_dict["scalars"]
    lines = [
        f"accuracy        : {s['accuracy']:.4f}",
        f"macro-F1        : {s['macro_f1']:.4f}",
        f"macro-precision : {s['macro_precision']:.4f}",
        f"macro-recall    : {s['macro_recall']:.4f}",
        f"MCC             : {s['mcc']:.4f}",
        f"Cohen's kappa   : {s['cohen_kappa']:.4f}",
    ]
    if "bootstrap_ci" in report_dict:
        lines.append("")
        lines.append("bootstrap 95% CIs:")
        for name, iv in report_dict["bootstrap_ci"].items():
            lines.append(f"  {name:<12}: {iv['point']:.4f} [{iv['low']:.4f}, {iv['high']:.4f}]")
    lines.append("")
    lines.append("per-class (precision / recall / f1 / support):")
    for name, m in report_dict["per_class"].items():
        lines.append(
            f"  {name:>3}: {m['precision']:.3f} / {m['recall']:.3f} / "
            f"{m['f1']:.3f} / {m['support']}"
        )
    return "\n".join(lines)


def evaluate_checkpoint(
    cfg: TrainConfig,
    checkpoint: str | Path,
    n_boot: int = 2000,
    device: torch.device | None = None,
) -> dict:
    """Load a checkpoint, run it on DS2, and return the full evaluation dict."""
    device = device or resolve_device(cfg.train.device)
    _, test_loader = build_dataloaders(
        batch_size=cfg.data.batch_size,
        oversample=False,  # never balance the test fold
        normalize=cfg.data.normalize,
        seed=cfg.train.seed,
        num_workers=cfg.data.num_workers,
        data_dir=Path(cfg.data.data_dir) if cfg.data.data_dir else None,
        train_records=tuple(cfg.data.train_records) if cfg.data.train_records else None,
        test_records=tuple(cfg.data.test_records) if cfg.data.test_records else None,
    )
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)

    y_true, y_pred = collect_predictions(model, test_loader, device)
    return evaluation_report(
        y_true, y_pred, cfg.model.n_classes, n_boot=n_boot, seed=cfg.train.seed
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an EchoFuseNet checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the run's JSON config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples.")
    parser.add_argument("--out", default=None, help="Where to write eval_report.json.")
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    report = evaluate_checkpoint(cfg, args.checkpoint, n_boot=args.n_boot)
    print(format_evaluation(report))

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
