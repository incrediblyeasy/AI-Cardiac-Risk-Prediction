"""Modality ablation study for EchoFuseNet (Day 12).

Quantifies each signal-to-image modality's contribution by training the fusion
model on every non-empty subset of ``{RP, GAF, MTF}`` and comparing on the same
DS2 test fold:

    single : RP        GAF        MTF
    pair   : RP+GAF    RP+MTF     GAF+MTF
    full   : RP+GAF+MTF                     <- reference

All seven configs are the *same* architecture with a different ``modalities``
setting (see ``EchoFuseNet``), so the comparison is clean. Crucially the Day-6
DataLoader is built **once** and reused across every config — it always yields
all three images; a subset model just ignores the branches it doesn't have — so
the expensive transform computation is shared, not repeated seven times.

Significance
------------
Each subset is compared to the full model with an exact **McNemar test** on the
per-sample DS2 correctness of the two models (both evaluated on the identical
test beats). The table annotates the p-value so "RP+GAF ≈ full" vs. "MTF ≪ full"
is a statistical statement, not eyeballing. (Across-fold paired tests from
``stats`` apply too, if the ablation is run under k-fold CV.)

CLI:
    python -m paper1_echofusenet.training.ablation --config configs/echofusenet_ds1ds2_baseline.json
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models import count_parameters
from .config import TrainConfig
from .crossval import _collect_predictions
from .metrics import ClassificationReport, classification_report
from .stats import TestResult, mcnemar_test
from .train import build_model, resolve_device, set_seed, train

# The seven modality subsets, in report order (singles, pairs, full reference).
MODALITY_SETS: tuple[tuple[str, ...], ...] = (
    ("rp",),
    ("gaf",),
    ("mtf",),
    ("rp", "gaf"),
    ("rp", "mtf"),
    ("gaf", "mtf"),
    ("rp", "gaf", "mtf"),
)

FULL_KEY = "rp+gaf+mtf"


def modality_key(modalities: tuple[str, ...]) -> str:
    """Canonical ``"rp+gaf+mtf"``-style key for a modality subset."""
    return "+".join(modalities)


@dataclass
class AblationEntry:
    """One config's ablation outcome on the shared test fold."""

    key: str
    modalities: tuple[str, ...]
    n_params: int
    accuracy: float
    macro_f1: float
    report: ClassificationReport
    y_true: np.ndarray
    y_pred: np.ndarray
    vs_full: TestResult | None = None  # McNemar vs the full model (None for full)


@dataclass
class AblationReport:
    per_config: list[AblationEntry]

    def format(self) -> str:
        lines = [
            f"{'config':>12} {'params':>9} {'acc':>7} {'macroF1':>8} {'p(vs full)':>11} {'':>4}",
            "-" * 56,
        ]
        for e in self.per_config:
            if e.vs_full is None:
                p_str, star = "  (ref)", ""
            else:
                p_str = f"{e.vs_full.pvalue:.3g}"
                star = _stars(e.vs_full.pvalue)
            lines.append(
                f"{e.key:>12} {e.n_params:>9,} {e.accuracy:7.4f} "
                f"{e.macro_f1:8.4f} {p_str:>11} {star:>4}"
            )
        lines.append("-" * 56)
        lines.append("p vs full: exact McNemar on per-sample DS2 correctness.")
        lines.append("stars: *** p<0.001  ** p<0.01  * p<0.05  (n.s. otherwise)")
        return "\n".join(lines)

    def summary_dict(self) -> dict:
        return {
            "configs": [
                {
                    "key": e.key,
                    "modalities": list(e.modalities),
                    "n_params": e.n_params,
                    "accuracy": e.accuracy,
                    "macro_f1": e.macro_f1,
                    "per_class_f1": e.report.f1.tolist(),
                    "mcnemar_vs_full": (
                        None
                        if e.vs_full is None
                        else {"statistic": e.vs_full.statistic, "pvalue": e.vs_full.pvalue}
                    ),
                }
                for e in self.per_config
            ]
        }


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def run_ablation(
    cfg: TrainConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    modality_sets: tuple[tuple[str, ...], ...] = MODALITY_SETS,
    device: torch.device | None = None,
) -> AblationReport:
    """Train every modality subset on the shared loaders and compare to full.

    The loaders are built once by the caller and reused for all configs. Returns
    an ``AblationReport`` with per-config metrics and McNemar p-values vs. the
    full model.
    """
    device = device or resolve_device(cfg.train.device)
    n_classes = cfg.model.n_classes
    out_root = Path(cfg.train.out_dir)

    entries: dict[str, AblationEntry] = {}
    for modalities in modality_sets:
        key = modality_key(modalities)

        save_dir = out_root / key
        best_model = save_dir / "best.pt"

        if best_model.exists():
            print(f"Skipping {key} (already completed)")
            continue
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.model.modalities = list(modalities)
        sub_cfg.train.out_dir = str(out_root / key)

        set_seed(cfg.train.seed)  # same init/order for every config
        model = build_model(sub_cfg)
        print(f"\n===== ablation: {key} ({count_parameters(model):,} params) =====")
        train(model, train_loader, test_loader, sub_cfg, device=device)

        report = evaluate_report(model, test_loader, device, n_classes)
        y_true, y_pred = _collect_predictions(model, test_loader, device)
        entries[key] = AblationEntry(
            key=key,
            modalities=modalities,
            n_params=count_parameters(model),
            accuracy=report.accuracy,
            macro_f1=report.macro_f1,
            report=report,
            y_true=y_true,
            y_pred=y_pred,
        )

    # McNemar of each subset vs the full model on identical test samples.
    if FULL_KEY in entries:
        full = entries[FULL_KEY]
        full_correct = full.y_true == full.y_pred
        for key, e in entries.items():
            if key == FULL_KEY:
                continue
            e.vs_full = mcnemar_test(full_correct, e.y_true == e.y_pred)

    report = AblationReport(per_config=[entries[modality_key(m)] for m in modality_sets])
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "ablation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(report.summary_dict(), fh, indent=2)
        fh.write("\n")
    return report


def evaluate_report(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, n_classes: int
) -> ClassificationReport:
    y_true, y_pred = _collect_predictions(model, loader, device)
    return classification_report(y_true, y_pred, n_classes)


def run_ablation_from_config(cfg: TrainConfig) -> AblationReport:
    """Build the DS1/DS2 loaders once and run the full ablation sweep."""
    from .train import set_seed as _seed  # local import keeps CLI import light

    _seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    from ..data.dataset import build_dataloaders
    from ..data.transform_cache import DiskTransformCache

    # Ablation already reuses one set of loaders across all 7 modality
    # configs (see run_ablation's docstring) -- so this cache's benefit here
    # is epoch-to-epoch reuse within a run, plus surviving a re-run of this
    # script, not "fewer configs recompute" (that redundancy didn't exist).
    transform_cache = DiskTransformCache(
        Path(cfg.train.out_dir) / "transform_cache",
        max_bytes=getattr(cfg.data, "transform_cache_max_gb", 4) * 1024**3,
    )

    data_kwargs = dict(
        batch_size=cfg.data.batch_size,
        oversample=cfg.data.oversample,
        normalize=cfg.data.normalize,
        seed=cfg.train.seed,
        num_workers=cfg.data.num_workers,
        train_records=tuple(cfg.data.train_records) if cfg.data.train_records else None,
        test_records=tuple(cfg.data.test_records) if cfg.data.test_records else None,
        transform_cache=transform_cache,
    )
    if cfg.data.data_dir:
        data_kwargs["data_dir"] = Path(cfg.data.data_dir)
    train_loader, test_loader = build_dataloaders(**data_kwargs)
    return run_ablation(cfg, train_loader, test_loader, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modality ablation for EchoFuseNet (Day 12).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per config.")
    parser.add_argument("--out-dir", default=None, help="Override config train.out_dir.")

    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir

    report = run_ablation_from_config(cfg)
    print("\n" + "=" * 56)
    print("Modality ablation study")
    print("=" * 56)
    print(report.format())


if __name__ == "__main__":
    main()
