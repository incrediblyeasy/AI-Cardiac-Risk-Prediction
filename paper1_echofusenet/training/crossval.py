"""Patient-grouped k-fold cross-validation for EchoFuseNet (Day 11).

The inter-patient protocol is preserved *inside* the CV loop: folds are formed by
**partitioning DS1 patients** (never beats), so no patient appears in both a
fold's train and validation sets. DS2 is never touched here — it stays the final
held-out test set from Day 10, so the "test protocol" is intact. CV runs entirely
within DS1 to estimate the *variance* of the model across patient subsets.

Pipeline per fold
-----------------
    DS1 beats --group by patient--> (train patients, val patients)
        train fold --(optional) oversample--> train loader
        val   fold --(natural distribution)--> val loader
        fresh EchoFuseNet -> train() -> evaluate() on the val fold

Each fold yields accuracy + macro-F1; ``mean_confidence_interval`` (see
``stats``) turns the k scores into a mean and a t-based CI, and the pooled
out-of-fold predictions give a bootstrap CI. Compare two configurations with the
paired tests in ``stats``.

CLI:
    python -m paper1_echofusenet.training.crossval --config configs/xxx.json --folds 5
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.aami import AAMI_CLASSES
from ..data.beats import BeatSegment, load_fold
from ..data.dataset import MultimodalBeatDataset, oversample_beats
from ..data.download import DEFAULT_DEST
from ..data.splits import assert_patient_disjoint
from .config import TrainConfig
from .metrics import ClassificationReport, accuracy_score, classification_report, macro_f1_score
from .stats import Interval, bootstrap_metric_ci, mean_confidence_interval
from .train import build_model, evaluate, resolve_device, set_seed, train


def patient_kfold(
    patient_ids: list[int], k: int, seed: int = 0
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Split patient ids into ``k`` (train_patients, val_patients) folds.

    Patients are shuffled deterministically (``seed``) then partitioned into ``k``
    near-equal groups; fold ``i`` validates on group ``i`` and trains on the
    rest. Because groups are disjoint sets of *patients*, every fold is
    inter-patient by construction.
    """
    ids = list(patient_ids)
    if k < 2:
        raise ValueError("k must be >= 2")
    if k > len(ids):
        raise ValueError(f"k={k} exceeds the number of patients ({len(ids)})")

    rng = np.random.default_rng(seed)
    order = np.array(ids)[rng.permutation(len(ids))]
    groups = np.array_split(order, k)

    splits: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for i in range(k):
        val = tuple(int(x) for x in groups[i])
        train = tuple(
            int(x) for j, g in enumerate(groups) if j != i for x in g
        )
        splits.append((train, val))
    return splits


@dataclass
class FoldResult:
    """One CV fold's held-out (validation) outcome."""

    fold: int
    val_patients: tuple[int, ...]
    accuracy: float
    macro_f1: float
    report: ClassificationReport
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class CrossValReport:
    """Aggregated k-fold CV results with confidence intervals."""

    per_fold: list[FoldResult]
    accuracy_ci: Interval
    macro_f1_ci: Interval
    pooled_accuracy_ci: Interval  # bootstrap over pooled out-of-fold predictions
    pooled_macro_f1_ci: Interval
    n_classes: int = 5

    def summary_dict(self) -> dict:
        return {
            "folds": [
                {
                    "fold": f.fold,
                    "val_patients": list(f.val_patients),
                    "accuracy": f.accuracy,
                    "macro_f1": f.macro_f1,
                }
                for f in self.per_fold
            ],
            "accuracy_mean_ci": _interval_dict(self.accuracy_ci),
            "macro_f1_mean_ci": _interval_dict(self.macro_f1_ci),
            "pooled_accuracy_bootstrap_ci": _interval_dict(self.pooled_accuracy_ci),
            "pooled_macro_f1_bootstrap_ci": _interval_dict(self.pooled_macro_f1_ci),
        }

    def format(self) -> str:
        lines = [
            f"{'fold':>4} {'val patients':<28} {'acc':>7} {'macroF1':>8}",
            "-" * 52,
        ]
        for f in self.per_fold:
            pats = ",".join(str(p) for p in f.val_patients)
            if len(pats) > 26:
                pats = pats[:25] + "…"
            lines.append(f"{f.fold:>4} {pats:<28} {f.accuracy:7.4f} {f.macro_f1:8.4f}")
        lines.append("-" * 52)
        lines.append(f"accuracy   (mean ± CI): {self.accuracy_ci}")
        lines.append(f"macro-F1   (mean ± CI): {self.macro_f1_ci}")
        lines.append(f"accuracy (bootstrap CI): {self.pooled_accuracy_ci}")
        lines.append(f"macro-F1 (bootstrap CI): {self.pooled_macro_f1_ci}")
        return "\n".join(lines)


def _interval_dict(iv: Interval) -> dict:
    return {"point": iv.point, "low": iv.low, "high": iv.high, "confidence": iv.confidence}


def cross_validate(
    cfg: TrainConfig,
    k: int = 5,
    seed: int = 0,
    data_dir: Path | None = None,
    records: tuple[int, ...] | None = None,
    device: torch.device | None = None,
    beats: list[BeatSegment] | None = None,
    n_boot: int = 2000,
) -> CrossValReport:
    """Run patient-grouped k-fold CV over DS1 and aggregate with CIs.

    ``beats`` may be supplied directly (used by tests) to bypass loading DS1 from
    disk. Otherwise DS1 is loaded via ``load_fold``. The DS2 test fold is never
    referenced here.
    """
    device = device or resolve_device(cfg.train.device)
    if beats is None:
        beats = load_fold("DS1", data_dir or DEFAULT_DEST, records=records)

    patients = sorted({b.record_id for b in beats})
    splits = patient_kfold(patients, k, seed)
    n_classes = cfg.model.n_classes
    out_root = Path(cfg.train.out_dir)

    per_fold: list[FoldResult] = []
    for i, (train_p, val_p) in enumerate(splits):
        assert_patient_disjoint(train_p, val_p)  # inter-patient guard, per fold
        train_set = set(train_p)
        val_set = set(val_p)
        train_beats = [b for b in beats if b.record_id in train_set]
        val_beats = [b for b in beats if b.record_id in val_set]
        if cfg.data.oversample:
            train_beats = oversample_beats(train_beats, seed=seed)

        gen = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            MultimodalBeatDataset(train_beats, cfg.data.normalize),
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            generator=gen,
        )
        val_loader = DataLoader(
            MultimodalBeatDataset(val_beats, cfg.data.normalize),
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
        )

        # Identical initialisation across folds so score spread reflects the data
        # split, not random init. Each fold checkpoints to its own subdirectory.
        set_seed(seed)
        model = build_model(cfg)
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.train.out_dir = str(out_root / f"fold_{i}")
        print(f"\n===== fold {i} | val patients {val_p} =====")
        train(model, train_loader, val_loader, fold_cfg, device=device)

        report = evaluate(model, val_loader, device, n_classes)
        y_true, y_pred = _collect_predictions(model, val_loader, device)
        per_fold.append(
            FoldResult(
                fold=i,
                val_patients=val_p,
                accuracy=report.accuracy,
                macro_f1=report.macro_f1,
                report=report,
                y_true=y_true,
                y_pred=y_pred,
            )
        )

    accs = [f.accuracy for f in per_fold]
    f1s = [f.macro_f1 for f in per_fold]
    pooled_true = np.concatenate([f.y_true for f in per_fold])
    pooled_pred = np.concatenate([f.y_pred for f in per_fold])

    report = CrossValReport(
        per_fold=per_fold,
        accuracy_ci=mean_confidence_interval(accs),
        macro_f1_ci=mean_confidence_interval(f1s),
        pooled_accuracy_ci=bootstrap_metric_ci(
            pooled_true, pooled_pred, accuracy_score, n_boot=n_boot, seed=seed
        ),
        pooled_macro_f1_ci=bootstrap_metric_ci(
            pooled_true,
            pooled_pred,
            lambda t, p: macro_f1_score(t, p, n_classes),
            n_boot=n_boot,
            seed=seed,
        ),
        n_classes=n_classes,
    )

    # Persist the summary next to the fold checkpoints.
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "cv_summary.json", "w", encoding="utf-8") as fh:
        json.dump(report.summary_dict(), fh, indent=2)
        fh.write("\n")
    return report


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient-grouped k-fold CV for EchoFuseNet (Day 11)."
    )
    parser.add_argument("--config", required=True, help="Path to a JSON training config.")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per fold.")
    parser.add_argument("--out-dir", default=None, help="Override config train.out_dir.")
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir

    report = cross_validate(cfg, k=args.folds, seed=cfg.train.seed)
    print("\n" + "=" * 52)
    print(f"{args.folds}-fold patient-grouped CV")
    print("=" * 52)
    print(report.format())


if __name__ == "__main__":
    main()
