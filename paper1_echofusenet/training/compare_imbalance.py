"""Head-to-head imbalance-recipe comparison for EchoFuseNet (§2 enhancement).

The checklist's core §2 deliverable: run the four class-imbalance strategies on
the **same** DS1/DS2 inter-patient split and pick the winner as the headline
recipe from evidence, not habit. The four configurations compared:

    oversample     — materialised minority oversampling, plain CE
    weighted_ce    — inverse-frequency-weighted CE, no oversampling
    focal          — focal loss (Lin et al.), no oversampling
    class_balanced — effective-number class-balanced focal (Cui et al.)

Each is trained from an identical seed/init on the identical split (only the
balancing/loss differs), then evaluated on DS2 with the full metric set
(accuracy, macro-F1, macro-P/R, MCC, kappa — see ``evaluate``). The recipe with
the best ``selection_metric`` (macro-F1 by default) is reported as the winner,
and a McNemar test (§5) compares the winner against the plain-oversample baseline
on the shared test fold to check the difference is not noise.

This deliberately reuses ``train`` / ``evaluate`` / ``stats`` rather than
re-implementing anything — it is an orchestration layer, so the four runs are
guaranteed identical apart from the recipe under test.

CLI:
    python -m paper1_echofusenet.training.compare_imbalance \\
        --config configs/echofusenet_ds1ds2_baseline.json --out runs/imbalance
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import build_dataloaders
from .config import TrainConfig
from .evaluate import collect_predictions, evaluation_report, format_evaluation
from .stats import mcnemar_test
from .train import build_model, resolve_device, set_seed, train

# The four recipes, expressed purely as (data, loss) config overrides so they all
# flow through the same train()/evaluate() path.
RECIPES: dict[str, dict] = {
    "oversample": {"oversample": True, "use_balanced_sampler": False, "loss": "ce"},
    "weighted_ce": {"oversample": False, "use_balanced_sampler": False, "loss": "weighted_ce"},
    "focal": {"oversample": False, "use_balanced_sampler": False, "loss": "focal"},
    "class_balanced": {"oversample": False, "use_balanced_sampler": False, "loss": "class_balanced"},
}


@dataclass
class RecipeResult:
    """One recipe's evaluation on the shared DS2 fold."""

    name: str
    report: dict                 # evaluation_report() output
    y_true: np.ndarray
    y_pred: np.ndarray

    @property
    def selection_score(self) -> float:
        return self.report["scalars"]["macro_f1"]


def _apply_recipe(cfg: TrainConfig, recipe: dict, out_dir: Path) -> TrainConfig:
    """Return a deep-copied config with one recipe's overrides applied."""
    c = copy.deepcopy(cfg)
    c.data.oversample = recipe["oversample"]
    c.data.use_balanced_sampler = recipe["use_balanced_sampler"]
    c.loss.name = recipe["loss"]
    c.train.class_weighted_loss = False  # recipe drives the loss explicitly
    c.train.out_dir = str(out_dir)
    return c


def compare_recipes(
    cfg: TrainConfig,
    out_root: str | Path = "runs/imbalance",
    selection_metric: str = "macro_f1",
    n_boot: int = 2000,
    device: torch.device | None = None,
) -> dict:
    """Train + evaluate all four recipes on one split; return a comparison dict.

    The DS1/DS2 split is rebuilt per recipe from the *same seed*, so the folds
    are byte-identical across recipes — the only moving part is the balancing
    strategy. Returns a dict with per-recipe metrics, the winner, and the
    McNemar test of winner-vs-oversample.
    """
    device = device or resolve_device(cfg.train.device)
    out_root = Path(out_root)
    results: list[RecipeResult] = []

    for name, recipe in RECIPES.items():
        rc = _apply_recipe(cfg, recipe, out_root / name)
        print(f"\n{'=' * 60}\nRecipe: {name}\n{'=' * 60}")
        set_seed(rc.train.seed)  # identical init across recipes
        train_loader, test_loader = build_dataloaders(
            batch_size=rc.data.batch_size,
            oversample=rc.data.oversample,
            use_balanced_sampler=rc.data.use_balanced_sampler,
            normalize=rc.data.normalize,
            seed=rc.train.seed,
            num_workers=rc.data.num_workers,
            data_dir=Path(rc.data.data_dir) if rc.data.data_dir else None,
            train_records=tuple(rc.data.train_records) if rc.data.train_records else None,
            test_records=tuple(rc.data.test_records) if rc.data.test_records else None,
        )
        model = build_model(rc)
        train(model, train_loader, test_loader, rc, device=device)

        y_true, y_pred = collect_predictions(model, test_loader, device)
        report = evaluation_report(
            y_true, y_pred, rc.model.n_classes, n_boot=n_boot, seed=rc.train.seed
        )
        results.append(RecipeResult(name, report, y_true, y_pred))
        print(f"\n[{name}] evaluation:\n" + format_evaluation(report))

    winner = max(results, key=lambda r: r.report["scalars"][selection_metric])
    baseline = next(r for r in results if r.name == "oversample")

    # §5 significance: winner vs the oversample baseline on the shared test fold.
    mcnemar = mcnemar_test(
        winner.y_true == winner.y_pred, baseline.y_true == baseline.y_pred
    )

    summary = {
        "selection_metric": selection_metric,
        "winner": winner.name,
        "recipes": {
            r.name: {
                "scalars": r.report["scalars"],
                "per_class": r.report["per_class"],
            }
            for r in results
        },
        "winner_vs_oversample_mcnemar": {
            "statistic": mcnemar.statistic,
            "pvalue": mcnemar.pvalue,
            "significant_at_0.05": mcnemar.significant(),
        },
    }

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "imbalance_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print("\n" + "=" * 60)
    print(f"Imbalance comparison — winner by {selection_metric}: {winner.name}")
    print("=" * 60)
    header = f"{'recipe':<16}{'acc':>8}{'macroF1':>9}{'MCC':>8}{'kappa':>8}"
    print(header)
    for r in results:
        s = r.report["scalars"]
        star = " *" if r.name == winner.name else ""
        print(
            f"{r.name:<16}{s['accuracy']:>8.4f}{s['macro_f1']:>9.4f}"
            f"{s['mcc']:>8.4f}{s['cohen_kappa']:>8.4f}{star}"
        )
    print(f"\nwinner vs oversample: {mcnemar}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare class-imbalance recipes on one DS1/DS2 split."
    )
    parser.add_argument("--config", required=True, help="Base JSON training config.")
    parser.add_argument("--out", default="runs/imbalance", help="Output root dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--metric", default="macro_f1", help="Selection metric.")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples.")
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    compare_recipes(
        cfg, out_root=args.out, selection_metric=args.metric, n_boot=args.n_boot
    )


if __name__ == "__main__":
    main()
