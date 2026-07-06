"""Optuna hyperparameter search for EchoFuseNet (§3 enhancement).

Wraps the Day-9 training loop in an Optuna objective so hyperparameters are
tuned by a documented, reproducible search instead of by hand. Every trial's
sampled config *and* its resulting score are logged (via ``shared.utils``'s
experiment ledger — the §8 "save experiment metadata" utility, built once and
reused here rather than duplicated), so the search is auditable after the fact.

Search space (checklist §3): learning rate, dropout, weight decay, batch size.
``image_size`` is deliberately **omitted** from the default space: in this
pipeline the transform image side length is fixed by the beat window (L = 256,
see ``data/beats.py``), so varying it would need a resize transform that does not
exist yet — flagged here rather than faked. Add it to ``SearchSpace`` only once a
resize path is wired.

**No test-fold leakage.** Tuning must never look at DS2. The default objective
scores a trial by patient-grouped k-fold CV *within DS1* (see
``training.crossval``), so DS2 stays the untouched final test set. The objective
is injected as a callable, so unit tests can score trials with a cheap synthetic
function instead of real training.

Optuna is an **optional** dependency (see ``pyproject.toml`` ``[tuning]`` extra);
importing this module without it raises a clear install hint only when a search
is actually run.

CLI:
    python -m paper1_echofusenet.tuning.optuna_search \\
        --config configs/echofusenet_ds1ds2_baseline.json --trials 30 --folds 3
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..training.config import TrainConfig

# An objective scores a resolved config and returns a scalar to *maximise*.
ObjectiveFn = Callable[[TrainConfig], float]


def _require_optuna():
    """Import optuna or raise an actionable install hint (optional dependency)."""
    try:
        import optuna

        return optuna
    except ImportError as exc:  # pragma: no cover - exercised only without optuna
        raise ImportError(
            "Optuna is required for hyperparameter search. Install it with "
            "`pip install optuna` or `pip install -e '.[tuning]'`."
        ) from exc


@dataclass
class SearchSpace:
    """Ranges for the tunable hyperparameters (checklist §3).

    Learning rate and weight decay are searched on a log scale (they span orders
    of magnitude); dropout on a linear scale; batch size over a discrete set.
    Bounds are intentionally conservative around the current defaults so the
    search refines rather than wanders.
    """

    lr: tuple[float, float] = (1e-4, 5e-3)
    dropout: tuple[float, float] = (0.1, 0.5)
    weight_decay: tuple[float, float] = (1e-6, 1e-3)
    batch_size: tuple[int, ...] = (16, 32, 64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lr": list(self.lr),
            "dropout": list(self.dropout),
            "weight_decay": list(self.weight_decay),
            "batch_size": list(self.batch_size),
        }


def suggest_config(trial, base_cfg: TrainConfig, space: SearchSpace) -> TrainConfig:
    """Sample one hyperparameter point and return a config with it applied.

    ``trial`` is an ``optuna.Trial``. The base config supplies everything not
    under search (architecture widths, data records, epochs, …); only the four
    searched fields are overwritten, so a trial differs from the baseline in
    exactly those knobs.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.optim.lr = trial.suggest_float("lr", *space.lr, log=True)
    cfg.optim.weight_decay = trial.suggest_float(
        "weight_decay", *space.weight_decay, log=True
    )
    cfg.model.dropout = trial.suggest_float("dropout", *space.dropout)
    cfg.data.batch_size = trial.suggest_categorical(
        "batch_size", list(space.batch_size)
    )
    return cfg


def build_trial_config(
    params: dict[str, Any], base_cfg: TrainConfig
) -> TrainConfig:
    """Rebuild the exact config for a set of trial params (e.g. the best trial).

    Mirrors ``suggest_config`` but from a plain params dict, so the winning
    configuration can be reconstructed and re-run outside Optuna.
    """
    cfg = copy.deepcopy(base_cfg)
    if "lr" in params:
        cfg.optim.lr = params["lr"]
    if "weight_decay" in params:
        cfg.optim.weight_decay = params["weight_decay"]
    if "dropout" in params:
        cfg.model.dropout = params["dropout"]
    if "batch_size" in params:
        cfg.data.batch_size = params["batch_size"]
    return cfg


def _default_cv_objective(
    folds: int, seed: int, metric: str, data_dir: Path | None
) -> ObjectiveFn:
    """DS1-internal k-fold CV objective (mean ``metric`` across folds).

    Imported lazily so the module is usable (and testable) without pulling in the
    heavy training/data stack until a real search runs.
    """
    from ..training.crossval import cross_validate

    def objective(cfg: TrainConfig) -> float:
        report = cross_validate(cfg, k=folds, seed=seed, data_dir=data_dir)
        iv = report.macro_f1_ci if metric == "macro_f1" else report.accuracy_ci
        return float(iv.point)

    return objective


def run_search(
    base_cfg: TrainConfig,
    space: SearchSpace | None = None,
    n_trials: int = 30,
    objective_fn: ObjectiveFn | None = None,
    seed: int = 0,
    folds: int = 3,
    metric: str = "macro_f1",
    out_dir: str | Path = "runs/optuna",
    data_dir: Path | None = None,
) -> dict:
    """Run an Optuna study maximising ``metric`` and return a result summary.

    ``objective_fn`` scores a resolved ``TrainConfig`` and returns the scalar to
    maximise; when omitted, DS1-internal k-fold CV is used (no DS2 leakage).
    Every trial appends ``{params, value}`` to ``trials.jsonl`` and to the shared
    experiment ledger; the best config is written to ``best_config.json``.
    """
    optuna = _require_optuna()
    space = space or SearchSpace()
    scorer = objective_fn or _default_cv_objective(folds, seed, metric, data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    trials_path.write_text("", encoding="utf-8")

    def _objective(trial) -> float:
        cfg = suggest_config(trial, base_cfg, space)
        value = float(scorer(cfg))
        record = {"number": trial.number, "params": trial.params, "value": value}
        with open(trials_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        # §8: log each trial's config + result to the shared ledger.
        try:
            from shared.utils import log_experiment

            log_experiment(
                name=f"optuna_trial_{trial.number}",
                seed=seed,
                config=cfg.to_dict(),
                metrics={metric: value},
                ledger=out_dir / "experiments.jsonl",
                extra={"optuna_params": trial.params},
            )
        except Exception:  # logging must never fail a trial
            pass
        return value

    # Seeded TPE sampler for a reproducible search.
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_objective, n_trials=n_trials)

    best_cfg = build_trial_config(study.best_params, base_cfg)
    best_cfg.to_file(out_dir / "best_config.json")

    summary = {
        "metric": metric,
        "n_trials": n_trials,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "search_space": space.to_dict(),
    }
    with open(out_dir / "search_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print("\n" + "=" * 52)
    print(f"Optuna search — {n_trials} trials, best {metric} = {study.best_value:.4f}")
    print("=" * 52)
    for k, v in study.best_params.items():
        print(f"  {k:<14}: {v}")
    print(f"\nbest config -> {out_dir / 'best_config.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for EchoFuseNet (§3).")
    parser.add_argument("--config", required=True, help="Base JSON training config.")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials.")
    parser.add_argument("--folds", type=int, default=3, help="DS1 CV folds per trial.")
    parser.add_argument("--metric", default="macro_f1", help="Metric to maximise.")
    parser.add_argument("--out", default="runs/optuna", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs/trial.")
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    run_search(
        cfg,
        n_trials=args.trials,
        folds=args.folds,
        metric=args.metric,
        out_dir=args.out,
        seed=cfg.train.seed,
    )


if __name__ == "__main__":
    main()
