"""§3 Optuna search: config plumbing (works without optuna) + gated run."""

import importlib.util

import pytest

from paper1_echofusenet.training.config import TrainConfig
from paper1_echofusenet.tuning import (
    SearchSpace,
    build_trial_config,
    run_search,
    suggest_config,
)

HAS_OPTUNA = importlib.util.find_spec("optuna") is not None


class _FakeTrial:
    """Minimal optuna.Trial stand-in so suggest_config is testable without optuna."""

    def __init__(self):
        self.params = {}

    def suggest_float(self, name, lo, hi, log=False):
        v = (lo * hi) ** 0.5 if log else (lo + hi) / 2
        self.params[name] = v
        return v

    def suggest_categorical(self, name, choices):
        self.params[name] = choices[0]
        return choices[0]


def test_suggest_config_applies_only_searched_fields():
    base = TrainConfig()
    cfg = suggest_config(_FakeTrial(), base, SearchSpace())
    assert base.optim.lr <= 5e-3
    assert 0.1 <= cfg.model.dropout <= 0.5
    assert cfg.data.batch_size in SearchSpace().batch_size
    # Untouched fields keep the base values.
    assert cfg.model.widths == base.model.widths


def test_build_trial_config_roundtrip():
    base = TrainConfig()
    params = {"lr": 2e-3, "weight_decay": 5e-5, "dropout": 0.25, "batch_size": 64}
    cfg = build_trial_config(params, base)
    assert cfg.optim.lr == 2e-3
    assert cfg.optim.weight_decay == 5e-5
    assert cfg.model.dropout == 0.25
    assert cfg.data.batch_size == 64


def test_search_space_serialises():
    d = SearchSpace().to_dict()
    assert set(d) == {"lr", "dropout", "weight_decay", "batch_size"}


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed (optional dep)")
def test_run_search_with_synthetic_objective(tmp_path):
    import math

    base = TrainConfig()

    def objective(cfg):
        # Peaks near lr=1e-3, dropout=0.3 — a smooth synthetic surface.
        return -abs(math.log10(cfg.optim.lr) + 3) - abs(cfg.model.dropout - 0.3)

    summary = run_search(
        base, SearchSpace(), n_trials=8, objective_fn=objective,
        out_dir=tmp_path / "optuna", seed=0,
    )
    assert "best_params" in summary
    assert (tmp_path / "optuna" / "best_config.json").exists()
    assert (tmp_path / "optuna" / "trials.jsonl").exists()
