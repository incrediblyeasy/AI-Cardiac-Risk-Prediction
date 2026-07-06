"""Hyperparameter optimisation for EchoFuseNet (§3 enhancement)."""

from .optuna_search import (
    SearchSpace,
    build_trial_config,
    run_search,
    suggest_config,
)

__all__ = ["SearchSpace", "build_trial_config", "run_search", "suggest_config"]
