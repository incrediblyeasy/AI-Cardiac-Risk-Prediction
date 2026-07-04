"""Causal-validation stack: target-trial protocol + sensitivity diagnostics.

What makes Paper 3's causal claims defensible (roadmap §4.8) rather than a
validation afterthought. Implemented and tested here: the E-value for unmeasured
confounding (VanderWeele & Ding), the pre-registration ``TargetTrialProtocol``
object, the stabilised-IPW ATE estimator, and the assumption diagnostics
(covariate-balance SMD, positivity, negative-control outcomes). These operate on
arrays, so they run on the MIMIC-IV cohort once it exists and are unit-tested on
synthetic data now.
"""

from .evalues import e_value, e_value_ci
from .protocol import TargetTrialProtocol
from .ipw import ipw_ate, stabilized_weights
from .diagnostics import (
    negative_control_test,
    positivity_check,
    standardized_mean_difference,
)

__all__ = [
    "e_value",
    "e_value_ci",
    "TargetTrialProtocol",
    "ipw_ate",
    "stabilized_weights",
    "standardized_mean_difference",
    "positivity_check",
    "negative_control_test",
]
