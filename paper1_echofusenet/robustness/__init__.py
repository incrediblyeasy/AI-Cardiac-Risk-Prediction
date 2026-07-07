"""§6 robustness — noise-injection and evaluation-under-noise for EchoFuseNet.

The publishable-fixes checklist (§6, "Improve robustness analysis") asks for a
*rigorous* noise study rather than a single Gaussian sweep: several noise levels
with **repeated runs and confidence intervals**, plus physiologically realistic
corruptions (baseline wander, powerline interference) that an ambulatory ECG
actually sees — not only white Gaussian noise.

Noise is added to the raw 1-D beat signal *before* the RP/GAF/MTF imaging, which
is where real acquisition noise enters; the transforms and model then run
unchanged. ``evaluate_under_noise`` is deliberately model-agnostic (it takes a
``predict_fn``), so it unit-tests without a trained checkpoint and, in a real run,
wraps the transform+model pipeline.
"""

from __future__ import annotations

from .noise import (
    NoiseLevelResult,
    NoiseSweepResult,
    add_baseline_wander,
    add_gaussian_noise,
    add_powerline_interference,
    apply_noise,
    evaluate_under_noise,
    measure_snr_db,
)

__all__ = [
    "add_gaussian_noise",
    "add_baseline_wander",
    "add_powerline_interference",
    "apply_noise",
    "measure_snr_db",
    "evaluate_under_noise",
    "NoiseLevelResult",
    "NoiseSweepResult",
]
