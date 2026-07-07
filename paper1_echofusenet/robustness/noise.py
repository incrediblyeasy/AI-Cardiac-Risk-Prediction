"""Signal-level noise injectors + an evaluation-under-noise harness (§6).

Three corruption models, all operating on the last axis so they vectorise over a
batch of beats ``(N, L)`` as well as a single beat ``(L,)``:

* **Gaussian** — white noise at a target signal-to-noise ratio (dB). The honest
  way to parametrise: noise power is set from the *signal's own* power so the
  requested SNR is achieved regardless of beat amplitude.
* **Baseline wander** — a slow (< 1 Hz) sinusoid modelling respiration / electrode
  motion; amplitude is expressed as a multiple of the beat's own std so it scales
  with signal size.
* **Powerline interference** — a 50/60 Hz sinusoid modelling mains pickup.

``evaluate_under_noise`` sweeps a metric across noise levels with **repeated
runs** and reports a confidence interval per level (Student-t across runs, or a
percentile bootstrap when a single run is requested) — exactly the rigour the
§6 checklist item asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..training.metrics import accuracy_score, macro_f1_score
from ..training.stats import Interval, bootstrap_metric_ci, mean_confidence_interval

# MIT-BIH is sampled at 360 Hz; used to place wander/powerline frequencies.
DEFAULT_FS: float = 360.0

# A prediction function maps a batch of (possibly noised) 1-D beats to labels.
PredictFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- #
# Noise injectors
# --------------------------------------------------------------------------- #
def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def add_gaussian_noise(
    signal: np.ndarray, snr_db: float, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Add white Gaussian noise at a target ``snr_db`` (per-signal power-matched).

    Noise power is ``P_signal / 10**(snr_db/10)`` computed per row, so every beat
    ends up at the requested SNR regardless of its amplitude. A silent (zero-power)
    beat is returned unchanged (no SNR is defined for it).
    """
    x = np.asarray(signal, dtype=np.float64)
    gen = _rng(rng)
    power = np.mean(x**2, axis=-1, keepdims=True)
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = gen.standard_normal(x.shape) * np.sqrt(noise_power)
    return x + noise


def add_baseline_wander(
    signal: np.ndarray,
    amplitude: float,
    fs: float = DEFAULT_FS,
    freq: float = 0.3,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Add a low-frequency (default 0.3 Hz) sinusoid modelling baseline wander.

    ``amplitude`` is a multiple of each beat's own std, so the drift scales with
    the signal. A random phase per beat avoids a fixed, learnable artefact.
    """
    x = np.asarray(signal, dtype=np.float64)
    gen = _rng(rng)
    L = x.shape[-1]
    t = np.arange(L) / fs
    std = np.std(x, axis=-1, keepdims=True)
    phase = gen.uniform(0.0, 2.0 * np.pi, size=x.shape[:-1] + (1,))
    wander = amplitude * std * np.sin(2.0 * np.pi * freq * t + phase)
    return x + wander


def add_powerline_interference(
    signal: np.ndarray,
    amplitude: float,
    fs: float = DEFAULT_FS,
    freq: float = 50.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Add a mains-frequency (default 50 Hz) sinusoid modelling powerline pickup.

    ``amplitude`` is a multiple of each beat's own std; phase is randomised per
    beat. Use ``freq=60.0`` for 60 Hz mains regions.
    """
    x = np.asarray(signal, dtype=np.float64)
    gen = _rng(rng)
    L = x.shape[-1]
    t = np.arange(L) / fs
    std = np.std(x, axis=-1, keepdims=True)
    phase = gen.uniform(0.0, 2.0 * np.pi, size=x.shape[:-1] + (1,))
    interference = amplitude * std * np.sin(2.0 * np.pi * freq * t + phase)
    return x + interference


def measure_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Empirical SNR in dB between a clean signal and its noised version."""
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noisy, dtype=np.float64) - clean
    sig_power = float(np.mean(clean**2))
    noise_power = float(np.mean(noise**2))
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * np.log10(sig_power / noise_power)


#: Dispatch table — the primary ``level`` argument means SNR(dB) for gaussian and
#: relative amplitude for the two sinusoidal corruptions.
_NOISE_KINDS = ("gaussian", "baseline_wander", "powerline")


def apply_noise(
    signal: np.ndarray,
    kind: str,
    level: float,
    *,
    fs: float = DEFAULT_FS,
    freq: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Apply a named corruption. ``level`` = SNR(dB) for gaussian, else amplitude."""
    gen = _rng(rng)
    if kind == "gaussian":
        return add_gaussian_noise(signal, level, gen)
    if kind == "baseline_wander":
        return add_baseline_wander(signal, level, fs=fs, freq=freq or 0.3, rng=gen)
    if kind == "powerline":
        return add_powerline_interference(signal, level, fs=fs, freq=freq or 50.0, rng=gen)
    raise ValueError(f"unknown noise kind '{kind}'; choose from {list(_NOISE_KINDS)}")


# --------------------------------------------------------------------------- #
# Evaluation under noise
# --------------------------------------------------------------------------- #
@dataclass
class NoiseLevelResult:
    """Accuracy + macro-F1 (with CIs across repeats) at one noise level."""

    level: float
    accuracy: Interval
    macro_f1: Interval
    n_runs: int


@dataclass
class NoiseSweepResult:
    """A full noise sweep: clean baseline + one result per level."""

    kind: str
    clean_accuracy: float
    clean_macro_f1: float
    levels: list[NoiseLevelResult]

    def summary_dict(self) -> dict:
        return {
            "kind": self.kind,
            "clean": {"accuracy": self.clean_accuracy, "macro_f1": self.clean_macro_f1},
            "levels": [
                {
                    "level": r.level,
                    "n_runs": r.n_runs,
                    "accuracy": {"point": r.accuracy.point, "low": r.accuracy.low, "high": r.accuracy.high},
                    "macro_f1": {"point": r.macro_f1.point, "low": r.macro_f1.low, "high": r.macro_f1.high},
                }
                for r in self.levels
            ],
        }


def evaluate_under_noise(
    predict_fn: PredictFn,
    signals: np.ndarray,
    labels: np.ndarray,
    *,
    kind: str,
    levels: list[float],
    n_classes: int = 5,
    n_runs: int = 3,
    fs: float = DEFAULT_FS,
    freq: float | None = None,
    seed: int = 0,
    confidence: float = 0.95,
    n_boot: int = 2000,
) -> NoiseSweepResult:
    """Sweep ``predict_fn``'s accuracy / macro-F1 across noise ``levels``.

    ``predict_fn`` maps a ``(N, L)`` batch of beats to ``(N,)`` predicted labels;
    it wraps the transform+model pipeline in a real run and is a trivial stub in
    tests. Each level is evaluated ``n_runs`` times with independent noise draws:
    with ``n_runs >= 2`` the CI is Student-t across runs (captures noise-draw
    variance); with ``n_runs == 1`` it falls back to a percentile bootstrap over
    the beats. The clean (noise-free) baseline is reported alongside.
    """
    if kind not in _NOISE_KINDS:
        raise ValueError(f"unknown noise kind '{kind}'; choose from {list(_NOISE_KINDS)}")
    signals = np.asarray(signals, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    if signals.shape[0] != labels.shape[0]:
        raise ValueError("signals and labels must have the same length")

    # Two-arg (y_true, y_pred) signature so these double as bootstrap metric_fns.
    def _acc(yt: np.ndarray, yp: np.ndarray) -> float:
        return accuracy_score(yt, yp)

    def _f1(yt: np.ndarray, yp: np.ndarray) -> float:
        return macro_f1_score(yt, yp, n_classes)

    clean_pred = np.asarray(predict_fn(signals)).ravel()
    clean_acc = _acc(labels, clean_pred)
    clean_f1 = _f1(labels, clean_pred)

    rng = np.random.default_rng(seed)
    level_results: list[NoiseLevelResult] = []
    for level in levels:
        accs: list[float] = []
        f1s: list[float] = []
        last_pred: np.ndarray | None = None
        for _ in range(max(1, n_runs)):
            noised = apply_noise(signals, kind, level, fs=fs, freq=freq, rng=rng)
            pred = np.asarray(predict_fn(noised)).ravel()
            last_pred = pred
            accs.append(_acc(labels, pred))
            f1s.append(_f1(labels, pred))

        if n_runs >= 2:
            acc_ci = mean_confidence_interval(accs, confidence)
            f1_ci = mean_confidence_interval(f1s, confidence)
        else:  # single run — bootstrap over beats instead
            acc_ci = bootstrap_metric_ci(labels, last_pred, _acc, n_boot=n_boot, confidence=confidence, seed=seed)
            f1_ci = bootstrap_metric_ci(labels, last_pred, _f1, n_boot=n_boot, confidence=confidence, seed=seed)

        level_results.append(
            NoiseLevelResult(level=level, accuracy=acc_ci, macro_f1=f1_ci, n_runs=max(1, n_runs))
        )

    return NoiseSweepResult(
        kind=kind,
        clean_accuracy=clean_acc,
        clean_macro_f1=clean_f1,
        levels=level_results,
    )
