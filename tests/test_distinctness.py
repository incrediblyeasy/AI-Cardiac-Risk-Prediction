"""Distinctness guard: RP, GAF and MTF must be genuinely different modalities.

The original design draft had a channel-duplication defect where the three
"modalities" were effectively the same image. This test locks in that RP, GAF
and MTF produce physically distinct fields for the *same* beat, so the three CNN
branches see different information.
"""

import numpy as np
import pytest

from paper1_echofusenet.transforms import (
    gramian_angular_field,
    markov_transition_field,
    recurrence_plot,
)


def _sample_signal(n=128, seed=7):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * np.pi, n)
    return (np.sin(t) + 0.3 * np.sin(3 * t) + 0.1 * rng.standard_normal(n)).astype(
        np.float32
    )


def _minmax(a):
    a = a.astype(np.float64)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _corr(a, b):
    return float(np.corrcoef(_minmax(a).ravel(), _minmax(b).ravel())[0, 1])


@pytest.fixture
def fields():
    x = _sample_signal()
    return {
        "RP": recurrence_plot(x),
        "GAF": gramian_angular_field(x),
        "MTF": markov_transition_field(x),
    }


def test_same_shape(fields):
    shapes = {k: v.shape for k, v in fields.items()}
    assert len(set(shapes.values())) == 1, shapes  # stackable into channels


def test_not_elementwise_equal(fields):
    names = list(fields)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = fields[names[i]], fields[names[j]]
            assert not np.array_equal(a, b), f"{names[i]} == {names[j]}"


def test_pairwise_correlation_below_duplication_threshold(fields):
    # |corr| ~ 1.0 would mean two channels are effectively the same image.
    names = list(fields)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = abs(_corr(fields[names[i]], fields[names[j]]))
            assert c < 0.95, f"{names[i]} vs {names[j]} correlation {c:.3f} too high"
