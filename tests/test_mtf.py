"""Markov Transition Field transform: shape, range, asymmetry, determinism."""

import numpy as np
import pytest

from paper1_echofusenet.transforms import markov_transition_field


def _sample_signal(n=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    return (np.sin(t) + 0.1 * rng.standard_normal(n)).astype(np.float32)


def test_output_shape_and_dtype():
    x = _sample_signal(64)
    mtf = markov_transition_field(x)
    assert mtf.shape == (64, 64)
    assert mtf.dtype == np.float32


def test_value_range_is_probability():
    x = _sample_signal(80)
    mtf = markov_transition_field(x)
    assert mtf.min() >= 0.0
    assert mtf.max() <= 1.0


def test_generally_not_symmetric():
    # MTF captures directional transitions, so it should not be symmetric.
    x = _sample_signal(60, seed=3)
    mtf = markov_transition_field(x)
    assert not np.allclose(mtf, mtf.T)


def test_deterministic():
    x = _sample_signal(32)
    assert np.array_equal(
        markov_transition_field(x), markov_transition_field(x)
    )


def test_constant_signal_is_single_state():
    # All samples land in one bin -> the only transition is that bin to itself.
    x = np.full(20, 1.5, dtype=np.float32)
    mtf = markov_transition_field(x)
    assert np.allclose(mtf, 1.0)


def test_invalid_n_bins_raises():
    with pytest.raises(ValueError):
        markov_transition_field(_sample_signal(10), n_bins=1)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        markov_transition_field(_sample_signal(10), strategy="bogus")


def test_too_short_raises():
    with pytest.raises(ValueError):
        markov_transition_field(np.array([1.0], dtype=np.float32))


def test_matches_pyts_quantile():
    pyts = pytest.importorskip("pyts.image")
    x = _sample_signal(64, seed=1)
    ours = markov_transition_field(x, n_bins=8, strategy="quantile")
    ref = pyts.MarkovTransitionField(
        n_bins=8, strategy="quantile"
    ).transform(x[None, :])[0]
    assert np.allclose(ours, ref, atol=1e-5)


def test_matches_pyts_uniform():
    pyts = pytest.importorskip("pyts.image")
    x = _sample_signal(64, seed=2)
    ours = markov_transition_field(x, n_bins=6, strategy="uniform")
    ref = pyts.MarkovTransitionField(
        n_bins=6, strategy="uniform"
    ).transform(x[None, :])[0]
    assert np.allclose(ours, ref, atol=1e-5)
