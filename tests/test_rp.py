"""Recurrence Plot transform: shape, symmetry, value range, determinism."""

import numpy as np
import pytest

from paper1_echofusenet.transforms import recurrence_plot


def _sample_signal(n=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    return (np.sin(t) + 0.1 * rng.standard_normal(n)).astype(np.float32)


def test_output_shape_dimension_1():
    x = _sample_signal(64)
    rp = recurrence_plot(x)
    assert rp.shape == (64, 64)
    assert rp.dtype == np.float32


def test_output_shape_with_embedding():
    x = _sample_signal(64)
    rp = recurrence_plot(x, dimension=3, time_delay=2)
    expected = 64 - (3 - 1) * 2
    assert rp.shape == (expected, expected)


def test_symmetric_with_zero_diagonal():
    x = _sample_signal(50)
    rp = recurrence_plot(x)
    assert np.allclose(rp, rp.T)
    assert np.allclose(np.diag(rp), 0.0)


def test_value_range_normalized():
    x = _sample_signal(80)
    rp = recurrence_plot(x, normalize=True)
    assert rp.min() >= 0.0
    assert rp.max() <= 1.0
    assert np.isclose(rp.max(), 1.0)  # min-max scaling hits 1 somewhere


def test_thresholded_is_binary_and_reflexive():
    x = _sample_signal(40)
    rp = recurrence_plot(x, threshold=0.2)
    assert set(np.unique(rp)).issubset({0.0, 1.0})
    # Every state recurs with itself (distance 0 <= threshold).
    assert np.all(np.diag(rp) == 1.0)


def test_deterministic():
    x = _sample_signal(32)
    assert np.array_equal(recurrence_plot(x), recurrence_plot(x))


def test_constant_signal_is_all_zero_unthresholded():
    x = np.full(20, 3.14, dtype=np.float32)
    rp = recurrence_plot(x)
    assert np.all(rp == 0.0)


def test_too_short_for_embedding_raises():
    with pytest.raises(ValueError):
        recurrence_plot(np.zeros(3), dimension=5, time_delay=2)


def test_matches_pyts_reference():
    # Cross-check the unthresholded, normalized RP against pyts.
    pyts = pytest.importorskip("pyts.image")
    x = _sample_signal(48, seed=1)
    ours = recurrence_plot(x, normalize=False)
    ref = pyts.RecurrencePlot(dimension=1, threshold=None).transform(x[None, :])[0]
    # Both are the raw distance matrix; compare up to float tolerance.
    assert np.allclose(ours, ref, atol=1e-5)
