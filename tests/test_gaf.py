"""Gramian Angular Field transform: shape, value range, symmetry, determinism."""

import numpy as np
import pytest

from paper1_echofusenet.transforms import gramian_angular_field


def _sample_signal(n=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    return (np.sin(t) + 0.1 * rng.standard_normal(n)).astype(np.float32)


def test_output_shape_and_dtype():
    x = _sample_signal(64)
    gaf = gramian_angular_field(x)
    assert gaf.shape == (64, 64)
    assert gaf.dtype == np.float32


def test_value_range_within_unit_interval():
    x = _sample_signal(80)
    gaf = gramian_angular_field(x)
    assert gaf.min() >= -1.0 - 1e-6
    assert gaf.max() <= 1.0 + 1e-6


def test_gasf_is_symmetric():
    x = _sample_signal(50)
    gasf = gramian_angular_field(x, method="summation")
    assert np.allclose(gasf, gasf.T, atol=1e-6)


def test_gadf_is_antisymmetric_with_zero_diagonal():
    x = _sample_signal(50)
    gadf = gramian_angular_field(x, method="difference")
    assert np.allclose(gadf, -gadf.T, atol=1e-6)
    assert np.allclose(np.diag(gadf), 0.0, atol=1e-6)


def test_deterministic():
    x = _sample_signal(32)
    assert np.array_equal(
        gramian_angular_field(x), gramian_angular_field(x)
    )


def test_constant_signal_maps_to_midpoint():
    # Constant -> scaled to midpoint of sample_range; GASF is a constant image.
    x = np.full(16, 2.71, dtype=np.float32)
    gasf = gramian_angular_field(x, sample_range=(-1.0, 1.0))
    # midpoint of (-1, 1) is 0 -> phi = arccos(0) = pi/2 -> cos(pi) = -1.
    assert np.allclose(gasf, -1.0, atol=1e-6)


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        gramian_angular_field(_sample_signal(10), method="bogus")


def test_invalid_sample_range_raises():
    with pytest.raises(ValueError):
        gramian_angular_field(_sample_signal(10), sample_range=(-2.0, 1.0))


def test_empty_signal_raises():
    with pytest.raises(ValueError):
        gramian_angular_field(np.array([], dtype=np.float32))


def test_matches_pyts_gasf():
    pyts = pytest.importorskip("pyts.image")
    x = _sample_signal(48, seed=1)
    ours = gramian_angular_field(x, method="summation", sample_range=(-1.0, 1.0))
    ref = pyts.GramianAngularField(
        method="summation", sample_range=(-1.0, 1.0)
    ).transform(x[None, :])[0]
    assert np.allclose(ours, ref, atol=1e-5)


def test_matches_pyts_gadf():
    pyts = pytest.importorskip("pyts.image")
    x = _sample_signal(48, seed=2)
    ours = gramian_angular_field(x, method="difference", sample_range=(-1.0, 1.0))
    ref = pyts.GramianAngularField(
        method="difference", sample_range=(-1.0, 1.0)
    ).transform(x[None, :])[0]
    assert np.allclose(ours, ref, atol=1e-5)
