"""§6 robustness: noise injectors achieve their spec + the sweep harness."""

import numpy as np

from paper1_echofusenet.robustness import (
    add_baseline_wander,
    add_gaussian_noise,
    add_powerline_interference,
    apply_noise,
    evaluate_under_noise,
    measure_snr_db,
)
from paper1_echofusenet.robustness.noise import DEFAULT_FS


def _sine(L=512, freq=5.0, fs=DEFAULT_FS):
    t = np.arange(L) / fs
    return np.sin(2.0 * np.pi * freq * t)


def test_gaussian_hits_target_snr():
    x = _sine()
    for snr in (20.0, 10.0, 5.0):
        noisy = add_gaussian_noise(x, snr, rng=0)
        measured = measure_snr_db(x, noisy)
        # Empirical SNR over a finite window is close to, not exactly, the target.
        assert abs(measured - snr) < 1.0


def test_gaussian_preserves_shape_and_is_seed_reproducible():
    x = _sine()
    a = add_gaussian_noise(x, 10.0, rng=42)
    b = add_gaussian_noise(x, 10.0, rng=42)
    c = add_gaussian_noise(x, 10.0, rng=7)
    assert a.shape == x.shape
    assert np.allclose(a, b)          # same seed -> identical
    assert not np.allclose(a, c)      # different seed -> different draw


def test_gaussian_vectorises_over_a_batch():
    batch = np.stack([_sine(freq=f) for f in (3.0, 5.0, 8.0)])  # (3, L)
    noisy = add_gaussian_noise(batch, 15.0, rng=1)
    assert noisy.shape == batch.shape
    # Each row independently near the target SNR.
    for i in range(batch.shape[0]):
        assert abs(measure_snr_db(batch[i], noisy[i]) - 15.0) < 1.5


def test_baseline_wander_energy_is_low_frequency():
    x = _sine(freq=8.0)
    wander_only = add_baseline_wander(np.zeros_like(x), amplitude=1.0, freq=0.3)
    spec = np.abs(np.fft.rfft(wander_only))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / DEFAULT_FS)
    dominant = freqs[int(np.argmax(spec))]
    assert dominant < 1.0             # wander concentrated below 1 Hz


def test_powerline_energy_sits_at_mains_frequency():
    x = np.zeros(1024)
    pl = add_powerline_interference(x + 1e-9, amplitude=1.0, freq=50.0)
    # add a nonzero std baseline so amplitude scaling is nonzero
    base = _sine(L=1024, freq=6.0)
    pl = add_powerline_interference(base, amplitude=0.5, freq=50.0) - base
    spec = np.abs(np.fft.rfft(pl))
    freqs = np.fft.rfftfreq(pl.size, d=1.0 / DEFAULT_FS)
    dominant = freqs[int(np.argmax(spec))]
    assert abs(dominant - 50.0) < 1.0


def test_apply_noise_dispatch_and_unknown_kind():
    x = _sine()
    assert apply_noise(x, "gaussian", 10.0, rng=0).shape == x.shape
    assert apply_noise(x, "baseline_wander", 0.5, rng=0).shape == x.shape
    assert apply_noise(x, "powerline", 0.5, rng=0).shape == x.shape
    try:
        apply_noise(x, "salt_and_pepper", 1.0)
    except ValueError as e:
        assert "unknown noise kind" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown kind")


def test_evaluate_under_noise_degrades_a_snr_sensitive_predictor():
    # A predictor that classifies by dominant frequency: clean beats classify
    # correctly; low-SNR Gaussian noise corrupts the spectrum -> the dominant bin
    # moves -> accuracy must drop. High SNR leaves it near-perfect.
    L = 256
    class0 = [_sine(L=L, freq=5.0) for _ in range(20)]     # low-frequency class
    class1 = [_sine(L=L, freq=20.0) for _ in range(20)]    # high-frequency class
    signals = np.stack(class0 + class1)
    labels = np.array([0] * 20 + [1] * 20)

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(batch, axis=-1))
        freqs = np.fft.rfftfreq(batch.shape[-1], d=1.0 / DEFAULT_FS)
        dominant = freqs[np.argmax(spec, axis=-1)]
        return (dominant >= 12.5).astype(int)  # split between 5 and 20 Hz

    result = evaluate_under_noise(
        predict_fn, signals, labels,
        kind="gaussian", levels=[20.0, -10.0], n_classes=2, n_runs=3, seed=0,
    )
    assert result.clean_accuracy == 1.0
    acc_mild = result.levels[0].accuracy.point   # 20 dB — barely perturbed
    acc_heavy = result.levels[1].accuracy.point  # -10 dB — spectrum swamped
    assert acc_mild >= acc_heavy
    assert acc_heavy < 1.0
    # CIs are populated across runs.
    for lvl in result.levels:
        assert lvl.n_runs == 3
        assert lvl.accuracy.low <= lvl.accuracy.point <= lvl.accuracy.high


def test_evaluate_under_noise_summary_dict_shape():
    signals = np.random.default_rng(1).standard_normal((12, 64))
    labels = np.zeros(12, dtype=int)
    result = evaluate_under_noise(
        lambda b: np.zeros(b.shape[0], dtype=int),
        signals, labels, kind="powerline", levels=[0.1, 0.5], n_classes=5, n_runs=1,
    )
    d = result.summary_dict()
    assert d["kind"] == "powerline"
    assert len(d["levels"]) == 2
    assert {"accuracy", "macro_f1"} <= d["levels"][0].keys()
