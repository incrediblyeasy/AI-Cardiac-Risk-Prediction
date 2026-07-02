"""Markov Transition Field (MTF) transform for a 1-D ECG beat.

An MTF encodes the *dynamics* of a signal: how the amplitude moves between
quantized states over time. The construction is:

1. Quantize the signal into ``n_bins`` amplitude bins (quantile bins by default,
   so each bin holds roughly equal mass).
2. Build the ``n_bins x n_bins`` Markov transition matrix ``W`` by counting
   transitions between the bins of successive samples, then row-normalizing so
   each row is a probability distribution.
3. Spread ``W`` back over time: ``M[i, j] = W[q_i, q_j]``, the probability of
   stepping from the bin at time ``i`` to the bin at time ``j``.

Unlike RP and GAF, the MTF is **not symmetric** (``P(a->b) != P(b->a)``) and its
diagonal is not fixed — it captures directional temporal transition structure,
which is exactly why it is a genuinely distinct third modality (see the
distinctness guard in `tests/test_distinctness.py`).

Because the Day-2 beats are z-scored, binning is data-driven (quantile/uniform on
the signal itself), so no particular input range is assumed.

Implemented in NumPy to mirror `pyts.image.MarkovTransitionField`
(``image_size=1.0``, i.e. no PAA aggregation); see `tests/test_mtf.py`.
"""

from __future__ import annotations

import numpy as np


def _digitize(signal: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Quantize a 1-D signal into integer bins ``0 .. n_bins-1``.

    Mirrors pyts' ``_digitize``: interior bin edges via quantile/uniform/normal
    strategy, then ``np.digitize(..., right=True)``.
    """
    if strategy == "uniform":
        edges = np.linspace(signal.min(), signal.max(), n_bins + 1)[1:-1]
    elif strategy == "quantile":
        edges = np.percentile(signal, np.linspace(0, 100, n_bins + 1)[1:-1])
    elif strategy == "normal":
        from scipy.stats import norm

        edges = norm.ppf(np.linspace(0, 1, n_bins + 1)[1:-1])
    else:
        raise ValueError(
            f"strategy must be 'quantile', 'uniform' or 'normal', got {strategy!r}"
        )
    return np.digitize(signal, edges, right=True).astype(np.int64)


def markov_transition_field(
    signal: np.ndarray,
    n_bins: int = 8,
    strategy: str = "quantile",
) -> np.ndarray:
    """Compute the Markov Transition Field of a 1-D signal.

    Parameters
    ----------
    signal:
        1-D array (e.g. ``BeatSegment.signal``). Any range; binning is
        data-driven.
    n_bins:
        Number of amplitude quantization bins (default 8).
    strategy:
        Binning strategy: ``"quantile"`` (default, equal-mass bins),
        ``"uniform"`` (equal-width) or ``"normal"`` (Gaussian quantiles).

    Returns
    -------
    np.ndarray
        ``(L, L)`` float32 matrix in ``[0, 1]`` of transition probabilities,
        where ``L = len(signal)``. Generally **not symmetric**.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = signal.shape[0]
    if n < 2:
        raise ValueError("signal must have at least 2 samples")

    binned = _digitize(signal, n_bins, strategy)

    # Markov transition matrix: count bin_t -> bin_{t+1}, then row-normalize.
    W = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(W, (binned[:-1], binned[1:]), 1.0)
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1.0  # rows never visited stay all-zero
    W /= row_sums[:, None]

    # Spread the transition probabilities back over the time axes.
    mtf = W[binned[:, None], binned[None, :]]
    return mtf.astype(np.float32)
