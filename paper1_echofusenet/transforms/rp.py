"""Recurrence Plot (RP) transform for a 1-D ECG beat.

A recurrence plot encodes how often the trajectory of a signal revisits similar
states. For a signal ``x`` we form (optionally time-delay-embedded) state
vectors and compute the pairwise distance matrix

    R[i, j] = || v_i - v_j ||_2

which is symmetric with a zero diagonal. Two output styles are supported:

* **unthresholded** (default): the distance matrix itself, min-max scaled to
  ``[0, 1]``. This grayscale form keeps texture that a CNN can exploit and is
  the variant used by most RP-based ECG classifiers.
* **thresholded**: a binary matrix ``R[i, j] = 1`` where the distance is below
  a threshold ``epsilon`` (the classic black-and-white recurrence plot).

Time-delay embedding (Takens) is available via ``dimension`` / ``time_delay``;
the default ``dimension=1`` uses the raw samples as states, giving an
``L x L`` image for a length-``L`` beat (256x256 for the Day-2 default window).

Implemented in NumPy so the behaviour is explicit and unit-testable; see
`tests/test_rp.py`. `pyts.image.RecurrencePlot` is an available cross-check.
"""

from __future__ import annotations

import numpy as np


def _embed(signal: np.ndarray, dimension: int, time_delay: int) -> np.ndarray:
    """Takens time-delay embedding.

    Returns an ``(n_states, dimension)`` array where
    ``v_i = [x_i, x_{i+τ}, ..., x_{i+(dimension-1)τ}]``. For ``dimension == 1``
    this is just the samples as a column vector.
    """
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    if time_delay < 1:
        raise ValueError("time_delay must be >= 1")

    n = signal.shape[0]
    n_states = n - (dimension - 1) * time_delay
    if n_states < 1:
        raise ValueError(
            f"signal too short (len={n}) for dimension={dimension}, "
            f"time_delay={time_delay}"
        )
    idx = np.arange(n_states)[:, None] + np.arange(dimension)[None, :] * time_delay
    return signal[idx]


def recurrence_plot(
    signal: np.ndarray,
    dimension: int = 1,
    time_delay: int = 1,
    threshold: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute the recurrence plot of a 1-D signal.

    Parameters
    ----------
    signal:
        1-D array (e.g. ``BeatSegment.signal``).
    dimension, time_delay:
        Takens embedding parameters. ``dimension=1`` (default) uses raw samples.
    threshold:
        If ``None`` (default), return the (optionally normalized) distance
        matrix. If a float, return the binary matrix ``distance <= threshold``.
        When ``normalize`` is True the threshold is applied in the ``[0, 1]``
        scaled space.
    normalize:
        Min-max scale the distance matrix to ``[0, 1]`` before thresholding /
        returning. Ignored effect on an all-constant signal (returns zeros).

    Returns
    -------
    np.ndarray
        Symmetric ``(L, L)`` float32 matrix, where
        ``L = len(signal) - (dimension - 1) * time_delay``.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    states = _embed(signal, dimension, time_delay)

    # Pairwise Euclidean distances between state vectors.
    diff = states[:, None, :] - states[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))

    if normalize:
        dmax = dist.max()
        if dmax > 0:
            dist = dist / dmax  # min is 0 (diagonal), so this is min-max scaling

    if threshold is not None:
        return (dist <= threshold).astype(np.float32)

    return dist.astype(np.float32)
