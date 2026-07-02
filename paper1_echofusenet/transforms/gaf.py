"""Gramian Angular Field (GAF) transform for a 1-D ECG beat.

A GAF encodes a signal's temporal correlations as a Gram-like matrix in a polar
angular basis. The signal is first min-max scaled to ``[-1, 1]`` and mapped to
angles ``phi = arccos(x_scaled)``; the image is then the pairwise trigonometric
sum/difference of those angles:

* **GASF** (``method="summation"``, default): ``G[i, j] = cos(phi_i + phi_j)``,
  which expands to ``x_i x_j - sqrt(1 - x_i^2) sqrt(1 - x_j^2)``. Symmetric, with
  the (rescaled) signal itself on the main diagonal. This is the variant used by
  most GAF-based ECG classifiers, so it is the default.
* **GADF** (``method="difference"``): ``G[i, j] = sin(phi_i - phi_j)``, which is
  anti-symmetric and captures directional temporal change.

Because the Day-2 beats are z-scored (unbounded) rather than in ``[-1, 1]``, the
scaling happens *inside* this transform — do not assume the input range.

Implemented in NumPy so the behaviour is explicit and unit-testable; see
`tests/test_gaf.py`. `pyts.image.GramianAngularField` is an available
cross-check.
"""

from __future__ import annotations

import numpy as np

_SUMMATION = {"summation", "gasf", "s"}
_DIFFERENCE = {"difference", "gadf", "d"}


def gramian_angular_field(
    signal: np.ndarray,
    method: str = "summation",
    sample_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Compute the Gramian Angular Field of a 1-D signal.

    Parameters
    ----------
    signal:
        1-D array (e.g. ``BeatSegment.signal``). Any range; it is min-max
        scaled to ``sample_range`` internally before the ``arccos`` step.
    method:
        ``"summation"`` (GASF, default) or ``"difference"`` (GADF). Aliases
        ``"gasf"``/``"gadf"`` (and ``"s"``/``"d"``) are accepted.
    sample_range:
        Target ``(low, high)`` for the internal min-max scaling. Must lie within
        ``[-1, 1]`` so that ``arccos`` is well defined. Default ``(-1, 1)``.

    Returns
    -------
    np.ndarray
        ``(L, L)`` float32 matrix in ``[-1, 1]``, where ``L = len(signal)``.
        GASF is symmetric; GADF is anti-symmetric.
    """
    low, high = sample_range
    if not (-1.0 <= low < high <= 1.0):
        raise ValueError(
            f"sample_range must satisfy -1 <= low < high <= 1, got {sample_range}"
        )

    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.shape[0] < 1:
        raise ValueError("signal must be non-empty")

    # Min-max scale to [low, high]; a constant signal maps to the midpoint.
    smin, smax = signal.min(), signal.max()
    if smax > smin:
        scaled = (signal - smin) / (smax - smin)  # [0, 1]
    else:
        scaled = np.full_like(signal, 0.5)
    scaled = scaled * (high - low) + low
    # Guard against tiny float excursions outside [-1, 1] before arccos.
    cos_phi = np.clip(scaled, -1.0, 1.0)
    sin_phi = np.sqrt(np.clip(1.0 - cos_phi**2, 0.0, 1.0))

    if method in _SUMMATION:
        # cos(phi_i + phi_j) = cos_i cos_j - sin_i sin_j
        gaf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
    elif method in _DIFFERENCE:
        # sin(phi_i - phi_j) = sin_i cos_j - cos_i sin_j
        gaf = np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)
    else:
        raise ValueError(
            f"method must be one of {sorted(_SUMMATION | _DIFFERENCE)}, got {method!r}"
        )

    return gaf.astype(np.float32)
