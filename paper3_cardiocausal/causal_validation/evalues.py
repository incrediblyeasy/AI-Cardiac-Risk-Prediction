"""E-values for unmeasured confounding (VanderWeele & Ding, 2017).

The E-value is the minimum strength of association (on the risk-ratio scale) that
an unmeasured confounder would need with *both* the treatment and the outcome to
fully explain away an observed effect. Larger E-value -> more robust causal claim.
It is a pure closed-form function of the reported effect, so it is implemented and
tested here directly (no cohort needed).

Formulas
--------
For a risk ratio ``RR`` (with ``RR < 1`` first inverted to ``1/RR`` so the result
is on a >= 1 scale)::

    E = RR + sqrt(RR * (RR - 1))

For a confidence interval, the E-value is computed for the interval limit
**closest to the null** (RR = 1); if the interval already crosses the null, its
E-value is 1 (the effect is not distinguishable from confounding).
"""

from __future__ import annotations

import math


def e_value(rr: float) -> float:
    """E-value for a risk-ratio point estimate.

    ``rr`` may be above or below 1; below-1 ratios are inverted first so the
    E-value is reported on the conventional >= 1 scale. ``rr = 1`` (null) -> 1.0.
    """
    if rr <= 0:
        raise ValueError("risk ratio must be positive")
    if rr < 1:
        rr = 1.0 / rr
    return rr + math.sqrt(rr * (rr - 1.0))


def e_value_ci(estimate: float, lo: float, hi: float) -> float:
    """E-value for the confidence limit closest to the null.

    Parameters
    ----------
    estimate, lo, hi:
        Risk-ratio point estimate and its lower/upper confidence limits.

    Returns 1.0 when the interval crosses the null (no confounding strength is
    required to explain a non-significant effect).
    """
    if lo > hi:
        raise ValueError("lo must be <= hi")
    if estimate >= 1:
        return 1.0 if lo <= 1 else e_value(lo)
    return 1.0 if hi >= 1 else e_value(hi)
