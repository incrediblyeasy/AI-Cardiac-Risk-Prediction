"""Causal-assumption diagnostics: covariate balance, positivity, negative controls.

These check the assumptions the IPW/SCM estimates rely on (roadmap §4.8) — treated
as core deliverables, not an afterthought:

* **Standardized mean difference (SMD)** — covariate balance between arms. |SMD|
  < 0.1 is the usual "balanced" rule of thumb; large SMD flags residual
  confounding the weighting failed to remove.
* **Positivity** — every unit must have propensity bounded away from 0 and 1, or
  the inverse weights blow up and the effect isn't identifiable. Reports the
  fraction of near-violations.
* **Negative-control outcome (NCO)** — an outcome known to have *no* causal link
  to the treatment should show ~zero estimated effect; a large NCO effect is
  evidence of residual confounding / bias.

Pure numpy so each is unit-testable on synthetic data.
"""

from __future__ import annotations

import numpy as np

from .ipw import ipw_ate


def standardized_mean_difference(covariate: np.ndarray, treatment: np.ndarray) -> float:
    """SMD of one covariate between treated and control arms.

    ``(mean_t - mean_c) / sqrt((var_t + var_c) / 2)`` — the pooled-SD standardised
    difference. Returns 0.0 when both arms have identical means.
    """
    x = np.asarray(covariate, dtype=np.float64).ravel()
    t = np.asarray(treatment, dtype=np.float64).ravel()
    xt, xc = x[t == 1], x[t == 0]
    if xt.size == 0 or xc.size == 0:
        return float("nan")
    diff = xt.mean() - xc.mean()
    pooled = np.sqrt((xt.var() + xc.var()) / 2.0)
    if pooled == 0:
        # Zero within-arm variance: identical means -> balanced (0); differing
        # means -> perfectly separated (undefined SMD, reported as inf).
        return 0.0 if diff == 0 else float("inf")
    return float(diff / pooled)


def positivity_check(propensity: np.ndarray, eps: float = 0.05) -> dict[str, float]:
    """Fraction of propensity scores within ``eps`` of 0 or 1 (near-violations).

    Returns min/max propensity and the violation fraction; a nonzero fraction
    warns that IPW estimates lean on poorly-supported regions.
    """
    e = np.asarray(propensity, dtype=np.float64).ravel()
    violations = np.mean((e < eps) | (e > 1 - eps))
    return {
        "min_propensity": float(e.min()),
        "max_propensity": float(e.max()),
        "violation_fraction": float(violations),
    }


def negative_control_test(
    treatment: np.ndarray,
    nco_outcome: np.ndarray,
    propensity: np.ndarray,
    tol: float = 0.1,
) -> dict[str, float | bool]:
    """IPW effect on a negative-control outcome; ``passes`` if it is ~zero.

    A negative-control outcome has no plausible causal link to the treatment, so a
    correctly-adjusted analysis should estimate an effect within ``tol`` of 0.
    A larger effect indicates residual confounding.
    """
    effect = ipw_ate(treatment, nco_outcome, propensity)
    return {"nco_effect": effect, "passes": bool(abs(effect) <= tol)}
