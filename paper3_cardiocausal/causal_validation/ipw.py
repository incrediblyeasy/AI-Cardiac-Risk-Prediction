"""Inverse-probability-weighting estimator for the average treatment effect.

IPW re-weights observed outcomes by the inverse of each unit's probability of
receiving the treatment it actually got, so the weighted treated/untreated groups
resemble the whole population — removing measured confounding *given* a correct
propensity model (roadmap §4.8). Pairs with the E-value (``evalues``) which then
bounds robustness to *unmeasured* confounding.

Pure numpy; takes a fitted propensity score (probability of treatment) as input so
the propensity model is decoupled from the estimator.
"""

from __future__ import annotations

import numpy as np


def stabilized_weights(treatment: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    """Stabilised IPW weights.

    ``w = p̄ / e(x)`` for treated, ``(1 - p̄) / (1 - e(x))`` for controls, where
    ``p̄`` is the marginal treatment prevalence. Stabilisation keeps weights near
    1 and reduces variance versus the raw ``1/e`` form.
    """
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    e = np.asarray(propensity, dtype=np.float64).ravel().clip(1e-6, 1 - 1e-6)
    p_bar = treatment.mean()
    return np.where(treatment == 1, p_bar / e, (1 - p_bar) / (1 - e))


def ipw_ate(
    treatment: np.ndarray, outcome: np.ndarray, propensity: np.ndarray
) -> float:
    """Stabilised-IPW estimate of ``E[Y | do(T=1)] - E[Y | do(T=0)]``.

    Computes the weighted mean outcome in each arm (Hájek estimator: weights
    normalised within arm) and returns their difference.
    """
    t = np.asarray(treatment, dtype=np.float64).ravel()
    y = np.asarray(outcome, dtype=np.float64).ravel()
    w = stabilized_weights(t, propensity)
    treated, control = t == 1, t == 0
    if not treated.any() or not control.any():
        return float("nan")
    mu1 = np.sum(w[treated] * y[treated]) / np.sum(w[treated])
    mu0 = np.sum(w[control] * y[control]) / np.sum(w[control])
    return float(mu1 - mu0)
