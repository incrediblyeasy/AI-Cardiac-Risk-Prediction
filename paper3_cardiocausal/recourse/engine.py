"""Recourse generation with a hard modifiable-variable constraint.

The one rule that makes a recommendation clinically valid: recourse may only move
**modifiable** variables. Recommending "be 10 years younger" is not recourse.
``ModifiabilityMask`` encodes which tabular features are actionable and projects
any proposed counterfactual back onto the feasible set by reverting changes to
immutable coordinates — so no downstream generator can accidentally suggest an
inadmissible intervention.

``generate_recourse`` is a gradient-based, Wachter-style search: starting from the
factual features it descends a loss that trades off *reaching a target risk* (BCE
of the risk model's output toward ``target_risk``) against *staying close to the
original* (L1 proximity), projecting through the ``ModifiabilityMask`` after every
step so immutable variables never move. It returns the recommended action and
whether the target risk was met. When paired with the SCM's ``do`` (``scm``) the
descent runs in the causal model's space so the recommendation is causally
consistent, not merely predictive.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch


class ModifiabilityMask:
    """Marks which tabular features are actionable and enforces it on edits.

    Parameters
    ----------
    feature_names:
        Ordered names of the tabular features (matches the tabular vector).
    modifiable:
        The subset of ``feature_names`` a recourse action may change. Anything not
        listed (e.g. ``age``, ``sex``) is immutable.
    """

    def __init__(self, feature_names: Sequence[str], modifiable: Sequence[str]) -> None:
        self.feature_names = list(feature_names)
        unknown = set(modifiable) - set(self.feature_names)
        if unknown:
            raise ValueError(f"modifiable features not in feature_names: {sorted(unknown)}")
        self.modifiable = set(modifiable)
        # 1.0 where a change is allowed, 0.0 where the feature is immutable.
        self.mask = torch.tensor(
            [1.0 if f in self.modifiable else 0.0 for f in self.feature_names]
        )

    def project(self, original: torch.Tensor, proposed: torch.Tensor) -> torch.Tensor:
        """Return ``proposed`` with immutable-feature changes reverted to ``original``.

        ``out = original + mask * (proposed - original)`` — modifiable coordinates
        take the proposed value; immutable coordinates are pinned to the original.
        """
        mask = self.mask.to(proposed)
        return original + mask * (proposed - original)

    def is_valid_action(
        self, original: torch.Tensor, proposed: torch.Tensor, tol: float = 1e-6
    ) -> bool:
        """True iff ``proposed`` changes no immutable feature beyond ``tol``."""
        mask = self.mask.to(proposed)
        immutable_delta = ((1.0 - mask) * (proposed - original)).abs()
        return bool((immutable_delta <= tol).all().item())


def generate_recourse(
    x: torch.Tensor,
    risk_fn: Callable[[torch.Tensor], torch.Tensor],
    mask: ModifiabilityMask,
    target_risk: float = 0.5,
    proximity_weight: float = 0.1,
    lr: float = 0.1,
    steps: int = 200,
) -> dict[str, torch.Tensor]:
    """Gradient-based recourse: minimal modifiable-only edit toward ``target_risk``.

    Parameters
    ----------
    x:
        Factual feature batch ``(B, F)`` (order matches ``mask.feature_names``).
    risk_fn:
        Differentiable map from features ``(B, F)`` to a risk **probability**
        ``(B,)`` in [0, 1] (e.g. a trained ``RiskHead`` composed with fusion).
    mask:
        Modifiability constraint; immutable coordinates are pinned to ``x``.
    target_risk:
        Desired post-recourse risk (the search pushes risk toward/below this).
    proximity_weight:
        Weight of the L1 proximity penalty (larger -> smaller, sparser edits).
    lr, steps:
        Optimisation schedule for the search.

    Returns a dict with ``recourse`` (the recommended feature vector, immutable
    coords unchanged), ``delta`` (the action ``recourse - x``), ``risk`` (achieved
    risk), and ``success`` (bool per sample: achieved risk <= target).
    """
    x = x.detach()
    target = torch.full((x.shape[0],), float(target_risk))
    # Optimise a free variable, then project so only modifiable coords move.
    z = x.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        proposed = mask.project(x, z)
        risk = risk_fn(proposed)
        # Push risk toward the target; L1 keeps the edit small and sparse.
        loss = torch.nn.functional.binary_cross_entropy(
            risk.clamp(1e-6, 1 - 1e-6), target
        ) + proximity_weight * (proposed - x).abs().sum(dim=1).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        recourse = mask.project(x, z.detach())
        achieved = risk_fn(recourse)
    return {
        "recourse": recourse,
        "delta": recourse - x,
        "risk": achieved,
        "success": achieved <= target + 1e-6,
    }
