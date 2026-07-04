"""Counterfactual-quality metrics — validity, proximity, sparsity.

These are built *before* claiming any counterfactual results (roadmap §3.3) so a
generated edit is judged on the same three axes the literature uses (Wachter et
al.; DiCE):

* **validity** — does the edit actually achieve the target class under the frozen
  decision head? (the necessary condition; a counterfactual that doesn't flip the
  class is not a counterfactual). Higher is better, in [0, 1].
* **proximity** — how *small* is the edit, as mean L1 distance in representation
  space between original and counterfactual. Lower is better.
* **sparsity** — how *few* representation dimensions moved (beyond a tolerance),
  as a fraction in [0, 1]. Lower is better — sparse edits are more interpretable.

All functions take ``torch.Tensor`` batches and return plain floats so they drop
straight into a metrics dict / results table. ``validity`` takes the decision
function (typically :meth:`FrozenEncoder.decision`) so the metric is measured
against the *same* fixed classifier everything else in Paper 2 uses.
"""

from __future__ import annotations

from typing import Callable

import torch


def validity(
    x_cf: torch.Tensor,
    target: torch.Tensor,
    decision_fn: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    """Fraction of counterfactuals whose predicted class equals ``target``.

    ``decision_fn`` maps a representation batch ``(B, D)`` to logits ``(B, C)``.
    ``target`` is class indices ``(B,)`` (or a one-hot, which is argmax-ed).
    """
    if target.dim() > 1:
        target = target.argmax(dim=1)
    with torch.no_grad():
        pred = decision_fn(x_cf).argmax(dim=1)
    return float((pred == target).float().mean().item())


def proximity(x: torch.Tensor, x_cf: torch.Tensor) -> float:
    """Mean L1 distance between originals and counterfactuals (lower = closer)."""
    return float((x - x_cf).abs().sum(dim=1).mean().item())


def sparsity(x: torch.Tensor, x_cf: torch.Tensor, tol: float = 1e-3) -> float:
    """Fraction of representation dimensions changed by more than ``tol``.

    Lower is sparser (fewer coordinates touched), which is the more interpretable
    edit. Averaged over the batch and the ``D`` representation dimensions.
    """
    changed = (x - x_cf).abs() > tol
    return float(changed.float().mean().item())


def counterfactual_report(
    x: torch.Tensor,
    x_cf: torch.Tensor,
    target: torch.Tensor,
    decision_fn: Callable[[torch.Tensor], torch.Tensor],
    tol: float = 1e-3,
) -> dict[str, float]:
    """Bundle the three metrics into one dict for logging / results tables."""
    return {
        "validity": validity(x_cf, target, decision_fn),
        "proximity": proximity(x, x_cf),
        "sparsity": sparsity(x, x_cf, tol=tol),
    }
