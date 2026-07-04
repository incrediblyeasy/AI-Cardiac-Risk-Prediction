"""Per-modality individual treatment effects via representation intervention.

Mechanism
---------
The fused representation is ``concat([emb_rp, emb_gaf, emb_mtf])`` (active
modalities only). To estimate modality *m*'s causal effect on the decision we
apply the do-operator to its block: replace ``emb_m`` with a **baseline** value
(the interventional reference) while holding the other blocks fixed, then push
both the factual and intervened representations through the *same frozen decision
head* and compare the predicted probabilities.

For sample *i*, class *k*, modality *m*::

    ITE_{i,m,k} = P(y=k | do(emb_m := factual))  -  P(y=k | do(emb_m := baseline))
                = softmax(decision(x_i))[k]  -  softmax(decision(x_i with emb_m := baseline))[k]

Averaging ``ITE_{·,m,k}`` over the cohort gives modality *m*'s average causal
effect on class *k* — the number that goes in the attribution table.

Baseline choice
---------------
The interventional reference for a modality block. Options:

* ``"mean"`` (default) — the batch mean of that block, i.e. "an average, class-
  uninformative version of this modality." This is the marginal-expectation
  baseline; it isolates the *information above average* that the modality carries.
* ``"zero"`` — ablate the block to zeros (encoder saw a blank image's embedding
  proxy). Simpler but off the data manifold; kept for sensitivity checks.
* an explicit tensor of shape ``(D_block,)`` or ``(B, D_block)``.

Everything runs against the frozen decision head, so no gradients or retraining
are involved — attribution is pure forward-mode intervention.
"""

from __future__ import annotations

from typing import Protocol

import torch


class _Decider(Protocol):
    """Minimal interface used here: a frozen encoder exposing decision + slices."""

    def decision(self, representation: torch.Tensor) -> torch.Tensor: ...
    def modality_slices(self) -> dict[str, slice]: ...


def _baseline_block(
    block: torch.Tensor, baseline: str | torch.Tensor
) -> torch.Tensor:
    """Resolve the interventional reference for one modality block ``(B, d)``."""
    if isinstance(baseline, torch.Tensor):
        ref = baseline.to(block)
        if ref.dim() == 1:
            ref = ref.unsqueeze(0).expand_as(block)
        return ref
    if baseline == "mean":
        return block.mean(dim=0, keepdim=True).expand_as(block)
    if baseline == "zero":
        return torch.zeros_like(block)
    raise ValueError(f"unknown baseline {baseline!r}; use 'mean', 'zero', or a tensor")


def intervene(
    representation: torch.Tensor,
    slices: dict[str, slice],
    modality: str,
    baseline: str | torch.Tensor = "mean",
) -> torch.Tensor:
    """Return a copy of ``representation`` with ``modality``'s block set to baseline.

    ``do(emb_modality := baseline)`` — the other modality blocks are untouched.
    """
    if modality not in slices:
        raise KeyError(f"modality {modality!r} not in {list(slices)}")
    out = representation.clone()
    sl = slices[modality]
    out[:, sl] = _baseline_block(representation[:, sl], baseline)
    return out


@torch.no_grad()
def modality_ite(
    representation: torch.Tensor,
    encoder: _Decider,
    modality: str,
    baseline: str | torch.Tensor = "mean",
) -> torch.Tensor:
    """Per-sample, per-class ITE of ablating ``modality`` — shape ``(B, n_classes)``.

    Positive ``ITE[i, k]`` means modality *m* was *pushing sample i toward class k*
    (removing it drops that class's probability).
    """
    slices = encoder.modality_slices()
    factual = torch.softmax(encoder.decision(representation), dim=1)
    ablated = torch.softmax(
        encoder.decision(intervene(representation, slices, modality, baseline)), dim=1
    )
    return factual - ablated


def attribution_table(
    representation: torch.Tensor,
    encoder: _Decider,
    baseline: str | torch.Tensor = "mean",
) -> dict[str, torch.Tensor]:
    """Mean per-class causal effect for every active modality.

    Returns ``{modality: mean_ITE_over_batch (n_classes,)}``. This is the
    apples-to-apples table compared against Grad-CAM/SHAP attributions in
    ``baselines`` to surface where causal and associational explanations diverge.
    """
    table: dict[str, torch.Tensor] = {}
    for m in encoder.modality_slices():
        table[m] = modality_ite(representation, encoder, m, baseline).mean(dim=0)
    return table
