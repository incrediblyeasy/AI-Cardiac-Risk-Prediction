r"""Exact Shapley modality attribution — associational baseline.

With only three modality "players" (RP / GAF / MTF), the Shapley value is exact
and cheap: enumerate all ``2^3`` coalitions rather than sampling (as Kernel SHAP
would). The value of a coalition ``S`` is the frozen decision head's predicted
probability for the target class when the modalities *outside* ``S`` are set to
their interventional baseline (reusing ``attribution.intervene``) — i.e. only the
modalities in ``S`` carry their factual information.

For player ``m`` the Shapley value is the usual weighted average of marginal
contributions over coalitions that exclude ``m``::

    φ_m = Σ_{S ⊆ M\{m}} |S|!(|M|-|S|-1)! / |M|!  · [ v(S ∪ {m}) - v(S) ]

This is the associational counterpart to the interventional ITE in
``attribution`` — comparing ``φ`` against the ITE table is exactly the "causal vs.
associational divergence" study (roadmap §3.5). It needs no ``shap`` dependency.
"""

from __future__ import annotations

import math
from itertools import combinations

import torch

from ..attribution.ite import intervene


def _coalition_value(
    representation: torch.Tensor,
    encoder,
    coalition: frozenset[str],
    target: torch.Tensor,
    baseline: str | torch.Tensor,
) -> torch.Tensor:
    """v(S): P(target) with modalities NOT in ``coalition`` set to baseline.

    Returns a per-sample probability vector ``(B,)`` for the target class.
    """
    slices = encoder.modality_slices()
    rep = representation
    for m in slices:
        if m not in coalition:
            rep = intervene(rep, slices, m, baseline)
    probs = torch.softmax(encoder.decision(rep), dim=1)
    return probs.gather(1, target.view(-1, 1)).squeeze(1)


@torch.no_grad()
def shap_modality_values(
    representation: torch.Tensor,
    encoder,
    target: torch.Tensor,
    baseline: str | torch.Tensor = "mean",
) -> dict[str, torch.Tensor]:
    """Exact per-modality Shapley values for the target class.

    Parameters
    ----------
    representation:
        Fused representation batch ``(B, D)`` from :meth:`FrozenEncoder.encode`.
    encoder:
        Provides ``decision`` + ``modality_slices`` (a :class:`FrozenEncoder`).
    target:
        Target class indices ``(B,)`` (or one-hot, which is argmax-ed).
    baseline:
        Interventional reference for absent modalities ("mean" | "zero" | tensor).

    Returns ``{modality: shapley_value (B,)}`` — the marginal-value attribution
    per sample. Summed over modalities, these equal ``v(full) - v(empty)`` (the
    Shapley efficiency property), which the tests check.
    """
    if target.dim() > 1:
        target = target.argmax(dim=1)
    modalities = list(encoder.modality_slices())
    n = len(modalities)

    # Cache every coalition value once (2^n of them) — each is reused across players.
    cache: dict[frozenset[str], torch.Tensor] = {}
    for k in range(n + 1):
        for combo in combinations(modalities, k):
            s = frozenset(combo)
            cache[s] = _coalition_value(representation, encoder, s, target, baseline)

    phi: dict[str, torch.Tensor] = {}
    for m in modalities:
        others = [x for x in modalities if x != m]
        acc = torch.zeros_like(cache[frozenset()])
        for k in range(len(others) + 1):
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for combo in combinations(others, k):
                s = frozenset(combo)
                acc = acc + weight * (cache[s | {m}] - cache[s])
        phi[m] = acc
    return phi
