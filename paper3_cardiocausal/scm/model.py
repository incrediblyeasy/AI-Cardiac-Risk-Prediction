r"""Structural causal model over the fused latent + calibrated risk head.

Two components:

``RiskHead`` (calibrated risk predictor)
    Fused latent -> single risk logit; ``P(event | latent)`` for the evaluation
    suite (AUROC/AUPRC/Brier/calibration). Small and data-independent.

``NeuralSCM`` (additive-noise structural causal model)
    A deep SCM over named scalar variables given a DAG (roadmap §4.3/§4.5). Each
    node follows an **additive-noise mechanism** ``X_j = f_j(pa(X_j)) + U_j`` with
    ``f_j`` a small MLP (a learnable constant for root nodes). Additive noise makes
    the model invertible, so Pearl's three-step counterfactual is exact:

        abduction  — infer the exogenous noise ``U`` from a factual observation
                     (``U_j = x_j - f_j(pa(x_j))``),
        action     — apply ``do(·)`` to fix intervened nodes,
        prediction — regenerate downstream nodes with the *same* ``U``.

    ``do`` supports the recourse engine (``recourse``); ``counterfactual`` answers
    "for *this* patient, what would the outcome have been under a different
    treatment?" — the per-individual query the causal-validation stack checks.

``fit_scm`` fits every mechanism by regressing each node on its parents (MSE),
which is the maximum-likelihood fit under the additive-noise assumption.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn


class RiskHead(nn.Module):
    """Calibrated binary-risk head over the fused latent.

    ``forward`` returns a **logit**; apply ``torch.sigmoid`` for a probability.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Risk logit ``(B,)`` for each latent vector."""
        return self.net(latent).squeeze(-1)

    @torch.no_grad()
    def risk(self, latent: torch.Tensor) -> torch.Tensor:
        """Calibrated risk probability ``(B,)`` in [0, 1]."""
        return torch.sigmoid(self.forward(latent))


def _topological_order(
    nodes: Sequence[str], parents: Mapping[str, Sequence[str]]
) -> list[str]:
    """Kahn's algorithm; raises on an unknown parent or a cycle."""
    parents = {n: list(parents.get(n, [])) for n in nodes}
    for n, ps in parents.items():
        unknown = set(ps) - set(nodes)
        if unknown:
            raise ValueError(f"node {n!r} has unknown parents {sorted(unknown)}")
    order: list[str] = []
    remaining = dict(parents)
    while remaining:
        ready = [n for n, ps in remaining.items() if not set(ps) & set(remaining)]
        if not ready:
            raise ValueError("DAG contains a cycle")
        for n in ready:
            order.append(n)
            del remaining[n]
    return order


class NeuralSCM(nn.Module):
    """Additive-noise structural causal model over named scalar nodes.

    Parameters
    ----------
    nodes:
        All variable names.
    parents:
        Mapping ``node -> list of parent nodes`` (the DAG edges). Omitted nodes
        are treated as roots (no parents).
    hidden_dim:
        Width of each non-root mechanism MLP.
    """

    def __init__(
        self,
        nodes: Sequence[str],
        parents: Mapping[str, Sequence[str]] | None = None,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        parents = {n: list((parents or {}).get(n, [])) for n in nodes}
        self.nodes = list(nodes)
        self.parents = parents
        self.order = _topological_order(nodes, parents)

        self.mechanisms = nn.ModuleDict()
        self.roots = nn.ParameterDict()
        for n in self.nodes:
            k = len(parents[n])
            if k == 0:
                self.roots[n] = nn.Parameter(torch.zeros(1))
            else:
                self.mechanisms[n] = nn.Sequential(
                    nn.Linear(k, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 1),
                )

    def _mechanism(self, node: str, values: Mapping[str, torch.Tensor], batch: int) -> torch.Tensor:
        """f_j(pa(X_j)): the deterministic part of node ``node``'s value."""
        ps = self.parents[node]
        if not ps:
            return self.roots[node].expand(batch)
        x = torch.stack([values[p] for p in ps], dim=1)  # (B, k)
        return self.mechanisms[node](x).squeeze(1)

    def _batch_size(self, some: Mapping[str, torch.Tensor]) -> int:
        return next(iter(some.values())).shape[0]

    def generate(
        self,
        noise: Mapping[str, torch.Tensor],
        interventions: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Structural forward pass: node values from exogenous ``noise``.

        ``interventions`` (``do``) override a node's value, cutting it off from its
        mechanism and parents. Evaluated in topological order.
        """
        interventions = dict(interventions or {})
        batch = self._batch_size(noise)
        values: dict[str, torch.Tensor] = {}
        for n in self.order:
            if n in interventions:
                values[n] = interventions[n]
            else:
                values[n] = self._mechanism(n, values, batch) + noise[n]
        return values

    def abduct(self, observations: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Infer exogenous noise ``U_j = x_j - f_j(pa(x_j))`` from a full observation."""
        missing = set(self.nodes) - set(observations)
        if missing:
            raise ValueError(f"abduction needs all nodes observed; missing {sorted(missing)}")
        batch = self._batch_size(observations)
        return {
            n: observations[n] - self._mechanism(n, observations, batch) for n in self.nodes
        }

    def do(
        self, interventions: Mapping[str, torch.Tensor], n_samples: int = 1
    ) -> dict[str, torch.Tensor]:
        """Sample the interventional distribution ``P(· | do(interventions))``.

        Roots draw standard-normal noise; deterministic given seeded RNG upstream.
        """
        any_iv = next(iter(interventions.values()))
        batch = any_iv.shape[0] if any_iv.dim() > 0 else n_samples
        noise = {n: torch.randn(batch) for n in self.nodes}
        return self.generate(noise, interventions)

    def counterfactual(
        self,
        observations: Mapping[str, torch.Tensor],
        interventions: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Pearl's abduction-action-prediction counterfactual for observed units.

        With no interventions this returns the observations exactly (the SCM is
        invertible under additive noise) — the identity check the tests use.
        """
        noise = self.abduct(observations)
        return self.generate(noise, interventions)


def fit_scm(
    scm: NeuralSCM,
    data: Mapping[str, torch.Tensor],
    epochs: int = 200,
    lr: float = 1e-2,
) -> dict[str, Any]:
    """Fit each mechanism by regressing every node on its parents (MSE).

    ``data`` maps every node name to an observed ``(N,)`` tensor. Returns a small
    history dict (final per-node and total loss). This is the maximum-likelihood
    fit under the additive-noise assumption; afterwards ``scm.counterfactual`` /
    ``scm.do`` give calibrated interventional answers.
    """
    missing = set(scm.nodes) - set(data)
    if missing:
        raise ValueError(f"data missing nodes {sorted(missing)}")
    opt = torch.optim.Adam(scm.parameters(), lr=lr)
    batch = scm._batch_size(data)
    last: dict[str, float] = {}
    for _ in range(epochs):
        opt.zero_grad()
        total = torch.zeros(())
        per_node: dict[str, float] = {}
        for n in scm.nodes:
            pred = scm._mechanism(n, data, batch)
            loss_n = torch.mean((pred - data[n]) ** 2)
            total = total + loss_n
            per_node[n] = float(loss_n.item())
        total.backward()
        opt.step()
        last = {"per_node": per_node, "total": float(total.item())}
    return last
