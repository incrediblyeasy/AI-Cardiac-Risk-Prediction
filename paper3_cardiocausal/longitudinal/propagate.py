"""Bounded latent-state propagation over serial ECG records.

Scope discipline (roadmap §4.7)
-------------------------------
This layer models short-horizon *retrospective* risk trajectories by propagating
the fused latent state forward a small, **explicitly bounded** number of steps. The
proposal deliberately avoids an unverifiable "digital twin" claim, so the horizon
is a hard cap enforced at call time — ``propagate`` raises if asked to roll out
past ``max_horizon``. This keeps scope creep from silently turning a bounded probe
into an open-ended simulator.

Model
-----
A single gated linear transition on the latent (``h_{t+1} = h_t + g ⊙ tanh(W h_t)``,
a residual step so states don't blow up or vanish over the short horizon). It is
intentionally minimal and data-independent so it is testable now; fitting it needs
serial-record cohorts from MIMIC-IV (``datasets``).
"""

from __future__ import annotations

import torch
from torch import nn


class LatentPropagator(nn.Module):
    """Roll the fused latent forward a bounded number of steps.

    Parameters
    ----------
    latent_dim:
        Width of the fused latent being propagated.
    max_horizon:
        Hard cap on rollout length. ``propagate`` refuses to exceed it — the
        bounded-scope guarantee.
    """

    def __init__(self, latent_dim: int, max_horizon: int = 4) -> None:
        super().__init__()
        if max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")
        self.latent_dim = latent_dim
        self.max_horizon = max_horizon
        self.transition = nn.Linear(latent_dim, latent_dim)
        self.gate = nn.Linear(latent_dim, latent_dim)

    def step(self, h: torch.Tensor) -> torch.Tensor:
        """One residual, gated transition step."""
        g = torch.sigmoid(self.gate(h))
        return h + g * torch.tanh(self.transition(h))

    def propagate(self, h0: torch.Tensor, horizon: int) -> torch.Tensor:
        """Return the trajectory ``(B, horizon + 1, latent_dim)`` including ``h0``.

        Raises ``ValueError`` if ``horizon`` exceeds ``max_horizon`` — the bounded
        in-silico guarantee (no open-ended "digital twin" rollout).
        """
        if horizon < 0:
            raise ValueError("horizon must be >= 0")
        if horizon > self.max_horizon:
            raise ValueError(
                f"horizon {horizon} exceeds max_horizon {self.max_horizon}; this "
                "layer is deliberately bounded (no open-ended simulation)"
            )
        states = [h0]
        h = h0
        for _ in range(horizon):
            h = self.step(h)
            states.append(h)
        return torch.stack(states, dim=1)
