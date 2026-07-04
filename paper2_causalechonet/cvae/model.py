"""Conditional VAE over the frozen representation space (~0.2M params).

Architecture
------------
A deliberately small MLP CVAE — the encoder it sits on already did the heavy
lifting, so this only has to model structure in a ``representation_dim``-D vector
(≈768 for the full 3-branch EchoFuseNet). Conditioning is by concatenating a
one-hot class vector to both the recognition and generative networks::

    encode:  [x ‖ c] -> h -> (mu, logvar)          (recognition q(z|x,c))
    z ~ N(mu, exp(logvar/2))                        (reparameterised sample)
    decode:  [z ‖ c] -> h -> x_hat                  (generator p(x|z,c))

Counterfactual generation re-conditions the decoder: encode ``x`` under its
*source* class, then decode the same latent under a *target* class. Because the
latent captures class-invariant structure and ``c`` injects the class, this yields
the **minimal** representation consistent with the target class — the edit whose
validity/proximity/sparsity Paper 2 measures (see ``metrics.py``).

Budget
------
Total system (frozen encoder + this CVAE + attribution) must stay < 1M params and
< 50ms CPU end-to-end (roadmap §3.6). With the defaults below the CVAE alone is
≈0.2M params; a test pins it under 0.35M as a guard-rail.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FeatureCVAE(nn.Module):
    """Conditional VAE on a fixed-length representation vector.

    Parameters
    ----------
    representation_dim:
        Width of the frozen encoder's fused representation (the CVAE's data dim).
    n_classes:
        Number of AAMI classes to condition on (one-hot). 5 for N/S/V/F/Q.
    latent_dim:
        Dimensionality of the VAE latent ``z``.
    hidden_dim:
        Width of the single hidden layer in both encoder and decoder MLPs.
    """

    def __init__(
        self,
        representation_dim: int,
        n_classes: int = 5,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.representation_dim = representation_dim
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        # Recognition network q(z | x, c).
        self.enc = nn.Sequential(
            nn.Linear(representation_dim + n_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Generative network p(x | z, c).
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + n_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, representation_dim),
        )

    # -- pieces ------------------------------------------------------------
    def _onehot(self, cond: torch.Tensor) -> torch.Tensor:
        """Accept either class indices ``(B,)`` or a ready one-hot ``(B, C)``."""
        if cond.dim() == 1:
            return F.one_hot(cond.long(), self.n_classes).float()
        return cond.float()

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, logvar)`` of ``q(z | x, c)``."""
        h = self.enc(torch.cat([x, self._onehot(cond)], dim=1))
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample ``z`` via the reparameterisation trick (identity in eval)."""
        if not self.training:
            return mu  # deterministic reconstruction / counterfactual at eval
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Reconstruct the representation from ``z`` conditioned on class ``c``."""
        return self.dec(torch.cat([z, self._onehot(cond)], dim=1))

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(x_hat, mu, logvar)`` for a reconstruction pass."""
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, cond), mu, logvar

    # -- counterfactuals ---------------------------------------------------
    @torch.no_grad()
    def counterfactual(
        self, x: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Generate a target-class counterfactual of ``x`` in representation space.

        Encodes ``x`` under its ``source`` class then decodes the recognised latent
        under the ``target`` class — the class-conditional edit that keeps
        everything else as close to ``x`` as the model allows. Deterministic
        (uses the posterior mean); call in ``eval`` mode.
        """
        mu, _ = self.encode(x, source)
        return self.decode(mu, target)


def cvae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """β-VAE loss: MSE reconstruction + β·KL(q(z|x,c) ‖ N(0, I)).

    Returns a dict with ``total`` (the value to backprop), ``recon`` and ``kl`` so
    the training loop can log each term. Both terms are per-sample means so ``beta``
    trades off on a scale-stable basis.
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return {"total": recon + beta * kl, "recon": recon, "kl": kl}
