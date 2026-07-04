"""Fuse the frozen ECG representation with tabular clinical context.

Design
------
Paper 3's inputs are heterogeneous: a dense ECG-beat representation from the
frozen Papers 1-2 encoder, and a tabular clinical vector (demographics, labs,
comorbidities, medications). A small two-tower projection maps each into a common
width, and their sum forms a shared latent that the SCM (``scm``) and recourse
engine (``recourse``) operate on::

    ecg_repr (D_ecg) --Linear--> h_ecg  \
                                          (+)  -> LayerNorm -> shared latent (D_latent)
    tabular  (D_tab) --Linear--> h_tab  /

Additive (rather than concat) fusion keeps the latent width fixed regardless of
which towers are present, so an ECG-only or tabular-only ablation drops in without
changing downstream shapes — useful for the "what does each modality add to risk"
study. LayerNorm stabilises the two towers onto a common scale.

This module is deliberately architecture-only and data-independent so it is
unit-testable now; fitting it needs the MIMIC-IV linked cohort (``datasets``).
"""

from __future__ import annotations

import torch
from torch import nn


class MultimodalFusion(nn.Module):
    """Two-tower additive fusion of an ECG representation and a tabular vector.

    Parameters
    ----------
    ecg_dim:
        Width of the frozen ECG representation (Paper 1/2 encoder output).
    tabular_dim:
        Number of tabular clinical features.
    latent_dim:
        Width of the shared latent space the SCM/recourse layers consume.
    """

    def __init__(self, ecg_dim: int, tabular_dim: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.ecg_dim = ecg_dim
        self.tabular_dim = tabular_dim
        self.latent_dim = latent_dim
        self.ecg_proj = nn.Linear(ecg_dim, latent_dim)
        self.tab_proj = nn.Linear(tabular_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, ecg_repr: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Return the shared latent ``(B, latent_dim)``."""
        return self.norm(self.ecg_proj(ecg_repr) + self.tab_proj(tabular))
