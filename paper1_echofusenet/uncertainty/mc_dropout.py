"""Monte Carlo Dropout for EchoFuseNet predictive uncertainty (§4 enhancement).

A trained network with dropout is an implicit ensemble: keeping dropout *active*
at inference and running the same input through ``T`` stochastic forward passes
yields ``T`` slightly different softmax vectors. Their **mean** is a better-
calibrated prediction and their **spread** is a usable uncertainty signal — all
from one model, so no extra parameters (the edge-deployment premise holds). The
one cost is latency: ``T`` passes instead of one, which must be checked against
the <15 ms budget (``benchmark.measure_latency_ms``) before this is called done.

Two uncertainty summaries are returned per beat:

* **predictive entropy** of the mean softmax — total uncertainty;
* **mutual information** (a.k.a. BALD) = predictive entropy minus the mean of
  per-pass entropies — the *epistemic* part, high when the passes *disagree*.

The functions take a model + inputs directly (no data loading) so they are pure
and unit-testable. ``enable_mc_dropout`` flips only ``Dropout`` layers to train
mode while leaving BatchNorm in eval mode — critical, since we want stochastic
dropout but *deterministic* normalisation statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode but re-enable only its ``Dropout`` layers.

    BatchNorm and everything else stay in eval mode (fixed running stats); only
    dropout is left stochastic, which is exactly what MC Dropout requires.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def _entropy(probs: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy (nats) of a probability array along ``axis``."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


@dataclass
class MCDropoutResult:
    """Aggregated MC-Dropout prediction over ``T`` passes for a batch.

    ``mean_probs`` is the ``(N, C)`` averaged softmax; ``prediction`` its argmax.
    ``predictive_entropy`` / ``mutual_information`` are ``(N,)`` per-beat
    uncertainty; ``std`` is the ``(N, C)`` per-class std across passes.
    """

    mean_probs: np.ndarray
    prediction: np.ndarray
    predictive_entropy: np.ndarray
    mutual_information: np.ndarray
    std: np.ndarray
    n_passes: int

    def confidence(self) -> np.ndarray:
        """Max class probability per beat — a simple confidence score."""
        return self.mean_probs.max(axis=1)


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_passes: int = 20,
    device: torch.device | None = None,
) -> MCDropoutResult:
    """Run ``n_passes`` stochastic forward passes and aggregate them.

    ``inputs`` is the ``(rp, gaf, mtf)`` image triple EchoFuseNet's ``forward``
    expects, each ``(N, 1, L, L)``. Returns an :class:`MCDropoutResult` with the
    mean prediction and per-beat uncertainty decomposition.
    """
    if n_passes < 1:
        raise ValueError("n_passes must be >= 1")
    device = device or torch.device("cpu")
    model = model.to(device)
    rp, gaf, mtf = (t.to(device) for t in inputs)
    enable_mc_dropout(model)

    per_pass: list[np.ndarray] = []
    for _ in range(n_passes):
        logits = model(rp, gaf, mtf)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        per_pass.append(probs)

    stacked = np.stack(per_pass, axis=0)          # (T, N, C)
    mean_probs = stacked.mean(axis=0)              # (N, C)
    predictive_entropy = _entropy(mean_probs, axis=1)          # H[E[p]]
    expected_entropy = _entropy(stacked, axis=2).mean(axis=0)  # E[H[p]]
    mutual_information = predictive_entropy - expected_entropy  # BALD (epistemic)

    return MCDropoutResult(
        mean_probs=mean_probs,
        prediction=mean_probs.argmax(axis=1),
        predictive_entropy=predictive_entropy,
        # Numerical noise can push MI marginally negative; clamp at 0.
        mutual_information=np.maximum(mutual_information, 0.0),
        std=stacked.std(axis=0),
        n_passes=n_passes,
    )
