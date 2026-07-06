"""Class-imbalance loss functions for EchoFuseNet (§2 enhancement).

Inter-patient MIT-BIH is dominated by the ``N`` class while ``S``/``F``/``Q`` are
rare, so the loss the model optimises — not just the metric it is judged on —
has to be chosen deliberately. This module collects the three loss recipes the
enhancement checklist asks us to *compare head-to-head* on the same DS1/DS2
split, so the "headline recipe" is picked from evidence rather than habit:

* **Weighted cross-entropy** — the simplest reweighting: scale each class's loss
  by an (optionally inverse-frequency) weight. Already available via
  ``TrainLoopConfig.class_weighted_loss``; exposed here as an explicit builder so
  all four recipes share one construction path.
* **Focal loss** (Lin et al., 2017) — down-weights easy, well-classified beats by
  a ``(1 - p_t)^gamma`` factor so training focuses on the hard minority beats.
  Supports an optional per-class ``alpha`` weight on top of the focal term.
* **Class-balanced loss** (Cui et al., 2019) — reweights by the *effective number
  of samples* ``(1 - beta^n) / (1 - beta)`` rather than raw frequency, which
  stops the weight of a class saturating once it has "enough" samples. Wraps
  either a CE or a focal base loss.

Design notes
------------
All losses subclass ``nn.Module`` and take ``(logits, target)`` with integer
targets, so they are drop-in replacements for ``nn.CrossEntropyLoss`` in the
Day-9 training loop. Class-frequency-derived weights are computed once from the
*training fold only* (never the test fold — see the leakage convention) via
``class_counts_from_loader`` and passed in, keeping the losses themselves free of
any data-loading side effects and therefore trivially unit-testable.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
# Class-frequency helpers (train fold only)
# --------------------------------------------------------------------------- #
def class_counts_from_loader(loader: DataLoader, n_classes: int) -> np.ndarray:
    """Per-class beat counts over a loader's labels.

    Iterates the loader once and tallies the integer label of every beat. The
    loader is expected to yield ``(..., labels)`` tuples (the multimodal loader
    yields ``(rp, gaf, mtf, labels)``); only the final element is read.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    for batch in loader:
        labels = batch[-1]
        vals, c = np.unique(np.asarray(labels).ravel(), return_counts=True)
        counts[vals] += c
    return counts


def inverse_frequency_weights(
    counts: np.ndarray, normalize: bool = True
) -> torch.Tensor:
    """Inverse-frequency class weights (optionally normalised to mean 1).

    Empty classes are floored to a count of 1 to avoid division by zero.
    Normalising to mean 1 keeps the loss magnitude comparable to an unweighted
    run, so the learning rate does not implicitly change with the weighting.
    """
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    inv = counts.sum() / counts
    if normalize:
        inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


def effective_number_weights(
    counts: np.ndarray, beta: float = 0.999, normalize: bool = True
) -> torch.Tensor:
    """Class-balanced weights from the effective number of samples.

    ``w_c ∝ (1 - beta) / (1 - beta^{n_c})`` (Cui et al., 2019). As ``beta -> 0``
    this collapses to no reweighting; as ``beta -> 1`` it approaches pure inverse
    frequency. ``beta`` is typically set very close to 1 (0.99–0.9999). Weights
    are normalised to mean 1 by default for scale-stability, matching
    ``inverse_frequency_weights``.
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    weights = 1.0 / effective_num
    if normalize:
        weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Loss modules
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """Multiclass focal loss (Lin et al., 2017).

    ``FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t)`` where ``p_t`` is the softmax
    probability of the true class. ``gamma`` (>= 0) controls how sharply easy
    examples are down-weighted (``gamma = 0`` recovers weighted cross-entropy).
    ``alpha`` is an optional per-class weight tensor (e.g. inverse-frequency),
    combining class reweighting with the focal down-weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        # Register alpha as a buffer so it moves with .to(device) and is saved.
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Per-sample CE (no reduction) reuses PyTorch's numerically-stable path,
        # including optional class weights (alpha) and label smoothing.
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        # p_t = prob of the true class = exp(-CE_unweighted); recover it from the
        # log-softmax directly so the focal term is unaffected by alpha weighting.
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        focal = (1.0 - p_t).pow(self.gamma) * ce
        if self.reduction == "sum":
            return focal.sum()
        if self.reduction == "none":
            return focal
        return focal.mean()


class ClassBalancedLoss(nn.Module):
    """Class-balanced loss (Cui et al., 2019) wrapping CE or focal.

    Applies effective-number-of-samples weights (see
    ``effective_number_weights``) as the per-class weight of an underlying loss.
    With ``base="ce"`` the weight enters a standard cross-entropy; with
    ``base="focal"`` it becomes the focal ``alpha`` term, giving the full
    class-balanced focal loss from the paper.
    """

    def __init__(
        self,
        class_counts: np.ndarray,
        beta: float = 0.999,
        base: str = "focal",
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        weights = effective_number_weights(class_counts, beta=beta)
        base = base.lower()
        if base == "ce":
            self.loss = nn.CrossEntropyLoss(
                weight=weights, label_smoothing=label_smoothing, reduction=reduction
            )
        elif base == "focal":
            self.loss = FocalLoss(
                gamma=gamma,
                alpha=weights,
                label_smoothing=label_smoothing,
                reduction=reduction,
            )
        else:
            raise ValueError(f"unknown base loss '{base}'; choose 'ce' or 'focal'")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target)


# --------------------------------------------------------------------------- #
# Factory — one construction path for all four imbalance recipes
# --------------------------------------------------------------------------- #
# Recognised ``loss.name`` values (see ``LossConfig`` in ``config.py``).
KNOWN_LOSSES: tuple[str, ...] = ("ce", "weighted_ce", "focal", "class_balanced")


def build_loss_fn(
    name: str,
    n_classes: int,
    class_counts: np.ndarray | None = None,
    *,
    gamma: float = 2.0,
    beta: float = 0.999,
    cb_base: str = "focal",
    label_smoothing: float = 0.0,
    device: torch.device | None = None,
) -> nn.Module:
    """Build one of the four imbalance-handling losses by name.

    ``class_counts`` (per-class beat counts from the *train* fold) is required
    for every recipe except plain ``"ce"``. This is the single entry point the
    training loop and the imbalance-comparison runner both go through, so the
    four recipes are guaranteed to be built identically.
    """
    name = name.lower()
    if name not in KNOWN_LOSSES:
        raise ValueError(f"unknown loss '{name}'; choose from {list(KNOWN_LOSSES)}")

    if name == "ce":
        loss: nn.Module = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        if class_counts is None:
            raise ValueError(f"loss '{name}' requires class_counts")
        if name == "weighted_ce":
            weights = inverse_frequency_weights(class_counts)
            loss = nn.CrossEntropyLoss(
                weight=weights, label_smoothing=label_smoothing
            )
        elif name == "focal":
            alpha = inverse_frequency_weights(class_counts)
            loss = FocalLoss(
                gamma=gamma, alpha=alpha, label_smoothing=label_smoothing
            )
        else:  # class_balanced
            loss = ClassBalancedLoss(
                class_counts,
                beta=beta,
                base=cb_base,
                gamma=gamma,
                label_smoothing=label_smoothing,
            )

    if device is not None:
        loss = loss.to(device)
    return loss
