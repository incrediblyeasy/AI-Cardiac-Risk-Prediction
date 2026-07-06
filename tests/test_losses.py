"""§2 imbalance losses: focal, class-balanced, weighted-CE, and the factory."""

import numpy as np
import pytest
import torch

from paper1_echofusenet.training.losses import (
    ClassBalancedLoss,
    FocalLoss,
    build_loss_fn,
    class_counts_from_loader,
    effective_number_weights,
    inverse_frequency_weights,
)


def test_focal_gamma_zero_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(16, 5)
    target = torch.randint(0, 5, (16,))
    focal = FocalLoss(gamma=0.0)(logits, target)
    ce = torch.nn.CrossEntropyLoss()(logits, target)
    assert torch.allclose(focal, ce, atol=1e-6)


def test_focal_downweights_easy_examples():
    # A confidently-correct (but not saturated) example: focal < CE because the
    # (1 - p_t)^gamma factor shrinks an already-small loss further.
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])
    focal = FocalLoss(gamma=2.0)(logits, target)
    ce = torch.nn.CrossEntropyLoss()(logits, target)
    assert 0.0 < float(focal) < float(ce)


def test_inverse_frequency_weights_upweight_minority():
    counts = np.array([1000, 10])
    w = inverse_frequency_weights(counts).numpy()
    assert w[1] > w[0]                       # rare class gets more weight
    assert np.isclose(w.mean(), 1.0)         # normalised to mean 1


def test_effective_number_weights_bounds():
    counts = np.array([1000, 50, 5])
    # beta -> 0 : ~uniform weights; beta near 1 : approaches inverse frequency.
    near_uniform = effective_number_weights(counts, beta=1e-9).numpy()
    assert np.allclose(near_uniform, 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        effective_number_weights(counts, beta=1.0)


def test_class_balanced_forward_finite():
    counts = np.array([1000, 50, 200, 20, 5])
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    for base in ("ce", "focal"):
        loss = ClassBalancedLoss(counts, beta=0.999, base=base)(logits, target)
        assert torch.isfinite(loss)


def test_build_loss_fn_all_recipes_and_unknown():
    counts = np.array([100, 10, 50, 5, 2])
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    for name in ("ce", "weighted_ce", "focal", "class_balanced"):
        loss = build_loss_fn(name, 5, counts)(logits, target)
        assert torch.isfinite(loss)
    with pytest.raises(ValueError):
        build_loss_fn("nope", 5, counts)
    # ce needs no counts; the others do.
    build_loss_fn("ce", 5)
    with pytest.raises(ValueError):
        build_loss_fn("focal", 5, class_counts=None)


def test_class_counts_from_loader():
    from torch.utils.data import DataLoader, TensorDataset

    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    x = torch.zeros(6, 1, 4, 4)
    ds = TensorDataset(x, x, x, labels)
    counts = class_counts_from_loader(DataLoader(ds, batch_size=4), n_classes=3)
    assert counts.tolist() == [3, 1, 2]
