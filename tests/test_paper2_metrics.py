"""Counterfactual-quality metrics: validity, proximity, sparsity."""

import torch

from paper2_causalechonet.cvae.metrics import (
    counterfactual_report,
    proximity,
    sparsity,
    validity,
)


def test_validity_perfect_and_zero():
    # decision_fn = identity logits; argmax of a one-hot-ish vector is the target.
    decision_fn = lambda z: z
    target = torch.tensor([2, 0, 4])
    hit = torch.zeros(3, 5)
    hit[torch.arange(3), target] = 10.0
    assert validity(hit, target, decision_fn) == 1.0

    miss = torch.zeros(3, 5)
    miss[torch.arange(3), (target + 1) % 5] = 10.0
    assert validity(miss, target, decision_fn) == 0.0


def test_validity_accepts_onehot_target():
    decision_fn = lambda z: z
    idx = torch.tensor([1, 3])
    onehot = torch.nn.functional.one_hot(idx, 5).float()
    logits = torch.zeros(2, 5)
    logits[torch.arange(2), idx] = 5.0
    assert validity(logits, onehot, decision_fn) == 1.0


def test_proximity_zero_for_identical():
    x = torch.randn(4, 10)
    assert proximity(x, x) == 0.0


def test_proximity_is_mean_l1():
    x = torch.zeros(2, 3)
    x_cf = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    # per-sample L1 sums = 3 and 0 -> mean 1.5
    assert abs(proximity(x, x_cf) - 1.5) < 1e-6


def test_sparsity_counts_changed_dims():
    x = torch.zeros(1, 4)
    x_cf = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # 1 of 4 dims changed
    assert abs(sparsity(x, x_cf) - 0.25) < 1e-6
    assert sparsity(x, x) == 0.0


def test_sparsity_respects_tolerance():
    x = torch.zeros(1, 4)
    x_cf = torch.tensor([[1e-4, 1e-4, 1e-4, 1e-4]])  # all below default tol
    assert sparsity(x, x_cf, tol=1e-3) == 0.0


def test_report_bundles_three_metrics():
    decision_fn = lambda z: z
    x = torch.zeros(2, 5)
    x_cf = torch.zeros(2, 5)
    target = torch.tensor([0, 0])
    x_cf[torch.arange(2), target] = 3.0
    rep = counterfactual_report(x, x_cf, target, decision_fn)
    assert set(rep) == {"validity", "proximity", "sparsity"}
    assert rep["validity"] == 1.0
