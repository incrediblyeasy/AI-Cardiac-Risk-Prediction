"""CNN branch: forward shapes, parameter budget, depthwise-separable savings."""

import torch
from torch import nn

from paper1_echofusenet.models import (
    CNNBranch,
    DepthwiseSeparableConv,
    count_parameters,
)


def test_forward_returns_embedding():
    branch = CNNBranch().eval()
    out = branch(torch.zeros(4, 1, 256, 256))
    assert out.shape == (4, branch.embedding_dim)
    assert out.shape == (4, 256)


def test_forward_with_classifier_head_returns_logits():
    branch = CNNBranch(n_classes=5).eval()
    out = branch(torch.zeros(3, 1, 256, 256))
    assert out.shape == (3, 5)


def test_handles_other_input_sizes():
    # AdaptiveAvgPool makes the branch size-agnostic.
    branch = CNNBranch().eval()
    out = branch(torch.zeros(2, 1, 128, 128))
    assert out.shape == (2, 256)


def test_deterministic_in_eval():
    branch = CNNBranch().eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        assert torch.equal(branch(x), branch(x))


def test_branch_within_parameter_budget():
    # A single branch should be ~0.18M so three branches + a small fusion head
    # stay under the ~0.7M total budget.
    params = count_parameters(CNNBranch())
    assert 0.15e6 < params < 0.22e6, params
    # Three branches alone must leave headroom for the fusion classifier.
    assert 3 * params < 0.7e6


def test_depthwise_separable_is_cheaper_than_full_conv():
    dsconv = DepthwiseSeparableConv(64, 128, stride=1)
    full = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
    # Depthwise-separable must use markedly fewer params than a plain 3x3 conv.
    assert count_parameters(dsconv) < count_parameters(full)


def test_gradients_flow():
    branch = CNNBranch(n_classes=5)
    out = branch(torch.zeros(2, 1, 256, 256))
    out.sum().backward()
    grads = [p.grad is not None for p in branch.parameters()]
    assert all(grads)
