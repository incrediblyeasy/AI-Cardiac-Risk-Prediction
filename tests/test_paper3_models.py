"""Paper 3 model layers: fusion, risk head, recourse mask, longitudinal bound."""

import pytest
import torch

from paper3_cardiocausal.fusion import MultimodalFusion
from paper3_cardiocausal.scm import RiskHead
from paper3_cardiocausal.recourse import ModifiabilityMask
from paper3_cardiocausal.longitudinal import LatentPropagator


# -- fusion ----------------------------------------------------------------
def test_fusion_output_shape():
    fuse = MultimodalFusion(ecg_dim=48, tabular_dim=10, latent_dim=32)
    ecg = torch.randn(4, 48)
    tab = torch.randn(4, 10)
    out = fuse(ecg, tab)
    assert out.shape == (4, 32)


def test_fusion_uses_both_modalities():
    fuse = MultimodalFusion(ecg_dim=8, tabular_dim=6, latent_dim=16).eval()
    ecg, tab = torch.randn(3, 8), torch.randn(3, 6)
    base = fuse(ecg, tab)
    assert not torch.allclose(base, fuse(torch.randn_like(ecg), tab))
    assert not torch.allclose(base, fuse(ecg, torch.randn_like(tab)))


# -- risk head -------------------------------------------------------------
def test_risk_head_logit_and_probability():
    head = RiskHead(latent_dim=16).eval()
    latent = torch.randn(5, 16)
    logit = head(latent)
    assert logit.shape == (5,)
    prob = head.risk(latent)
    assert prob.shape == (5,)
    assert torch.all((prob >= 0) & (prob <= 1))


# -- recourse modifiability mask ------------------------------------------
def test_mask_pins_immutable_features():
    names = ["age", "sex", "sbp", "ldl"]
    mask = ModifiabilityMask(names, modifiable=["sbp", "ldl"])
    original = torch.tensor([[60.0, 1.0, 140.0, 3.5]])
    proposed = torch.tensor([[40.0, 0.0, 120.0, 2.0]])  # tries to change age & sex too
    projected = mask.project(original, proposed)
    # age and sex reverted to original; sbp and ldl take proposed values.
    assert projected[0, 0] == 60.0 and projected[0, 1] == 1.0
    assert projected[0, 2] == 120.0 and projected[0, 3] == 2.0


def test_mask_validity_check():
    names = ["age", "sbp"]
    mask = ModifiabilityMask(names, modifiable=["sbp"])
    original = torch.tensor([[60.0, 140.0]])
    good = torch.tensor([[60.0, 120.0]])       # only sbp changed
    bad = torch.tensor([[55.0, 120.0]])        # age changed -> invalid
    assert mask.is_valid_action(original, good)
    assert not mask.is_valid_action(original, bad)


def test_mask_rejects_unknown_modifiable():
    with pytest.raises(ValueError):
        ModifiabilityMask(["age", "sbp"], modifiable=["weight"])


# -- bounded longitudinal propagation -------------------------------------
def test_propagate_trajectory_shape_includes_start():
    prop = LatentPropagator(latent_dim=12, max_horizon=4)
    h0 = torch.randn(3, 12)
    traj = prop.propagate(h0, horizon=3)
    assert traj.shape == (3, 4, 12)          # horizon + 1 states
    assert torch.allclose(traj[:, 0], h0)    # first state is the input


def test_propagate_horizon_zero_returns_only_start():
    prop = LatentPropagator(latent_dim=5)
    h0 = torch.randn(2, 5)
    traj = prop.propagate(h0, horizon=0)
    assert traj.shape == (2, 1, 5)


def test_propagate_refuses_to_exceed_bound():
    prop = LatentPropagator(latent_dim=5, max_horizon=3)
    with pytest.raises(ValueError):
        prop.propagate(torch.randn(2, 5), horizon=4)
