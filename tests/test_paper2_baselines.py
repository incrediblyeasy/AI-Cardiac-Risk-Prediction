"""Associational baselines: exact-Shapley properties + Grad-CAM plumbing."""

import torch

from paper1_echofusenet.models import EchoFuseNet
from paper2_causalechonet.encoder import FrozenEncoder
from paper2_causalechonet.baselines import branch_gradcam, shap_modality_values
from paper2_causalechonet.attribution.ite import intervene


def _encoder():
    return FrozenEncoder(EchoFuseNet(widths=(8, 16, 16), fusion_hidden=16))


def _images(b=4, L=64):
    g = torch.Generator().manual_seed(0)
    make = lambda: torch.rand(b, 1, L, L, generator=g)
    return make(), make(), make()


# -- exact Shapley ---------------------------------------------------------
def test_shap_covers_all_modalities():
    enc = _encoder()
    rep = torch.randn(5, enc.representation_dim)
    target = torch.randint(0, 5, (5,))
    phi = shap_modality_values(rep, enc, target)
    assert set(phi) == {"rp", "gaf", "mtf"}
    for v in phi.values():
        assert v.shape == (5,)


def test_shap_efficiency_property():
    # Σ_m φ_m == v(full) - v(empty): the Shapley efficiency axiom.
    enc = _encoder()
    slices = enc.modality_slices()
    rep = torch.randn(6, enc.representation_dim)
    target = torch.randint(0, 5, (6,))

    phi = shap_modality_values(rep, enc, target, baseline="mean")
    total = phi["rp"] + phi["gaf"] + phi["mtf"]

    v_full = torch.softmax(enc.decision(rep), dim=1).gather(1, target.view(-1, 1)).squeeze(1)
    empty = rep
    for m in slices:
        empty = intervene(empty, slices, m, "mean")
    v_empty = torch.softmax(enc.decision(empty), dim=1).gather(1, target.view(-1, 1)).squeeze(1)

    assert torch.allclose(total, v_full - v_empty, atol=1e-5)


def test_shap_accepts_onehot_target():
    enc = _encoder()
    rep = torch.randn(3, enc.representation_dim)
    idx = torch.tensor([0, 2, 4])
    onehot = torch.nn.functional.one_hot(idx, 5).float()
    a = shap_modality_values(rep, enc, idx)
    b = shap_modality_values(rep, enc, onehot)
    assert torch.allclose(a["rp"], b["rp"], atol=1e-6)


# -- Grad-CAM --------------------------------------------------------------
def test_gradcam_importance_shape_and_sign():
    enc = _encoder()
    rp, gaf, mtf = _images()
    target = torch.randint(0, 5, (4,))
    imp = branch_gradcam(enc, rp, gaf, mtf, target)
    assert set(imp) == {"rp", "gaf", "mtf"}
    for v in imp.values():
        assert v.shape == (4,)
        assert torch.all(v >= 0)  # Grad-CAM is ReLU'd -> non-negative


def test_gradcam_can_return_maps():
    enc = _encoder()
    rp, gaf, mtf = _images(b=2)
    out = branch_gradcam(enc, rp, gaf, mtf, torch.tensor([0, 1]), return_maps=True)
    assert "rp_map" in out and out["rp_map"].dim() == 3  # (B, h, w)


def test_gradcam_does_not_leak_grads_to_encoder():
    # The frozen guarantee: Grad-CAM backprops to activations, never to weights.
    enc = _encoder()
    rp, gaf, mtf = _images(b=2)
    branch_gradcam(enc, rp, gaf, mtf, torch.tensor([0, 1]))
    assert all(p.grad is None for p in enc.model.parameters())
    assert not enc.model.training  # stays in eval
