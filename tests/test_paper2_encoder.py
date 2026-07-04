"""FrozenEncoder: checkpoint round-trip, freeze guarantee, representation geometry."""

import torch

from paper1_echofusenet.models import EchoFuseNet
from paper1_echofusenet.training.config import TrainConfig
from paper2_causalechonet.encoder import FrozenEncoder, load_frozen_encoder


def _tiny_echofusenet():
    # Small widths keep the test fast; geometry/plumbing is identical to full size.
    return EchoFuseNet(widths=(8, 16, 16), fusion_hidden=16)


def _dummy_images(b=4, L=64):
    g = torch.Generator().manual_seed(0)
    make = lambda: torch.rand(b, 1, L, L, generator=g)
    return make(), make(), make()


def test_encode_shape_matches_representation_dim():
    enc = FrozenEncoder(_tiny_echofusenet())
    rp, gaf, mtf = _dummy_images()
    rep = enc.encode(rp, gaf, mtf)
    assert rep.shape == (4, enc.representation_dim)
    # 3 modalities x embedding_dim (16) = 48.
    assert enc.representation_dim == 3 * 16


def test_all_parameters_frozen():
    enc = FrozenEncoder(_tiny_echofusenet())
    assert all(not p.requires_grad for p in enc.model.parameters())
    assert not enc.model.training  # eval mode pinned


def test_freeze_survives_downstream_backward():
    # Backprop through the decision head into a representation must NOT create
    # grads on the encoder weights (the core no-leakage guarantee).
    enc = FrozenEncoder(_tiny_echofusenet())
    rp, gaf, mtf = _dummy_images(b=2)
    rep = enc.encode(rp, gaf, mtf).clone().requires_grad_(True)
    logits = enc.decision(rep)
    logits.sum().backward()
    assert rep.grad is not None                       # grad reaches the representation
    assert all(p.grad is None for p in enc.model.parameters())  # ...but never the weights


def test_train_mode_cannot_unfreeze_encoder():
    enc = FrozenEncoder(_tiny_echofusenet())
    enc.train()  # a downstream loop may call this
    assert not enc.model.training  # encoder stays in eval regardless


def test_modality_slices_partition_representation():
    enc = FrozenEncoder(_tiny_echofusenet())
    slices = enc.modality_slices()
    assert set(slices) == {"rp", "gaf", "mtf"}
    covered = sorted(i for sl in slices.values() for i in range(sl.start, sl.stop))
    assert covered == list(range(enc.representation_dim))  # non-overlapping, complete


def test_decision_matches_full_model_forward():
    model = _tiny_echofusenet().eval()
    enc = FrozenEncoder(model)
    rp, gaf, mtf = _dummy_images()
    with torch.no_grad():
        direct = model(rp, gaf, mtf)
        via_enc = enc.classify(rp, gaf, mtf)
    assert torch.allclose(direct, via_enc, atol=1e-5)


def test_from_checkpoint_roundtrip(tmp_path):
    # Save a Paper-1-format checkpoint, reload as a frozen encoder, weights match.
    model = _tiny_echofusenet()
    cfg = TrainConfig()
    cfg.model.widths = [8, 16, 16]
    cfg.model.fusion_hidden = 16
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "metric": 0.5,
            "config": cfg.to_dict(),
        },
        ckpt,
    )
    enc = load_frozen_encoder(ckpt)
    rp, gaf, mtf = _dummy_images()
    with torch.no_grad():
        expected = model.eval()(rp, gaf, mtf)
        got = enc.classify(rp, gaf, mtf)
    assert torch.allclose(expected, got, atol=1e-5)
    assert all(not p.requires_grad for p in enc.model.parameters())


def test_from_checkpoint_rejects_bad_dict(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"not": "a checkpoint"}, bad)
    try:
        load_frozen_encoder(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on a malformed checkpoint")
