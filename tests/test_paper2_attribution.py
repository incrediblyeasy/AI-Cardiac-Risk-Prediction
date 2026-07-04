"""Modality attribution: intervention correctness + ITE shapes/behaviour."""

import torch

from paper1_echofusenet.models import EchoFuseNet
from paper2_causalechonet.encoder import FrozenEncoder
from paper2_causalechonet.attribution import (
    attribution_table,
    intervene,
    modality_ite,
)


def _encoder():
    return FrozenEncoder(EchoFuseNet(widths=(8, 16, 16), fusion_hidden=16))


def test_intervene_replaces_only_target_block():
    enc = _encoder()
    slices = enc.modality_slices()
    rep = torch.randn(4, enc.representation_dim)
    out = intervene(rep, slices, "gaf", baseline="zero")
    # gaf block zeroed...
    assert torch.count_nonzero(out[:, slices["gaf"]]) == 0
    # ...rp and mtf untouched.
    for m in ("rp", "mtf"):
        assert torch.allclose(out[:, slices[m]], rep[:, slices[m]])


def test_intervene_mean_baseline_sets_block_to_column_mean():
    enc = _encoder()
    slices = enc.modality_slices()
    rep = torch.randn(6, enc.representation_dim)
    out = intervene(rep, slices, "rp", baseline="mean")
    block_mean = rep[:, slices["rp"]].mean(dim=0)
    assert torch.allclose(out[:, slices["rp"]], block_mean.expand(6, -1), atol=1e-6)


def test_intervene_unknown_modality_raises():
    enc = _encoder()
    try:
        intervene(torch.randn(2, enc.representation_dim), enc.modality_slices(), "xyz")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown modality")


def test_modality_ite_shape_and_probability_bounds():
    enc = _encoder()
    rep = torch.randn(5, enc.representation_dim)
    ite = modality_ite(rep, enc, "mtf")
    assert ite.shape == (5, 5)  # (batch, n_classes)
    # ITE is a difference of two probability vectors -> each entry in [-1, 1].
    assert ite.abs().max().item() <= 1.0 + 1e-6


def test_modality_ite_rows_sum_to_zero():
    # Both factual and ablated softmax rows sum to 1, so the ITE row sums to 0.
    enc = _encoder()
    rep = torch.randn(5, enc.representation_dim)
    ite = modality_ite(rep, enc, "rp")
    assert torch.allclose(ite.sum(dim=1), torch.zeros(5), atol=1e-5)


def test_attribution_table_covers_all_modalities():
    enc = _encoder()
    rep = torch.randn(8, enc.representation_dim)
    table = attribution_table(rep, enc)
    assert set(table) == {"rp", "gaf", "mtf"}
    for v in table.values():
        assert v.shape == (5,)
