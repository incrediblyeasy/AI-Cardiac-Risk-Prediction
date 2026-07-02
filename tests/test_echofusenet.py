"""EchoFuseNet: forward shape, ~0.7M budget, independent branches, integration."""

import numpy as np
import torch

from paper1_echofusenet.models import CNNBranch, EchoFuseNet, count_parameters


def _dummy_batch(b=4, L=256):
    g = torch.Generator().manual_seed(0)
    make = lambda: torch.rand(b, 1, L, L, generator=g)
    return make(), make(), make()


def test_forward_returns_logits():
    model = EchoFuseNet(n_classes=5).eval()
    rp, gaf, mtf = _dummy_batch()
    with torch.no_grad():
        out = model(rp, gaf, mtf)
    assert out.shape == (4, 5)


def test_total_parameter_budget_is_about_0_7M():
    params = count_parameters(EchoFuseNet())
    # Assembled model must sit at ~0.7M (below the budget, with a sane floor).
    assert 0.5e6 < params < 0.7e6, params


def test_three_independent_branches():
    model = EchoFuseNet()
    # Branches must be distinct modules (no weight sharing across modalities).
    assert model.branch_rp is not model.branch_gaf
    assert model.branch_gaf is not model.branch_mtf
    ids = {id(model.branch_rp), id(model.branch_gaf), id(model.branch_mtf)}
    assert len(ids) == 3


def test_branches_respond_to_their_own_input():
    # Changing only one modality must change the output -> all three are wired in.
    model = EchoFuseNet().eval()
    rp, gaf, mtf = _dummy_batch()
    with torch.no_grad():
        base = model(rp, gaf, mtf)
        alt_rp = model(torch.rand_like(rp), gaf, mtf)
        alt_mtf = model(rp, gaf, torch.rand_like(mtf))
    assert not torch.allclose(base, alt_rp)
    assert not torch.allclose(base, alt_mtf)


def test_gradients_flow_to_all_branches():
    model = EchoFuseNet(n_classes=5)
    rp, gaf, mtf = _dummy_batch(b=2)
    logits = model(rp, gaf, mtf)
    loss = torch.nn.functional.cross_entropy(
        logits, torch.tensor([0, 3])
    )
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_consumes_dataloader_style_tuple():
    # Integration with the Day-6 (rp, gaf, mtf, label) tuple contract.
    from paper1_echofusenet.data.dataset import MultimodalBeatDataset
    from paper1_echofusenet.data.beats import BeatSegment

    def fake_beat(label, seed):
        rng = np.random.default_rng(seed)
        sig = (
            np.sin(np.linspace(0, 4 * np.pi, 64)) + 0.1 * rng.standard_normal(64)
        ).astype(np.float32)
        return BeatSegment(sig, "NSVFQ"[label], label, 1, 30, "DS1")

    ds = MultimodalBeatDataset([fake_beat(i % 5, i) for i in range(6)])
    loader = torch.utils.data.DataLoader(ds, batch_size=3)
    rp, gaf, mtf, labels = next(iter(loader))

    model = EchoFuseNet().eval()
    with torch.no_grad():
        out = model(rp, gaf, mtf)
    assert out.shape == (3, 5)
    assert labels.shape == (3,)
