"""Modality-subset model behaviour + the Day-12 ablation sweep driver."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from paper1_echofusenet.data.beats import BeatSegment
from paper1_echofusenet.data.dataset import MultimodalBeatDataset
from paper1_echofusenet.models import EchoFuseNet, count_parameters
from paper1_echofusenet.training.ablation import (
    FULL_KEY,
    MODALITY_SETS,
    modality_key,
    run_ablation,
)
from paper1_echofusenet.training.config import TrainConfig


def _batch(b=4, L=256):
    g = torch.Generator().manual_seed(0)
    make = lambda: torch.rand(b, 1, L, L, generator=g)
    return make(), make(), make()


# ------------------------------ model subsets ------------------------------ #
def test_single_modality_builds_one_branch():
    model = EchoFuseNet(modalities=("gaf",)).eval()
    assert model.branch_rp is None
    assert model.branch_mtf is None
    assert model.branch_gaf is not None
    assert model.modalities == ("gaf",)
    rp, gaf, mtf = _batch()
    with torch.no_grad():
        out = model(rp, gaf, mtf)
    assert out.shape == (4, 5)


def test_subset_is_smaller_than_full():
    full = count_parameters(EchoFuseNet())
    one = count_parameters(EchoFuseNet(modalities=("rp",)))
    two = count_parameters(EchoFuseNet(modalities=("rp", "mtf")))
    assert one < two < full


def test_inactive_modality_is_ignored():
    # A single-GAF model must be invariant to the RP and MTF inputs.
    model = EchoFuseNet(modalities=("gaf",)).eval()
    rp, gaf, mtf = _batch()
    with torch.no_grad():
        base = model(rp, gaf, mtf)
        changed = model(torch.rand_like(rp), gaf, torch.rand_like(mtf))
    assert torch.allclose(base, changed)


def test_modalities_stored_in_canonical_order():
    # Input order shouldn't matter; storage + branches follow (rp, gaf, mtf).
    model = EchoFuseNet(modalities=("mtf", "rp"))
    assert model.modalities == ("rp", "mtf")
    assert model.branch_gaf is None


def test_unknown_modality_raises():
    with pytest.raises(ValueError, match="unknown modalities"):
        EchoFuseNet(modalities=("rp", "xyz"))


def test_default_is_full_three_branches():
    model = EchoFuseNet()
    assert model.modalities == ("rp", "gaf", "mtf")
    assert all(b is not None for b in (model.branch_rp, model.branch_gaf, model.branch_mtf))


# ------------------------------ ablation driver ---------------------------- #
def _loader(n=20, seed=0, batch_size=16):
    rng = np.random.default_rng(seed)

    def beat(label, s):
        r = np.random.default_rng(s)
        sig = (
            np.sin(np.linspace(0, (label + 1) * np.pi, 64))
            + 0.05 * r.standard_normal(64)
        ).astype(np.float32)
        return BeatSegment(sig, "NSVFQ"[label], label, 1, 30, "DS1")

    beats = [beat(i % 5, int(rng.integers(0, 1 << 30))) for i in range(n)]
    return DataLoader(MultimodalBeatDataset(beats), batch_size=batch_size)


def test_modality_key():
    assert modality_key(("rp", "gaf", "mtf")) == FULL_KEY
    assert modality_key(("gaf",)) == "gaf"


def test_run_ablation_covers_all_configs_with_significance(tmp_path):
    cfg = TrainConfig.from_dict(
        {
            "data": {"batch_size": 16},
            "model": {"widths": [8, 16, 16], "fusion_hidden": 16},
            "optim": {"lr": 0.01},
            "train": {
                "epochs": 1,
                "device": "cpu",
                "out_dir": str(tmp_path / "abl"),
                "log_interval": 0,
            },
        }
    )
    train_loader = _loader(seed=0)
    test_loader = _loader(seed=1)

    report = run_ablation(cfg, train_loader, test_loader, device=torch.device("cpu"))

    # All seven configs present, in report order.
    assert [e.key for e in report.per_config] == [modality_key(m) for m in MODALITY_SETS]

    for e in report.per_config:
        assert 0.0 <= e.accuracy <= 1.0
        if e.key == FULL_KEY:
            assert e.vs_full is None            # the reference has no self-comparison
        else:
            assert e.vs_full is not None        # every subset is tested vs full
            assert 0.0 <= e.vs_full.pvalue <= 1.0

    # Parameter counts strictly increase from singles to the full model.
    params = {e.key: e.n_params for e in report.per_config}
    assert params["rp"] < params["rp+gaf"] < params[FULL_KEY]

    # Summary artifact written and well-formed.
    import json

    summary = json.loads((tmp_path / "abl" / "ablation_summary.json").read_text())
    assert len(summary["configs"]) == 7
