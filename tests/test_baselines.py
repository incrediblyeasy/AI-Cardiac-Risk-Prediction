"""§6 baselines: adapter channel-stacking + comparison-table assembly.

The torchvision backbones are an optional dependency, so the architecture-build
path is gated; the adapter logic and table assembly are tested with a fake
backbone and an injected train/eval so they run without torchvision.
"""

import importlib.util

import numpy as np
import pytest
import torch
from torch import nn

from paper1_echofusenet.baselines.compare import compare_baselines, format_comparison
from paper1_echofusenet.baselines.models import (
    BASELINE_ARCHITECTURES,
    BaselineClassifier,
    build_baseline,
)
from paper1_echofusenet.training.config import TrainConfig

HAS_TV = importlib.util.find_spec("torchvision") is not None


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None
        self.proj = nn.Conv2d(3, 5, 1)

    def forward(self, x):
        self.seen = tuple(x.shape)
        return self.proj(x).mean(dim=(2, 3))


def test_adapter_stacks_three_into_one_3channel_input():
    bb = _FakeBackbone()
    clf = BaselineClassifier(bb)
    out = clf(torch.rand(4, 1, 32, 32), torch.rand(4, 1, 32, 32), torch.rand(4, 1, 32, 32))
    assert bb.seen[1] == 3            # backbone received 3 channels
    assert out.shape == (4, 5)


def test_adapter_resizes_when_input_size_set():
    bb = _FakeBackbone()
    clf = BaselineClassifier(bb, input_size=16)
    clf(torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32))
    assert bb.seen[2:] == (16, 16)


def test_compare_table_assembles_with_injected_eval(tmp_path):
    cfg = TrainConfig()
    cfg.train.device = "cpu"
    rng = np.random.default_rng(0)

    def fake_te(model, c, dev):
        yt = rng.integers(0, 5, 120)
        yp = yt.copy()
        m = rng.random(120) < 0.2
        yp[m] = rng.integers(0, 5, int(m.sum()))
        return yt, yp

    rows = compare_baselines(
        cfg, baseline_names=[], include_echofusenet=True,
        out_dir=tmp_path / "b", image_size=32, latency_iter=5,
        train_and_eval=fake_te,
    )
    assert len(rows) == 1
    assert rows[0].name == "EchoFuseNet"
    assert 0.0 <= rows[0].macro_f1 <= 1.0
    assert rows[0].n_params > 0
    assert (tmp_path / "b" / "comparison.json").exists()
    assert "EchoFuseNet" in format_comparison(rows)


def test_build_baseline_unknown_name_raises():
    with pytest.raises(ValueError):
        build_baseline("not_a_real_net")


@pytest.mark.skipif(not HAS_TV, reason="torchvision not installed (optional dep)")
def test_build_resnet18_forward():
    model = build_baseline("resnet18", n_classes=5)
    out = model(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 5)
    assert "resnet18" in BASELINE_ARCHITECTURES
