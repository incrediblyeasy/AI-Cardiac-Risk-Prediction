"""Representation caching: encode a beat loader, round-trip the disk cache."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper1_echofusenet.data.beats import BeatSegment
from paper1_echofusenet.data.dataset import MultimodalBeatDataset
from paper1_echofusenet.models import EchoFuseNet
from paper2_causalechonet.data import build_representation_dataset, encode_loader
from paper2_causalechonet.encoder import FrozenEncoder


def _beat_loader(n=6, batch=3):
    def fake_beat(i):
        rng = np.random.default_rng(i)
        sig = (np.sin(np.linspace(0, 4 * np.pi, 64)) + 0.1 * rng.standard_normal(64)).astype(
            np.float32
        )
        return BeatSegment(sig, "NSVFQ"[i % 5], i % 5, 1, 30, "DS1")

    ds = MultimodalBeatDataset([fake_beat(i) for i in range(n)])
    return DataLoader(ds, batch_size=batch)


def _encoder():
    return FrozenEncoder(EchoFuseNet(widths=(8, 16, 16), fusion_hidden=16))


def test_encode_loader_shapes():
    enc = _encoder()
    reps, labels = encode_loader(enc, _beat_loader(n=6))
    assert reps.shape == (6, enc.representation_dim)
    assert labels.shape == (6,)


def test_build_dataset_and_cache_roundtrip(tmp_path):
    enc = _encoder()
    cache = tmp_path / "repr.pt"
    ds1 = build_representation_dataset(enc, _beat_loader(), cache_path=cache)
    assert cache.exists()
    # Second call loads from cache and yields identical tensors.
    ds2 = build_representation_dataset(enc, _beat_loader(), cache_path=cache)
    x1, y1 = ds1[0]
    x2, y2 = ds2[0]
    assert torch.allclose(x1, x2)
    assert int(y1) == int(y2)


def test_stale_cache_detected(tmp_path):
    enc = _encoder()
    cache = tmp_path / "repr.pt"
    build_representation_dataset(enc, _beat_loader(), cache_path=cache)
    # A differently-sized encoder must reject the stale cache.
    other = FrozenEncoder(EchoFuseNet(widths=(8, 16, 32), fusion_hidden=16))
    try:
        build_representation_dataset(other, _beat_loader(), cache_path=cache)
    except ValueError:
        return
    raise AssertionError("expected ValueError on a stale cache")
