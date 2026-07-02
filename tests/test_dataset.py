"""Multimodal DataLoader: item shapes, oversampling correctness, no leakage.

Two levels, mirroring the leakage guard:
  1. Pure/synthetic: oversampling math + Dataset item structure (no data needed).
  2. End-to-end: a real (subset) split's DataLoaders — batch shapes, train class
     balance after oversampling, and an untouched natural test distribution.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from paper1_echofusenet.data import beats as beats_mod
from paper1_echofusenet.data.beats import BeatSegment, class_counts
from paper1_echofusenet.data.dataset import (
    MultimodalBeatDataset,
    build_dataloaders,
    oversample_beats,
    oversample_indices,
)
from paper1_echofusenet.data.download import DEFAULT_DEST


# ---- level 1: synthetic, always runs --------------------------------------

def _fake_beat(label: int, record_id: int = 1, n: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    sig = (np.sin(np.linspace(0, 4 * np.pi, n)) + 0.1 * rng.standard_normal(n)).astype(
        np.float32
    )
    return BeatSegment(
        signal=sig,
        aami="NSVFQ"[label],
        label=label,
        record_id=record_id,
        r_peak=100,
        fold="DS1",
    )


def test_oversample_balances_classes():
    # Imbalanced: 10 of class 0, 3 of class 1, 1 of class 2.
    labels = np.array([0] * 10 + [1] * 3 + [2] * 1)
    idx = oversample_indices(labels, seed=0)
    balanced = labels[idx]
    _, counts = np.unique(balanced, return_counts=True)
    assert set(counts) == {10}  # every class raised to the majority count
    assert len(idx) == 30


def test_oversample_is_superset_of_originals():
    labels = np.array([0] * 5 + [1] * 2)
    idx = oversample_indices(labels, seed=1)
    # Every original index must still appear (we only add duplicates).
    assert set(range(len(labels))).issubset(set(idx.tolist()))


def test_oversample_deterministic():
    labels = np.array([0] * 8 + [1] * 3 + [2] * 2)
    assert np.array_equal(
        oversample_indices(labels, seed=7), oversample_indices(labels, seed=7)
    )


def test_oversample_empty():
    assert oversample_indices(np.array([], dtype=int)).size == 0


def test_dataset_item_shapes_and_range():
    ds = MultimodalBeatDataset([_fake_beat(0), _fake_beat(1)], normalize=True)
    rp, gaf, mtf, label = ds[0]
    for ch in (rp, gaf, mtf):
        assert ch.shape == (1, 32, 32)
        assert ch.dtype == torch.float32
        assert float(ch.min()) >= 0.0 and float(ch.max()) <= 1.0  # common scale
    assert label.dtype == torch.long
    assert int(label) == 0


def test_dataset_default_collate_batches():
    ds = MultimodalBeatDataset([_fake_beat(i % 3, seed=i) for i in range(6)])
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    rp, gaf, mtf, labels = next(iter(loader))
    assert rp.shape == (4, 1, 32, 32)
    assert gaf.shape == (4, 1, 32, 32)
    assert mtf.shape == (4, 1, 32, 32)
    assert labels.shape == (4,)


# ---- level 2: needs downloaded data ---------------------------------------

_HAS_DATA = (Path(DEFAULT_DEST) / "101.dat").exists() and (
    Path(DEFAULT_DEST) / "100.dat"
).exists()

needs_data = pytest.mark.skipif(
    not _HAS_DATA,
    reason="MIT-BIH not downloaded; run `python -m paper1_echofusenet.data.download`",
)


@needs_data
def test_end_to_end_batch_shapes():
    train_loader, test_loader = build_dataloaders(
        batch_size=8,
        train_records=(101, 106),
        test_records=(100, 103),
        oversample=True,
    )
    rp, gaf, mtf, labels = next(iter(train_loader))
    L = beats_mod.WINDOW_BEFORE + beats_mod.WINDOW_AFTER
    assert rp.shape == (8, 1, L, L)
    assert gaf.shape == (8, 1, L, L)
    assert mtf.shape == (8, 1, L, L)
    assert labels.shape == (8,)


@needs_data
def test_train_oversampled_balanced_test_untouched():
    train_records = (101, 106)
    test_records = (100, 103)

    # Natural (pre-oversampling) reference counts for both folds.
    raw_train, raw_test = beats_mod.build_split(
        train_records=train_records, test_records=test_records
    )
    raw_train_counts = class_counts(raw_train)

    train_loader, test_loader = build_dataloaders(
        batch_size=32,
        train_records=train_records,
        test_records=test_records,
        oversample=True,
    )

    # Train: every present class raised to the majority count (balanced).
    train_beats = train_loader.dataset.beats
    train_counts = class_counts(train_beats)
    assert len(set(train_counts.values())) == 1  # all equal
    majority = max(raw_train_counts.values())
    assert set(train_counts.values()) == {majority}

    # Test: identical to the natural distribution — not resampled.
    test_beats = test_loader.dataset.beats
    assert class_counts(test_beats) == class_counts(raw_test)


@needs_data
def test_no_leakage_between_loaders():
    train_loader, test_loader = build_dataloaders(
        train_records=(101, 106), test_records=(100, 103)
    )
    train_ids = {b.record_id for b in train_loader.dataset.beats}
    test_ids = {b.record_id for b in test_loader.dataset.beats}
    assert train_ids.isdisjoint(test_ids)


@needs_data
def test_oversample_disabled_keeps_natural_train():
    raw_train, _ = beats_mod.build_split(
        train_records=(101,), test_records=(100,)
    )
    train_loader, _ = build_dataloaders(
        train_records=(101,), test_records=(100,), oversample=False
    )
    assert class_counts(train_loader.dataset.beats) == class_counts(raw_train)
