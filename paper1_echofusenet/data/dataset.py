"""Multimodal beat Dataset / DataLoader for EchoFuseNet.

Combines the three Day 3-5 signal-to-image transforms (RP, GAF, MTF) into one
PyTorch ``Dataset`` that yields ``(rp, gaf, mtf, label)`` tuples — one channel
per late-fusion CNN branch — plus a ``build_dataloaders`` helper that assembles
the inter-patient split and applies oversampling **correctly**.

Two rules from the project protocol are enforced here:

* **Split before balancing.** Oversampling runs only on the DS1 (training) beats
  and only *after* ``build_split`` has produced patient-disjoint folds. The DS2
  (test) fold is never resampled — it keeps its natural class distribution so
  evaluation is honest. Because oversampling only duplicates beats that already
  belong to training patients, it cannot introduce cross-fold leakage.
* **Leakage guard still applies.** ``build_dataloaders`` goes through
  ``build_split``, which runs ``assert_patient_disjoint`` on the extracted beats.

Each transform maps a 1-D beat to an ``(L, L)`` image; with the default Day-2
window ``L = 256``. Channels are optionally rescaled to a common ``[0, 1]`` range
(GAF is natively ``[-1, 1]``) so a CNN sees consistent input scales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..transforms import (
    gramian_angular_field,
    markov_transition_field,
    recurrence_plot,
)
from .beats import BeatSegment, build_split
from .download import DEFAULT_DEST

CHANNEL_NAMES: tuple[str, ...] = ("RP", "GAF", "MTF")


def beat_to_channels(
    signal: np.ndarray, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a 1-D beat to its (RP, GAF, MTF) images.

    With ``normalize=True`` every channel is put on a common ``[0, 1]`` scale:
    RP and MTF are already in ``[0, 1]``; GAF (natively ``[-1, 1]``) is rescaled
    via ``(x + 1) / 2``.
    """
    rp = recurrence_plot(signal)  # [0, 1]
    gaf = gramian_angular_field(signal)  # [-1, 1]
    mtf = markov_transition_field(signal)  # [0, 1]
    if normalize:
        gaf = (gaf + 1.0) / 2.0
    return rp, gaf, mtf


class MultimodalBeatDataset(Dataset):
    """Yields ``(rp, gaf, mtf, label)`` for each beat.

    ``rp``/``gaf``/``mtf`` are ``(1, L, L)`` float32 tensors (a single-channel
    image per CNN branch); ``label`` is a scalar ``long`` tensor in ``0..4``.
    Images are computed lazily in ``__getitem__`` so memory stays flat regardless
    of fold size (important after oversampling).
    """

    def __init__(self, beats: list[BeatSegment], normalize: bool = True) -> None:
        self.beats = beats
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.beats)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        beat = self.beats[index]
        rp, gaf, mtf = beat_to_channels(beat.signal, normalize=self.normalize)
        to_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0)
        return (
            to_tensor(rp),
            to_tensor(gaf),
            to_tensor(mtf),
            torch.tensor(beat.label, dtype=torch.long),
        )


def oversample_indices(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices that oversample every minority class up to the majority count.

    Keeps all original samples and adds with-replacement draws from each smaller
    class until all classes match the largest one, then shuffles. Deterministic
    for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.empty(0, dtype=np.int64)

    classes, counts = np.unique(labels, return_counts=True)
    target = int(counts.max())

    pieces: list[np.ndarray] = []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        deficit = target - idx.size
        if deficit > 0:
            extra = rng.choice(idx, size=deficit, replace=True)
            idx = np.concatenate([idx, extra])
        pieces.append(idx)

    out = np.concatenate(pieces)
    rng.shuffle(out)
    return out.astype(np.int64)


def oversample_beats(beats: list[BeatSegment], seed: int = 0) -> list[BeatSegment]:
    """Class-balanced oversampling of a beat list (training fold only)."""
    labels = np.array([b.label for b in beats])
    order = oversample_indices(labels, seed=seed)
    return [beats[i] for i in order]


def build_dataloaders(
    batch_size: int = 32,
    oversample: bool = True,
    normalize: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    data_dir: Path = DEFAULT_DEST,
    train_records: tuple[int, ...] | None = None,
    test_records: tuple[int, ...] | None = None,
    use_balanced_sampler: bool = False,
    **extract_kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Build (train, test) multimodal DataLoaders under the inter-patient split.

    Class balancing of the **train** fold (only, and only after ``build_split``
    has produced patient-disjoint folds) can be done two ways:

    * ``oversample=True`` — materialise duplicated minority beats up front.
    * ``use_balanced_sampler=True`` — attach a per-epoch class-balanced
      ``WeightedRandomSampler`` and leave the beat list untouched.

    The sampler takes precedence when both are set (they are mutually exclusive —
    the sampler already rebalances). The test fold is always returned with its
    natural DS2 distribution. Returns ``(train_loader, test_loader)``.
    """
    train_beats, test_beats = build_split(
        data_dir, train_records, test_records, **extract_kwargs
    )
    if oversample and not use_balanced_sampler:
        train_beats = oversample_beats(train_beats, seed=seed)

    train_ds = MultimodalBeatDataset(train_beats, normalize=normalize)
    test_ds = MultimodalBeatDataset(test_beats, normalize=normalize)

    # Train ordering: a class-balanced sampler (mutually exclusive with shuffle),
    # otherwise deterministic seeded shuffling.
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if use_balanced_sampler:
        from .sampler import make_balanced_sampler

        sampler = make_balanced_sampler(train_ds, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,  # never shuffle/resample the test fold
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, test_loader
