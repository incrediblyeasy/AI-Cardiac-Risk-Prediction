"""Class-balanced sampling for the EchoFuseNet training fold (§2 enhancement).

The Day-6 pipeline already balances classes by *materialising* an oversampled
beat list (``dataset.oversample_beats``): minority beats are physically
duplicated before the ``Dataset`` is built. That is simple and leakage-safe, but
it (a) fixes the oversampled set for the whole run and (b) grows the in-memory
beat list. This module offers the standard alternative the checklist asks for — a
proper ``Sampler`` class that rebalances *per epoch at draw time* via
``WeightedRandomSampler``, leaving the underlying ``Dataset`` untouched.

Both strategies obey the same protocol rule: **balancing is a train-fold-only
operation applied after the patient split**. A sampler is only ever attached to
the training ``DataLoader``; the test loader keeps its natural DS2 distribution.

Two entry points:

* ``class_balanced_sampler`` — build a ``WeightedRandomSampler`` from an explicit
  label array (weights ∝ 1 / class-frequency, so each class is drawn roughly
  equally in expectation).
* ``make_balanced_sampler`` — convenience wrapper that reads labels off a
  ``MultimodalBeatDataset`` (or any dataset exposing ``.beats`` with ``.label``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def class_weights_for_samples(labels: np.ndarray) -> np.ndarray:
    """Per-sample sampling weights that equalise class draw probability.

    Every sample of class ``c`` gets weight ``1 / count(c)``, so the expected
    number of draws per class is equal regardless of how imbalanced the classes
    are. Empty classes cannot occur here (a label only appears if a sample has
    it), but counts are floored to 1 defensively.
    """
    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        return np.empty(0, dtype=np.float64)
    classes, counts = np.unique(labels, return_counts=True)
    count_of = {int(c): int(n) for c, n in zip(classes, counts)}
    per_class_w = {c: 1.0 / max(n, 1) for c, n in count_of.items()}
    return np.array([per_class_w[int(l)] for l in labels], dtype=np.float64)


def class_balanced_sampler(
    labels: np.ndarray,
    num_samples: int | None = None,
    seed: int = 0,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that draws classes ~uniformly.

    ``num_samples`` defaults to ``len(labels)`` so one epoch sees as many beats
    as the natural dataset (but rebalanced). The RNG is seeded deterministically
    for reproducibility (project convention). ``replacement=True`` is required
    for genuine oversampling of minority classes.
    """
    weights = class_weights_for_samples(labels)
    n = num_samples if num_samples is not None else int(weights.size)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=n,
        replacement=replacement,
        generator=generator,
    )


def _labels_of(dataset: Dataset) -> np.ndarray:
    """Extract the integer label array from a beat dataset without loading images.

    Reads ``dataset.beats[i].label`` when available (the cheap path — no
    transform computation); otherwise falls back to iterating ``__getitem__`` and
    taking the last tuple element.
    """
    beats = getattr(dataset, "beats", None)
    if beats is not None:
        return np.array([b.label for b in beats], dtype=np.int64)
    return np.array([int(dataset[i][-1]) for i in range(len(dataset))], dtype=np.int64)


def make_balanced_sampler(
    dataset: Dataset, seed: int = 0, num_samples: int | None = None
) -> WeightedRandomSampler:
    """Class-balanced sampler for a beat ``Dataset`` (labels read cheaply)."""
    return class_balanced_sampler(_labels_of(dataset), num_samples=num_samples, seed=seed)
