"""§2 class-balanced sampler: weights, draw balance, dataset integration."""

import numpy as np

from paper1_echofusenet.data.beats import BeatSegment
from paper1_echofusenet.data.dataset import MultimodalBeatDataset
from paper1_echofusenet.data.sampler import (
    class_balanced_sampler,
    class_weights_for_samples,
    make_balanced_sampler,
)


def test_weights_inverse_to_class_size():
    labels = np.array([0, 0, 0, 0, 1])   # class 0 x4, class 1 x1
    w = class_weights_for_samples(labels)
    assert np.isclose(w[0], 1 / 4)
    assert np.isclose(w[4], 1 / 1)


def test_sampler_balances_draws():
    # Heavily imbalanced: 100 / 10 / 2. Balanced sampler should even the draws.
    labels = np.array([0] * 100 + [1] * 10 + [2] * 2)
    sampler = class_balanced_sampler(labels, num_samples=3000, seed=0)
    drawn = labels[np.array(list(sampler))]
    _, counts = np.unique(drawn, return_counts=True)
    # Each class should get roughly a third of the draws (within 30%).
    frac = counts / counts.sum()
    assert np.all(np.abs(frac - 1 / 3) < 0.1)


def test_sampler_is_deterministic_given_seed():
    labels = np.array([0] * 20 + [1] * 5)
    a = list(class_balanced_sampler(labels, seed=7))
    b = list(class_balanced_sampler(labels, seed=7))
    assert a == b


def test_make_balanced_sampler_reads_dataset_labels():
    def beat(label):
        sig = np.zeros(64, dtype=np.float32)
        return BeatSegment(sig, "NSVFQ"[label], label, 1, 30, "DS1")

    beats = [beat(0)] * 10 + [beat(1)] * 2
    ds = MultimodalBeatDataset(beats)
    sampler = make_balanced_sampler(ds, seed=0)
    assert sampler.num_samples == len(ds)
