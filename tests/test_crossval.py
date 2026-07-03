"""Patient-grouped k-fold CV: split invariants + tiny end-to-end run (no data)."""

import numpy as np

from paper1_echofusenet.data.beats import BeatSegment
from paper1_echofusenet.training.config import TrainConfig
from paper1_echofusenet.training.crossval import cross_validate, patient_kfold


# --------------------------- split invariants ------------------------------ #
def test_patient_kfold_is_disjoint_and_covers_all():
    patients = list(range(1, 23))  # 22 DS1-like patients
    folds = patient_kfold(patients, k=5, seed=0)
    assert len(folds) == 5

    covered_val: set[int] = set()
    for train_p, val_p in folds:
        # No patient shared between a fold's train and val (inter-patient).
        assert set(train_p).isdisjoint(val_p)
        # Train + val together account for every patient exactly.
        assert set(train_p) | set(val_p) == set(patients)
        covered_val |= set(val_p)
    # Every patient is validated in exactly one fold (partition of the val role).
    assert covered_val == set(patients)


def test_patient_kfold_val_groups_partition():
    patients = list(range(1, 11))
    folds = patient_kfold(patients, k=5, seed=1)
    val_groups = [set(v) for _, v in folds]
    # Val groups are pairwise disjoint and near-equal in size.
    for i in range(len(val_groups)):
        for j in range(i + 1, len(val_groups)):
            assert val_groups[i].isdisjoint(val_groups[j])
    sizes = [len(v) for v in val_groups]
    assert max(sizes) - min(sizes) <= 1


def test_patient_kfold_deterministic():
    a = patient_kfold(list(range(1, 13)), k=4, seed=7)
    b = patient_kfold(list(range(1, 13)), k=4, seed=7)
    assert a == b


def test_patient_kfold_rejects_bad_k():
    import pytest

    with pytest.raises(ValueError):
        patient_kfold([1, 2, 3], k=1)
    with pytest.raises(ValueError):
        patient_kfold([1, 2, 3], k=5)  # more folds than patients


# --------------------------- end-to-end (synthetic) ------------------------ #
def _synthetic_ds1(n_patients=6, per_class=4, seed=0):
    """Patient-labelled beats with class-separable frequency content."""
    rng = np.random.default_rng(seed)
    beats: list[BeatSegment] = []
    for pid in range(1, n_patients + 1):
        for label in range(5):
            for _ in range(per_class):
                r = np.random.default_rng(int(rng.integers(0, 1 << 30)))
                sig = (
                    np.sin(np.linspace(0, (label + 1) * np.pi, 64))
                    + 0.05 * r.standard_normal(64)
                ).astype(np.float32)
                beats.append(BeatSegment(sig, "NSVFQ"[label], label, pid, 30, "DS1"))
    return beats


def test_cross_validate_end_to_end(tmp_path):
    cfg = TrainConfig.from_dict(
        {
            "data": {"batch_size": 16, "oversample": False},
            "model": {"widths": [8, 16, 16], "fusion_hidden": 16},
            "optim": {"lr": 0.01},
            "train": {
                "epochs": 1,
                "device": "cpu",
                "out_dir": str(tmp_path / "cv"),
                "log_interval": 0,
            },
        }
    )
    beats = _synthetic_ds1(n_patients=6)
    report = cross_validate(cfg, k=3, seed=0, beats=beats, n_boot=100)

    # Three folds, each validated on a disjoint set of patients.
    assert len(report.per_fold) == 3
    val_sets = [set(f.val_patients) for f in report.per_fold]
    assert set().union(*val_sets) == {1, 2, 3, 4, 5, 6}
    for f in report.per_fold:
        assert 0.0 <= f.accuracy <= 1.0

    # CIs are well-formed and ordered.
    for iv in (report.accuracy_ci, report.macro_f1_ci, report.pooled_accuracy_ci):
        assert iv.low <= iv.point <= iv.high

    # Summary artifact was written and round-trips.
    import json

    summary = json.loads((tmp_path / "cv" / "cv_summary.json").read_text())
    assert len(summary["folds"]) == 3
    assert "accuracy_mean_ci" in summary
