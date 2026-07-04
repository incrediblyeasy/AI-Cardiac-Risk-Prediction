"""MIMIC-IV linkage algorithm: time-zero selection + subject-level split."""

import pytest

from paper3_cardiocausal.datasets import (
    build_linked_cohort,
    link_ecg_ehr,
    subject_level_split,
)


def test_link_keeps_first_eligible_ecg_per_subject():
    ecg = [
        {"subject_id": 1, "time": 20.0, "ecg": "b"},
        {"subject_id": 1, "time": 10.0, "ecg": "a"},   # earlier -> time zero
        {"subject_id": 2, "time": 5.0, "ecg": "c"},
        {"subject_id": 3, "time": 1.0, "ecg": "d"},    # subject 3 not in EHR -> dropped
    ]
    ehr = {
        1: {"age": 60, "treatment": 1, "outcome": 0},
        2: {"age": 70, "treatment": 0, "outcome": 1},
    }
    rows = link_ecg_ehr(ecg, ehr)
    assert [r["subject_id"] for r in rows] == [1, 2]     # subject 3 dropped
    assert next(r for r in rows if r["subject_id"] == 1)["ecg"] == "a"  # earliest kept
    assert rows[0]["age"] == 60                          # EHR merged in


def test_link_applies_eligibility_before_time_zero():
    ecg = [
        {"subject_id": 1, "time": 1.0, "quality": "bad"},
        {"subject_id": 1, "time": 2.0, "quality": "good"},
    ]
    ehr = {1: {"outcome": 1}}
    rows = link_ecg_ehr(ecg, ehr, eligibility=lambda rec, e: rec["quality"] == "good")
    assert len(rows) == 1
    assert rows[0]["time"] == 2.0     # earliest *eligible* ECG is time zero


def test_subject_level_split_is_disjoint():
    rows = [{"subject_id": s, "row": i} for i in range(2) for s in range(20)]
    split = subject_level_split(rows, fractions=(0.6, 0.2, 0.2), seed=0)
    subj = {k: {r["subject_id"] for r in v} for k, v in split.items()}
    # No subject appears in more than one fold.
    assert subj["train"] & subj["val"] == set()
    assert subj["train"] & subj["test"] == set()
    assert subj["val"] & subj["test"] == set()
    # Every subject placed exactly once.
    assert subj["train"] | subj["val"] | subj["test"] == set(range(20))


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        subject_level_split([{"subject_id": 1}], fractions=(0.5, 0.2, 0.2))


def test_build_linked_cohort_is_credentialing_gated():
    with pytest.raises(NotImplementedError):
        build_linked_cohort()
