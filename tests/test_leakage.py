"""Leakage guard — the check that must run on every future data-pipeline change.

Two levels:
  1. Pure list-level: DS1/DS2 record ids are disjoint (no data needed).
  2. Beat-level: a real (subset) split's extracted beats carry no patient id in
     both folds (skipped when MIT-BIH is not downloaded).
"""

from pathlib import Path

import pytest

from paper1_echofusenet.data import beats, splits
from paper1_echofusenet.data.download import DEFAULT_DEST


# ---- level 1: always runs -------------------------------------------------

def test_record_lists_are_disjoint():
    splits.assert_patient_disjoint(splits.DS1_PATIENTS, splits.DS2_PATIENTS)


def test_guard_raises_on_overlap():
    with pytest.raises(AssertionError):
        splits.assert_patient_disjoint([100, 101], [101, 200])


def test_guard_passes_on_disjoint():
    splits.assert_patient_disjoint([100, 101], [200, 201])  # must not raise


# ---- level 2: needs downloaded data ---------------------------------------

_HAS_DATA = (Path(DEFAULT_DEST) / "101.dat").exists() and (Path(DEFAULT_DEST) / "100.dat").exists()

needs_data = pytest.mark.skipif(
    not _HAS_DATA,
    reason="MIT-BIH not downloaded; run `python -m paper1_echofusenet.data.download`",
)


@needs_data
def test_extracted_beats_are_patient_disjoint():
    # Small subset (2 records per fold) keeps this fast but exercises the real path.
    train, test = beats.build_split(
        train_records=(101, 106),
        test_records=(100, 103),
    )
    train_ids = {b.record_id for b in train}
    test_ids = {b.record_id for b in test}
    assert train_ids == {101, 106}
    assert test_ids == {100, 103}
    assert train_ids.isdisjoint(test_ids)


@needs_data
def test_build_split_rejects_cross_fold_record():
    # Asking DS1 to load a DS2 record must fail before any silent leakage.
    with pytest.raises(ValueError):
        beats.load_fold("DS1", records=(100,))  # 100 is a DS2 record
