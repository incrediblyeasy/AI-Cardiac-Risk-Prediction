"""Smoke test that a downloaded MIT-BIH record loads and maps to AAMI beats.

Skipped automatically when the raw data has not been downloaded yet, so the
suite stays green on a fresh checkout."""

from pathlib import Path

import pytest

from paper1_echofusenet.data import mitbih
from paper1_echofusenet.data.download import DEFAULT_DEST

# Record 100 is in DS2 and is one of the smallest headers to check.
_REC = 100
_HAS_DATA = (Path(DEFAULT_DEST) / f"{_REC}.dat").exists()

pytestmark = pytest.mark.skipif(
    not _HAS_DATA,
    reason="MIT-BIH not downloaded; run `python -m paper1_echofusenet.data.download`",
)


def test_load_record_100():
    rec = mitbih.load_record(_REC)
    assert rec.record_id == _REC
    assert rec.fs == 360.0                    # MIT-BIH is sampled at 360 Hz
    assert rec.signal.ndim == 2
    assert rec.signal.shape[1] == len(rec.channels)
    assert rec.n_beats > 0
    # Every retained beat must carry a valid AAMI label.
    assert all(b.aami in ("N", "S", "V", "F", "Q") for b in rec.beats)


def test_class_distribution_nonempty():
    rec = mitbih.load_record(_REC)
    dist = mitbih.class_distribution(rec)
    assert sum(dist.values()) == rec.n_beats
