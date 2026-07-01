"""Beat extraction: fixed-length windows, R-peak centering, labels, edges."""

from pathlib import Path

import numpy as np
import pytest

from paper1_echofusenet.data import beats
from paper1_echofusenet.data.download import DEFAULT_DEST
from paper1_echofusenet.data.mitbih import load_record

_REC = 100
_HAS_DATA = (Path(DEFAULT_DEST) / f"{_REC}.dat").exists()

pytestmark = pytest.mark.skipif(
    not _HAS_DATA,
    reason="MIT-BIH not downloaded; run `python -m paper1_echofusenet.data.download`",
)


def test_segments_have_fixed_window_length():
    rec = load_record(_REC)
    segs = beats.extract_beats(rec, fold="DS2")
    assert len(segs) > 0
    expected = beats.WINDOW_BEFORE + beats.WINDOW_AFTER
    assert all(s.signal.shape == (expected,) for s in segs)
    assert all(s.signal.dtype == np.float32 for s in segs)


def test_labels_are_valid_and_consistent():
    rec = load_record(_REC)
    segs = beats.extract_beats(rec, fold="DS2")
    for s in segs:
        assert s.aami in ("N", "S", "V", "F", "Q")
        assert 0 <= s.label <= 4
        assert s.record_id == _REC
        assert s.fold == "DS2"


def test_edge_beats_are_skipped_not_truncated():
    # No extracted window may reference samples outside the signal.
    rec = load_record(_REC)
    n = rec.signal.shape[0]
    segs = beats.extract_beats(rec, fold="DS2")
    for s in segs:
        assert s.r_peak - beats.WINDOW_BEFORE >= 0
        assert s.r_peak + beats.WINDOW_AFTER <= n


def test_zscore_normalization_applied():
    rec = load_record(_REC)
    segs = beats.extract_beats(rec, fold="DS2", normalize=True)
    means = np.array([s.signal.mean() for s in segs])
    stds = np.array([s.signal.std() for s in segs])
    assert np.allclose(means, 0.0, atol=1e-4)
    # std is ~1 except for any (rare) flat segment guarded to std 0.
    assert np.all((np.isclose(stds, 1.0, atol=1e-3)) | (stds < 1e-6))


def test_raw_mode_preserves_amplitude():
    rec = load_record(_REC)
    raw = beats.extract_beats(rec, fold="DS2", normalize=False)
    norm = beats.extract_beats(rec, fold="DS2", normalize=True)
    # Raw beats should not all be unit-variance (they carry real mV amplitudes).
    raw_stds = np.array([s.signal.std() for s in raw])
    assert not np.allclose(raw_stds, 1.0, atol=1e-2)
    assert len(raw) == len(norm)


def test_prefers_mlii_lead():
    # Record 100 lists MLII first; the extractor should select it.
    rec = load_record(_REC)
    assert beats._lead_index(rec, "MLII") == rec.channels.index("MLII")
