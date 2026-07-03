"""INCART loader: record naming, 257->256 resampling, lead II, AAMI labels."""

import numpy as np

from paper1_echofusenet.data.aami import class_index
from paper1_echofusenet.data.incart import (
    INCART_RECORDS,
    TARGET_LEN,
    extract_incart_beats,
    record_id_of,
)
from paper1_echofusenet.data.mitbih import Beat, Record

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _synthetic_incart(fs=257.0, seconds=12, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    signal = rng.standard_normal((n, 12)).astype(np.float32)
    # Mark lead II (index 1) so we can tell which lead was picked.
    signal[:, 1] += 5.0
    beats = [
        Beat(sample=300, symbol="N", aami="N"),
        Beat(sample=800, symbol="V", aami="V"),
        Beat(sample=1500, symbol="A", aami="S"),
        Beat(sample=5, symbol="N", aami="N"),  # window falls off the start -> skip
    ]
    return Record(record_id=7, signal=signal, fs=fs, channels=list(LEADS_12), beats=beats)


def test_record_naming():
    assert INCART_RECORDS[0] == "I01"
    assert INCART_RECORDS[-1] == "I75"
    assert len(INCART_RECORDS) == 75
    assert record_id_of("I07") == 7
    assert record_id_of("I75") == 75


def test_beats_resampled_to_target_len():
    rec = _synthetic_incart()
    segs = extract_incart_beats(rec)
    # Three in-bounds beats; the edge beat (sample=5) is dropped.
    assert len(segs) == 3
    for s in segs:
        assert s.signal.shape == (TARGET_LEN,)
        assert s.signal.dtype == np.float32
        assert s.fold == "INCART"
        assert s.record_id == 7


def test_labels_match_aami():
    rec = _synthetic_incart()
    segs = extract_incart_beats(rec)
    aamis = [s.aami for s in segs]
    assert aamis == ["N", "V", "S"]
    for s in segs:
        assert s.label == class_index(s.aami)


def test_lead_ii_is_selected():
    rec = _synthetic_incart()
    # Lead II (index 1) has a +5 offset; z-scored output removes the mean, but
    # picking a different lead would change the extracted values. Compare the
    # default (II) against an explicit lead-I extraction.
    seg_ii = extract_incart_beats(rec, normalize=False)[0].signal
    seg_i = extract_incart_beats(rec, lead="I", normalize=False)[0].signal
    assert not np.allclose(seg_ii, seg_i)


def test_zscore_normalization_default():
    rec = _synthetic_incart()
    seg = extract_incart_beats(rec, normalize=True)[0].signal
    # z-scored -> ~zero mean, unit std.
    assert abs(seg.mean()) < 1e-4
    assert abs(seg.std() - 1.0) < 1e-2


def test_lead_fallback_when_ii_absent():
    # A record without a named "II" lead should fall back, not crash.
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((3000, 3)).astype(np.float32)
    rec = Record(
        record_id=1,
        signal=sig,
        fs=257.0,
        channels=["A", "B", "C"],  # no "II"
        beats=[Beat(sample=300, symbol="N", aami="N")],
    )
    segs = extract_incart_beats(rec)
    assert len(segs) == 1
    assert segs[0].signal.shape == (TARGET_LEN,)
