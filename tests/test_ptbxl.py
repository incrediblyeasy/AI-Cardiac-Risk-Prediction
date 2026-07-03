"""PTB-XL probe: NORM record selection (CSV) + beat-from-peaks resampling."""

import numpy as np

from paper1_echofusenet.data.ptbxl import (
    TARGET_LEN,
    beats_from_peaks,
    select_norm_records,
)

# A tiny stand-in for ptbxl_database.csv with the two columns we read.
_CSV = """ecg_id,scp_codes,filename_lr
1,"{'NORM': 100.0, 'SR': 0.0}",records100/00000/00001_lr
2,"{'MI': 100.0}",records100/00000/00002_lr
3,"{'NORM': 50.0, 'STTC': 80.0}",records100/00000/00003_lr
4,"{'NORM': 100.0}",records100/00000/00004_lr
5,"{'NORM': 15.0, 'SR': 0.0}",records100/00000/00005_lr
"""


def test_select_norm_records_picks_clean_normals(tmp_path):
    csv_path = tmp_path / "ptbxl_database.csv"
    csv_path.write_text(_CSV, encoding="utf-8")
    norm = select_norm_records(csv_path, min_confidence=80.0)
    ids = [ecg_id for ecg_id, _ in norm]
    # 1 and 4 are confident, NORM-dominant. 2 is MI. 3 has STTC more likely.
    # 5's NORM confidence (15) is below threshold.
    assert ids == [1, 4]
    assert norm[0][1] == "records100/00000/00001_lr"


def test_select_norm_records_limit(tmp_path):
    csv_path = tmp_path / "ptbxl_database.csv"
    csv_path.write_text(_CSV, encoding="utf-8")
    assert len(select_norm_records(csv_path, limit=1, min_confidence=80.0)) == 1


def test_beats_from_peaks_resamples_and_skips_edges():
    fs = 100.0
    trace = np.sin(np.linspace(0, 40 * np.pi, 1000)).astype(np.float32)
    # One valid peak in the middle; two near the edges that must be dropped.
    peaks = np.array([2, 500, 999])
    beats = beats_from_peaks(trace, fs, peaks)
    assert len(beats) == 1
    assert beats[0].shape == (TARGET_LEN,)


def test_beats_from_peaks_normalization():
    fs = 100.0
    trace = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    beats = beats_from_peaks(trace, fs, np.array([500]), normalize=True)
    assert abs(beats[0].mean()) < 1e-4
    assert abs(beats[0].std() - 1.0) < 1e-2


def test_beats_from_peaks_empty_when_all_edges():
    trace = np.zeros(300, dtype=np.float32)
    # With a 100 Hz physical window (~36 samples each side), peak 5 falls off.
    beats = beats_from_peaks(trace, 100.0, np.array([5]))
    assert beats == []
