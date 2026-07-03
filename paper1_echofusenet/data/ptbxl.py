"""PTB-XL domain-transfer probe for EchoFuseNet (Day 15).

**Important scope note.** PTB-XL is a *record-level diagnostic* dataset (10-second
12-lead clinical ECGs labelled with SCP statements: NORM, MI, STTC, CD, HYP). It
carries **no per-beat AAMI annotations and no R-peak markers**, unlike MIT-BIH /
INCART. So a beat-level 5-class arrhythmia model cannot be scored beat-for-beat
against PTB-XL truth.

What *is* valid is a **normal-beat transfer probe**: take PTB-XL records labelled
NORM (normal ECG), detect R-peaks with a standard QRS detector, extract beats
through the same RP/GAF/MTF pipeline, and measure how often the frozen DS1 model
still calls those beats "N" under a brand-new domain (different country, hardware,
100 Hz vs 360 Hz). N-recall on NORM beats + the predicted-class distribution is
an honest domain-shift signal; we do **not** claim full 5-class metrics here.

Sampling/lead handling mirrors ``incart.py``: the 100 Hz PTB-XL beat is taken over
the same *physical* window as a MIT-BIH beat and resampled to 256 samples, lead II
by default.
"""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import numpy as np
import wfdb
from wfdb import processing as wfdb_processing

from .aami import class_index
from .beats import BeatSegment
from .incart import AFTER_S, BEFORE_S, TARGET_LEN, _zscore

PHYSIONET_DB = "ptb-xl"
DEFAULT_PTBXL_DEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "ptbxl"
DATABASE_CSV = "ptbxl_database.csv"
SAMPLING_LR: float = 100.0  # low-res PTB-XL sampling rate (records100/)


def select_norm_records(
    csv_path: Path, limit: int | None = None, min_confidence: float = 80.0
) -> list[tuple[int, str]]:
    """Return ``(ecg_id, filename_lr)`` for records that are confidently NORM.

    A record qualifies when its ``scp_codes`` contains ``NORM`` with likelihood
    >= ``min_confidence`` and NORM is the highest-likelihood code — i.e. a clean
    normal ECG, the only PTB-XL subset with a defensible beat-level expectation
    (all beats should be N).
    """
    selected: list[tuple[int, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                codes = ast.literal_eval(row["scp_codes"])
            except (ValueError, SyntaxError, KeyError):
                continue
            if not isinstance(codes, dict) or "NORM" not in codes:
                continue
            norm_conf = float(codes["NORM"])
            if norm_conf < min_confidence:
                continue
            if codes and max(codes.values()) > norm_conf:
                continue  # some other code is more likely -> not a clean normal
            selected.append((int(row["ecg_id"]), row["filename_lr"]))
            if limit is not None and len(selected) >= limit:
                break
    return selected


def detect_rpeaks(trace: np.ndarray, fs: float) -> np.ndarray:
    """R-peak sample indices via wfdb's XQRS detector."""
    return np.asarray(
        wfdb_processing.xqrs_detect(sig=np.asarray(trace, dtype=np.float64), fs=fs, verbose=False),
        dtype=np.int64,
    )


def beats_from_peaks(
    trace: np.ndarray,
    fs: float,
    peaks: np.ndarray,
    before_s: float = BEFORE_S,
    after_s: float = AFTER_S,
    target_len: int = TARGET_LEN,
    normalize: bool = True,
) -> list[np.ndarray]:
    """Extract resampled beat windows around given R-peaks (pure/testable).

    Each beat is the physical ``[before_s, after_s]`` window at ``fs``, resampled
    to ``target_len`` samples. Peaks whose window falls off the edge are skipped.
    """
    from scipy.signal import resample

    trace = np.asarray(trace, dtype=np.float64)
    n = trace.shape[0]
    before = round(before_s * fs)
    after = round(after_s * fs)

    out: list[np.ndarray] = []
    for peak in peaks:
        start = int(peak) - before
        stop = int(peak) + after
        if start < 0 or stop > n:
            continue
        seg = trace[start:stop]
        if seg.shape[0] != target_len:
            seg = resample(seg, target_len)
        out.append(_zscore(seg) if normalize else seg.astype(np.float32))
    return out


def load_ptbxl_norm_beats(
    data_dir: Path = DEFAULT_PTBXL_DEST,
    limit: int | None = 200,
    lead: str = "II",
    normalize: bool = True,
    min_confidence: float = 80.0,
) -> list[BeatSegment]:
    """Load beats from NORM PTB-XL records, labelled N (the transfer probe truth).

    ``limit`` caps the number of records (PTB-XL is large); ``None`` uses all
    NORM records. Every returned beat has AAMI label N, since these are normal
    ECGs — the model's N-recall on them is the domain-transfer number.
    """
    data_dir = Path(data_dir)
    records = select_norm_records(data_dir / DATABASE_CSV, limit=limit, min_confidence=min_confidence)

    beats: list[BeatSegment] = []
    for ecg_id, filename_lr in records:
        rec = wfdb.rdrecord(str(data_dir / filename_lr))
        signal = np.asarray(rec.p_signal, dtype=np.float32)
        fs = float(rec.fs)
        channel = _lead_index_by_name(list(rec.sig_name), lead, signal)
        trace = signal[:, channel] if signal.ndim == 2 else signal

        peaks = detect_rpeaks(trace, fs)
        for seg in beats_from_peaks(trace, fs, peaks, normalize=normalize):
            beats.append(
                BeatSegment(
                    signal=seg,
                    aami="N",
                    label=class_index("N"),
                    record_id=ecg_id,
                    r_peak=0,
                    fold="PTBXL",
                )
            )
    return beats


def _lead_index_by_name(channels: list[str], prefer: str, signal: np.ndarray) -> int:
    try:
        return channels.index(prefer)
    except ValueError:
        return 1 if signal.ndim == 2 and signal.shape[1] > 1 else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PTB-XL (large!) for the domain-transfer probe."
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_PTBXL_DEST)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PTB-XL to {args.dest} (this is several GB)...")
    wfdb.dl_database(PHYSIONET_DB, dl_dir=str(args.dest))
    print("Done.")


if __name__ == "__main__":
    main()
