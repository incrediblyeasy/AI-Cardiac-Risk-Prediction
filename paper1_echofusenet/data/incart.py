"""St. Petersburg INCART 12-lead Arrhythmia DB — external-validation loader (Day 14).

INCART is a *completely external* test set: 75 records the model never saw and
that come from different patients, hardware, and country than MIT-BIH. Running
the DS1-trained EchoFuseNet on INCART with **no retraining** is the honest test
of cross-dataset generalisation.

Two differences from MIT-BIH must be reconciled so the *same* RP/GAF/MTF pipeline
and the *same* trained weights apply unchanged:

* **Sampling rate.** INCART is 257 Hz, MIT-BIH 360 Hz. Extracting a fixed *sample*
  window would give a different physical duration and a distorted morphology.
  Instead we take the same *physical* window as the MIT-BIH beat (128/360 s before
  and after the R-peak) and resample it to exactly ``TARGET_LEN`` (256) samples,
  so each beat yields the same 256×256 image the model trained on.
* **Leads.** INCART has 12 leads (I, II, III, aVR, …, V6); MIT-BIH training used
  MLII. Lead **II** is the closest analogue, so it is the default.

Beat symbols map through the same ``aami`` table. Records are named ``I01``…``I75``;
the numeric part is used as ``BeatSegment.record_id``. The DS1/DS2 split and the
leakage guard are MIT-BIH concepts and do **not** apply here — INCART is used
whole, as an external fold labelled ``"INCART"``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample

from .aami import class_index, symbol_to_aami
from .beats import WINDOW_AFTER, WINDOW_BEFORE, BeatSegment
from .mitbih import Beat, Record

# PhysioNet slug + default git-ignored destination (mirrors download.py).
PHYSIONET_DB = "incartdb"
DEFAULT_INCART_DEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "incartdb"

# All 75 INCART records: I01 … I75.
INCART_RECORDS: tuple[str, ...] = tuple(f"I{n:02d}" for n in range(1, 76))

# MIT-BIH reference sampling rate + physical window the model trained on.
MITBIH_FS: float = 360.0
BEFORE_S: float = WINDOW_BEFORE / MITBIH_FS   # ~0.356 s pre-R-peak
AFTER_S: float = WINDOW_AFTER / MITBIH_FS     # ~0.356 s post-R-peak
TARGET_LEN: int = WINDOW_BEFORE + WINDOW_AFTER  # 256 -> 256x256 images

_RECORD_EXTS = (".dat", ".hea", ".atr")


def _record_present(dest: Path, name: str) -> bool:
    return all((dest / f"{name}{ext}").exists() for ext in _RECORD_EXTS)


def download_incart(
    dest: Path = DEFAULT_INCART_DEST, records: tuple[str, ...] = INCART_RECORDS
) -> Path:
    """Download INCART records via wfdb (idempotent; skips present records)."""
    dest.mkdir(parents=True, exist_ok=True)
    to_fetch = [r for r in records if not _record_present(dest, r)]
    if not to_fetch:
        print(f"All {len(records)} INCART records already present in {dest}")
        return dest
    print(f"Downloading {len(to_fetch)} INCART record(s) to {dest} ...")
    wfdb.dl_database(PHYSIONET_DB, dl_dir=str(dest), records=to_fetch)
    print("Done.")
    return dest


def record_id_of(name: str) -> int:
    """Numeric id for an INCART record name (``"I07" -> 7``)."""
    return int(name.lstrip("I").lstrip("i"))


def load_incart_record(name: str, data_dir: Path = DEFAULT_INCART_DEST) -> Record:
    """Load one INCART record and its AAMI-mappable beats (12-lead signal)."""
    path = str(Path(data_dir) / name)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    beats: list[Beat] = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        aami = symbol_to_aami(symbol)
        if aami is None:
            continue
        beats.append(Beat(sample=int(sample), symbol=symbol, aami=aami))

    return Record(
        record_id=record_id_of(name),
        signal=np.asarray(rec.p_signal, dtype=np.float32),
        fs=float(rec.fs),
        channels=list(rec.sig_name),
        beats=beats,
    )


def _lead_index(record: Record, prefer: str = "II") -> int:
    """Index of the preferred lead, else the second lead (or 0 if 1-D)."""
    try:
        return record.channels.index(prefer)
    except ValueError:
        return 1 if record.signal.ndim == 2 and record.signal.shape[1] > 1 else 0


def _zscore(segment: np.ndarray) -> np.ndarray:
    """Per-beat z-score with a zero-variance guard (matches beats.py)."""
    mean = segment.mean()
    std = segment.std()
    if std < 1e-8:
        return (segment - mean).astype(np.float32)
    return ((segment - mean) / std).astype(np.float32)


def extract_incart_beats(
    record: Record,
    before_s: float = BEFORE_S,
    after_s: float = AFTER_S,
    target_len: int = TARGET_LEN,
    lead: str = "II",
    normalize: bool = True,
) -> list[BeatSegment]:
    """Extract resampled, AAMI-labeled beats from an INCART record.

    Each beat is the physical ``[before_s, after_s]`` window around the R-peak at
    INCART's native rate, resampled to ``target_len`` samples so it matches the
    MIT-BIH beat length (and therefore the model's expected image size). Beats
    whose window falls off the record edge are skipped.
    """
    fs = record.fs
    before = round(before_s * fs)
    after = round(after_s * fs)
    channel = _lead_index(record, lead)
    trace = record.signal[:, channel] if record.signal.ndim == 2 else record.signal
    n_samples = trace.shape[0]

    segments: list[BeatSegment] = []
    for beat in record.beats:
        start = beat.sample - before
        stop = beat.sample + after
        if start < 0 or stop > n_samples:
            continue
        seg = np.asarray(trace[start:stop], dtype=np.float64)
        if seg.shape[0] != target_len:
            seg = resample(seg, target_len)  # match MIT-BIH beat length
        seg = _zscore(seg) if normalize else seg.astype(np.float32)
        segments.append(
            BeatSegment(
                signal=seg,
                aami=beat.aami,
                label=class_index(beat.aami),
                record_id=record.record_id,
                r_peak=beat.sample,
                fold="INCART",
            )
        )
    return segments


def load_incart(
    data_dir: Path = DEFAULT_INCART_DEST,
    records: tuple[str, ...] | None = None,
    **extract_kwargs,
) -> list[BeatSegment]:
    """Load and extract all beats from the requested INCART records."""
    use = records if records is not None else INCART_RECORDS
    beats: list[BeatSegment] = []
    for name in use:
        record = load_incart_record(name, data_dir)
        beats.extend(extract_incart_beats(record, **extract_kwargs))
    return beats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download St. Petersburg INCART DB")
    parser.add_argument("--dest", type=Path, default=DEFAULT_INCART_DEST)
    args = parser.parse_args()
    download_incart(args.dest)


if __name__ == "__main__":
    main()
