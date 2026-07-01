"""Load MIT-BIH Arrhythmia records and enumerate AAMI-labeled beats.

Day-1 scope: prove the raw data is loadable and that annotations map cleanly to
AAMI classes. Beat *windowing* around R-peaks (fixed-length segments) is a Day-2
concern and lives with the split/extraction pipeline; here we only expose the
raw signal, header metadata, and the list of beat annotations with their AAMI
labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from .aami import symbol_to_aami
from .download import DEFAULT_DEST


@dataclass
class Beat:
    """A single annotated heartbeat."""

    sample: int          # R-peak sample index into the signal
    symbol: str          # raw MIT-BIH annotation symbol
    aami: str            # mapped AAMI class (N/S/V/F/Q)


@dataclass
class Record:
    """A loaded MIT-BIH record."""

    record_id: int
    signal: np.ndarray   # shape (n_samples, n_channels); channel 0 is MLII
    fs: float            # sampling frequency (Hz)
    channels: list[str]  # signal lead names
    beats: list[Beat]    # AAMI-mappable beats only

    @property
    def n_beats(self) -> int:
        return len(self.beats)


def load_record(record_id: int, data_dir: Path = DEFAULT_DEST) -> Record:
    """Load one MIT-BIH record and its AAMI-labeled beats.

    Non-beat annotations (rhythm/quality markers) and symbols with no AAMI
    mapping are dropped, so ``Record.beats`` contains only classifiable beats.
    """
    path = str(Path(data_dir) / str(record_id))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    beats: list[Beat] = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        aami = symbol_to_aami(symbol)
        if aami is None:
            continue
        beats.append(Beat(sample=int(sample), symbol=symbol, aami=aami))

    return Record(
        record_id=record_id,
        signal=np.asarray(rec.p_signal, dtype=np.float32),
        fs=float(rec.fs),
        channels=list(rec.sig_name),
        beats=beats,
    )


def class_distribution(record: Record) -> dict[str, int]:
    """Count beats per AAMI class within a record."""
    counts: dict[str, int] = {}
    for beat in record.beats:
        counts[beat.aami] = counts.get(beat.aami, 0) + 1
    return counts
