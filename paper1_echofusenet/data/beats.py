"""R-peak-windowed beat extraction and patient-level fold assembly.

Day-2 scope: turn loaded MIT-BIH records into fixed-length, AAMI-labeled beat
segments, and assemble the inter-patient DS1 (train) / DS2 (test) folds with a
hard leakage guard.

Windowing
---------
Each beat is a fixed-length window of ``WINDOW_BEFORE + WINDOW_AFTER`` samples
taken around the annotated R-peak. The R-peak sits at index ``WINDOW_BEFORE``
inside the segment. Beats whose window would fall off either end of the record
are skipped (they cannot yield a full-length segment).

The lead used is **MLII** when the record has it (all MIT-BIH records except a
handful list MLII as channel 0; e.g. record 114 lists it second), otherwise
channel 0. This keeps the morphology consistent across records.

Leakage
-------
Folds are built strictly from the de Chazal DS1/DS2 record lists, and
``build_split`` calls the patient-disjoint guard on the *extracted* beats — not
just the record lists — so any accidental cross-fold beat is caught here, not
silently trained on. Oversampling/resampling is explicitly NOT done here; that
belongs to the training fold only and lands on Day 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .aami import class_index
from .download import DEFAULT_DEST
from .mitbih import Record, load_record
from .splits import assert_patient_disjoint, records_for_fold

# Default window around the R-peak (samples). MIT-BIH is 360 Hz, so 128+128
# spans ~0.71 s — enough to cover the QRS complex plus surrounding P/T context,
# and a clean size for the Day 3-5 signal-to-image transforms.
WINDOW_BEFORE = 128
WINDOW_AFTER = 128


@dataclass
class BeatSegment:
    """One fixed-length, AAMI-labeled heartbeat."""

    signal: np.ndarray   # shape (WINDOW_BEFORE + WINDOW_AFTER,), float32
    aami: str            # AAMI class string (N/S/V/F/Q)
    label: int           # integer class label (0..4)
    record_id: int       # source patient/record — used by the leakage guard
    r_peak: int          # R-peak sample index in the original record
    fold: str            # "DS1" or "DS2"


def _lead_index(record: Record, prefer: str = "MLII") -> int:
    """Index of the preferred lead, or 0 if the record does not carry it."""
    try:
        return record.channels.index(prefer)
    except ValueError:
        return 0


def _zscore(segment: np.ndarray) -> np.ndarray:
    """Per-beat z-score normalization with a zero-variance guard."""
    mean = segment.mean()
    std = segment.std()
    if std < 1e-8:
        return (segment - mean).astype(np.float32)
    return ((segment - mean) / std).astype(np.float32)


def extract_beats(
    record: Record,
    fold: str,
    before: int = WINDOW_BEFORE,
    after: int = WINDOW_AFTER,
    lead: str = "MLII",
    normalize: bool = True,
) -> list[BeatSegment]:
    """Extract fixed-length beat segments from a loaded record.

    Beats whose R-peak window would exceed the signal bounds are skipped.
    """
    channel = _lead_index(record, lead)
    trace = record.signal[:, channel]
    n_samples = trace.shape[0]
    window = before + after

    segments: list[BeatSegment] = []
    for beat in record.beats:
        start = beat.sample - before
        stop = beat.sample + after
        if start < 0 or stop > n_samples:
            continue  # incomplete window at the record edge
        seg = trace[start:stop]
        if seg.shape[0] != window:  # defensive; should not happen
            continue
        if normalize:
            seg = _zscore(seg)
        else:
            seg = seg.astype(np.float32)
        segments.append(
            BeatSegment(
                signal=seg,
                aami=beat.aami,
                label=class_index(beat.aami),
                record_id=record.record_id,
                r_peak=beat.sample,
                fold=fold,
            )
        )
    return segments


def load_fold(
    fold: str,
    data_dir: Path = DEFAULT_DEST,
    records: tuple[int, ...] | None = None,
    **extract_kwargs,
) -> list[BeatSegment]:
    """Load and extract all beats for a fold (``"DS1"`` or ``"DS2"``).

    ``records`` overrides the fold's record list (useful for fast tests). All
    ids must still belong to the requested fold.
    """
    fold_records = records_for_fold(fold)
    use = fold_records if records is None else records
    unknown = set(use) - set(fold_records)
    if unknown:
        raise ValueError(f"records {sorted(unknown)} are not in fold {fold}")

    beats: list[BeatSegment] = []
    for record_id in use:
        record = load_record(record_id, data_dir)
        beats.extend(extract_beats(record, fold, **extract_kwargs))
    return beats


def build_split(
    data_dir: Path = DEFAULT_DEST,
    train_records: tuple[int, ...] | None = None,
    test_records: tuple[int, ...] | None = None,
    **extract_kwargs,
) -> tuple[list[BeatSegment], list[BeatSegment]]:
    """Build the (train, test) beat sets and enforce the leakage guard.

    Returns ``(ds1_beats, ds2_beats)``. Raises ``AssertionError`` if any patient
    id ends up in both folds — the guard runs on the extracted beats, so it is
    impossible to silently train on a test patient.
    """
    train = load_fold("DS1", data_dir, records=train_records, **extract_kwargs)
    test = load_fold("DS2", data_dir, records=test_records, **extract_kwargs)

    assert_patient_disjoint(
        {b.record_id for b in train},
        {b.record_id for b in test},
    )
    return train, test


def class_counts(beats: list[BeatSegment]) -> dict[str, int]:
    """Count extracted beats per AAMI class."""
    counts: dict[str, int] = {}
    for beat in beats:
        counts[beat.aami] = counts.get(beat.aami, 0) + 1
    return counts
