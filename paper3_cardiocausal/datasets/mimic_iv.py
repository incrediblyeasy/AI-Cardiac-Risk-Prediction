"""MIMIC-IV-ECG + EHR linkage — real algorithm, credentialed file I/O gated.

Split by readiness:

* ``link_ecg_ehr`` / ``subject_level_split`` (**implemented + tested**) — the
  linkage *algorithm*: join ECG records to per-subject EHR, apply eligibility +
  the time-zero rule (keep each subject's first eligible ECG), and split at the
  **subject** level so no subject appears in two folds (the same inter-patient
  discipline Paper 1 enforces). These take already-loaded records, so they run on
  synthetic data now and on real MIMIC-IV once it's available.

* ``build_linked_cohort`` (**gated**) — the thin wrapper that *reads* MIMIC-IV
  files off disk. MIMIC-IV is credentialed PhysioNet data (identity verification +
  CITI training), a lead-time bottleneck to start early (roadmap §4.1). Only the
  file-reading is gated; the linkage logic it would call is already done.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np


def link_ecg_ehr(
    ecg_records: Sequence[Mapping[str, Any]],
    ehr_by_subject: Mapping[Any, Mapping[str, Any]],
    eligibility: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Build cohort rows by linking ECGs to EHR under the time-zero rule.

    Parameters
    ----------
    ecg_records:
        Each a mapping with at least ``subject_id`` and ``time`` (comparable), plus
        any ECG payload (e.g. a waveform reference or precomputed representation).
    ehr_by_subject:
        ``subject_id -> EHR mapping`` (covariates, treatment, outcome, follow-up).
        ECGs whose subject is absent here are dropped (no linkage).
    eligibility:
        Optional ``(ecg_record, ehr) -> bool`` predicate. ECGs failing it are
        skipped *before* the time-zero selection, so time-zero is the first
        *eligible* ECG.

    Returns one row per subject — the earliest eligible ECG (**time zero**) merged
    with that subject's EHR — sorted by ``subject_id`` for determinism.
    """
    by_subject: dict[Any, Mapping[str, Any]] = {}
    for rec in ecg_records:
        sid = rec["subject_id"]
        ehr = ehr_by_subject.get(sid)
        if ehr is None:
            continue
        if eligibility is not None and not eligibility(rec, ehr):
            continue
        # Keep the earliest eligible ECG per subject (time-zero alignment).
        current = by_subject.get(sid)
        if current is None or rec["time"] < current["time"]:
            by_subject[sid] = rec

    rows: list[dict[str, Any]] = []
    for sid in sorted(by_subject):
        row: dict[str, Any] = {"subject_id": sid}
        row.update(dict(ehr_by_subject[sid]))
        row.update(dict(by_subject[sid]))  # ECG fields win on key clash
        rows.append(row)
    return rows


def subject_level_split(
    rows: Sequence[Mapping[str, Any]],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Partition rows into train/val/test with **no subject in two folds**.

    Splitting is on unique ``subject_id`` (not rows), so the inter-patient
    guarantee holds even if a subject contributes multiple rows. Deterministic for
    a given ``seed``.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1; got {fractions}")
    subjects = sorted({r["subject_id"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    assign = {
        **{s: "train" for s in subjects[:n_train]},
        **{s: "val" for s in subjects[n_train : n_train + n_val]},
        **{s: "test" for s in subjects[n_train + n_val :]},
    }
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for r in rows:
        out[assign[r["subject_id"]]].append(dict(r))
    return out


def build_linked_cohort(*args: Any, **kwargs: Any):
    """Read MIMIC-IV files and link them — gated on credentialed access.

    The linkage/splitting logic is already implemented (``link_ecg_ehr``,
    ``subject_level_split``); only reading the credentialed files is gated.
    """
    raise NotImplementedError(
        "Reading MIMIC-IV off disk requires credentialed PhysioNet access "
        "(identity verification + CITI training) — start it now (roadmap §4.1). "
        "Once files are present: load MIMIC-IV-ECG records + EHR tables, then call "
        "link_ecg_ehr(...) and subject_level_split(...) (both already implemented "
        "and tested)."
    )
