# Day 1 — Environment & Repo Setup + Data Acquisition

## Goal
Get the repo scaffolded, dependencies installed, and MIT-BIH data loaded with the DS1/DS2 protocol documented.

## Tasks
- [x] Create repo structure per CLAUDE.md (paper1_echofusenet/, shared/, docs/)
- [x] Install deps: torch, numpy, scipy, wfdb, pyts (or equivalent RP/GAF/MTF library), pytest
- [x] Download MIT-BIH Arrhythmia Database
- [x] Document the de Chazal DS1/DS2 patient-ID split (which patients go in DS1 vs DS2) in a config/reference file
- [x] Verify AAMI class mapping (N, S, V, F, Q) against MIT-BIH annotation codes

## Deliverable / Definition of Done
- Working environment (requirements.txt / pyproject.toml committed)
- Raw MIT-BIH data downloaded and loadable
- DS1/DS2 patient ID lists documented in code

## Dependencies
None — this is the starting point.

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-01

**Status:** ✅ Done

**What I completed:**
- Scaffolded the repo per CLAUDE.md: `paper1_echofusenet/` (with `data/`), `shared/`, `docs/`, `tests/`, plus `.gitignore`, `requirements.txt`, `pyproject.toml`, and a new `CLAUDE.md` (it did not exist yet).
- Installed and validated deps on Python 3.11: torch 2.12, numpy 2.4, scipy 1.17, pytest 9.1 (pre-existing) + wfdb 4.3.1 and pyts 0.13.0 (newly installed).
- Downloaded the full MIT-BIH Arrhythmia DB (48 records: .dat/.hea/.atr) via a scripted, idempotent downloader (`paper1_echofusenet/data/download.py`) into `data/raw/mitdb/` (git-ignored).
- Documented the de Chazal DS1/DS2 inter-patient split in code (`data/splits.py`) and in `docs/ds1_ds2_split.md`: DS1 = 22 train records, DS2 = 22 test records, paced 102/104/107/217 excluded. Includes `assert_no_leakage()` self-check.
- Encoded and verified the AAMI 5-class mapping (N/S/V/F/Q) in `data/aami.py` + `docs/aami_mapping.md`, checked against MIT-BIH annotation symbols.
- Wrote a loader (`data/mitbih.py`) that returns signal + metadata + AAMI-labeled beats. Verified end-to-end: rec 101 (DS1) → 1865 beats, rec 100 (DS2) → 2273 beats, both fs=360 Hz, 2 leads.
- Test suite: **12 passing** (`pytest`) — split integrity, AAMI mapping, and real data-load smoke tests (auto-skip when data absent).

**Blockers / issues:**
- None. Note the repo lives inside a large drive-wide git repo; actual dataset files are git-ignored and must never be committed.

**Notes for next day (Day 2):**
- Split logic already exists in `data/splits.py`; Day 2 extends the leakage guard from patient-id lists to *extracted beats* (assert no patient appears in both folds) and adds R-peak-centered windowing. Beat annotations are already exposed as `Beat(sample, symbol, aami)` via `mitbih.load_record`.
- MIT-BIH R-peaks are the annotation `sample` indices; window around those. Signal channel 0 is MLII for all records except a couple — confirm lead selection per record on Day 2.
- Class Q/F/S are very rare per-record (expected under de Chazal) — keep this in mind for the oversampling decision on Day 6, not before the split.

**Time spent:** ~1 session (setup + data acquisition + tests)
