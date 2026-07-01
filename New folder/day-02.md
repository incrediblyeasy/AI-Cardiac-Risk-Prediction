# Day 2 — Patient-Level Split & Leakage Guard

## Goal
Implement the inter-patient split correctly and build an automated check that makes leakage impossible to miss.

## Tasks
- [x] Implement patient-level split logic strictly following DS1/DS2
- [x] Build a leakage-check utility that asserts zero patient-ID overlap between train and test
- [x] Extract beat segments per patient with correct windowing around R-peaks
- [x] Add this leakage check as a unit test that runs on every future data pipeline change

## Deliverable / Definition of Done
- Split module producing DS1 (train) / DS2 (test) patient sets
- Passing automated leakage test
- Beat extraction pipeline producing labeled beat segments

## Dependencies
Day 1 (raw data + DS1/DS2 documentation).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-01

**Status:** ✅ Done

**What I completed:**
- Added patient-level fold helpers to `data/splits.py`: `records_for_fold("DS1"|"DS2")` plus a general `assert_patient_disjoint(train, test)` leakage guard that works on any id iterables (record lists *or* extracted-beat ids).
- New `data/beats.py`: R-peak-windowed beat extraction. Each beat → fixed 256-sample window (`WINDOW_BEFORE=128`, `WINDOW_AFTER=128`) centered on the R-peak; per-beat z-score normalization (with zero-variance guard); MLII lead auto-selected (falls back to channel 0), which correctly handles records that list MLII second (e.g. 114). Edge beats whose window overruns the signal are skipped, never truncated.
- `load_fold(fold, records=None)` assembles a fold and rejects any record not belonging to it; `build_split()` builds (DS1, DS2) beat sets and runs the leakage guard on the *extracted beats* — so training on a test patient is structurally impossible.
- Leakage check added as unit tests (`tests/test_leakage.py`) that run on every `pytest` invocation: list-level always, beat-level on a fast 2-records-per-fold subset. Plus `tests/test_beats.py` covering window length, R-peak centering, valid labels, edge-skipping, and normalization.
- **Test suite: 23 passing** (up from 12).
- Full-split sanity on real data: DS1 = 51,000 beats / 22 patients, DS2 = 49,693 beats / 22 patients, **zero patient overlap**. Class distribution matches the expected de Chazal imbalance — train N/S/V/F/Q = 45846/944/3788/414/8, test = 44241/1837/3220/388/7.

**Blockers / issues:**
- None. Class Q is near-empty (8 train / 7 test) because the paced records that dominate Q are excluded under the inter-patient protocol — expected, not a bug. Keep this in mind for the Day 6 oversampling decision.

**Notes for next day (Day 3 — RP transform):**
- Feed the RP/GAF/MTF transforms the per-beat 1-D signal `BeatSegment.signal` (length 256, float32, already z-scored). `beats.extract_beats(record, fold)` or `beats.build_split()` gives labeled segments.
- For the Day-3 per-class visualizations, pull one `BeatSegment` per AAMI class from DS1; Q will be scarce (only 8 beats) so grab whatever exists.
- Normalization is ON by default — if a transform prefers raw amplitudes, pass `normalize=False`.
- Window length 256 is a config default (`WINDOW_BEFORE`/`WINDOW_AFTER`); revisit only if a transform needs a different size.

**Time spent:** ~1 session (split logic + beat extraction + leakage tests)
