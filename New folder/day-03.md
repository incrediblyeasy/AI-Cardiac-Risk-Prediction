# Day 3 — Recurrence Plot (RP) Transform

## Goal
Implement and validate the RP signal-to-image transform.

## Tasks
- [x] Implement RP transform from an ECG beat signal
- [x] Visualize sample RP images across all 5 AAMI classes
- [x] Unit test: verify output shape and value range are consistent

## Deliverable / Definition of Done
- RP transform module
- Sample RP visualizations for each class saved to docs/ or notebooks/

## Dependencies
Day 2 (beat extraction pipeline).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-01

**Status:** ✅ Done

**What I completed:**
- New `transforms/` package with `transforms/rp.py`: `recurrence_plot(signal, dimension=1, time_delay=1, threshold=None, normalize=True)`. Implemented in NumPy — Takens time-delay embedding + pairwise Euclidean distance matrix, min-max scaled to [0,1] (unthresholded/grayscale default) or binary when a threshold is given. Output is a symmetric (L, L) float32 image; for the Day-2 256-sample beat that's 256x256.
- Unit tests (`tests/test_rp.py`, 9 tests, no data needed): output shape (with and without embedding), symmetry + zero diagonal, [0,1] value range, binary/reflexive thresholded mode, determinism, constant-signal edge case, too-short-for-embedding error, and a **cross-check against `pyts.image.RecurrencePlot`** (matches to 1e-5).
- Visualization script `scripts/visualize_rp.py`: scans DS1, grabs one beat per AAMI class, renders beat trace + RP, saves `docs/figures/rp_samples.png`. Generated with all 5 classes present (N/S from rec 101, V rec 106, F rec 108, Q rec 101).
- Added `matplotlib` to `requirements.txt` and the `dev` extra.
- **Full test suite: 32 passing** (up from 23).

**Blockers / issues:**
- None. The RP images are visually class-discriminative: N/S show a sharp R-peak cross, V a broad aberrant texture, F a fusion pattern, Q the notched paced-style morphology — good sign the modality carries signal for the CNN.

**Notes for next day (Day 4 — GAF transform):**
- Mirror this module layout: add `transforms/gaf.py` exporting `gramian_angular_field(signal, ...)`, register it in `transforms/__init__.py`, add `tests/test_gaf.py` (shape/range/determinism + pyts `GramianAngularField` cross-check), and extend the viz to a GAF row (or a `scripts/visualize_gaf.py`).
- GAF expects the signal scaled to [-1, 1] before the arccos step; the Day-2 beats are z-scored (not bounded), so scale inside the transform (min-max to [-1,1]) rather than assuming input range.
- Keep transforms pure `(1-D array) -> 2-D array` so Day 6 can compose RP+GAF+MTF into the multimodal DataLoader uniformly.

**Time spent:** ~1 session (RP transform + tests + per-class visualization)
