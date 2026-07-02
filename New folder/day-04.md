# Day 4 — Gramian Angular Field (GAF) Transform

## Goal
Implement and validate the GAF signal-to-image transform.

## Tasks
- [x] Implement GAF transform (decide GASF vs GADF, or both, and document the choice)
- [x] Visualize sample GAF images across all 5 AAMI classes
- [x] Unit test output shape/range

## Deliverable / Definition of Done
- GAF transform module
- Sample GAF visualizations for each class

## Dependencies
Day 2 (beat extraction pipeline). Independent of Day 3.

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-02

**Status:** ✅ Done

**What I completed:**
- New `transforms/gaf.py`: `gramian_angular_field(signal, method="summation", sample_range=(-1, 1))`. Pure NumPy — min-max scales the signal into `sample_range`, maps to angles `phi = arccos(x)`, then builds the field from the angle sum/difference. **Default is GASF** (`cos(phi_i + phi_j)`, symmetric, signal on the diagonal) — the variant used by most GAF-based ECG classifiers; **GADF** (`sin(phi_i - phi_j)`, anti-symmetric) is available via `method="difference"`. Output is an `(L, L)` float32 image in `[-1, 1]` — 256x256 for the Day-2 beat.
- Scaling is done **inside** the transform (Day-3 note): Day-2 beats are z-scored/unbounded, so GAF can't assume `[-1, 1]` input. Constant signals map to the range midpoint instead of dividing by zero.
- Registered `gramian_angular_field` in `transforms/__init__.py`.
- Unit tests (`tests/test_gaf.py`, 11 tests, no data needed): shape/dtype, `[-1, 1]` value range, GASF symmetry, GADF anti-symmetry + zero diagonal, determinism, constant-signal edge case, invalid method/sample_range/empty-signal errors, and **cross-checks against `pyts.image.GramianAngularField`** for both GASF and GADF (match to 1e-5).
- Visualization script `scripts/visualize_gaf.py` (`--method summation|difference`): one beat per AAMI class from DS1, renders beat trace + GAF, saves `docs/figures/gaf_samples.png`. Generated with all 5 classes (N/S/Q rec 101, V rec 106, F rec 108).
- **Full test suite: 43 passing** (up from 32).

**Blockers / issues:**
- None. The GASF images are visually class-discriminative — N/S show a strong warm R-peak block, V a cool aberrant field, F a mixed fusion texture, Q the notched double-cross paced morphology.

**Notes for next day (Day 5 — MTF transform):**
- Mirror this layout once more: `transforms/mtf.py` exporting `markov_transition_field(signal, n_bins=..., ...)`, register it in `transforms/__init__.py`, add `tests/test_mtf.py` (shape/range/determinism + pyts `MarkovTransitionField` cross-check), and add `scripts/visualize_mtf.py`.
- MTF quantizes amplitudes into bins and builds a transition-probability field, so unlike RP/GAF it is **not** symmetric and its diagonal is not fixed — assert the right invariants (row/window structure, `[0, 1]` range) rather than symmetry.
- After Day 5 all three modalities exist; Day 6 composes RP+GAF+MTF into the multimodal DataLoader — keep the `(1-D array) -> 2-D array` signature uniform.

**Time spent:** ~1 session (GAF transform + tests + per-class visualization)
