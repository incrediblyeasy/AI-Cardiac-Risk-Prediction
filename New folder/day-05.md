# Day 5 — Markov Transition Field (MTF) Transform

## Goal
Implement and validate the MTF signal-to-image transform, and confirm all three representations are genuinely distinct.

## Tasks
- [x] Implement MTF transform
- [x] Visualize sample MTF images across all 5 AAMI classes
- [x] Unit test output shape/range
- [x] Write a distinctness check: confirm RP, GAF, MTF outputs for the same beat are NOT duplicates of each other (this guards against the channel-duplication defect from the original draft)

## Deliverable / Definition of Done
- MTF transform module
- Distinctness verification report (RP vs GAF vs MTF are physically different per beat)

## Dependencies
Day 2 (beat extraction pipeline). Independent of Days 3-4.

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-02

**Status:** ✅ Done

**What I completed:**
- New `transforms/mtf.py`: `markov_transition_field(signal, n_bins=8, strategy="quantile")`. Pure NumPy — quantizes amplitudes into bins (quantile/uniform/normal), builds the row-normalized `n_bins x n_bins` Markov transition matrix from successive-sample transitions, then spreads it back over time as `M[i,j] = W[q_i, q_j]`. Output is `(L, L)` float32 in `[0, 1]` — 256x256 for the Day-2 beat.
- Unlike RP/GAF, the MTF is **not symmetric** and its diagonal is not fixed (directional transition probabilities), so the tests assert the *right* invariants: `[0, 1]` probability range and asymmetry rather than symmetry.
- Registered `markov_transition_field` in `transforms/__init__.py` — all three modalities now exported.
- Unit tests (`tests/test_mtf.py`, 10 tests, no data needed): shape/dtype, `[0, 1]` range, asymmetry, determinism, constant-signal edge case (single state → all ones), invalid `n_bins`/`strategy`/too-short errors, and **cross-checks against `pyts.image.MarkovTransitionField`** for both quantile and uniform strategies (match to 1e-5).
- **Distinctness guard** (`tests/test_distinctness.py`, 3 tests): computes RP+GAF+MTF for the same beat, confirms identical shape (stackable into channels), asserts no pair is element-wise equal, and that every pair's min-max-normalized correlation is `< 0.95` — this locks out the channel-duplication defect from the original draft.
- Visualization script `scripts/visualize_mtf.py` → `docs/figures/mtf_samples.png` (one beat per AAMI class, all 5 present).
- **Distinctness verification report** `scripts/check_distinctness.py` → prints a per-class pairwise-correlation table and saves `docs/figures/transforms_comparison.png` (RP/GAF/MTF rows × 5 class columns). Result: **max pairwise correlation 0.758 → DISTINCT**. RP-GAF are the most correlated pair (~0.67–0.76, both smooth similarity fields); anything involving MTF is far lower (≤0.44), confirming it adds independent information.
- **Full test suite: 56 passing** (up from 43).

**Blockers / issues:**
- None. The comparison figure makes the three modalities visibly different per class: RP smooth distance texture, GAF angular correlation blocks, MTF sparse transition structure.

**Notes for next day (Day 6 — multimodal DataLoader):**
- All three transforms now share the uniform `(1-D array) -> (L, L) 2-D array` signature and are exported from `paper1_echofusenet.transforms`, so Day 6 can stack them into a 3-channel tensor per beat directly.
- Mind the differing native ranges when composing channels: RP `[0,1]`, GAF `[-1,1]`, MTF `[0,1]`. Decide per-channel normalization in the DataLoader (e.g. rescale GAF to `[0,1]` or standardize each channel) rather than feeding mixed ranges to the CNN.
- Keep transforms on the **training fold only after** the patient split (leakage guard from Day 2 still applies to any augmentation).

**Time spent:** ~1 session (MTF transform + tests + distinctness guard/report + per-class visualization)
