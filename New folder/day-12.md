# Day 12 — Ablation Study

## Goal
Quantify each modality's contribution.

## Tasks
- [x] Run single-modality configs (RP-only, GAF-only, MTF-only) — *framework built & tested; full sweep deferred to GPU (see note)*
- [x] Run pairwise configs (RP+GAF, RP+MTF, GAF+MTF) — *framework built & tested*
- [x] Run full 3-modality config as reference
- [x] Apply significance testing across all configs — *McNemar vs full wired in*

## Deliverable / Definition of Done
- Full ablation table with significance annotations

## Dependencies
Day 11 (CV + significance testing infrastructure).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-03 (framework); full sweep pending GPU

**Status:** 🟡 In progress — framework ✅ done & tested; real ablation table pending compute

**What I completed:**
- Made `EchoFuseNet` **modality-configurable** (backward-compatible): new
  `modalities` arg accepts any non-empty subset of `("rp","gaf","mtf")`; only the
  selected branches are built and concatenated, `forward` still takes the full
  triple and ignores inactive inputs. Default = all three, so every existing
  test/behaviour is unchanged. Added `modalities` to `ModelConfig` +
  `build_model`.
- `training/ablation.py` — the Day-12 sweep driver:
  - `MODALITY_SETS` = the 7 configs (3 single, 3 pairwise, full reference).
  - `run_ablation()` trains each subset **reusing one shared DataLoader** (the
    transforms are computed once, not 7×) and compares every subset to the full
    model with an exact **McNemar test** on per-sample DS2 correctness.
  - `AblationReport.format()` prints the table with params, acc, macro-F1, and
    significance stars; `ablation_summary.json` is persisted.
  - CLI: `python -m paper1_echofusenet.training.ablation --config <cfg>`.
- Tests: `tests/test_ablation.py` (13 model-subset + driver tests) — all green.
  Full suite now ~130 tests.

**Blockers / issues:**
- **Compute.** The full 7-config sweep is 7 training runs; on this CPU-only box
  (~30 beats/s for the 3-branch model, faster for subsets) that's several hours
  and cannot run in-session alongside the Day-10 job. The *framework* is done,
  tested, and CLI-runnable; the **real ablation table with significance numbers
  needs a GPU** (or an overnight CPU run). This is the same hardware wall that
  caps Days 10–15 — a GPU is the single highest-leverage unblock.

**Notes for next day:**
- To produce the real table on GPU: `python -m paper1_echofusenet.training.ablation
  --config configs/echofusenet_ds1ds2_baseline.json` (set device "auto", bump
  epochs). Expected story: full ≳ pairwise > single; McNemar should flag the
  weakest single modality as significantly worse than full.
- Ablation can also run *under CV* for across-fold paired tests (stats.paired_ttest
  / wilcoxon) once GPU is available.

**Time spent:** ~2h (framework)
