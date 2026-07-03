# Day 11 — k-Fold Cross-Validation + Statistical Significance

## Goal
Add statistical rigor on top of the baseline.

## Tasks
- [ ] Implement k-fold CV on top of the inter-patient protocol (fold across training patients, keep test protocol intact)
- [ ] Add statistical significance testing across folds (e.g. bootstrap CIs or paired significance test)

## Deliverable / Definition of Done
- k-fold CV results with confidence intervals / significance report

## Dependencies
Day 10 (baseline training run working).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-03 (framework + DS2 CIs); full CV deferred to GPU

**Status:** 🟡 Framework ✅ done, tested, and CIs computed on the real model; full 5-fold CV deferred (GPU)

**What I completed:**
- `training/crossval.py` — patient-grouped k-fold (folds partition DS1 *patients*,
  never beats → inter-patient preserved per fold; DS2 untouched) + `cross_validate`
  driver with per-fold train/eval and CI aggregation.
- `training/stats.py` — Student-t CI across folds, percentile **bootstrap CI**,
  paired t / Wilcoxon (across folds), exact **McNemar** (per sample).
- `scripts/evaluate_ci.py` — bootstrap CIs on the final DS2 model. **Result
  (2000 resamples):** accuracy 0.9293 **[0.9271, 0.9315]**, macro-F1 0.3793
  **[0.3748, 0.3844]**. Artifact `runs/ds1ds2_baseline/ds2_bootstrap_ci.json`.
- Tests: `tests/test_stats.py`, `tests/test_crossval.py` — all green.

**Blockers / issues:**
- Full 5-fold CV = 5 training runs ≈ 7.5h on this CPU-only box → deferred to GPU.
  The framework is delivered + tested + CLI-runnable; the bootstrap CIs above give
  a real significance number now without the 7.5h retrain.

**Notes for next day:**
- On GPU: `python -m paper1_echofusenet.training.crossval --config
  configs/echofusenet_ds1ds2_baseline.json --folds 5` for mean±CI across folds.

**Time spent:** ~2.5h (framework + CI run).
