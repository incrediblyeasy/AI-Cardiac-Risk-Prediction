# Day 9 — Training Loop & Config System

## Goal
Build a reproducible, config-driven training script.

## Tasks
- [x] Build train.py with config-driven hyperparameters (no hardcoded constants)
- [x] Implement loss function, optimizer, LR schedule, checkpointing
- [x] Add metrics logging (accuracy, per-class F1, confusion matrix; TensorBoard/W&B optional)

## Deliverable / Definition of Done
- Working training script + at least one config file

## Dependencies
Day 8 (assembled model) + Day 6 (data pipeline).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-03

**Status:** ✅ Done

**What I completed:**
- New `paper1_echofusenet/training/` package with three modules:
  - `config.py` — nested dataclass config (`TrainConfig` → data/model/optim/train),
    JSON (de)serialisation, and strict unknown-key rejection so config typos fail
    loudly. No hyperparameter is hardcoded in the loop; a run is fully described
    by `config.json` + seed.
  - `metrics.py` — confusion matrix, per-class precision/recall/F1, and macro-F1
    (the honest headline metric for imbalanced MIT-BIH), NumPy-only, with
    divide-by-zero guards (0.0, never NaN). macro-F1 averages only over classes
    with support > 0.
  - `train.py` — CrossEntropy (optional class weighting + label smoothing),
    AdamW/Adam/SGD, cosine/step/none LR schedule, optional grad clipping;
    best.pt + last.pt checkpointing (weights/optimizer/epoch/config), a config.json
    snapshot, and per-epoch `history.jsonl`. Optional TensorBoard (silently skipped
    if unavailable). CLI: `python -m paper1_echofusenet.training.train --config ...`
    with `--epochs`/`--out-dir` overrides. `train()`/`evaluate()` take loaders+model
    directly so they're unit-testable without downloaded data.
- Two config files: `configs/echofusenet_default.json` (full 30-epoch DS1/DS2 run)
  and `configs/echofusenet_smoke.json` (2-record, 2-epoch fast check).
- Tests: `tests/test_metrics.py`, `tests/test_train_config.py`, `tests/test_train.py`
  (17 new). Full suite = **96 passed**.
- Verified the CLI end-to-end on the smoke config: data loaded, trained, printed
  per-class report + confusion matrix, and wrote best.pt/last.pt/config.json/
  history.jsonl. (Smoke metrics are meaningless by design — 2 N-dominated records.)

**Blockers / issues:**
- `training/__init__.py` re-exports a `train` function that shadows the `train`
  submodule — import the function by name (`from ...training.train import train`),
  not via a module alias. Noted so Day 10 doesn't trip on it.

**Notes for next day:**
- Day 10 = first full DS1/DS2 run: `python -m paper1_echofusenet.training.train
  --config configs/echofusenet_default.json`. Watch the macro-F1 band and the S/V
  recall — the smoke run collapsed S entirely (expected on 2 records; the real
  run has the full DS1 distribution + oversampling).
- Consider adding a validation split carved from DS1 for best-checkpoint selection
  so `best.pt` isn't chosen on the DS2 test fold (currently evaluated on DS2 for
  convenience; fine for Day 9 plumbing, revisit for honest early-stopping).

**Time spent:** ~2.5h
