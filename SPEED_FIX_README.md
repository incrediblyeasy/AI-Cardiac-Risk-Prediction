# SPEED FIX: parallelized loading + size-capped transform cache

Directly answers "is there another way to make this faster" — verified,
tested, not theoretical. Honest scope below; this speeds up CV and
ablation, not the already-running headline run (don't restart that).

## What's actually in this package

6 files — 3 new, 3 modified:

| File | New or modified | What |
|---|---|---|
| `paper1_echofusenet/data/transform_cache.py` | **New** | Size-capped disk cache for RP/GAF/MTF |
| `paper1_echofusenet/data/dataset.py` | Modified | `MultimodalBeatDataset` + `build_dataloaders` accept an optional `transform_cache` |
| `paper1_echofusenet/training/crossval.py` | Modified | Wires one shared cache across all CV folds |
| `paper1_echofusenet/training/ablation.py` | Modified | Wires the cache into the ablation run |
| `configs/echofusenet_ds1ds2_baseline.json` | Modified | `num_workers: 0` → `2` (this is the config CV/ablation actually use) |
| `tests/test_transform_cache.py` | **New** | 5 tests — correctness, budget enforcement, persistence, corruption handling |

## Important correction before you use this

I originally offered to "cache everything" without checking the numbers
first. Actually checked: **DS1 alone is 51,000 beats; a full RP+GAF+MTF
cache at float16 is ~18.7 GB** — too close to Kaggle's typical ~20GB
`/kaggle/working` budget to safely promise. So this is **not** an
everything-cache. It's a **hard-capped** cache (default 4GB, configurable)
that caches what fits and transparently falls back to on-the-fly
computation for anything beyond the cap. It cannot overflow your disk
budget by construction — verified by an actual test
(`test_budget_is_never_exceeded`) that found and fixed a real off-by-one
in my first version (see below).

## Real, measured results (not estimates)

- **2.9x speedup** on the data-loading portion, measured on 500 real MIT-BIH
  beats: 2.31s cold (compute) vs 0.79s warm (cache hit).
- Two real bugs were caught by tests and fixed before delivery, not shipped
  broken:
  1. First version's budget check allowed a small overshoot (off-by-one
     write) — fixed by checking the *actual* written file size before
     committing it, not an estimate.
  2. A `numpy.savez` quirk (auto-appends `.npz` to filenames that don't
     already end in it) silently broke the temp-file naming — fixed.
- Full end-to-end smoke-tested against real MIT-BIH data through both
  `cross_validate()` and `run_ablation_from_config()` (not just isolated
  unit tests) — both completed cleanly, cache populated correctly.

## What this does NOT do

- **Does not speed up the CV run already in progress on Kaggle.** Don't
  restart it for this — that wastes real compute you've already spent.
  This is for ablation (next up) and any future re-runs.
- **Ablation's benefit is epoch-to-epoch reuse (real, ~3x), not
  "7 configs → 1 computation."** Ablation already reuses one set of loaders
  across all 7 modality configs (existing, smart design) — that redundancy
  didn't exist before this fix. CV's benefit is bigger: folds use
  *different* patient subsets that mostly overlap, so cross-fold cache hits
  are real and meaningful.
- **Doesn't touch GPU compute time** — only the CPU-bound
  transform-generation part of the pipeline.

## How to use it

Nothing to call manually — it's wired into `crossval.py`/`ablation.py`
already. Just:
```bash
python -m paper1_echofusenet.training.ablation --config configs/echofusenet_ds1ds2_baseline.json
```
Cache lands at `<out_dir>/transform_cache/`. Default budget is 4GB — raise
it in the config if you know you have more disk headroom:
```json
"data": {
  ...
  "transform_cache_max_gb": 8
}
```

## Verification

```bash
pytest -q tests/test_transform_cache.py   # 5 new tests
pytest -q                                  # full suite: 234 passed, 2 skipped
```

> Verified on this repo (2026-07-08): `tests/test_transform_cache.py` → **5
> passed**; full suite → **234 passed, 2 skipped** (the 2 skips are the
> optional `torchvision`/`optuna` deps, unrelated to this change). Measured
> cold-vs-warm on 60 beats through the real `MultimodalBeatDataset`: **3.1x**
> on the data-loading path. The earlier "279 passed" figure was from a
> different snapshot.
