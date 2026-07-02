# Day 6 — Data Pipeline Integration

## Goal
Combine RP/GAF/MTF into one multimodal DataLoader with oversampling applied correctly.

## Tasks
- [x] Build a unified multimodal Dataset/DataLoader producing (RP, GAF, MTF, label) tuples
- [x] Apply oversampling ONLY to the training fold, and only AFTER the patient-level split (critical correction — do not oversample before splitting)
- [x] Sanity-check batch shapes and post-oversampling class balance on train; confirm test fold is untouched/natural distribution

## Deliverable / Definition of Done
- End-to-end DataLoader producing correctly balanced train batches and untouched test batches

## Dependencies
Days 3, 4, 5 (all three transforms working).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-02

**Status:** ✅ Done

**What I completed:**
- New `data/dataset.py` with the multimodal pipeline:
  - `MultimodalBeatDataset(Dataset)` — yields `(rp, gaf, mtf, label)` per beat: three `(1, L, L)` float32 tensors (one channel per late-fusion branch) plus a scalar `long` label. Transforms are computed **lazily** in `__getitem__` so memory stays flat regardless of fold size (the oversampled train fold is ~229k beats). Default collate stacks them into `(B, 1, L, L)` batches.
  - `beat_to_channels(signal, normalize=True)` — RP/GAF/MTF on a common `[0, 1]` scale (RP & MTF native `[0,1]`; GAF `[-1,1]` rescaled via `(x+1)/2`), resolving the Day-5 mixed-range handoff note.
  - `oversample_indices` / `oversample_beats` — class-balanced oversampling that keeps all originals and adds with-replacement draws until every class matches the majority count; deterministic per `seed`.
  - `build_dataloaders(...)` — assembles the split via `build_split` (which runs the Day-2 leakage guard on extracted beats), oversamples the **train fold only, after the split**, and returns `(train_loader, test_loader)`. Test loader is `shuffle=False` and never resampled; train shuffle uses a seeded generator for reproducibility.
- All exported from `paper1_echofusenet.data`.
- Tests (`tests/test_dataset.py`, 10 tests — 4 synthetic always-run + 6 data-backed, mirroring the leakage-test two-level pattern): oversampling balances classes / is a superset of originals / is deterministic / handles empty; Dataset item shapes+range+dtype and default-collate batching; end-to-end batch shapes `(B,1,256,256)`; **train balanced to the majority count while test equals the natural extracted distribution**; folds patient-disjoint; `oversample=False` leaves train natural.
- Sanity-check script `scripts/check_dataloader.py` (`--subset` for a 2-record smoke run). Full-split run confirms:
  - Batch shapes `RP/GAF/MTF (32,1,256,256)`, labels `[0,4]`, channels in `[0,1]`.
  - DS1 train **before** oversampling `N:45846 S:944 V:3788 F:414 Q:8` → **after** `45846` for every class (balanced, total 229,230).
  - DS2 test **UNTOUCHED**: `N:44241 S:1837 V:3220 F:388 Q:7` in both natural extraction and the loader.
  - Train/test folds patient-disjoint (22 vs 22 patients).
- **Full test suite: 66 passing** (up from 56).

**Blockers / issues:**
- None functionally, but a real modeling caveat surfaced: DS1 is *extremely* imbalanced (Q has only **8** beats, F only 414 vs N's 45,846). Pure random oversampling duplicates the 8 Q beats ~5,730× — the model will see almost no genuine Q variety. Flagging for Day 7+: consider augmentation on the minority draws, class-weighted loss, or a focal loss instead of (or alongside) raw oversampling. The current `oversample` flag is easy to turn off to compare.

**Notes for next day (Day 7 — CNN branch architecture):**
- Consume `build_dataloaders()` directly; each branch takes a single `(B, 1, L, L)` input (`L=256`). The three-branch tuple order is fixed as `(rp, gaf, mtf, label)` — keep that contract when wiring the model's forward signature.
- Channels are already normalized to `[0,1]`; if a branch wants per-channel standardization instead, do it in the model/transform, not by re-touching the test fold.
- For fast iteration during architecture bring-up, use `build_dataloaders(train_records=(101,106), test_records=(100,103))` or the raw `MultimodalBeatDataset` on a handful of beats — the full oversampled train fold is 229k beats and computes transforms on the fly (CPU-heavy per epoch; consider caching or precomputing to disk once the architecture is fixed).

**Time spent:** ~1 session (multimodal Dataset/DataLoader + correct post-split oversampling + tests + end-to-end sanity script)
