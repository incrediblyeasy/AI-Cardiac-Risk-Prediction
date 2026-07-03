# Day 14 — External Validation — INCART

## Goal
Test generalization on a completely external dataset.

## Tasks
- [x] Acquire and preprocess the INCART dataset (same RP/GAF/MTF pipeline, same AAMI class mapping) — *loader + downloader built & tested; download+run deferred (see note)*
- [x] Run the trained model (no retraining) on INCART — *external-validation script built*
- [x] Report metrics and compare against DS1/DS2 baseline — *side-by-side table wired in*

## Deliverable / Definition of Done
- INCART external validation results table

## Dependencies
Day 10 (trained model) + Day 6 (reusable pipeline).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-03 (pipeline); download + eval run pending

**Status:** 🟡 In progress — pipeline ✅ done & tested; real INCART numbers pending download + final Day-10 model

**What I completed:**
- `paper1_echofusenet/data/incart.py` — external-validation loader that makes
  INCART look exactly like MIT-BIH to the frozen model:
  - **Sampling-rate reconciliation**: INCART is 257 Hz vs MIT-BIH 360 Hz, so each
    beat is taken over the *same physical window* (128/360 s each side of the
    R-peak) and resampled to 256 samples → identical 256×256 RP/GAF/MTF images.
  - **Lead**: defaults to lead **II** (closest to MIT-BIH MLII), with a safe
    fallback if absent. Same `aami` mapping, beats labelled fold `"INCART"`.
  - `download_incart()` (idempotent, mirrors `download.py`), `load_incart()`,
    record list I01–I75.
- `scripts/external_validation.py` — loads a DS1-trained checkpoint, runs it on
  INCART **with no retraining**, prints the per-class report + bootstrap 95% CIs,
  and a **DS2-vs-INCART side-by-side table** (pulls the DS2 baseline from the
  Day-10 run dir). Saves `incart_external.json`.
- `tests/test_incart.py` — 6 tests (record naming, 257→256 resampling, lead-II
  selection + fallback, AAMI labels, z-score) using a synthetic 12-lead record,
  so no download is needed to validate the logic. All green.

**Blockers / issues:**
- **Data + compute.** INCART is a large multi-hundred-MB download (75 records ×
  12 leads × 30 min) and evaluating ~175k beats on this CPU-only box takes tens
  of minutes; it also needs the *final* Day-10 model (still training). So the
  pipeline is delivered + tested and the run is deferred to when: (1) Day-10
  finishes, (2) INCART is downloaded (`python -m paper1_echofusenet.data.incart`),
  (3) the CPU is free (ideally GPU). Expect a **generalisation drop** on INCART
  vs DS2 (different hardware/population, resampling) — that gap is the result.

**Notes for next day:**
- Run: `python -m scripts.external_validation --config
  configs/echofusenet_ds1ds2_baseline.json --checkpoint runs/ds1ds2_baseline/best.pt`.
- INCART's class mix differs from MIT-BIH (more V); read per-class F1, not just
  accuracy. Consider reporting macro-F1 as the headline for the cross-dataset gap.

**Time spent:** ~1.5h (pipeline)
