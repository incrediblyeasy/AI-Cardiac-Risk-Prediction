# Day 15 — Domain-Transfer — PTB-XL + Consolidation

## Goal
Run the domain-transfer experiment and consolidate everything built so far into one results package.

## Tasks
- [x] Acquire/preprocess relevant PTB-XL subset for domain-transfer evaluation — *NORM-subset probe built & tested; download+run deferred*
- [x] Run domain-transfer experiment, report results — *`domain_transfer.py` built; run deferred*
- [x] Consolidate all results from Days 10-14 into one results summary document — **`docs/RESULTS_SUMMARY.md`**
- [x] Write a readiness checklist for what's left before manuscript drafting begins — *in RESULTS_SUMMARY.md*

## Deliverable / Definition of Done
- PTB-XL domain-transfer results
- Consolidated Day 1-15 results summary doc
- Manuscript-readiness checklist

## Dependencies
Day 6 (pipeline) + Day 10 (trained model).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-03 (probe + consolidation); PTB-XL run pending

**Status:** 🟡 In progress — consolidation ✅ done; PTB-XL probe ✅ built & tested; real transfer numbers pending download + final model

**What I completed:**
- `paper1_echofusenet/data/ptbxl.py` — PTB-XL **domain-transfer probe**, with the
  key methodological call documented up front: **PTB-XL has no per-beat AAMI truth**
  (it's record-level SCP-code diagnostic data), so a beat classifier can't be
  scored beat-for-beat. The valid probe is the **NORM subset** — select confident
  normal ECGs via the CSV (stdlib parsing, no pandas dependency), detect R-peaks
  (wfdb XQRS), resample beats 100→256 (same physical window as MIT-BIH), lead II,
  and measure the frozen model's **N-recall** + predicted-class distribution under
  domain shift.
- `scripts/domain_transfer.py` — runs the frozen DS1 model on PTB-XL NORM beats,
  reports N-recall + bootstrap CI + predicted distribution, saves JSON.
- `tests/test_ptbxl.py` — 5 tests (NORM CSV selection incl. confidence/dominance
  rules, beat-from-peaks resampling + edge-skip + z-score) on synthetic data, no
  download needed. All green.
- **`docs/RESULTS_SUMMARY.md`** — the consolidated Day 1–15 results document:
  results-at-a-glance table, per-day summaries with real numbers where available
  (model size 2.596 MB ✅; Day-10 epoch-1 DS2 acc 0.8714 in-band), reproducibility
  commands, and a **Manuscript-Readiness Checklist** (done / framework-pending /
  before-drafting), flagging GPU as the single biggest unblock.

**Blockers / issues:**
- **PTB-XL is a several-GB download and lacks beat-level labels** → only the
  NORM-subset transfer probe is scientifically valid, and its real numbers need
  the download + final Day-10 model + compute. Framework is delivered + tested.
- Consolidation doc's Day-10 row is interim (epoch 1); refresh with final 3-epoch
  `best.pt`, and ideally re-run the whole battery at full scale on GPU.

**Notes for next day (post-sprint / Phase 1):**
- First cheap win: run `scripts/evaluate_ci.py` on the final `best.pt` for DS2 CIs,
  then `scripts/benchmark.py` on an idle CPU for the latency number.
- Then, on GPU: full 30-epoch oversampled baseline, 5-fold CV, 7-config ablation,
  INCART + PTB-XL runs → drop the numbers into `docs/RESULTS_SUMMARY.md`.

**Time spent:** ~2h (probe + consolidation doc)
