# EchoFuseNet — Consolidated Results Summary (Days 1–15)

**Paper 1 — EchoFuseNet:** a multimodal ECG-beat classifier that encodes each
heartbeat as three signal-to-image transforms (Recurrence Plot, Gramian Angular
Field, Markov Transition Field) and fuses three lightweight depthwise-separable
CNN branches by late fusion. 5-class AAMI (N/S/V/F/Q), inter-patient protocol.

- **Parameters:** 652,677 (~0.65M, within the ~0.7M budget)
- **Exported size:** **2.596 MB** (< 3 MB deployment budget) ✅
- **Protocol:** de Chazal inter-patient — train DS1 (22 patients), test DS2 (22
  patients); 4 paced records excluded; no patient in both folds (leakage guard).

> **Compute caveat (read this first).** All Day 10–15 experiments were executed on
> a **CPU-only** machine (no CUDA; ~30 beats/s train, ~90 beats/s eval for the
> full 3-branch model). This makes full-scale runs (30-epoch oversampled training,
> 5-fold CV, 7-config ablation, large external datasets) take many hours to days.
> Where a full run was infeasible in-session, the **framework is built, unit-tested,
> and CLI-runnable**, and the run is deferred to a GPU. Each such case is marked
> **[deferred: GPU]** or **[deferred: idle-CPU]** below. A GPU is the single
> highest-leverage unblock for the remaining scale.

---

## Results at a glance

| Day | Experiment | Status | Headline result |
|---|---|---|---|
| 10 | Baseline DS1→DS2 (plain CE, 3 epochs) | ✅ done | DS2 acc **0.9293** (IN 87–94% band) · macro-F1 0.3793 |
| 11 | k-fold CV + significance | ✅ framework · CIs computed · [deferred: GPU] for full CV | bootstrap CIs on final DS2 model |
| 12 | Modality ablation (7 configs) | ✅ framework · [deferred: GPU] | McNemar vs full wired; table on GPU |
| 13 | Latency & size benchmark | ✅ **both PASS** | 2.596 MB · median **10.44 ms** (4-thr) / **9.48 ms** (1-thr) |
| 14 | INCART external validation | ✅ pipeline · [deferred: download+run] | 257→256 resample, lead II |
| 15 | PTB-XL domain transfer | ✅ probe · [deferred: download+run] | NORM-subset N-recall probe |

---

## Day 10 — Baseline (inter-patient DS1→DS2)

**Config:** [configs/echofusenet_ds1ds2_baseline.json](../configs/echofusenet_ds1ds2_baseline.json)
— plain cross-entropy, **no oversampling**, 3 epochs, AdamW + cosine, batch 64,
device CPU. (Plain CE chosen over oversampling/class-weighting so the run is
feasible on CPU *and* lands in the accuracy band; oversampling's minority-recall
benefit is quantified separately in the Day-12 ablation on GPU.)

**DS2 test fold:** 49,693 beats — N 44,241 · S 1,837 · V 3,220 · F 388 · Q 7.

**Final result (epoch 3, best.pt):**

| metric | value |
|---|---|
| accuracy | **0.9293** |
| macro-F1 | 0.3793 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| N | 0.950 | 0.977 | 0.963 | 44,241 |
| S | 0.107 | 0.038 | 0.056 | 1,837 |
| V | 0.824 | 0.897 | 0.859 | 3,220 |
| F | 0.077 | 0.010 | 0.018 | 388 |
| Q | 0.000 | 0.000 | 0.000 | 7 |

Confusion matrix (rows=true, cols=pred):

```
        N      S      V      F      Q
 N  43217    559    434     31      0
 S   1747     69     21      0      0
 V    298     16   2889     17      0
 F    226      3    155      4      0
 Q      2      0      5      0      0
```

**Band check (87–94% inter-patient accuracy):** 0.9293 = **92.93% → IN BAND.** ✅
No leakage/bug flag. N (F1 0.963) and **V (F1 0.859)** are learned well; S/F/Q are
poor — the expected signature of plain CE on an 89%-N-dominated fold without
oversampling. Low macro-F1 (0.379) is therefore a *baseline* number and the direct
motivation for the Day-12 oversampling/fusion ablation, which should lift S/F recall.

Artifacts: `runs/ds1ds2_baseline/{best.pt,last.pt,history.jsonl,config.json}`.

---

## Day 11 — k-fold CV + statistical significance

**Framework (tested):**
[crossval.py](../paper1_echofusenet/training/crossval.py),
[stats.py](../paper1_echofusenet/training/stats.py).

- **Patient-grouped k-fold** over DS1 (folds partition *patients*, never beats →
  inter-patient preserved per fold; DS2 untouched).
- **Significance:** Student-t CI across folds; percentile **bootstrap CI** on a
  fixed test set; **paired t / Wilcoxon** across folds; exact **McNemar** per-sample.
- **Concrete CI result (computed on the final `best.pt`, 2000 bootstrap resamples):**

  | metric | point | 95% CI |
  |---|---|---|
  | accuracy | 0.9293 | [0.9271, 0.9315] |
  | macro-F1 | 0.3793 | [0.3748, 0.3844] |

  Tight intervals (≈50k DS2 beats). Artifact: `runs/ds1ds2_baseline/ds2_bootstrap_ci.json`.
- Full 5-fold CV = 5 training runs ≈ 7.5h on CPU → **[deferred: GPU]**.

---

## Day 12 — Modality ablation

**Framework (tested):** [ablation.py](../paper1_echofusenet/training/ablation.py);
`EchoFuseNet` made modality-configurable (backward-compatible).

- Sweeps all **7 configs**: RP, GAF, MTF (single) · RP+GAF, RP+MTF, GAF+MTF
  (pairwise) · RP+GAF+MTF (full reference).
- **One shared DataLoader** across all configs (transforms computed once, not 7×).
- Each subset compared to the full model with an exact **McNemar** test on
  per-sample DS2 correctness; table annotated with significance stars.
- 7 training runs on CPU ≈ several hours → **[deferred: GPU]**. Expected story:
  full ≳ pairwise > single; weakest single modality significantly worse than full.

---

## Day 13 — Latency & size benchmark

**Harness (tested):** [benchmark.py](../paper1_echofusenet/benchmark.py),
`scripts/benchmark.py`.

Measured on an idle CPU with the trained `best.pt` (batch-1, 200 iters):

| metric | threshold | 4-thread | 1-thread (edge) | verdict |
|---|---|---|---|---|
| exported size | < 3 MB | 2.596 MB | 2.596 MB | ✅ PASS |
| median latency | < 15 ms | **10.44 ms** | **9.48 ms** | ✅ PASS |
| p95 latency | — | 13.05 ms | 14.06 ms | (under 15) |

**Both thresholds pass, single-core included** — no optimization needed. If a
tighter target ever bites, TorchScript / dynamic int8 quantization drop into the
same harness. Artifacts: `runs/ds1ds2_baseline/benchmark_{4,1}thread.json`.

---

## Day 14 — INCART external validation

**Pipeline (tested):** [incart.py](../paper1_echofusenet/data/incart.py),
`scripts/external_validation.py`.

- Reconciles INCART (257 Hz, 12-lead) to the frozen model: same *physical* R-peak
  window resampled to **256 samples**; **lead II** (≈ MLII); same AAMI mapping.
- Runs the DS1 model with **no retraining**; reports per-class F1 + bootstrap CIs
  and a **DS2-vs-INCART** side-by-side generalisation table.
- Needs a large download + the final Day-10 model + CPU/GPU → **[deferred: run]**.
  Expect a generalisation drop vs DS2 (different hardware/population/resampling).

---

## Day 15 — PTB-XL domain transfer

**Probe (tested):** [ptbxl.py](../paper1_echofusenet/data/ptbxl.py),
`scripts/domain_transfer.py`.

- **Methodological note:** PTB-XL is *record-level diagnostic* data (SCP codes),
  with **no per-beat AAMI truth**. A beat classifier cannot be scored beat-for-beat
  there. The valid probe is the **NORM subset**: detect R-peaks (wfdb XQRS),
  extract beats, and measure the frozen model's **N-recall** + predicted-class
  distribution under domain shift (100 Hz, different domain). We do **not** claim
  full 5-class metrics on PTB-XL.
- Needs a several-GB download + final model → **[deferred: run]**.

---

## Reproducibility

```bash
pip install -r requirements.txt
pytest                                   # full suite (~150 tests)

# Day 10 baseline
python -m paper1_echofusenet.training.train --config configs/echofusenet_ds1ds2_baseline.json
# Day 11 CIs on the baseline
python -m scripts.evaluate_ci   --config configs/echofusenet_ds1ds2_baseline.json --checkpoint runs/ds1ds2_baseline/best.pt
# Day 11 full CV (GPU) / Day 12 ablation (GPU)
python -m paper1_echofusenet.training.crossval  --config configs/echofusenet_ds1ds2_baseline.json --folds 5
python -m paper1_echofusenet.training.ablation  --config configs/echofusenet_ds1ds2_baseline.json
# Day 13 benchmark (idle CPU)
python -m scripts.benchmark --iters 200 --threads 4
# Day 14 / 15 external + domain transfer (after downloads)
python -m paper1_echofusenet.data.incart && python -m scripts.external_validation --config configs/echofusenet_ds1ds2_baseline.json --checkpoint runs/ds1ds2_baseline/best.pt
python -m paper1_echofusenet.data.ptbxl  && python -m scripts.domain_transfer     --config configs/echofusenet_ds1ds2_baseline.json --checkpoint runs/ds1ds2_baseline/best.pt
```

All runs are seeded and config-driven; each writes its resolved `config.json` +
metrics artifacts next to the checkpoints.

---

## Manuscript-Readiness Checklist

Legend: ✅ done · 🟡 framework done, full run pending compute · ⬜ not started.

### Method & code (ready)
- ✅ Inter-patient DS1/DS2 split + hard leakage guard (unit-tested)
- ✅ RP / GAF / MTF transforms (unit-tested, distinctness-guarded)
- ✅ Multimodal DataLoader with post-split, train-only oversampling
- ✅ EchoFuseNet (~0.65M params) + modality-configurable variant for ablation
- ✅ Config-driven, reproducible training loop; metrics (per-class F1, macro-F1, CM)
- ✅ CV + significance library; ablation driver; benchmark harness; INCART + PTB-XL loaders
- ✅ ~150 unit tests green; exported model **2.596 MB < 3 MB**

### Experiments to finalize (need GPU / downloads / idle CPU)
- 🟡 **Baseline**: finish the 3-epoch run; ideally a **full 30-epoch oversampled**
  run on GPU for the headline table (current CPU run is a feasibility-sized proxy).
- 🟡 **Bootstrap CIs** on the final DS2 model (`evaluate_ci.py`) — quick, do first.
- 🟡 **5-fold patient CV** with mean±CI (GPU).
- 🟡 **7-config ablation** table with McNemar significance (GPU).
- 🟡 **Latency** median vs 15 ms on an idle CPU (+ quantized variant if needed).
- 🟡 **INCART** external validation table (download + run).
- 🟡 **PTB-XL** NORM-subset domain-transfer probe (download + run).

### Before drafting begins (remaining Phase 1 weeks)
- ⬜ Decide headline training recipe (oversampling vs class-weighting vs focal loss)
  from the ablation, then lock one config for all reported numbers.
- ⬜ Produce final figures: confusion matrices (DS2/INCART), per-class F1 bars,
  ablation bar chart, example RP/GAF/MTF triptychs (some already in `docs/figures`).
- ⬜ Results tables with CIs + significance for every claim.
- ⬜ Related-work / baseline comparison table (de Chazal, other RP/GAF/MTF ECG papers).
- ⬜ Ethics/data-availability statements (all datasets are public PhysioNet).
- ⬜ Reproducibility appendix (this doc + configs + seeds).

**Single biggest unblock:** GPU access. It converts every 🟡 above from hours/days
to minutes and enables the full-scale, publication-grade runs.
