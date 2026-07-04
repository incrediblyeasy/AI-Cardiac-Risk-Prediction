# PROJECT STATUS & ROADMAP — vs. Portfolio Review Proposal

Verified against `AI-Cardiac-Risk-Prediction-main.zip` by actually running the
code (not just reading it) on 2026-07-03. Compared against
`PhD_Portfolio_Review_and_Capstone_Proposal.docx`.

## 0. Bottom line

**Paper 1 (EchoFuseNet) is real and correct as far as it goes — roughly
70-80% done.** Core framework is fully built and unit-tested; what's left is
GPU-scale runs, two external dataset downloads, and manuscript writing.

**Paper 2 (CausalEchoNet) and Paper 3 (CardioCausal) do not exist.** No files,
no stubs, 0% code. This means **PhD Objective 1 (causal counterfactual
reasoning) is currently 0% delivered** — Paper 1 only supplies the encoder
backbone Objective 1 depends on. Objective 2 is partially served by Paper 1's
edge-deployability result but needs Paper 3's longitudinal layer to be
complete.

The repo is internally honest about this — its own `RESULTS_SUMMARY.md`
correctly marks deferred items and never claims something it didn't run.
That discipline should carry forward into Papers 2 and 3.

> **Update (2026-07-04): Papers 2 & 3 code is complete.** `paper2_causalechonet/`
> and `paper3_cardiocausal/` are fully implemented and unit-tested, matching Paper
> 1's conventions. Paper 2: frozen-encoder loader (freeze guarantee), feature-space
> CVAE + counterfactual metrics, per-modality ITE attribution, **exact-Shapley +
> Grad-CAM baselines**, representation caching, and a **wired end-to-end training
> loop**. Paper 3: multimodal fusion, calibrated risk head, an **additive-noise
> `NeuralSCM` with `do` + exact counterfactuals and `fit_scm`**, **gradient recourse**
> with a modifiable-only constraint, bounded longitudinal layer, and the full
> causal-validation stack (**E-values, stabilised-IPW ATE, covariate-balance /
> positivity / negative-control diagnostics**, target-trial protocol object) plus
> AUROC/AUPRC/Brier/ECE and the **MIMIC-IV linkage + subject-level split algorithm**.
> Full suite: **228 tests pass** (133 Paper 1 + 95 new). What remains is *only* the
> things code cannot supply: the GPU-scale Paper 1 headline run + frozen-encoder
> export, credentialed/downloaded data (MIMIC-IV, external cohorts), human clinical
> review, and manuscript prose — every runtime-gated entry point raises with its
> exact prerequisite, and **no results are fabricated**. Details:
> `docs/PAPER2_PAPER3_SCAFFOLD.md`. Still 0% *delivered science* — this is the
> framework the gated runs will use, not the runs themselves.

---

## 1. Independently verified evidence (I ran these myself)

| Claim in repo docs | Verified? | How |
|---|---|---|
| 133 tests pass | ✅ Confirmed | Ran `pytest` after downloading MIT-BIH — 133/133 pass, 0 skipped |
| Model = 652,677 params (~0.65M, within 0.7M budget) | ✅ Confirmed | Reran `scripts/model_summary.py` independently |
| RP/GAF/MTF branches are distinct (no channel-duplication defect) | ✅ Confirmed | Reran `scripts/check_distinctness.py` — max pairwise correlation 0.758, flagged DISTINCT |
| DS1/DS2 patient-level split, zero leakage | ✅ Confirmed | `test_leakage.py`, `test_splits.py` pass against real downloaded data |
| Benchmark: 2.596 MB, ~10ms CPU latency | 🟡 Not independently rerun (no checkpoint shipped — correctly gitignored) | Code path exists and is unit-tested; number comes from repo's own prior run |
| Day 10 baseline accuracy 92.93% (in 87-94% band) | 🟡 Not independently rerun (no checkpoint shipped) | Same — trust but not re-verified in this session |
| INCART / PTB-XL pipelines | ✅ Code confirmed present and tested | `incart.py`, `ptbxl.py`, `external_validation.py`, `domain_transfer.py` all exist; actual runs need downloads |
| Paper 2 / Paper 3 code | ❌ Confirmed absent | `find . -iname "*paper2*" -o -iname "*causal*" -o -iname "*paper3*"` → zero results |

---

## 2. Paper 1 — EchoFuseNet: what's left

Everything below is genuinely remaining work, not busywork — the repo's own
`docs/RESULTS_SUMMARY.md` checklist is accurate and I'm reusing its framing.

### 2a. Needs GPU (biggest single unblock)
- [ ] Full 30-epoch oversampled training run for the headline table (current
      run is a 3-epoch CPU feasibility proxy, plain CE, no oversampling)
- [ ] Full 5-fold patient-grouped CV with mean ± CI
- [ ] Full 7-config modality ablation (single/pairwise/full) with McNemar
      significance table

### 2b. Needs external dataset downloads + a run
- [ ] INCART external validation — pipeline built, needs download + run
- [ ] PTB-XL domain-transfer probe — pipeline built, needs download + run
      (correctly scoped to NORM-subset N-recall only — PTB-XL has no
      per-beat AAMI ground truth, so no full 5-class claim is possible there)

### 2c. Manuscript-only (no new code, but real work)
- [ ] Lock one headline training recipe (oversampling vs. class-weighting vs.
      focal loss) based on the ablation results
- [ ] Final figures: DS2/INCART confusion matrices, per-class F1 bars,
      ablation bar chart, RP/GAF/MTF triptychs (some already in `docs/figures`)
- [ ] Results tables with CIs + significance for every claim
- [ ] Related-work / baseline comparison table (de Chazal, other RP/GAF/MTF papers)
- [ ] Ethics / data-availability statement (all datasets are public PhysioNet)
- [ ] Reproducibility appendix
- [ ] Submit to *Biomedical Signal Processing and Control*

**Frozen encoder export:** once the GPU headline run is locked, export the
checkpoint as the frozen encoder Paper 2 will import. This is the hard
dependency gate between Paper 1 and Paper 2 — nothing in Paper 2 should start
training against a moving Paper 1 target.

---

## 3. Paper 2 — CausalEchoNet: full build plan (0% → start here after Paper 1 encoder freezes)

**Full title:** *CausalEchoNet: Modality-Specific Counterfactual Explanations
for Multimodal ECG Arrhythmia Classification via Conditional VAE and Causal
Attribution*
**Target journal:** *IEEE Journal of Biomedical and Health Informatics*

### Proposed structure
```
paper2_causalechonet/
├── encoder/          # loads Paper 1's frozen checkpoint, no retraining
├── cvae/              # feature-space Conditional VAE (~0.2M params)
├── attribution/       # modality-level ITE / causal attribution (RP/GAF/MTF)
├── baselines/          # Grad-CAM, SHAP for comparison
├── training/
├── configs/
└── tests/
```

### Build sequence
1. **Encoder loader.** Wrap Paper 1's frozen checkpoint behind a clean
   interface (`encode(beat) -> representation`). Add a test asserting the
   loaded encoder's weights are frozen (no gradient updates ever reach it).
2. **Feature-space CVAE (~0.2M params).** Trains in the representation space
   produced by the frozen encoder, not raw signal space. Conditions on
   target class to answer: "what minimal change shifts class A → class B?"
3. **Counterfactual quality metrics.** Validity (does the edit actually flip
   the class), proximity (how small is the edit), sparsity — build these as
   reusable metrics before claiming any results.
4. **Modality-level causal attribution via ITE.** Intervene on RP, GAF, MTF
   branches independently; measure each branch's causal effect on the
   decision. This is the paper's core novelty — don't let it become an
   afterthought bolted onto the CVAE.
5. **Baseline comparison suite.** Implement Grad-CAM and SHAP on the same
   frozen encoder, so the "causal vs. associational divergence" comparison
   is apples-to-apples.
6. **Budget checks.** Total system (frozen encoder + CVAE + attribution) must
   stay under 1M params combined and under 50ms CPU end-to-end — build the
   benchmark harness early (reuse/extend Paper 1's `benchmark.py` pattern),
   not as an afterthought before submission.

### Definition of done
- Counterfactual generator produces valid, minimal, class-A→B edits with
  measured validity/proximity/sparsity.
- ITE-based per-modality attribution pipeline with quantified causal effects.
- Grad-CAM/SHAP comparison table showing where causal and associational
  attributions diverge, with concrete cases.
- Param/latency budget verified (<1M params, <50ms CPU).
- CVAE + attribution modules exported in a form Paper 3 can import.

---

## 4. Paper 3 — CardioCausal (Capstone): full build plan (0% → largest remaining scope)

**Full title:** *CardioCausal: A Structural-Causal Multimodal Engine for
Personalized Cardiac Risk, Counterfactual Intervention Recommendation, and
Pathway-Level Explanation*
**Target journal:** *Artificial Intelligence in Medicine* (primary)

### Proposed structure
```
paper3_cardiocausal/
├── fusion/              # ECG repr (Paper 1/2) + tabular EHR → shared latent space
├── scm/                 # structural causal model
├── recourse/            # CVAE extension (from Paper 2) for intervention recs
├── longitudinal/         # bounded in-silico latent-state-propagation layer
├── causal_validation/    # target trial emulation, IPW, E-values, negative controls
├── datasets/
│   ├── ptbxl/            # representation dev + benchmarking
│   ├── mimic_iv/          # ECG+EHR linkage — backbone for causal estimation
│   └── external/          # Chapman-Shaoxing, CODE-15%
└── tests/
```

### Build sequence (matches the proposal's own phase plan, Weeks 21-40)
1. **PhysioNet credentialing for MIMIC-IV** — this is a real access-control
   step (identity verification + training course), not a technical task.
   Start it early; it's a lead-time bottleneck, not something code can route
   around. Nothing in the causal-validation stack can begin without it.
2. **ECG↔EHR linked cohort construction** from MIMIC-IV-ECG + MIMIC-IV EHR.
3. **Pre-register the target-trial-emulation protocol** (PICOT, DAG,
   confounder set, time-zero rule) *before* running any causal estimation —
   this ordering matters for the validity of the causal claims, not just
   for tidiness.
4. **Multimodal fusion layer**: combine the ECG representation (Papers 1-2)
   with tabular clinical context (demographics, labs, comorbidities,
   medications) into a shared latent space.
5. **Structural causal model** over the fused latent space producing
   calibrated risk + supporting the counterfactual layer.
6. **Extend Paper 2's CVAE** into a recourse engine: causally-consistent
   counterfactual intervention recommendations over *modifiable* variables
   only (not all variables — e.g. age isn't a valid intervention target).
7. **Bounded in-silico longitudinal layer**: latent-state propagation over
   serial ECG records for short-horizon retrospective risk trajectories.
   Keep this explicitly bounded — the proposal is deliberately avoiding an
   unverifiable "digital twin" claim; don't let scope creep re-introduce it.
8. **Causal validation stack**: target trial emulation, IPW, E-values,
   negative-control outcomes, positivity checks, covariate-balance
   diagnostics. This is what makes the paper's causal claims defensible —
   treat it as core deliverable work, not a validation afterthought.

### External validation
- [ ] Chapman-Shaoxing + CODE-15% (345,779 exams with mortality follow-up)

### Evaluation suite
- [ ] Risk discrimination: AUROC, AUPRC with bootstrap CIs
- [ ] Calibration: slope/intercept, Brier score, decision-curve analysis
- [ ] Causal-effect estimation with sensitivity analysis (E-values, negative
      controls, positivity checks)
- [ ] Counterfactual quality (validity, proximity, sparsity, plausibility,
      actionability) benchmarked against DiCE, Wachter et al., GCX
- [ ] Clinical-alignment review of generated counterfactuals — this needs a
      human clinical/causal-inference reviewer, flag it for supervisor input
      per the proposal's own "Requests to the Supervisor" section

### Definition of done
- MIMIC-IV cohort built, causal protocol pre-registered.
- Fusion + SCM + recourse + longitudinal layers built on frozen Paper 1/2 components.
- Full evaluation suite run and logged.
- External validation on Chapman-Shaoxing and CODE-15% complete.
- Manuscript drafted for *Artificial Intelligence in Medicine*.

---

## 5. Objective coverage — current state vs. required

| PhD Objective | Required from | Current status |
|---|---|---|
| **Obj 1** — causal counterfactual reasoning & pathway explanations | Papers 2 + 3 | **0% delivered.** Paper 1 supplies only the encoder backbone this depends on. |
| **Obj 2** — multimodal early-warning modeling | Papers 1 + 3 | **Partially delivered.** Paper 1's edge-deployability (<15ms, <3MB) is verified ✅. The short-horizon risk / longitudinal piece needs Paper 3 — not started. |

**Practical implication:** the thesis cannot be considered on-track for either
objective until Paper 2 exists — it's the dependency both the explanation
requirement and (via its CVAE) Paper 3's recourse engine sit on.

---

## 6. Recommended priority order

1. **Finish Paper 1's GPU-gated items** (§2a) — this unblocks the frozen
   encoder that everything downstream depends on. Don't start Paper 2 code
   against an unfrozen/still-changing Paper 1 model.
2. **Run INCART + PTB-XL** (§2b) in parallel with #1 if compute allows — they
   don't block Paper 2 the way the frozen encoder does.
3. **Start PhysioNet MIMIC-IV credentialing now**, in parallel with #1-2 —
   it's pure lead time, not compute, so there's no reason to wait.
4. **Write Paper 1's manuscript** (§2c) once results are locked.
5. **Scaffold and build Paper 2** (§3) once the frozen encoder is exported.
6. **Scaffold and build Paper 3** (§4) once Paper 2's CVAE/attribution
   modules are exportable and MIMIC-IV credentialing has cleared.

---

*Cross-referenced against: `AI-Cardiac-Risk-Prediction-main.zip` (code, run
2026-07-03) and `PhD_Portfolio_Review_and_Capstone_Proposal.docx` (April 2026).*
