# Papers 2 & 3 — Implementation Notes

Full code for **Paper 2 (CausalEchoNet)** and **Paper 3 (CardioCausal)**, built
from the plans in `PROJECT_STATUS_AND_ROADMAP.md` §3–§4. Both follow Paper 1's
conventions (rationale-heavy docstrings, nested-dataclass + JSON configs,
`from __future__ import annotations`, tests under top-level `tests/`, numpy/scipy/
torch only — no sklearn).

**Layout choice:** the roadmap's illustrative trees nest `configs/` and `tests/`
inside each paper package. The *actual* repo (and `pyproject`/`pytest` config)
keeps `tests/` and `configs/` at the top level, so these packages do too — that
keeps every test discovered by `pytest` and every config beside Paper 1's.

## Status legend
- ✅ **implemented + tested** — real, exercised code, green under `pytest`.
- 🔒 **runtime-gated** — code is complete; it *runs* only when a real dependency
  is supplied (a frozen Paper 1 checkpoint, or credentialed/downloaded data). It
  fails loudly with the exact prerequisite rather than fabricating output. The
  gate is data/compute/access — **not** missing code.

## Paper 2 — `paper2_causalechonet/`
| Module | What | Status |
|---|---|---|
| `encoder/frozen.py` | Load Paper-1 checkpoint, freeze all weights (eval-pinned), expose `encode`/`decision`/`modality_slices` | ✅ (freeze-survives-backward guard) |
| `cvae/model.py` | `FeatureCVAE` (conditional VAE in representation space) + `cvae_loss` | ✅ |
| `cvae/metrics.py` | Counterfactual `validity` / `proximity` / `sparsity` | ✅ |
| `attribution/ite.py` | Per-modality intervention + ITE + attribution table | ✅ |
| `baselines/shap_baseline.py` | **Exact** Shapley over 2³ modality coalitions (no `shap` dep) | ✅ (efficiency-axiom test) |
| `baselines/gradcam.py` | Per-branch Grad-CAM saliency → per-modality scalar | ✅ (no-grad-leak test) |
| `data.py` | Cache frozen-encoder representations once, reuse across epochs | ✅ |
| `training/train.py` | `train_one_epoch` + full `run_from_config` (encode → train → checkpoint on validity) | ✅ code · 🔒 needs Paper-1 checkpoint to run |

**Runtime gate:** `run_from_config` raises `ValueError` unless
`cfg.encoder.checkpoint` points at a **frozen** Paper-1 model — do not train
against a moving target (roadmap §2). Everything else runs today.

Config: `configs/causalechonet_cvae_smoke.json`.

## Paper 3 — `paper3_cardiocausal/`
| Module | What | Status |
|---|---|---|
| `fusion/model.py` | `MultimodalFusion` — ECG repr + tabular → shared latent | ✅ |
| `scm/model.py` | `RiskHead` + `NeuralSCM` (additive-noise; `do`, exact counterfactuals) + `fit_scm` | ✅ (invertibility + fit tests) |
| `recourse/engine.py` | `ModifiabilityMask` + gradient `generate_recourse` (modifiable-only, risk-lowering) | ✅ |
| `longitudinal/propagate.py` | `LatentPropagator` — bounded rollout (hard `max_horizon`) | ✅ |
| `causal_validation/evalues.py` | E-value + CI E-value (VanderWeele & Ding) | ✅ |
| `causal_validation/ipw.py` | Stabilised-IPW ATE (Hájek) | ✅ (recovers effect under confounding) |
| `causal_validation/diagnostics.py` | Covariate-balance SMD, positivity, negative-control | ✅ |
| `causal_validation/protocol.py` | `TargetTrialProtocol` pre-registration object | ✅ |
| `evaluation/metrics.py` | AUROC, average precision, Brier, ECE (numpy/scipy) | ✅ |
| `datasets/mimic_iv.py` | `link_ecg_ehr` + `subject_level_split` (time-zero, inter-patient) | ✅ · `build_linked_cohort` 🔒 credentialed I/O |
| `datasets/external.py` | Chapman-Shaoxing + CODE-15% loaders | 🔒 downloads |
| `training/config.py` | `CardioCausalConfig` | ✅ |

**Runtime gates:** MIMIC-IV file reading needs credentialed PhysioNet access
(identity + CITI training — start now, roadmap §4.1); the *linkage/split algorithm*
it feeds is already implemented and tested. External-cohort loaders need the
downloads. The SCM/recourse/causal machinery all run today on in-memory data.

## Tests
`tests/test_paper2_*.py` and `tests/test_paper3_*.py` — all green; full repo suite
**228 passed**. Runtime-gated entry points are themselves tested (they must raise
their specific error), so a gate can't silently rot into a no-op.

## What is still NOT here (inherently needs you)
- **Headline results** — the GPU-scale Paper 1 runs + the frozen-encoder export
  that unlocks Paper 2/3 training. No fabricated numbers.
- **Credentialed / downloaded data** — MIMIC-IV access, INCART/PTB-XL/
  Chapman-Shaoxing/CODE-15% downloads.
- **Human-only steps** — clinical-alignment review of counterfactuals; journal
  submission.
- **Manuscript prose** — methods can be written now; results tables wait on the
  gated runs.

This mirrors Paper 1's own discipline (`RESULTS_SUMMARY.md` marks every deferred
item rather than claiming an unrun result).
