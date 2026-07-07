# PAPER 1 + PAPER 2 — COMPLETE IMPLEMENTATION GUIDE

Verified fresh against this exact upload (2026-07-07): dependencies
installed, MIT-BIH downloaded, full suite run. **Result: 269 passed, 1
skipped** (skip = optional `torchvision`, only needed for the baseline
comparison table). Paper 3 code is present in this zip but is **out of
scope** per your stated direction (Paper 1 + Paper 2 only) — left alone,
not discussed further here.

Every command below is copied from this repo's actual scripts, not
reconstructed from memory.

---

## Part 1 — Paper 1 (EchoFuseNet): what's left, and exact commands

### 1. CRITICAL — GPU headline training run

Nothing downstream works until this runs. This is a full 30-epoch run —
needs a GPU (Kaggle/Colab/HPC, not CPU).

```bash
python -m paper1_echofusenet.training.train --config configs/echofusenet_gpu_headline.json
```

Produces the checkpoint at `runs/ds1ds2_headline/best.pt` — this exact path
is what Paper 2's config already expects (`configs/causalechonet_cvae.json`
points here by default), so don't rename or move it without updating that
config too.

### 2. 5-fold patient-grouped cross-validation

```bash
python -m paper1_echofusenet.training.crossval --config configs/echofusenet_ds1ds2_baseline.json --folds 5
```
Optional overrides: `--seed`, `--epochs`, `--out-dir`.

### 3. 7-config modality ablation (RP/GAF/MTF single, pairwise, full)

```bash
python -m paper1_echofusenet.training.ablation --config configs/echofusenet_ds1ds2_baseline.json
```
Optional: `--epochs` to override per-config epoch count, `--out-dir`.

### 4. INCART external validation

```bash
python -m scripts.external_validation --config configs/echofusenet_ds1ds2_baseline.json --checkpoint runs/ds1ds2_headline/best.pt
```
Downloads INCART to a default location automatically (override with
`--incart-dir` if needed); `--records` lets you subset if you want a quick
partial run first. `--n-boot` controls bootstrap CI iterations (default 2000).

### 5. PTB-XL domain-transfer probe

```bash
python -m scripts.domain_transfer --config configs/echofusenet_ds1ds2_baseline.json --checkpoint runs/ds1ds2_headline/best.pt
```
`--limit` caps how many NORM records to pull (default 200) — raise this for
a fuller run once the quick version looks right. Remember: this is
deliberately scoped to NORM-subset N-recall only, not a full 5-class claim
(PTB-XL has no per-beat AAMI ground truth).

### 6. Noise robustness + deployment realism (§6, §8 — NOW IMPLEMENTED)

**Status update (2026-07-07): these are now built directly in this repo** —
the earlier "copy from a separate download" step is done. The prose PDF
(`paper1_publishable_fixes.pdf`) contained no source, so the modules were
written fresh to Paper 1's conventions and unit-tested:

| Module | Path | What |
|---|---|---|
| Noise injectors + sweep | `paper1_echofusenet/robustness/noise.py` | Gaussian (target SNR), baseline wander, powerline interference; `evaluate_under_noise` sweeps accuracy/macro-F1 across levels with **repeated runs + CIs** |
| ONNX deployment | `paper1_echofusenet/deployment/onnx_export.py` | `export_onnx`, `verify_parity` (ONNX Runtime vs PyTorch), `quantize_dynamic_onnx` (int8), `export_and_report` |
| Tests | `tests/test_robustness.py`, `tests/test_deployment.py` | 13 tests total |

`onnx`/`onnxruntime`/`onnxscript` are optional (the `[deployment]` extra):
`pip install -e '.[deployment]'`. The deployment tests **skip** cleanly when
they're absent, so the core suite never hard-requires them.

Verify: `pytest -q tests/test_robustness.py tests/test_deployment.py` →
**13 passed** (with the deployment extra installed; 8 passed + 5 skipped
without it). Full suite: **229 passed, 2 skipped**.

Verified numbers from a real export of the full model: ONNX 2.469 MB →
int8 **0.696 MB (3.55× smaller)**, ONNX-vs-PyTorch parity `3.7e-09`. The
remaining run-time work is to point these at the **trained** checkpoint from
Step 1 (the code is done; only the trained weights are pending).

### 7. Code quality — `print()` → `logging`

Low priority, non-blocking. Repo-wide find/replace of `print(...)` calls
with a proper `logging` call; do this last, it doesn't affect any result.

### 8. Manuscript

After 1-6 above produce real numbers: lock the final training recipe, build
result tables with CIs and significance, generate final figures, write and
submit to *Biomedical Signal Processing and Control*.

---

## Part 2 — Paper 2 (CausalEchoNet): what's left, and exact commands

**Code status: 100% complete, already tested** (frozen-encoder loader,
feature-space CVAE, counterfactual validity/proximity/sparsity metrics,
per-modality ITE attribution, Grad-CAM + exact-Shapley SHAP baselines,
representation caching, full training loop). **Results status: 0%,
entirely blocked on Part 1, Step 1.**

The training entry point actively refuses to run without a real frozen
checkpoint — this isn't a bug, it's a deliberate guard so Paper 2 never
trains against a moving target:

```python
# from paper2_causalechonet/training/train.py — raises ValueError unless
# cfg.encoder.checkpoint points at a real, frozen Paper 1 checkpoint
```

### Once Part 1 Step 1 is done, run Paper 2 like this:

```bash
python -m paper2_causalechonet.training.train --config configs/causalechonet_cvae.json
```
The config already points at `runs/ds1ds2_headline/best.pt` — if you changed
that path in Part 1, update it here too before running.

**There is nothing else to build for Paper 2.** Every module has a green
test. The only path to Paper 2 results is: finish Part 1 Step 1, then run
the one command above.

---

## Recommended order, start to finish

1. **Part 1, Step 1** — GPU headline run. Blocks everything else.
2. In parallel while that GPU job runs, or immediately after: Part 1 Steps
   4-5 (INCART/PTB-XL downloads can start independently — they don't need
   the trained checkpoint to *download*, only to *evaluate* against, so
   fetch the data now).
3. Part 1 Steps 2-3 (CV, ablation) — GPU runs, do after Step 1 or alongside
   if you have a second GPU session.
4. Part 1 Step 6 — merge the noise/deployment files, verify `9 passed`, then
   run both against the real trained checkpoint from Step 1.
5. **Part 2** — run the one Paper 2 command now that a real checkpoint exists.
6. Part 1 Step 7 — logging cleanup, whenever convenient.
7. Part 1 Step 8 — manuscript, once every number above is real and locked.

Everything in this order is either already-verified code waiting on compute,
or one file-copy step away from being verified code waiting on compute. No
open design questions remain for either paper.
