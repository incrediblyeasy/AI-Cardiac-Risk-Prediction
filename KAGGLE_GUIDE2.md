# Kaggle Notebooks — run the headline results (step by step)

**Updated:** now includes early stopping (stops automatically at the best
epoch, not always 30) and resume support (survives disconnects/quota
cutoffs, at both the epoch level and — new — the CV fold level).

Same job as the Colab guide, but on Kaggle. Copy each cell into a Kaggle
notebook and run top to bottom.

**Before you start:** the code must be on GitHub (it is —
`https://github.com/incrediblyeasy/AI-Cardiac-Risk-Prediction.git`). Create a new
notebook: **kaggle.com → Create → New Notebook.**

---

## 0. Turn on the GPU **and Internet**
In the right-hand panel (click **⋮ / Settings** or the "Session options" gear):
- **Accelerator → GPU T4 x2** (or **GPU P100**). Save.
- **Internet → On.**

Then run:
```python
!nvidia-smi
```
You should see a GPU (e.g. Tesla T4 or P100). If not, the accelerator didn't take.

> Kaggle GPU sessions run up to ~12h and there's a weekly GPU quota (~30h).
> With early stopping now on (see step 5), the headline run should finish
> well before either limit in most cases — but if quota runs out mid-run
> anyway, step 5's `--resume` flag picks up where it left off rather than
> restarting from epoch 0.

## 1. Get the code
```python
%cd /kaggle/working
!git clone https://github.com/incrediblyeasy/AI-Cardiac-Risk-Prediction.git
%cd AI-Cardiac-Risk-Prediction
```
(If you cloned earlier in the same session, instead run
`%cd /kaggle/working/AI-Cardiac-Risk-Prediction` then `!git pull`.)

## 2. Install dependencies
```python
!pip install -q -r requirements.txt
!pip install -q onnx onnxruntime onnxscript   # needed for the deployment module (step 6)
```

## 3. Download MIT-BIH (the training data)
```python
!python -m paper1_echofusenet.data.download
```
(If this fails with a network error, Internet is still Off — go back to step 0.)

## 4. (Optional) Sanity-check the code
```python
!pytest -q
```
Expect everything green (~243 tests; the exact count varies slightly with
optional deps — e.g. `torchvision`, preinstalled on Kaggle, un-skips the
baseline-comparison tests).

## 5. Paper 1 — headline training (the main result)

**What changed:** `configs/echofusenet_gpu_headline.json` now has early
stopping on (`patience: 5`, `min_delta: 0.001`) — training stops once
macro-F1 hasn't improved for 5 straight epochs, instead of always grinding
through all 30. Based on the actual headline run's own result (best epoch
was 7 of 30), expect this to typically finish around epoch 10-15, not 30 —
faster, and the *reported* result is unaffected either way (`best.pt` was
always the best epoch, never just the last one).

```python
!python -m paper1_echofusenet.training.train --config configs/echofusenet_gpu_headline.json
```

**If the session disconnects or quota runs out mid-run:** don't restart
from scratch —
```python
!python -m paper1_echofusenet.training.train --config configs/echofusenet_gpu_headline.json --resume runs/ds1ds2_headline/last.pt
```
This restores the optimizer/scheduler/early-stopping state too, not just
the model weights — a real continuation, not a fresh run wearing old weights.

Outputs land in `runs/ds1ds2_headline/` — `best.pt` (the model + **frozen encoder**),
`history.jsonl`, `config.json`. The final line prints the best macro-F1.

## 6. Paper 1 — CV, ablation, CIs, benchmark

**New: CV now resumes at the fold level too**, not just mid-epoch. If the
whole process dies between folds (e.g. a quota cutoff after fold 2
finishes but before fold 3 starts), rerunning the *same command* below
automatically skips folds that already completed and only trains what's
missing — it checks `runs/.../fold_N/fold_result.npz` before retraining
each fold. No separate resume flag needed for this — it's on by default.
(Force a full clean rerun with `--no-resume` if you ever want that instead.)

```python
# 5-fold patient-grouped cross-validation (now resumable per-fold + early-stopping per-fold)
!python -m paper1_echofusenet.training.crossval --config configs/echofusenet_gpu_headline.json --folds 5
# 7-config modality ablation with McNemar significance
!python -m paper1_echofusenet.training.ablation --config configs/echofusenet_gpu_headline.json
# Bootstrap confidence intervals on the headline model
!python -m scripts.evaluate_ci --config configs/echofusenet_gpu_headline.json --checkpoint runs/ds1ds2_headline/best.pt
# Latency + size benchmark
!python -m scripts.benchmark --iters 200 --threads 4
# Noise robustness (Gaussian / baseline-wander / powerline)
# Deployment: ONNX export + ONNX Runtime latency + dynamic quantization
# (see HOW_TO_RUN_ON_KAGGLE.md for the exact wiring — these need the trained
# checkpoint from step 5 plumbed into their function calls, not a bare CLI yet)
```

**Why CV took much longer than expected before this fix:** it wasn't a
CPU/GPU bug (this config was always `device: "auto"`, correctly GPU-enabled)
— it's that `oversample: true` in this same config genuinely multiplies
every training epoch to roughly 4-5x the raw DS1 beat count, on top of
running the full 30 epochs with no early stopping. Both fixes above (early
stopping + fold-level resume) target that combination directly.

## 7. Paper 2 — train the CVAE on the frozen encoder
`configs/causalechonet_cvae.json` already points at `runs/ds1ds2_headline/best.pt`,
so this just works once step 5 is done:
```python
!python -m paper2_causalechonet.training.train --config configs/causalechonet_cvae.json
```
Outputs land in `runs/causalechonet_cvae/` (counterfactual validity/proximity/sparsity in `history.jsonl`).

## 8. (Optional) External validation — after those datasets are downloaded
```python
# INCART
!python -m paper1_echofusenet.data.incart
!python -m scripts.external_validation --config configs/echofusenet_gpu_headline.json --checkpoint runs/ds1ds2_headline/best.pt
# PTB-XL (NORM-subset domain-transfer probe)
!python -m paper1_echofusenet.data.ptbxl
!python -m scripts.domain_transfer --config configs/echofusenet_gpu_headline.json --checkpoint runs/ds1ds2_headline/best.pt
```

## 9. Save the results back to your computer
```python
import shutil
shutil.make_archive("/kaggle/working/runs", "zip", "runs")
print("Wrote /kaggle/working/runs.zip")
```
Then either:
- **Quick way:** right panel → **Output → /kaggle/working**, find
  `runs.zip`, **⋮ → Download**.
- **Reliable way:** click **Save Version** → *Save & Run All* (or *Quick
  Save*). This is also what makes results survive a session ending — do
  this proactively before you plan to stop for the day, not only after a
  crash.

---

## Notes / gotchas specific to Kaggle

- **Internet must be On** (step 0) or nothing that reaches the network works.
- **Only `/kaggle/working` persists** and is downloadable. The clone/run
  happens there, so you're fine by default.
- **Session resets wipe the clone.** Re-run steps 1-3 after a fresh session,
  then use `--resume` (step 5) or just rerun the CV command as-is (step 6 —
  it self-skips completed folds) rather than starting over.
- **GPU quota:** if the accelerator dropdown is greyed out, you've hit the
  weekly limit — resets weekly. With early stopping + resume now in place,
  a single quota cutoff should cost you at most the current epoch/fold's
  partial progress, not the whole run.
- **`git pull` for updates:** `%cd /kaggle/working/AI-Cardiac-Risk-Prediction && !git pull`.
- **"Failed to save draft" / can't commit:** usually means this same
  notebook is open in more than one tab — close duplicates, keep one.
