# Kaggle Notebooks — run the headline results (step by step)

Same job as the Colab guide, but on Kaggle (use this since Colab isn't working).
It runs Paper 1's GPU headline training, exports the frozen encoder, and trains
Paper 2's CVAE — the runs this machine (CPU-only) can't do. Copy each cell into a
Kaggle notebook and run top to bottom.

**Before you start:** the code must be on GitHub (it is —
`https://github.com/incrediblyeasy/AI-Cardiac-Risk-Prediction.git`). Create a new
notebook: **kaggle.com → Create → New Notebook.**

---

## 0. Turn on the GPU **and Internet**
In the right-hand panel (click **⋮ / Settings** or the "Session options" gear):
- **Accelerator → GPU T4 x2** (or **GPU P100**). Save.
- **Internet → On.** *(This is required — `git clone`, `pip install`, and the data
  download all need internet. Kaggle asks you to verify your phone number once to
  enable it.)*

Then run:
```python
!nvidia-smi
```
You should see a GPU (e.g. Tesla T4 or P100). If not, the accelerator didn't take.

> Kaggle GPU sessions run up to ~12h and there's a weekly GPU quota (~30h). The
> headline run fits easily, but don't leave idle sessions open.

## 1. Get the code
Kaggle's writable working directory is `/kaggle/working`, so clone there:
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
Expect everything green (~228 tests). Skip if you're in a hurry.

## 5. Paper 1 — headline training (the main result)
30-epoch oversampled run on GPU. On a T4 this is roughly ~1–2 hours; keep the tab open.
```python
!python -m paper1_echofusenet.training.train --config configs/echofusenet_gpu_headline.json
```
Outputs land in `runs/ds1ds2_headline/` — `best.pt` (the model + **frozen encoder**),
`history.jsonl`, `config.json`. The final line prints the best macro-F1.

## 6. Paper 1 — CV, ablation, CIs, benchmark
```python
# 5-fold patient-grouped cross-validation
!python -m paper1_echofusenet.training.crossval --config configs/echofusenet_gpu_headline.json --folds 5
# 7-config modality ablation with McNemar significance
!python -m paper1_echofusenet.training.ablation --config configs/echofusenet_gpu_headline.json
# Bootstrap confidence intervals on the headline model
!python -m scripts.evaluate_ci --config configs/echofusenet_gpu_headline.json --checkpoint runs/ds1ds2_headline/best.pt
# Latency + size benchmark
!python -m scripts.benchmark --iters 200 --threads 4
```

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
Kaggle has no `files.download()` — instead you write the zip into `/kaggle/working`
(the notebook's persisted output) and grab it from the **Output** panel.
```python
import shutil
shutil.make_archive("/kaggle/working/runs", "zip", "runs")
print("Wrote /kaggle/working/runs.zip")
```
Then either:
- **Quick way:** in the right panel open **Output → /kaggle/working**, find
  `runs.zip`, and click the **⋮ → Download**.
- **Reliable way:** click **Save Version** (top right) → *Save & Run All* (or
  *Quick Save*). When it finishes, open the notebook's **Output** tab and download
  `runs.zip` from there. Saving a version also keeps your results if the session
  closes.

**Send me `runs.zip`** (or share it via Google Drive). It has `best.pt`, the metrics,
and the CVAE history. From there I generate the figures/tables and we move to
Paper 3 (which needs MIMIC-IV — Track A).

---

## Notes / gotchas specific to Kaggle
- **Internet must be On** (step 0) or nothing that reaches the network works. If a
  cell hangs or errors on a download, that's almost always the cause.
- **Only `/kaggle/working` persists** and is downloadable. Anything written
  elsewhere (e.g. `/tmp`) is lost when the session ends — the guide clones and runs
  under `/kaggle/working`, so you're fine.
- **Session resets wipe the clone.** If you close the tab and come back, re-run
  steps 1–3 (clone + install + download) before continuing. "Save Version" preserves
  *outputs*, not the running filesystem.
- **GPU quota:** if the accelerator dropdown is greyed out, you've hit the weekly
  GPU hours limit — it resets weekly, or run CPU-only (much slower, not recommended
  for step 5).
- **`git pull` for updates:** if I push new code, just
  `%cd /kaggle/working/AI-Cardiac-Risk-Prediction && !git pull` instead of
  re-cloning.
