# Google Colab — run the headline results (step by step)

This runs Paper 1's GPU headline training, exports the frozen encoder, and trains
Paper 2's CVAE — the runs this machine (CPU-only) can't do. Copy each cell into a
Colab notebook and run top to bottom.

**Before you start:** the code must be on GitHub. From your PC, push the current
project once (see "Pushing your code" at the bottom). The repo is
`https://github.com/incrediblyeasy/AI-Cardiac-Risk-Prediction.git`.

---

## 0. Turn on the GPU
In Colab: **Runtime → Change runtime type → Hardware accelerator → GPU → Save.**
Then run:
```python
!nvidia-smi
```
You should see a GPU (e.g. Tesla T4). If not, the runtime type didn't take.

## 1. Get the code
```python
!git clone https://github.com/incrediblyeasy/AI-Cardiac-Risk-Prediction.git
%cd AI-Cardiac-Risk-Prediction
```
(If you cloned earlier in the session, instead run `%cd AI-Cardiac-Risk-Prediction` then `!git pull`.)

## 2. Install dependencies
```python
!pip install -q -r requirements.txt
```

## 3. Download MIT-BIH (the training data)
```python
!python -m paper1_echofusenet.data.download
```

## 4. (Optional) Sanity-check the code
```python
!pytest -q
```
Expect everything green (~228 tests). Skip if you're in a hurry.

## 5. Paper 1 — headline training (the main result)
30-epoch oversampled run on GPU. On a T4 this is roughly ~1–2 hours; leave the tab open.
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
```python
!zip -r runs.zip runs
from google.colab import files
files.download("runs.zip")
```
**Send me `runs.zip`** (or push it — see below). It has `best.pt`, the metrics,
and the CVAE history. From there I generate the figures/tables and we move to
Paper 3 (which needs MIMIC-IV — Track A).

---

## Pushing your code (one time, from your PC)
The new Paper 2/3 code is not on GitHub yet. In a terminal in the project folder:
```bash
git add -A
git commit -m "Papers 2 & 3 implementation + Colab configs"
git push origin main
```
If `git push` asks for a password, use a **GitHub personal access token** (GitHub
Settings → Developer settings → Personal access tokens), not your account password.

### Getting results back via GitHub instead of download (optional)
Checkpoints are large and git-ignored, so the download in step 9 is simplest. If
you prefer, upload `runs.zip` to a Google Drive folder and share the link with me.
