# REMOVE PAPER 3 + REMAINING FIXES — Do-It-Yourself Guide

Every step below was actually run and verified against your repo before
being written here — not guessed. Scope after this: **Paper 1 (EchoFuseNet)
+ Paper 2 (CausalEchoNet)**. Paper 3 (CardioCausal) is fully removed.

Starting point assumed: the v2 zip (the one with `paper2_causalechonet/`
and `paper3_cardiocausal/` both present, plus the noise/deployment fixes
from a previous session already in `paper1_echofusenet/robustness/` and
`paper1_echofusenet/deployment/`).

---

## Step 1 — Delete Paper 3

```bash
cd AI-Cardiac-Risk-Prediction-main
rm -rf paper3_cardiocausal
rm -f tests/test_paper3_*.py
rm -f configs/cardiocausal_smoke.json
```

**Verified clean:** Paper 2 has zero imports of anything in
`paper3_cardiocausal` — removing it breaks nothing in Paper 2. (Checked with
`grep -rln "paper3_cardiocausal" paper2_causalechonet/` → empty.)

## Step 2 — Fix `pyproject.toml`

Find this line:
```toml
include = ["paper1_echofusenet*", "paper2_causalechonet*", "paper3_cardiocausal*", "shared*"]
```
Change it to:
```toml
include = ["paper1_echofusenet*", "paper2_causalechonet*", "shared*"]
```

## Step 3 — Replace the Paper 2/3 scaffold doc

```bash
rm docs/PAPER2_PAPER3_SCAFFOLD.md
```
Then create `docs/PAPER2_SCAFFOLD.md` with the Paper 2 table only (drop the
entire "## Paper 3" section, and the Paper-3-specific lines in "## Tests" /
"## What is still NOT here"). If you want the exact rewritten version
without doing this by hand, say so and I'll hand you the finished file
directly instead of the instructions.

## Step 4 — Flag the scope change in `PROJECT_STATUS_AND_ROADMAP.md`

At the very top, right after `## 0. Bottom line`, add:
```markdown
> **SCOPE CHANGE (this update):** Paper 3 (CardioCausal), §4 below, has been
> **removed from this repo**. This is now a 2-paper scope: Paper 1
> (EchoFuseNet) + Paper 2 (CausalEchoNet). §4 is left in place below as
> historical planning context only — nothing in `paper3_cardiocausal/`
> exists in the repo anymore.
```
(§4 itself can stay as-is below this — it's just historical context now, not
live scope. Not worth renumbering every section for this.)

## Step 5 — Verify nothing broke

```bash
pip install -r requirements.txt
python -m paper1_echofusenet.data.download
pytest -q
```
**Expected result (verified just now on your actual code):
`227 passed, 1 skipped`.** The 1 skip is optional `torchvision` (only needed
for the ResNet/DenseNet/etc. baseline comparison table) — not a problem.

If you get a different number, something didn't match what's described
here — paste the actual pytest output and I'll tell you what's different.

---

## Remaining fixes status (PDF checklist)

Already done, nothing to add:

| # | Fix | Status |
|---|---|---|
| 1, 2, 4, 5 | Split, modality pipeline, ablation, config sync | ✅ Pre-existing |
| 6 | Noise robustness | ✅ Already in `paper1_echofusenet/robustness/noise.py` |
| 8 | Deployment (ONNX/quantization) | ✅ Already in `paper1_echofusenet/deployment/onnx_export.py` |

Still open:

| # | Fix | What's left |
|---|---|---|
| 3 | External validation | Framework's built (INCART/PTB-XL loaders exist) — just needs the actual downloads + a GPU run |
| 7 | Richer interpretability | Grad-CAM/SHAP already exist in `paper2_causalechonet/baselines/` (kept there deliberately — Paper 2's job, not Paper 1's, per the original design). **Nothing new needed here** unless you want a Paper-1-side illustration figure too, which is a smaller, optional add-on, not a checklist requirement in this 2-paper scope |

Only genuine remaining work: run the GPU headline training, then run §3
(external validation) and §6/§8 (noise/deployment) against the real trained
checkpoint — none of these are code gaps anymore, just runs that haven't
happened yet.

---

## If you'd rather I just do this and hand you the finished zip

Say so — I've already done and verified this exact removal in a working
copy (that's how the "227 passed" number above is real, not estimated). It's
a one-message turnaround to package it if you want the finished result
instead of doing steps 1-4 by hand.
