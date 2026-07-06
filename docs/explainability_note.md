# Explainability in Paper 1 — scope note

**Status:** intentionally minimal. This is the §1 checklist item, built to the
flag: a *single qualitative note plus at most one figure*, **not** a first-class
Grad-CAM / Score-CAM / SHAP / attention module. That infrastructure is Paper 2's
contribution and lives in `paper2_causalechonet/baselines/`.

## Why Paper 1 stops here

The approved proposal assigns explainability to **Paper 2 (CausalEchoNet)**,
whose whole novelty is *causal* attribution (via ITE) benchmarked **against**
associational saliency (Grad-CAM / SHAP) to show the two diverge. Building a full
saliency module into Paper 1 would either duplicate that comparison or blunt
Paper 2's "first counterfactual framework…" claim. So Paper 1 deliberately does
not ship one.

## What Paper 1 *may* include (discussion / limitations only)

- One illustrative Grad-CAM overlay — a single representative beat per AAMI class
  (N/S/V/F/Q) — placed in the discussion, purely to show the fused branches
  attend to physiologically plausible regions of the RP/GAF/MTF images.
- A one-paragraph limitations note stating, verbatim in spirit:

  > EchoFuseNet's predictions are here interpreted only qualitatively via a
  > single Grad-CAM overlay. A full, *causal* explanation framework — separating
  > associational saliency from counterfactual attribution — is the subject of
  > follow-up work (Paper 2, CausalEchoNet), and is deliberately out of scope
  > here to keep Paper 1 focused on the lightweight multimodal classifier.

## Explicitly NOT built in Paper 1

- No Grad-CAM/Score-CAM/SHAP/attention-visualisation module in the Paper 1
  pipeline or `paper1_echofusenet/`.
- No saliency baselines table — that belongs to Paper 2.

If a reviewer asks Paper 1 for more interpretability, the response is to point at
the Paper 2 roadmap rather than expand Paper 1's scope. See
[`PAPER1_ENHANCEMENT_CHECKLIST.md`](../PAPER1_ENHANCEMENT_CHECKLIST.md) §1.
