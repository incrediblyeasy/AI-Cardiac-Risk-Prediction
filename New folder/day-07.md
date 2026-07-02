# Day 7 — Branch 1 CNN (Depthwise Separable Convs)

## Goal
Build the first lightweight CNN branch and validate its parameter budget.

## Tasks
- [x] Build the lightweight CNN branch architecture using depthwise separable convolutions
- [x] Run a forward-pass smoke test with dummy input
- [x] Log parameter count for this single branch against the overall ~0.7M total budget

## Deliverable / Definition of Done
- Branch1 module (e.g. for RP) with logged parameter count

## Dependencies
Day 6 (pipeline producing correctly shaped inputs).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-02

**Status:** ✅ Done

**What I completed:**
- New `paper1_echofusenet/models/` package with `models/branch.py`:
  - `DepthwiseSeparableConv(in, out, stride)` — MobileNet-style 3x3 depthwise conv (`groups=in_channels`) + 1x1 pointwise conv, each with BatchNorm + ReLU. Replaces a `k*k*Cin*Cout` conv with `k*k*Cin + Cin*Cout` params (~9x cheaper for 3x3), which is what keeps the branch lightweight.
  - `CNNBranch(in_channels=1, widths=(32,64,128,256,256), n_classes=None)` — stride-2 stem conv then a chain of stride-2 depthwise-separable blocks (spatial 256→128→64→32→16→8→4), global-average-pooled to a `widths[-1]`-D embedding. Returns the **embedding** by default (for Day-8 late fusion); pass `n_classes` to attach a standalone linear head returning logits.
  - `count_parameters(module)` utility.
- Forward-pass smoke test (`scripts/branch_summary.py`): dummy `(2,1,256,256)` → embedding `(2,256)`, OK. Also prints the per-stage param breakdown and the full-model budget projection.
- **Parameter budget logged:** single branch = **184,448 params (0.184M)**. Projected full model = 3×branch (553,344) + estimated fusion head (99,077 for `3*256→128→5`) = **652,421 (0.652M)** → **WITHIN the ~0.7M budget**.
- Unit tests (`tests/test_branch.py`, 7 tests): embedding shape, classifier-head logits shape, size-agnostic via `AdaptiveAvgPool`, eval-mode determinism, **parameter budget band (0.15M–0.22M and 3×branch < 0.7M)**, depthwise-separable uses fewer params than a plain 3x3 conv, and gradients flow to every parameter.
- **Full test suite: 73 passing** (up from 66).
- Updated the sprint tracker in `New folder/README.md` (Days 4–7 marked ✅ with dates).

**Blockers / issues:**
- None. Note the branch (0.184M) came in comfortably under budget with headroom; widths were tuned so that three branches + the planned fusion head land at ~0.65M, close to the ~0.7M spec without exceeding it.

**Notes for next day (Day 8 — Branch 2 & 3 + Late Fusion):**
- `CNNBranch` is modality-agnostic (all three inputs are `(B,1,256,256)`), so Branches 2 & 3 are just three separate `CNNBranch()` instances — do **not** share weights (each modality has distinct texture statistics). Instantiate one per modality and consume the `(rp, gaf, mtf, label)` tuple order from Day 6.
- Fusion: concat the three 256-D embeddings → 768-D → small MLP (`768→128→5`) as projected here (~99k params). That keeps the assembled model at ~0.652M; if it needs to sit closer to 0.7M, widen the fusion hidden dim rather than the branches.
- The `n_classes` head on `CNNBranch` is only for standalone/smoke use — the fused model will take the branch **embeddings** (default `n_classes=None`) and classify after concatenation.

**Time spent:** ~1 session (depthwise-separable branch + smoke test + param-budget report + tests)
