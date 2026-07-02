# Day 8 — Branch 2 & 3 + Late Fusion

## Goal
Complete the three-branch architecture and fuse it.

## Tasks
- [x] Replicate/adapt the branch architecture for GAF and MTF inputs
- [x] Implement the late fusion layer combining all three branches
- [x] Verify total model parameter count is ~0.7M as specified

## Deliverable / Definition of Done
- Full EchoFuseNet model assembled end-to-end
- Parameter budget verification report

## Dependencies
Day 7 (Branch 1 pattern established).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-02

**Status:** ✅ Done

**What I completed:**
- New `models/echofusenet.py` with `EchoFuseNet(n_classes=5, widths=(32,64,128,256,256), fusion_hidden=128, dropout=0.3)`:
  - **Three independent `CNNBranch` instances** (`branch_rp`, `branch_gaf`, `branch_mtf`) — Day-7 architecture reused for all modalities but with **no weight sharing** (each image type has distinct texture statistics).
  - **Late fusion**: concatenates the three 256-D branch embeddings → 768-D → MLP head `Linear(768→128) + BatchNorm1d + ReLU + Dropout(0.3) + Linear(128→5)` → class logits.
  - `forward(rp, gaf, mtf)` consumes the fixed Day-6 DataLoader tuple order (`(rp, gaf, mtf, label)`), each input `(B, 1, 256, 256)`, returns `(B, 5)` logits.
- Exported `EchoFuseNet` from `paper1_echofusenet.models`.
- **Parameter budget verified (`scripts/model_summary.py`):**
  | component | params | share |
  |---|---|---|
  | branch_rp | 184,448 | 28.3% |
  | branch_gaf | 184,448 | 28.3% |
  | branch_mtf | 184,448 | 28.3% |
  | fusion | 99,333 | 15.2% |
  | **TOTAL** | **652,677 (0.653M)** | **WITHIN ~0.7M budget** |

  Forward smoke test: `3 × (2,1,256,256)` → logits `(2,5)`, OK.
- Unit tests (`tests/test_echofusenet.py`, 6 tests): logits shape; **total-parameter budget band (0.5M–0.7M)**; three branches are distinct objects (no sharing); perturbing any single modality changes the output (all three wired into the head); gradients flow to every parameter; and an **integration test** feeding a real Day-6 `MultimodalBeatDataset` batch through the model.
- **Full test suite: 79 passing** (up from 73).
- Updated the sprint tracker in `New folder/README.md` (Day 8 ✅).

**Blockers / issues:**
- None. Assembled model is 0.653M — comfortably under the 0.7M budget with the three branches dominating (85%); the fusion head is only 15%. If the paper wants to sit nearer 0.7M exactly, widen `fusion_hidden` rather than the branches.

**Notes for next day (Day 9 — training loop & config system):**
- Model + DataLoader now compose directly: `train_loader` (oversampled DS1) and `test_loader` (natural DS2) from `build_dataloaders()`, and `EchoFuseNet()(rp, gaf, mtf)` → logits vs `label`. Use `CrossEntropyLoss` on the logits.
- Per CLAUDE.md: **config-driven, no hardcoded hyperparameters** — put lr, batch_size, epochs, widths, dropout, oversample flag, seed into a config (dataclass or YAML under `configs/`). Wire deterministic seeds (torch + numpy) for reproducibility.
- Recall the Day-6 imbalance caveat (Q has only 8 real beats): consider class-weighted or focal `CrossEntropyLoss` as an alternative to pure oversampling, and log per-class metrics (macro-F1), not just accuracy, since N dominates ~90%.
- The model runs on CPU here (torch 2.12.1+cpu); the training loop should be device-agnostic (`.to(device)`) so it moves to GPU when available.

**Time spent:** ~1 session (three-branch assembly + late-fusion head + budget verification + tests)
