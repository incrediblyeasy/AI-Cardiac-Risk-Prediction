# Day 6 — Data Pipeline Integration

## Goal
Combine RP/GAF/MTF into one multimodal DataLoader with oversampling applied correctly.

## Tasks
- [ ] Build a unified multimodal Dataset/DataLoader producing (RP, GAF, MTF, label) tuples
- [ ] Apply oversampling ONLY to the training fold, and only AFTER the patient-level split (critical correction — do not oversample before splitting)
- [ ] Sanity-check batch shapes and post-oversampling class balance on train; confirm test fold is untouched/natural distribution

## Deliverable / Definition of Done
- End-to-end DataLoader producing correctly balanced train batches and untouched test batches

## Dependencies
Days 3, 4, 5 (all three transforms working).

---

## Daily Update (fill in when done)

**Date completed:**

**Status:** ⬜ Not started · 🟡 In progress · ✅ Done

**What I completed:**
-

**Blockers / issues:**
-

**Notes for next day:**
-

**Time spent:**
