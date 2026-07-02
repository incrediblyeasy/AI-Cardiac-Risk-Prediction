# 15-Day Sprint Plan — EchoFuseNet (Paper 1)

Scope: this covers the core technical build of Paper 1 (EchoFuseNet) — environment through domain-transfer evaluation. Manuscript drafting and final submission polish come after Day 15, inside the remaining Phase 1 weeks (see `CLAUDE.md`).

## How to use this

1. Each day has its own file: `day-01.md` … `day-15.md`.
2. Work only the day you're on. Check off tasks as you go.
3. At end of day, fill in the **Daily Update** section at the bottom of that day's file (status, what you completed, blockers, time spent).
4. Update the tracker table below as you finish each day.
5. Days generally depend on the prior day's output — check the "Dependencies" line in each file before starting.

## Progress Tracker

| Day | Focus | Status | Date Completed |
|---|---|---|---|
| 1 | Environment & Repo Setup + Data Acquisition | ✅ | 2026-07-01 |
| 2 | Patient-Level Split & Leakage Guard | ✅ | 2026-07-01 |
| 3 | Recurrence Plot (RP) Transform | ✅ | 2026-07-01 |
| 4 | Gramian Angular Field (GAF) Transform | ✅ | 2026-07-02 |
| 5 | Markov Transition Field (MTF) Transform | ✅ | 2026-07-02 |
| 6 | Data Pipeline Integration | ✅ | 2026-07-02 |
| 7 | Branch 1 CNN (Depthwise Separable Convs) | ✅ | 2026-07-02 |
| 8 | Branch 2 & 3 + Late Fusion | ✅ | 2026-07-02 |
| 9 | Training Loop & Config System | ⬜ | |
| 10 | First Full Training Run (DS1/DS2) | ⬜ | |
| 11 | k-Fold CV + Statistical Significance | ⬜ | |
| 12 | Ablation Study | ⬜ | |
| 13 | Latency & Size Benchmark | ⬜ | |
| 14 | External Validation — INCART | ⬜ | |
| 15 | Domain-Transfer — PTB-XL + Consolidation | ⬜ | |

Status legend: ⬜ Not started · 🟡 In progress · ✅ Done

## Notes

- Days 3, 4, 5 (RP/GAF/MTF) can run in parallel if you have bandwidth — they all only depend on Day 2's beat extraction, not on each other.
- Day 13 (benchmark) only depends on Day 8's assembled model — can be pulled earlier if you want the deployability check done sooner.
- If any day's result looks off (especially Day 10's accuracy band, or Day 2's leakage test), stop and fix before moving forward — don't stack more work on a broken foundation.
