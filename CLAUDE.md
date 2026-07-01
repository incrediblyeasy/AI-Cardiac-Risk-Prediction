# EchoFuseNet — Project Guide (CLAUDE.md)

AI system for personalized cardiac risk prediction. **Paper 1 = EchoFuseNet**: a
multimodal ECG-beat classifier that encodes each heartbeat as three
signal-to-image transforms (Recurrence Plot, Gramian Angular Field, Markov
Transition Field) and fuses three lightweight CNN branches via late fusion.

The build is scheduled as a 15-day sprint. See `New folder/README.md` for the
day-by-day plan and `New folder/day-XX.md` for each day's tasks and daily log.

## Repository structure

```
paper1_echofusenet/          # Paper 1 package
    data/
        aami.py              # MIT-BIH symbol -> AAMI class (N/S/V/F/Q) mapping
        splits.py            # de Chazal inter-patient DS1 (train) / DS2 (test) split
        download.py          # MIT-BIH Arrhythmia DB downloader (via wfdb)
        mitbih.py            # Record loading + AAMI-labeled beat extraction
    transforms/              # RP / GAF / MTF encoders            (Days 3-5)
    models/                  # CNN branches + late fusion         (Days 7-8)
    training/                # train loop, config system          (Day 9+)
shared/                      # code reused across papers
docs/                        # reference docs (DS1/DS2, AAMI, protocol notes)
tests/                       # pytest suite (leakage guard lives here from Day 2)
configs/                     # experiment configs                 (Day 9+)
```

## Conventions

- **Inter-patient protocol is mandatory.** Train on DS1 patients, test on DS2
  patients. No patient ID may appear in both. The 4 paced records (102, 104,
  107, 217) are excluded per AAMI recommendation.
- **No leakage.** Any resampling/oversampling happens *after* the patient split
  and *only* on the training fold. A leakage unit test guards this (Day 2).
- **Config-driven.** No hardcoded hyperparameters in training code (Day 9).
- **Reproducible.** Deterministic seeds; data download is scripted, not manual.

## Commands

```bash
pip install -r requirements.txt          # or: pip install -e ".[dev]"
python -m paper1_echofusenet.data.download   # download MIT-BIH into data/raw/mitdb
pytest                                    # run the test suite
```

## Data layout (git-ignored)

Raw databases live under `data/raw/` and are **not** committed:
`data/raw/mitdb/` (MIT-BIH), later `data/raw/incartdb/`, `data/raw/ptbxl/`.
