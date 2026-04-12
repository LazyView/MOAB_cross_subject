# Custom Motor Imagery Dataset — Description for Dataset Comparison

## Recording

| Property | Value |
|---|---|
| Hardware | BrainProducts V-AMP 16 amplifier |
| File format | BrainVision (.eeg / .vhdr / .vmrk) |
| Original channels | 18 EEG |
| Channels used for MI | 3 (C3, C4, Cz — 10–20 system) |
| Sampling rate (Mochura subjects) | 500 Hz |
| Sampling rate (Saleh subjects) | 1000 Hz (downsampled to 500 Hz) |
| Online filtering | None stated |

## Paradigm

| Property | Value |
|---|---|
| Task | Binary left-hand vs right-hand motor imagery |
| Paradigm type | **Endogenous, self-paced / detection-based** — no visual cue, MI precedes a force-onset marker |
| MI onset detection | Force sensor (Event 4 / Event 5) |
| Trial structure | 10 s rest phase → 20 s movement phase (multiple MI events per phase) |
| Feedback | None (offline recording) |
| Runs per subject | 2–4 (1–2 per hand), recorded as separate files |
| Sessions per subject | 1 (single recording day per subject) |

## Subjects

| Property | Value |
|---|---|
| Total subjects | 29 |
| After outlier exclusion | 27 |
| Excluded subjects | 2 (atypical lateralisation / BCI illiteracy, confirmed by EEG diagnostics) |
| Population | Mixed (healthy volunteers, dates 2020–2023) |
| Subject identification | Composite key `DDMMYYYY_ID` (ID only unique within recording date) |

## Epochs

| Property | Value |
|---|---|
| Epoch window (CSP experiments) | −3.5 s to +0.5 s relative to movement onset (2001 samples) |
| Epoch window (DL / transfer learning) | −3.5 s to −0.5 s relative to movement onset (1501 samples, active MI only) |
| Baseline correction | −3.5 s to −3.0 s |
| Bandpass filter | 8–30 Hz |
| Trials per class (mean) | ~51.5 left-hand, ~56.0 right-hand per subject |
| Class balance | Slight imbalance (~8% more right-hand trials) |
| Total epochs (27 subjects) | ~1391 left-hand, ~1511 right-hand |

## Classification Performance (LOSO, 27 subjects)

| Pipeline | Mean accuracy | Std |
|---|---|---|
| CSP + LDA | 0.703 | — |
| EEGNet (custom-only) | 0.510 | 0.094 |
| EEGNet (fine-tuned from BCI IV-2a) | 0.514 | 0.068 |
| Chance level | 0.500 | — |

Per-subject range with CSP+LDA: **0.324 – 1.000** (high inter-subject variability)

## Key Characteristics to Match When Looking for a Source Dataset

- **Paradigm type**: Ideally another detection/movement-onset paradigm (not purely cue-triggered) — this is rare, most public datasets are cue-based
- **Channels**: Only C3, C4, Cz are used, so the source dataset needs at least these three (any standard 10–20 cap will have them)
- **Sampling rate**: 500 Hz or higher (can be resampled down)
- **Task**: Binary left/right hand MI
- **Epoch duration**: ~3 s active MI window
- **Size**: More trials per subject than ~54 is the main goal — BCI IV-2a's 144/class is already better; datasets with 200+ trials/class/subject would be ideal
