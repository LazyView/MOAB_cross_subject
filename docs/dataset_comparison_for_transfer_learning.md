# Source Dataset Comparison for Transfer Learning to Custom MI Dataset

## 1. Custom Dataset Summary (Target Domain)

| Property | Value |
|---|---|
| Paradigm | Endogenous, self-paced / detection-based (force sensor onset) |
| Task | Binary left/right hand MI |
| Channels used | 3 (C3, C4, Cz) |
| Sampling rate | 500 Hz |
| Subjects | 27 (after exclusion) |
| Trials per class | ~52 left, ~56 right per subject |
| Epoch window | −3.5 to −0.5 s relative to movement onset (3 s active MI) |
| Best result | CSP+LDA: 0.703 mean acc; EEGNet fine-tuned from BCI IV-2a: 0.514 |

---

## 2. Candidate Datasets — Detailed Profiles

### 2.1 BCI Competition IV Dataset 2a (Brunner et al., 2008)

| Property | Value |
|---|---|
| Reference | Tangermann et al., 2012; Brunner et al., 2008 |
| Subjects | 9 healthy |
| Channels | 22 EEG + 3 EOG |
| Sampling rate | 250 Hz |
| Paradigm | Cue-based (visual arrow) |
| Classes | 4 (left hand, right hand, feet, tongue) — filterable to 2 |
| Trials per class | 72 per session (144 total across 2 sessions) |
| Sessions | 2 (different days) |
| MI window | ~4 s (0.5–4.5 s after cue onset, typically 0.5–2.5 s used) |
| MOABB class | `BNCI2014_001` |
| Data format | GDF |
| Access | http://bnci-horizon-2020.eu/database/data-sets |

**Strengths:** Gold-standard benchmark; extensively used in MI transfer learning literature; good trial count per subject (144/class); includes C3, C4, Cz; already tested by the user as a source for EEGNet fine-tuning.

**Weaknesses:** Only 9 subjects (severely limits cross-subject pre-training); 250 Hz sampling rate requires resampling alignment; cue-based paradigm mismatch with the custom self-paced paradigm; user already achieved only 0.514 accuracy fine-tuning from this source.

### 2.2 PhysioNet EEG Motor Movement/Imagery Dataset (Schalk et al., 2004)

| Property | Value |
|---|---|
| Reference | Schalk et al., 2004 (BCI2000 system) |
| Subjects | 109 (103 after quality control) |
| Channels | 64 EEG |
| Sampling rate | 160 Hz |
| Paradigm | Cue-based (visual target on screen) |
| Classes | Left fist / right fist imagery (also both fists / both feet) |
| Trials per class | ~21–24 per subject (3 runs × ~7–8 trials per run) |
| Sessions | 1 |
| MI window | ~4 s per trial |
| MOABB class | `PhysionetMI` |
| Data format | EDF+ |
| Access | https://physionet.org/content/eegmmidb/1.0.0/ |

**Strengths:** Largest subject pool of any candidate (103+ subjects); 64-channel coverage includes C3/C4/Cz; extensively used in cross-subject studies; freely available.

**Weaknesses:** Very low sampling rate (160 Hz) — below the custom dataset's 500 Hz and below Nyquist for beta-band features at higher frequencies; critically low trial count per subject (~21–24 per class, lower than the custom dataset's ~54); the task is "open/close fist" rather than pure hand MI, introducing a subtle paradigm difference; known data quality issues in several subjects (6 dropped for anomalies in curated versions).

### 2.3 OpenBMI Dataset (Lee et al., 2019) — GigaDB 100542

| Property | Value |
|---|---|
| Reference | Lee et al., 2019 (GigaScience) |
| Subjects | 54 healthy (ages 24–35) |
| Channels | 62 EEG (Ag/AgCl) |
| Sampling rate | 1000 Hz |
| Paradigm | Cue-based (visual arrow, left/right) |
| Classes | 2 (left/right hand grasping imagery) |
| Trials per class | 100 per session (training: 50/class, test: 50/class) |
| Sessions | 2 (different days) |
| MI window | 4 s (after cue onset) |
| MOABB class | `Lee2019_MI` |
| Data format | MAT |
| Access | http://gigadb.org/dataset/100542 |
| Hardware | BrainAmp (Brain Products) |

**Strengths:** Large subject pool (54); high trial count (100/class/session × 2 sessions = 200/class total); high sampling rate (1000 Hz, easily downsampled to 500 Hz); pure binary left/right hand MI task — exact match to the custom dataset; recorded with BrainAmp (same manufacturer as the custom dataset's V-AMP); 62 channels include all necessary sensorimotor electrodes; already integrated into MOABB; includes BCI illiteracy investigation data.

**Weaknesses:** Cue-based paradigm (like all candidates); visual cue-triggered MI vs. the custom dataset's self-paced onset; the imagery task is hand "grasping" rather than generic hand MI, though the motor cortex activation patterns should be highly similar.

### 2.4 MI Stroke Dataset (Liu et al., 2024)

| Property | Value |
|---|---|
| Reference | Liu et al., 2024 (Scientific Data) |
| Subjects | 50 acute stroke patients |
| Channels | 29 EEG + 2 EOG (semi-dry saline electrodes) |
| Sampling rate | 500 Hz |
| Paradigm | Cue-based (audio instruction) |
| Classes | 2 (left/right hand MI) |
| Trials per class | 20 per subject |
| Sessions | 1 |
| MI window | 4 s |
| MOABB class | `Liu2024` |
| Data format | BIDS (BrainVision) |
| Access | via paper DOI / MOABB |

**Strengths:** Matching sampling rate (500 Hz); binary left/right hand MI; channels include C3/C4/Cz; integrated into MOABB; relatively large subject pool (50).

**Weaknesses:** Critical population mismatch — acute stroke patients exhibit fundamentally different neural activation patterns compared to healthy subjects, including weaker and often atypical lateralization of MI signals. This makes it a poor source for transfer learning to healthy volunteer data. Very low trial count (20/class), even lower than the custom dataset. Motion artifacts reported in 13 of 50 subjects. Semi-dry saline electrodes introduce signal characteristic differences from standard wet Ag/AgCl electrodes.

### 2.5 World Robot Conference MI-BCI Dataset (Yang et al., 2025)

| Property | Value |
|---|---|
| Reference | Yang et al., 2025 (Scientific Data) |
| Subjects | 62 total (51 for 2-class, 11 for 3-class) |
| Channels | 58 EEG |
| Sampling rate | 1000 Hz |
| Paradigm | Cue-based (visual cue) |
| Classes (2C) | Left/right hand grasping |
| Trials per class | 100 per session (2C paradigm) |
| Sessions | 3 (different days) |
| MI window | 4 s |
| MOABB class | Not yet integrated (as of early 2025) |
| Data format | MAT (raw + preprocessed) |
| Access | https://doi.org/10.25452/figshare.plus.22671172 |

**Strengths:** Highest total trial count of all candidates (100/class × 3 sessions = 300/class per subject for the 2C paradigm); large subject pool for the 2C task (51 subjects); multi-day recordings enable cross-session analysis; high sampling rate (1000 Hz); binary left/right hand grasping MI; high reported classification accuracy (85.3% with EEGNet for 2C) suggests strong signal quality.

**Weaknesses:** Very recent publication (March 2025) — not yet integrated into MOABB, requiring custom loading code; cue-based paradigm; dataset availability and community adoption are still limited; the 3C subset only has 11 subjects; data collected in a competition context which may affect ecological validity.

---

## 3. Comparative Matrix

| Criterion | BCI IV-2a | PhysioNet | OpenBMI | Liu2024 Stroke | Yang2025 WRC |
|---|---|---|---|---|---|
| **Task match** (L/R hand MI) | ✅ (from 4-class) | ⚠️ (fist open/close) | ✅ (hand grasp) | ✅ | ✅ (hand grasp) |
| **Channels** (≥ C3, C4, Cz) | ✅ 22 ch | ✅ 64 ch | ✅ 62 ch | ✅ 29 ch | ✅ 58 ch |
| **Sampling rate** (≥500 Hz) | ❌ 250 Hz | ❌ 160 Hz | ✅ 1000 Hz | ✅ 500 Hz | ✅ 1000 Hz |
| **Trials/class/subject** | 144 | ~22 | 200 | 20 | 300 |
| **Subjects** | 9 | 103 | 54 | 50 | 51 (2C) |
| **Total L/R trials** | ~2,592 | ~4,532 | ~21,600 | ~2,000 | ~30,600 |
| **Sessions** | 2 | 1 | 2 | 1 | 3 |
| **Population** | Healthy | Healthy | Healthy | Stroke ❌ | Healthy |
| **Paradigm type** | Cue-based | Cue-based | Cue-based | Cue-based | Cue-based |
| **MOABB integration** | ✅ | ✅ | ✅ | ✅ | ❌ (not yet) |
| **Hardware family** | Unspecified | BCI2000 | BrainAmp ✅ | Neuracle | Neuracle |

> Note: None of the candidate datasets use a self-paced / detection-based paradigm matching the custom dataset. This is acknowledged as a fundamental limitation — self-paced MI datasets are extremely rare in the public domain.

---

## 4. Transfer Learning Suitability Analysis

### Key factors for transfer learning success

For EEGNet-based (or similar DL) transfer learning from a source to the custom target dataset, the most impactful factors are:

1. **Volume of source data** — More pre-training trials produce more robust feature representations. Cross-subject pre-training benefits from both many subjects AND many trials per subject.
2. **Task similarity** — Binary left/right hand MI is the target task; the source should match exactly.
3. **Signal compatibility** — Matching or higher sampling rate avoids information loss during resampling. Shared channel space (C3/C4/Cz minimum) is essential.
4. **Population match** — Healthy volunteers in both source and target ensures similar neural activation patterns.
5. **Paradigm compatibility** — While no candidate matches the self-paced paradigm, cue-based sources can still provide useful spatial filter initialization (CSP patterns over C3/C4/Cz should be broadly similar for left/right MI regardless of paradigm).

### Ranking

**1st — OpenBMI (Lee2019_MI)** ⭐ Recommended

This dataset offers the best overall balance for transfer learning to the custom dataset. It provides 54 subjects with 200 trials per class each, totalling approximately 21,600 left/right MI trials for pre-training — a 7.4× increase over the custom dataset's total trial count. The 1000 Hz sampling rate exceeds the target's 500 Hz, allowing clean downsampling. The task (left/right hand grasping imagery) is a close match to the custom dataset's left/right hand MI. The BrainAmp hardware (Brain Products) is from the same manufacturer family as the custom dataset's V-AMP, which may reduce hardware-specific signal characteristic differences. Critically, it is already fully integrated into MOABB as `Lee2019_MI`, making pipeline integration straightforward.

**2nd — Yang2025 WRC**

This dataset provides the highest raw data volume: 51 subjects × 300 trials/class × 3 sessions = ~30,600 total L/R trials. The multi-session design also provides natural data augmentation through cross-session variability. However, it was published in March 2025 and is not yet available in MOABB, meaning the user would need to write custom loading code. If MOABB integration becomes available or custom loading is feasible, this dataset could outperform OpenBMI as a source due to its sheer volume.

**3rd — BCI Competition IV 2a**

Already tested by the user with EEGNet fine-tuning (result: 0.514 accuracy). The limited subject count (9) restricts the diversity of learned representations. The 250 Hz sampling rate requires upsampling or downsampling alignment. While this is the most-cited MI benchmark, its small scale makes it suboptimal as a sole pre-training source for cross-subject transfer learning.

**4th — PhysioNet EEGMMIDB**

Despite having 103 subjects, the critically low trial count per subject (~22/class) and low sampling rate (160 Hz) severely limit its utility. The data quality is also questionable for several subjects. The large subject pool could theoretically help with learning cross-subject invariances, but the sparse per-subject data makes this advantage marginal.

**5th — Liu2024 Stroke Dataset** ❌ Not recommended

The acute stroke patient population exhibits fundamentally different cortical activation patterns from healthy volunteers. Transfer from stroke patients to healthy subjects would likely introduce negative transfer — the model would learn signal characteristics (weaker ERD, atypical lateralization, compensatory activity) that do not generalize to the target domain. The very low trial count (20/class) further disqualifies this dataset.

---

## 5. Recommendation

**Use OpenBMI (Lee2019_MI) as the primary source dataset for transfer learning.**

The rationale in summary:

- 54 subjects × 200 trials/class provides substantial pre-training volume
- 1000 Hz → 500 Hz downsampling is clean and lossless for the 8–30 Hz band of interest
- Binary left/right hand MI task matches the custom dataset
- Same hardware manufacturer family (Brain Products)
- Full MOABB integration (`Lee2019_MI`) enables immediate use in existing pipelines
- Two-session design provides natural cross-session variability during pre-training

**Potential secondary source:** If additional pre-training data is desired, consider combining OpenBMI with BCI IV-2a (despite its small subject pool, its 144 trials/class adds diversity). Multi-source transfer learning — pre-training on a combined pool from both datasets — could further improve robustness, though this adds complexity to the pipeline.

**Future consideration:** Monitor MOABB integration of the Yang2025 WRC dataset. If it becomes available, its 300 trials/class × 51 subjects × 3 sessions makes it an even more powerful pre-training source, and it could either replace or augment OpenBMI as the primary source.

---

## 6. Paradigm Mismatch Note

All five candidates use cue-based paradigms, while the custom dataset uses a self-paced / detection-based paradigm. This is the single largest source of domain shift for transfer learning. The practical impact is:

- **Temporal structure differs:** In cue-based paradigms, MI onset is time-locked to a visual cue. In the custom dataset, MI onset is detected by a force sensor, meaning the temporal relationship between MI planning and execution is unconstrained.
- **Epoch alignment differs:** Cue-based epochs are aligned to cue onset; custom epochs are aligned to movement onset (working backward in time).
- **Mitigation:** Focus transfer learning on learning spatial filters (CSP-like patterns over C3/C4) rather than temporal dynamics. EEGNet's depthwise spatial convolution layer is the component most likely to transfer well, as the spatial topography of left/right MI (contralateral ERD over C3/C4) is paradigm-independent. Consider freezing early spatial layers and fine-tuning only temporal and classification layers.

---

*Document prepared for Filip's Bachelor's thesis — Motor Imagery BCI Transfer Learning*
*Based on web research conducted April 2026*
