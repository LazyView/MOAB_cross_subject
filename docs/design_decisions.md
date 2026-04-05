# Design Decisions

Decisions made during implementation planning. Update this file when a decision changes.

---

## ML/DL Model

**Decision:** EEGNet via braindecode  
**Alternatives considered:** ShallowConvNet, EEG Conformer  
**Reason:** EEGNet is explicitly designed for small EEG datasets (~840 epochs/class, 27 subjects); most cited compact CNN for BCI; ~2k parameters avoids overfitting risk.  
**One model only** — adding more models is out of scope for the thesis.

---

## Dataset Analysis

**Datasets:** Custom (BrainVision, 29 subjects) + BCI Competition IV-2a (MOABB: `BNCI2014_001`)  
**Channel strategy:** Use C3, C4, Cz for both datasets. For BCI IV-2a, select these three channels from the full 22-channel recording.  
**Reason:** Transfer learning requires identical input dimensionality across datasets.

---

## Epoch Window (cross-dataset alignment)

**Problem:** The two datasets use different epoch anchors:
- Custom dataset: anchored to **movement onset** → MI activity is *before* the marker
- BCI IV-2a: anchored to **cue onset** → MI activity is *after* the cue

**Decision:** Use a shared 3-second active-MI window (Option A):

| Dataset      | tmin   | tmax   | Duration | Samples (500 Hz) |
|--------------|--------|--------|----------|------------------|
| Custom       | -3.5 s | -0.5 s | 3 s      | 1500             |
| BCI IV-2a    | +0.5 s | +3.5 s | 3 s      | 1500             |

**Impact on existing pipeline:** CSP baselines were computed with the old custom window (-3.5 to +0.5 s). The DL pipeline uses a new window; CSP results are unchanged and remain comparable on their own terms.  
**BCI IV-2a resample target:** 500 Hz (same as custom dataset).

---

## Transfer Learning

**Strategy:** Variant B — pretrain + fine-tune  
- **Pretrain:** BCI IV-2a (larger, public benchmark)  
- **Fine-tune:** Custom dataset  
- **Comparison baselines:** (1) EEGNet trained only on custom dataset (no pretraining), (2) direct transfer without fine-tuning (Variant A, zero-shot)

---

## Thesis

**Limitations section:** Added as a subsection inside the Discussion chapter.  
**Existing content:** May be modified if it improves quality (this overrides the earlier "do not modify" rule).
