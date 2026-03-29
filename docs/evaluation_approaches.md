# Evaluation Approaches

This document describes the evaluation framework and classification pipelines used in the cross-subject Motor Imagery BCI pipeline.

---

## 1. Evaluation Strategy: Leave-One-Subject-Out (LOSO)

**Implemented via:** `moabb.evaluations.CrossSubjectEvaluation`

### What it does

The dataset contains 29 subjects. LOSO iterates over each subject: in each fold, **one subject is held out as the test set** and the remaining 28 subjects form the training set. The classifier is trained from scratch on each fold. Final performance is the mean accuracy across all 29 folds.

### Why use it

- Directly measures **generalization to unseen subjects** — the core challenge of cross-subject BCI
- No data from the test subject leaks into training at any point
- The result answers the thesis question: *"Can a model trained on other people decode motor imagery for a new user without calibration?"*

### Limitations

- High variance: one poorly-recorded subject can swing results significantly
- Pessimistic estimate — in practice, even a few calibration trials from the new subject would improve accuracy
- Computationally expensive: N full training runs for N subjects

### Difference from within-subject evaluation

Within-subject evaluation trains and tests on the **same subject** (e.g., via k-fold on that subject's sessions). It measures how well the system works for a known user. LOSO measures how well it transfers to an unknown user — a harder and more realistic scenario for a calibration-free BCI.

---

## 2. Feature Extraction: Common Spatial Patterns (CSP)

**Implemented via:** `mne.decoding.CSP(n_components=4, log=True)`

### What it does

CSP is a supervised spatial filtering algorithm for two-class EEG problems. Given epochs from two classes (left hand vs. right hand MI), it learns a set of spatial filters (linear combinations of electrodes) that **maximize the variance ratio between classes**.

The output of CSP for each epoch is a feature vector of `n_components` values. With `log=True`, each feature is the log-variance of the filtered signal, which stabilizes the distribution and tends to improve classifier performance.

With `n_components=4`, the pipeline uses the 2 most discriminative filters for each class (the 2 with highest and 2 with lowest eigenvalues).

### Why use it

- Standard and well-validated for Motor Imagery EEG
- Exploits the known neuroscience: left-hand MI causes ERD (power decrease) over C4 and ERS (power increase) over C3, and vice versa for right-hand
- Reduces the feature space dramatically: from raw epochs `(channels × time_samples)` down to 4 scalar values
- Works well even with only 3 channels (C3, C4, Cz)

### Limitations

- Supervised: requires class labels, so it is fit on the training set and must not see test data (MOABB handles this correctly inside the pipeline)
- Sensitive to artifacts — large-amplitude noise can dominate the covariance matrices
- Assumes stationarity: spatial patterns learned from training subjects may not fully transfer to the test subject (this is the fundamental cross-subject challenge)
- With only 3 channels, the spatial filter space is constrained — CSP has limited degrees of freedom compared to full 64-channel setups

---

## 3. Classifiers

All three classifiers receive the same 4-dimensional CSP log-variance feature vector as input.

---

### 3.1 CSP + LDA (Linear Discriminant Analysis)

**Result: 0.674 ± 0.218**

#### What it does

LDA fits a Gaussian model to each class and classifies by finding the linear decision boundary that maximizes the ratio of between-class to within-class scatter. It assumes both classes share the same covariance matrix.

#### Benefits

- Naturally paired with CSP: CSP log-variance features are approximately Gaussian, satisfying LDA's core assumption
- No hyperparameters to tune
- Very fast to fit and predict
- Interpretable decision boundary

#### Limitations

- Assumes equal covariance across classes and Gaussian feature distributions — violated when features are noisy or non-Gaussian
- Linear boundary only — cannot capture non-linear class structure

---

### 3.2 CSP + Logistic Regression (LR)

**Result: 0.674 ± 0.218**

#### What it does

LR models the posterior probability `P(class | features)` directly by fitting a logistic sigmoid to a linear combination of features. It is discriminative rather than generative.

#### Benefits

- Produces calibrated probability outputs (useful if you need confidence estimates)
- Slightly more robust than LDA when the Gaussian assumption is violated
- L2 regularization by default (via `C` parameter) — helps when training set is small relative to feature dimensionality

#### Limitations

- With only 4 CSP features, LDA and LR are nearly equivalent — this explains the **identical scores** in results
- Requires feature scaling for reliable convergence (StandardScaler is included in the pipeline)
- One more hyperparameter (`C`) compared to LDA, though the default `C=1.0` was used

#### Note on identical LDA/LR results

With 4 features and the class balance seen here, LDA and LR converge to the same decision boundary. LR adds no practical benefit over LDA in this setup. **LDA is preferred for the thesis** as it is the canonical CSP companion with no extra preprocessing required.

---

### 3.3 CSP + SVM (Support Vector Machine, RBF kernel)

**Result: 0.631 ± 0.214**

#### What it does

SVM finds the maximum-margin hyperplane separating classes. With `kernel='rbf'`, it implicitly maps features into a higher-dimensional space, allowing non-linear decision boundaries. `C=1.0` controls the margin/error trade-off; `gamma='scale'` sets the RBF kernel width relative to feature variance.

#### Benefits

- Can model non-linear relationships between CSP features and class labels
- Good generalization when margin is maximized
- In theory, better suited to cases where class distributions overlap non-linearly

#### Limitations

- With only 4 features, the non-linear capacity of RBF is rarely needed — the feature space is already simple
- More sensitive to the choice of `C` and `gamma` than LDA/LR
- Observed **inconsistent per-subject behavior**: better than LDA on some subjects, considerably worse on others (e.g., subjects 3, 15)
- Does not produce calibrated probabilities without additional post-processing
- **Not recommended for the thesis** without hyperparameter tuning (grid search over `C`, `gamma`)

---

## 4. Summary Comparison

| | CSP+LDA | CSP+LR | CSP+SVM |
|---|---|---|---|
| Mean accuracy | 0.674 | 0.674 | 0.631 |
| Decision boundary | Linear | Linear | Non-linear (RBF) |
| Hyperparameters | None | `C` (default 1.0) | `C`, `gamma` |
| Scaling required | No | Yes | Yes |
| Calibrated probabilities | No | Yes | No (by default) |
| Tuning needed | No | No | Yes |
| Recommended for thesis | **Yes** | No (redundant) | Only after tuning |

---

## 5. Pipeline Configuration

All pipelines share:

- **Channels:** C3, C4, Cz
- **Bandpass filter:** 8–30 Hz (mu + beta bands)
- **Epoch window:** −3.5 s to +0.5 s relative to movement onset
- **Baseline correction:** −3.5 s to −3.0 s
- **Resampling:** 500 Hz
- **CSP components:** 4
- **Evaluation:** Leave-One-Subject-Out (29 folds)
- **Chance level:** 0.500 (binary classification)
