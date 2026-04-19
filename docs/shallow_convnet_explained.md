# ShallowConvNet (ShallowFBCSPNet) — Architecture and Rationale

## Overview

ShallowConvNet is a compact convolutional neural network designed specifically for EEG-based Motor Imagery classification. It was introduced by Schirrmeister et al. (2017) as a deep learning analogue to the classical Filter Bank Common Spatial Patterns (FBCSP) pipeline.

Unlike general-purpose CNNs adapted for EEG, ShallowConvNet's architecture directly mirrors the signal processing steps that FBCSP performs — making it interpretable in terms of known BCI signal features.

---

## Architecture

The network consists of three learned layers and one aggregation stage:

```
Input: (batch, channels, time)
   │
   ▼
[Temporal convolution]   — learns bandpass-like filters across time
   │
   ▼
[Spatial convolution]    — learns spatial filters across EEG channels (analogous to CSP)
   │
   ▼
[Squaring + mean pooling] — computes log-band-power features
   │
   ▼
[Log activation]
   │
   ▼
[Fully connected + softmax]  — classification
```

### Layer details

| Layer | What it learns | CSP/FBCSP analogue |
|---|---|---|
| Temporal conv (40 filters, size 25) | Narrow-band spectral filters | Filter bank (mu/beta band isolation) |
| Spatial conv (40 filters, size n_channels) | Weighted combinations of channels | Common Spatial Patterns (CSP) |
| Squaring + avg pool | Signal envelope (instantaneous power) | Band power estimation |
| Log | Log-power transform | Log-variance in CSP features |
| Dense | Linear classifier | LDA / LR |

The design means that if the network learns perfectly, it recovers what FBCSP would compute — but it can also learn deviations from that fixed pipeline when the data supports them.

---

## Why it works for Motor Imagery

Motor Imagery produces Event-Related (De)Synchronization (ERD/ERS) in the mu (8–12 Hz) and beta (13–30 Hz) bands, lateralized over sensorimotor cortex. ShallowConvNet's architecture is tuned to capture exactly this:

- **Temporal conv** isolates mu/beta activity by learning narrow bandpass filters
- **Spatial conv** extracts lateralization patterns between C3, C4, Cz (the same channels used here)
- **Power envelope** captures ERD/ERS magnitude, which is the actual MI signal
- **Log transform** normalizes the power distribution, improving linear separability

---

## Comparison to EEGNet and CSP+LDA

| Property | CSP+LDA | ShallowConvNet | EEGNet |
|---|---|---|---|
| Type | Classical | Shallow CNN | Deep CNN |
| Parameters | ~few (4–8 spatial filters) | ~few thousand | ~more than Shallow |
| Inductive bias | Strong (spatial filters) | Moderate (mirrors FBCSP) | Weak (general) |
| Interpretability | High | Medium | Low |
| Data requirement | Low | Medium | Higher |
| Result (custom, 27 subjects LOSO) | 0.703 ± 0.218 | 0.680 ± 0.165 | 0.510 ± 0.094 |

**Key observation:** On the custom dataset, ShallowConvNet (0.680) nearly matches CSP+LDA (0.703) in mean accuracy while achieving substantially lower variance (0.165 vs 0.218). This suggests it generalizes more consistently across subjects, even though its mean is slightly lower. EEGNet underperforms both — likely because it has less inductive bias and the dataset is too small to compensate.

---

## Hyperparameters (in this project)

Defaults are set in `config/dataset_config.yaml` under `dl:` and can be overridden per-method in `config/pipeline.yaml`.

| Param | Default | Effect |
|---|---|---|
| `max_epochs` | 150 | Training duration — more epochs = more fitting, risk of overfitting |
| `lr` | 0.001 | Adam learning rate — lower = more stable, slower convergence |
| `batch_size` | 64 | Mini-batch size — larger = faster, less noisy gradient estimates |

---

## Epoch window

ShallowConvNet uses the **active MI window**: `tmin=-3.5s` to `tmax=-0.5s` relative to movement onset (3 seconds, 1501 samples at 500 Hz). This window captures the MI preparation phase, where ERD is maximal. This is the same window used by EEGNet in this project.

CSP+LDA uses a wider window that includes the full trial — the distinction is set via paradigm overrides in the pipeline config.

---

## Implementation note

The network is wrapped in braindecode's `EEGClassifier`, which provides an sklearn-compatible `fit`/`predict` interface. A `cast_to_float32` preprocessing step is prepended in the pipeline because MOABB returns `float64` arrays and PyTorch requires `float32`.

See `methods/shallow.py` for the factory implementation.

---

## Reference

Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420.
