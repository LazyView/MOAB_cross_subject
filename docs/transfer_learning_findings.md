# Transfer Learning Findings

## Results summary

| Condition            | Mean  | Std   |
|----------------------|-------|-------|
| CSP+LDA (baseline)   | 0.703 | —     |
| EEGNet custom-only   | 0.510 | 0.094 |
| EEGNet direct transfer | 0.519 | 0.080 |
| EEGNet fine-tuned    | 0.514 | 0.068 |
| Chance               | 0.500 | —     |

Full per-subject scores: `results/transfer_learning.csv`

## Key findings

1. **Transfer learning does not improve mean accuracy** — all three EEGNet conditions sit at chance regardless of pretraining strategy.

2. **Std decreases with transfer** — 0.094 (custom-only) → 0.080 (direct) → 0.068 (fine-tuned). Pretraining stabilises predictions across subjects even without improving the mean. This is the primary measurable benefit of transfer learning in this setup.

3. **Subject-level heterogeneity** — transfer helps some subjects substantially (e.g. subject 5: 0.270 → 0.625 direct transfer) while hurting others (e.g. subject 14: 0.569 → 0.391 fine-tuned). High inter-subject variance makes average-level conclusions insufficient on their own.

4. **CSP+LDA dominates by a large margin** — 0.703 vs ~0.514 best EEGNet condition.

## Why EEGNet underperforms in this setup

- **Only 3 channels** — EEGNet is designed to exploit spatial information across many electrodes; restricting to C3/C4/Cz eliminates most of that advantage.
- **Cross-paradigm domain shift** — BCI IV-2a is cue-triggered (MI follows a cue), the custom dataset is movement-detection based (MI precedes an onset marker). Epoch alignment reduces but does not eliminate this mismatch.
- **Small pretraining corpus** — 9 BCI IV-2a subjects (~2592 epochs) is modest for deep learning; CSP can generalise from the same data with far less.

## Thesis narrative

Classical spatial filtering (CSP+LDA) is more robust than EEGNet when channels and training data are limited. Transfer learning from BCI IV-2a to the custom dataset reduces prediction variance but does not overcome the fundamental domain gap. These results motivate future work with full-cap recordings and larger pretraining corpora.
