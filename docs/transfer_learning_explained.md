# Transfer Learning — Theory and Code Walkthrough

## 1. What is Transfer Learning?

Transfer learning reuses a model trained on one dataset (the **source domain**) when working on a different but related dataset (the **target domain**). The bet is that the model has already learned useful representations from the source data, so you don't need to train from scratch on the target.

In our case:
- **Source domain**: BCI Competition IV-2a (9 subjects, cue-based MI, 144 trials/class/subject)
- **Target domain**: Custom dataset (27 subjects, detection-based MI, ~54 trials/class/subject)

The motivation is simple: the custom dataset is small. EEGNet needs enough data to learn subject-invariant EEG patterns — 54 trials/class is borderline. BCI IV-2a has 144/class and represents the same underlying task (left vs. right hand MI), so pretraining there gives the model a head start.

---

## 2. What is EEGNet? (briefly)

EEGNet (`braindecode.models.EEGNet`) is a small CNN specifically designed for EEG. It has ~1700 parameters in our setup (3 channels, 1501 samples). Three blocks:

1. **Temporal conv** — learns which frequencies matter (like a learnable bandpass filter bank)
2. **Depthwise spatial conv** — learns how to combine the 3 electrodes (like a learnable spatial filter)
3. **Separable temporal conv** — integrates features into a compact representation

Output → Dense → Softmax → class probabilities.

It's intentionally tiny so it can generalise from small EEG datasets. Despite this, CSP+LDA still beats it here because CSP is hand-designed for exactly the ERD/ERS signal structure we're exploiting.

---

## 3. The Three Conditions Compared

| Condition | What happens | Why |
|-----------|-------------|-----|
| **custom-only** | Random init → train on custom LOSO | Baseline, no transfer |
| **direct transfer** | Pretrain on BCI IV-2a → evaluate on custom, no adaptation | Measures raw transferability |
| **fine-tuned** | Pretrain on BCI IV-2a → continue training on custom train fold → evaluate | Adaptation to target domain |

Results: `0.510 ± 0.094` → `0.519 ± 0.080` → `0.514 ± 0.068`. Mean barely moves, std drops.

---

## 4. Code Walkthrough — `evaluation/transfer_learning.py`

### Step 1: Load BCI IV-2a

```python
bci2a_paradigm = MotorImagery(
    events=['left_hand', 'right_hand'],
    n_classes=2,
    fmin=8, fmax=30,
    tmin=0.5, tmax=3.5,        # active MI after cue
    channels=['C3', 'C4', 'Cz'],
    resample=500,
)
X_bci, y_bci, _ = bci2a_paradigm.get_data(BNCI2014_001())
X_bci = X_bci[:, :, :1501].astype(np.float32)  # trim 1502→1501
```

Why trim? MOABB resamples BCI IV-2a from 250 Hz to 500 Hz. A 3-second window at 250 Hz gives 751 samples; at 500 Hz that's 1502 (MNE includes both endpoints). The custom dataset at native 500 Hz gives exactly 1501. One sample is trimmed to make shapes match for the shared EEGNet.

### Step 2: Load Custom Dataset

```python
custom_paradigm = MotorImagery(
    tmin=-3.5, tmax=-0.5,   # active MI before movement onset (custom convention)
    baseline=(-3.5, -3.0),
)
X_custom, y_custom, meta_custom = custom_paradigm.get_data(custom_dataset)
```

Both datasets end up as arrays of shape `(n_epochs, 3, 1501)` with the same label encoding. The `LabelEncoder` is fitted once on BCI IV-2a labels and reused for the custom dataset — this guarantees `left_hand=0, right_hand=1` is consistent across both.

### Step 3: Pretrain on all BCI IV-2a

```python
pretrain_clf = EEGClassifier(
    module=EEGNet,                    # class, not instance — required for sklearn clone()
    module__n_chans=3,
    module__n_outputs=2,
    module__n_times=1501,
    module__final_conv_length='auto',
    max_epochs=300,
    lr=0.0001,
    batch_size=64,
    train_split=None,                 # no internal val set, MOABB owns the splits
    device='cuda',
    verbose=0,
)
pretrain_clf.fit(X_bci, y_bci_enc)
pretrained_state = deepcopy(pretrain_clf.module_.state_dict())
```

`pretrain_clf.module_` is the actual PyTorch `nn.Module` (EEGNet). After fitting, `state_dict()` returns a dictionary of all weight tensors. `deepcopy` is critical — without it, later `load_state_dict` calls would modify the same object in memory, so you'd lose the original pretrained weights.

### Step 4: LOSO Loop

For each test subject, two variants are evaluated:

#### Variant A — Direct Transfer

```python
pretrain_clf.module_.load_state_dict(pretrained_state)
y_pred_A = pretrain_clf.predict(X_test)
```

Restore the pretrained weights, then call predict directly without any fitting. The model sees the custom-dataset test subject for the first time with BCI IV-2a knowledge only.

#### Variant B — Fine-Tuning

```python
finetune_clf = EEGClassifier(
    module=EEGNet,
    ...
    max_epochs=100,          # fewer epochs than pretraining
    lr=1e-5,                 # much lower LR (100× smaller)
    warm_start=True,         # KEY: prevents skorch from reinitialising weights on fit()
    device=device,
)
finetune_clf.initialize()                             # allocates module + optimizer
finetune_clf.module_.load_state_dict(pretrained_state)  # inject pretrained weights
finetune_clf.fit(X_train, y_train)                    # fine-tune on custom train fold
y_pred_B = finetune_clf.predict(X_test)
```

**Why a new `EEGClassifier` object?** We can't reuse `pretrain_clf` for fine-tuning because we need a different `lr` and `max_epochs`. Skorch doesn't let you change those after fitting without resetting.

**Why `warm_start=True`?** By default, calling `.fit()` on an `EEGClassifier` that has already been `initialize()`d would reinitialise the module weights (random). `warm_start=True` tells skorch to keep the current weights and just continue training — which is what we want after loading `pretrained_state`.

**Why lower LR (1e-5 vs 1e-4)?** Fine-tuning with a large learning rate risks catastrophic forgetting — the optimizer overwrites the pretrained representations with noise from the small custom dataset. A low LR makes small adjustments that adapt the model to the target domain without destroying source-domain knowledge.

**Why fewer epochs (100 vs 300)?** The custom training fold has ~54 × 26 ≈ 1400 epochs total. Training for 300 epochs on this would heavily overfit. 100 epochs with a low LR is a conservative adaptation.

---

## 5. Key Engineering Detail — `module=EEGNet` (class, not instance)

In sklearn-compatible code, `.clone()` is used to create fresh copies of estimators (e.g., at each LOSO fold). `clone()` works by reading the constructor arguments of the estimator and calling `__init__` again. This requires `module` to be a **class**, not an instance — if you passed `EEGNet(...)`, clone would try to copy the instance object and fail.

The `module__*` prefix (double underscore) is skorch's convention for passing arguments to the `module` constructor. At fit time, skorch calls `EEGNet(n_chans=3, n_outputs=2, n_times=1501, ...)` internally.

---

## 6. Why These Results Make Sense

**Mean accuracy doesn't improve (~0.51 across all conditions)** because:
- Only 3 channels — EEGNet's spatial conv can't exploit wider topographic patterns
- Cross-paradigm domain shift — cue-based (BCI IV-2a) vs detection-based (custom) creates a mismatch that pretraining can't fully bridge
- Small pretraining corpus — 9 subjects × ~2592 trials is modest for a neural network

**Std decreases (0.094 → 0.068)** because:
- The pretrained model has seen many EEG patterns and tends to predict more conservatively (closer to 0.5) rather than wildly wrong
- It stabilises predictions across subjects even without improving the average

**Per-subject heterogeneity is high**:
- Subject 5: 0.270 → 0.625 with direct transfer (transfer helps massively for this subject)
- Subject 14: 0.569 → 0.391 after fine-tuning (fine-tuning hurts this subject)
- This is expected — inter-subject variability dominates, and no fixed cross-subject model works equally well for all users
