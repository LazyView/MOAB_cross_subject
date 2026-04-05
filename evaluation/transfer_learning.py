"""
Transfer learning evaluation: EEGNet pretrained on BCI IV-2a, tested on custom dataset.

Three conditions compared (all LOSO on the custom dataset):
  1. custom-only    — loaded from existing results CSV (task 1)
  2. direct-transfer — pretrained on BCI IV-2a, tested with no adaptation
  3. fine-tuned     — pretrained on BCI IV-2a, fine-tuned on custom training subjects per fold

Pretraining uses all 9 BCI IV-2a subjects. Fine-tuning uses a lower LR for fewer epochs.
BCI IV-2a epochs are trimmed from 1502 to 1501 samples to match the custom dataset.

Results saved to results/transfer_learning.csv
Run from project root: python evaluation/transfer_learning.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
import mne
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from braindecode.models import EEGNet
from braindecode.classifier import EEGClassifier
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from dataset.custom_dataset import MotorImageryDataset

mne.set_log_level('WARNING')

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

EXCLUDE_OUTLIERS = True
OUTLIER_SUBJECTS = [7, 8]

# --- Config ---
with open("config/dataset_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

sig_cfg = config['signal']
dl_cfg = config['dl']
N_TIMES = dl_cfg['n_times']   # 1501 — canonical size for both datasets
N_CHANS = len(sig_cfg['channels'])  # 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 1. Load BCI IV-2a
# ---------------------------------------------------------------------------
print("\n[1/4] Loading BCI IV-2a...")

bci2a_paradigm = MotorImagery(
    events=['left_hand', 'right_hand'],
    n_classes=2,
    fmin=sig_cfg['filter']['l_freq'],
    fmax=sig_cfg['filter']['h_freq'],
    tmin=0.5,
    tmax=3.5,
    channels=sig_cfg['channels'],
    resample=sig_cfg['sfreq'],
)

X_bci, y_bci, _ = bci2a_paradigm.get_data(BNCI2014_001())

# Trim 1502 → 1501 samples (resampling artefact, see docs/design_decisions.md)
X_bci = X_bci[:, :, :N_TIMES].astype(np.float32)

le = LabelEncoder()
y_bci_enc = le.fit_transform(y_bci)

print(f"    BCI IV-2a: {X_bci.shape}  classes: {le.classes_}")

# ---------------------------------------------------------------------------
# 2. Load custom dataset
# ---------------------------------------------------------------------------
print("[2/4] Loading custom dataset...")

custom_dataset = MotorImageryDataset(
    data_path=config['data_path'],
    config_path="config/dataset_config.yaml"
)
if EXCLUDE_OUTLIERS:
    custom_dataset.subject_list = [
        s for s in custom_dataset.subject_list if s not in OUTLIER_SUBJECTS
    ]

custom_paradigm = MotorImagery(
    events=['left_hand', 'right_hand'],
    n_classes=2,
    fmin=sig_cfg['filter']['l_freq'],
    fmax=sig_cfg['filter']['h_freq'],
    tmin=dl_cfg['tmin'],
    tmax=dl_cfg['tmax'],
    baseline=tuple(config['epoch']['baseline']),
)

X_custom, y_custom, meta_custom = custom_paradigm.get_data(custom_dataset)
X_custom = X_custom.astype(np.float32)
y_custom_enc = le.transform(y_custom)   # same encoder as BCI IV-2a

subject_ids = meta_custom['subject'].values
unique_subjects = np.unique(subject_ids)

print(f"    Custom:    {X_custom.shape}  subjects: {len(unique_subjects)}")

# ---------------------------------------------------------------------------
# 3. Pretrain EEGNet on all BCI IV-2a data
# ---------------------------------------------------------------------------
print(f"\n[3/4] Pretraining EEGNet on BCI IV-2a ({dl_cfg['max_epochs']} epochs)...")

pretrain_clf = EEGClassifier(
    module=EEGNet,
    module__n_chans=N_CHANS,
    module__n_outputs=2,
    module__n_times=N_TIMES,
    module__final_conv_length='auto',
    max_epochs=dl_cfg['max_epochs'],
    lr=dl_cfg['lr'],
    batch_size=dl_cfg['batch_size'],
    train_split=None,
    device=device,
    verbose=0,
)
pretrain_clf.fit(X_bci, y_bci_enc)
pretrained_state = deepcopy(pretrain_clf.module_.state_dict())
print("    Pretraining done.")

# ---------------------------------------------------------------------------
# 4. LOSO evaluation on custom dataset
# ---------------------------------------------------------------------------
print(f"\n[4/4] LOSO on custom dataset ({len(unique_subjects)} subjects)...")
print(f"      Direct transfer: no adaptation")
print(f"      Fine-tuning    : {dl_cfg['fine_tune_epochs']} epochs, lr={dl_cfg['fine_tune_lr']}")
print()

records = []

for i, test_subj in enumerate(unique_subjects):
    train_mask = subject_ids != test_subj
    test_mask  = subject_ids == test_subj

    X_train, y_train = X_custom[train_mask], y_custom_enc[train_mask]
    X_test,  y_test  = X_custom[test_mask],  y_custom_enc[test_mask]

    # --- Variant A: direct transfer (restore pretrained weights, no training) ---
    pretrain_clf.module_.load_state_dict(pretrained_state)
    y_pred_A = pretrain_clf.predict(X_test)
    score_A = accuracy_score(y_test, y_pred_A)

    # --- Variant B: fine-tune on custom training subjects ---
    finetune_clf = EEGClassifier(
        module=EEGNet,
        module__n_chans=N_CHANS,
        module__n_outputs=2,
        module__n_times=N_TIMES,
        module__final_conv_length='auto',
        max_epochs=dl_cfg['fine_tune_epochs'],
        lr=dl_cfg['fine_tune_lr'],
        batch_size=dl_cfg['batch_size'],
        train_split=None,
        warm_start=True,   # prevents skorch from reinitialising the module on fit()
        device=device,
        verbose=0,
    )
    finetune_clf.initialize()                                    # fresh module + optimizer
    finetune_clf.module_.load_state_dict(pretrained_state)       # load pretrained weights
    finetune_clf.fit(X_train, y_train)                           # fine-tune
    y_pred_B = finetune_clf.predict(X_test)
    score_B = accuracy_score(y_test, y_pred_B)

    records.append({
        'subject':         test_subj,
        'direct_transfer': score_A,
        'fine_tuned':      score_B,
    })
    print(f"  [{i+1:2d}/{len(unique_subjects)}] subject {test_subj:2d}  "
          f"direct={score_A:.3f}  fine-tuned={score_B:.3f}")

# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(records)

# Load custom-only baseline from task 1
custom_only_path = RESULTS_DIR / "dl_cross_subject_eval_excluded_outliers.csv"
if custom_only_path.exists():
    custom_only = pd.read_csv(custom_only_path)
    custom_only_scores = custom_only[custom_only['pipeline'] == 'EEGNet'].set_index('subject')['score']
    results_df['custom_only'] = results_df['subject'].map(custom_only_scores)
else:
    print(f"\nWARNING: {custom_only_path} not found — run dl_cross_subject_eval.py first.")

print("\n=== Transfer Learning Results ===")
cols = [c for c in ['custom_only', 'direct_transfer', 'fine_tuned'] if c in results_df.columns]
print(results_df[['subject'] + cols].to_string(index=False, float_format='{:.3f}'.format))

print("\n=== Mean ± Std ===")
for col in cols:
    m, s = results_df[col].mean(), results_df[col].std()
    print(f"  {col:<20} {m:.3f} ± {s:.3f}")
print(f"  {'chance':<20} 0.500")

out_path = RESULTS_DIR / "transfer_learning.csv"
results_df.to_csv(out_path, index=False)
print(f"\nResults saved → {out_path}")
