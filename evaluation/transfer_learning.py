"""
Transfer learning evaluation: EEGNet pretrained on BCI IV-2a, tested on custom dataset.

Three conditions compared (all LOSO on the custom dataset, same script and seed):
  1. custom-only    — fresh EEGNet trained from scratch on custom training subjects
  2. direct-transfer — pretrained on BCI IV-2a, tested with no adaptation
  3. fine-tuned     — pretrained on BCI IV-2a, fine-tuned on custom training subjects per fold

Pretraining uses all 9 BCI IV-2a subjects. Fine-tuning uses a lower LR for fewer epochs.
BCI IV-2a epochs are trimmed from 1502 to 1501 samples to match the custom dataset.
A fixed seed makes all three conditions reproducible and paired per subject.

Results saved to results/transfer_learning.csv.

Can be run standalone:
    python evaluation/transfer_learning.py

Or called from run_pipeline.py when transfer_learning.enabled: true in pipeline.yaml.
"""

import sys
from datetime import datetime
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

SEED = 42


def _set_seed(seed: int) -> None:
    """Reset PyTorch / NumPy RNGs so every fresh classifier starts from the same state."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(dataset_cfg: dict, pipeline_cfg: dict) -> None:
    """
    Run transfer learning evaluation.

    Parameters
    ----------
    dataset_cfg  : dict  Contents of dataset_config.yaml.
    pipeline_cfg : dict  Contents of pipeline.yaml (for output path).
    """
    sig_cfg  = dataset_cfg['signal']
    dl_cfg   = dataset_cfg['dl']
    out_cfg  = pipeline_cfg['output']

    n_times  = dl_cfg['n_times']          # 1501 — canonical size for both datasets
    n_chans  = len(sig_cfg['channels'])   # 3
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'

    results_dir = Path(out_cfg['results_dir'])
    results_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("Running: Transfer Learning (BCI IV-2a → custom)")
    print(f"{'='*60}")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load BCI IV-2a
    # -------------------------------------------------------------------------
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
    X_bci = X_bci[:, :, :n_times].astype(np.float32)   # trim 1502 → 1501

    le = LabelEncoder()
    y_bci_enc = le.fit_transform(y_bci)
    print(f"    BCI IV-2a: {X_bci.shape}  classes: {le.classes_}")

    # -------------------------------------------------------------------------
    # 2. Load custom dataset
    # -------------------------------------------------------------------------
    print("[2/4] Loading custom dataset...")

    custom_dataset = MotorImageryDataset(
        data_path=dataset_cfg['data_path'],
        config_path="config/dataset_config.yaml"
    )

    custom_paradigm = MotorImagery(
        events=['left_hand', 'right_hand'],
        n_classes=2,
        fmin=sig_cfg['filter']['l_freq'],
        fmax=sig_cfg['filter']['h_freq'],
        tmin=dl_cfg['tmin'],
        tmax=dl_cfg['tmax'],
        baseline=tuple(dataset_cfg['epoch']['baseline']),
    )

    X_custom, y_custom, meta_custom = custom_paradigm.get_data(custom_dataset)
    X_custom = X_custom.astype(np.float32)
    y_custom_enc = le.transform(y_custom)

    subject_ids     = meta_custom['subject'].values
    unique_subjects = np.unique(subject_ids)
    print(f"    Custom:    {X_custom.shape}  subjects: {len(unique_subjects)}")

    # -------------------------------------------------------------------------
    # 3. Pretrain EEGNet on all BCI IV-2a data
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Pretraining EEGNet on BCI IV-2a ({dl_cfg['max_epochs']} epochs)...")

    _set_seed(SEED)
    pretrain_clf = EEGClassifier(
        module=EEGNet,
        module__n_chans=n_chans,
        module__n_outputs=2,
        module__n_times=n_times,
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

    # -------------------------------------------------------------------------
    # 4. LOSO evaluation on custom dataset
    # -------------------------------------------------------------------------
    print(f"\n[4/4] LOSO on custom dataset ({len(unique_subjects)} subjects)...")
    print(f"      Custom-only     : train from scratch, {dl_cfg['max_epochs']} epochs, lr={dl_cfg['lr']}")
    print(f"      Direct transfer : no adaptation")
    print(f"      Fine-tuning     : {dl_cfg['fine_tune_epochs']} epochs, lr={dl_cfg['fine_tune_lr']}")
    print()

    records = []

    for i, test_subj in enumerate(unique_subjects):
        train_mask = subject_ids != test_subj
        test_mask  = subject_ids == test_subj

        X_train, y_train = X_custom[train_mask], y_custom_enc[train_mask]
        X_test,  y_test  = X_custom[test_mask],  y_custom_enc[test_mask]

        # Variant 0: custom-only (fresh EEGNet, no BCI IV-2a)
        _set_seed(SEED)
        custom_only_clf = EEGClassifier(
            module=EEGNet,
            module__n_chans=n_chans,
            module__n_outputs=2,
            module__n_times=n_times,
            module__final_conv_length='auto',
            max_epochs=dl_cfg['max_epochs'],
            lr=dl_cfg['lr'],
            batch_size=dl_cfg['batch_size'],
            train_split=None,
            device=device,
            verbose=0,
        )
        custom_only_clf.fit(X_train, y_train)
        y_pred_0 = custom_only_clf.predict(X_test)
        score_0  = accuracy_score(y_test, y_pred_0)

        # Variant A: direct transfer
        pretrain_clf.module_.load_state_dict(pretrained_state)
        y_pred_A = pretrain_clf.predict(X_test)
        score_A  = accuracy_score(y_test, y_pred_A)

        # Variant B: fine-tune on custom training subjects
        _set_seed(SEED)
        finetune_clf = EEGClassifier(
            module=EEGNet,
            module__n_chans=n_chans,
            module__n_outputs=2,
            module__n_times=n_times,
            module__final_conv_length='auto',
            max_epochs=dl_cfg['fine_tune_epochs'],
            lr=dl_cfg['fine_tune_lr'],
            batch_size=dl_cfg['batch_size'],
            train_split=None,
            warm_start=True,
            device=device,
            verbose=0,
        )
        finetune_clf.initialize()
        finetune_clf.module_.load_state_dict(pretrained_state)
        finetune_clf.fit(X_train, y_train)
        y_pred_B = finetune_clf.predict(X_test)
        score_B  = accuracy_score(y_test, y_pred_B)

        records.append({
            'subject':         test_subj,
            'custom_only':     score_0,
            'direct_transfer': score_A,
            'fine_tuned':      score_B,
        })
        print(f"  [{i+1:2d}/{len(unique_subjects)}] subject {test_subj:2d}  "
              f"custom-only={score_0:.3f}  direct={score_A:.3f}  fine-tuned={score_B:.3f}")

    # -------------------------------------------------------------------------
    # 5. Results
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(records)

    print("\n=== Transfer Learning Results ===")
    cols = ['custom_only', 'direct_transfer', 'fine_tuned']
    print(results_df[['subject'] + cols].to_string(index=False, float_format='{:.3f}'.format))

    print("\n=== Mean ± Std ===")
    for col in cols:
        m, s = results_df[col].mean(), results_df[col].std()
        print(f"  {col:<20} {m:.3f} ± {s:.3f}")
    print(f"  {'chance':<20} 0.500")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = results_dir / f"transfer_learning_{timestamp}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    with open("config/dataset_config.yaml", 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    with open("config/pipeline.yaml", 'r') as f:
        pipeline_cfg = yaml.safe_load(f)
    run(dataset_cfg, pipeline_cfg)
