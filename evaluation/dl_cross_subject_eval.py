"""
Cross-subject evaluation using EEGNet (braindecode) with MOABB CrossSubjectEvaluation.

Epoch window: tmin to tmax from config (default -3.5 to -0.5 s relative to movement onset).
This is the active MI window (3 s × 500 Hz = 1501 samples), aligned with BCI IV-2a for
transfer learning. CSP baselines use a different window and are not affected.

Results saved to results/dl_cross_subject_eval[_excluded_outliers].csv.
Run from project root: python evaluation/dl_cross_subject_eval.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import mne
import yaml
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from braindecode.models import EEGNet
from braindecode.classifier import EEGClassifier
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import MotorImagery

from dataset.custom_dataset import MotorImageryDataset

mne.set_log_level('WARNING')

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

EXCLUDE_OUTLIERS = True
OUTLIER_SUBJECTS = [7, 8]

# --- Load config ---
with open("config/dataset_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

dl_cfg = config['dl']
sig_cfg = config['signal']

# --- Derived constants ---
n_chans = len(sig_cfg['channels'])  # 3 (C3, C4, Cz)
# MNE Epochs include both endpoints: n_times = round((tmax - tmin) * sfreq) + 1
n_times = int(round((dl_cfg['tmax'] - dl_cfg['tmin']) * sig_cfg['sfreq'])) + 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"EEGNet input: {n_chans} channels × {n_times} samples")

# --- Init dataset ---
dataset = MotorImageryDataset(
    data_path=config['data_path'],
    config_path="config/dataset_config.yaml"
)

# --- Init paradigm with active-MI epoch window ---
paradigm = MotorImagery(
    events=['left_hand', 'right_hand'],
    n_classes=2,
    fmin=sig_cfg['filter']['l_freq'],
    fmax=sig_cfg['filter']['h_freq'],
    tmin=dl_cfg['tmin'],
    tmax=dl_cfg['tmax'],
    baseline=tuple(config['epoch']['baseline']),
)

# --- Optionally exclude outlier subjects ---
if EXCLUDE_OUTLIERS:
    dataset.subject_list = [s for s in dataset.subject_list if s not in OUTLIER_SUBJECTS]
    print(f"Excluded subjects: {OUTLIER_SUBJECTS} → {len(dataset.subject_list)} subjects remaining")


def cast_to_float32(X: np.ndarray) -> np.ndarray:
    """Cast epoch array to float32 for PyTorch compatibility."""
    return X.astype(np.float32)


# --- Build EEGNet pipeline ---
# module is passed as a class (not instance) so sklearn clone() works correctly
# across LOSO folds. Hyperparams use the module__* prefix (skorch convention).
clf = EEGClassifier(
    module=EEGNet,
    module__n_chans=n_chans,
    module__n_outputs=2,
    module__n_times=n_times,
    module__final_conv_length='auto',
    max_epochs=dl_cfg['max_epochs'],
    lr=dl_cfg['lr'],
    batch_size=dl_cfg['batch_size'],
    train_split=None,   # MOABB handles train/test split; no internal val set
    device=device,
    verbose=0,
)

pipelines = {
    'EEGNet': Pipeline([
        ('cast', FunctionTransformer(cast_to_float32)),
        ('eegnet', clf),
    ])
}

# --- Run cross-subject evaluation (leave-one-subject-out) ---
evaluation = CrossSubjectEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    overwrite=True,
)

results: pd.DataFrame = evaluation.process(pipelines)

# --- Report results ---
pivot = results.pivot(index='subject', columns='pipeline', values='score')
print("\n=== DL Cross-Subject Evaluation Results ===")
print(pivot.to_string())

scores = results[results['pipeline'] == 'EEGNet']['score']
print(f"\n  EEGNet  {scores.mean():.3f} ± {scores.std():.3f}")
print(f"  Chance level: 0.500")

# --- Save results ---
suffix = "_excluded_outliers" if EXCLUDE_OUTLIERS else ""
out_path = RESULTS_DIR / f"dl_cross_subject_eval{suffix}.csv"
results.to_csv(out_path, index=False)
print(f"\nResults saved → {out_path}")
