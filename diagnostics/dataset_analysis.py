"""
Dataset statistics for both datasets used in this thesis.

Prints: subjects, channels, sampling rate, epoch window, total epochs, class balance.
Note: first run downloads BCI IV-2a (~200 MB) to ~/mne_data automatically.

Run from project root: python diagnostics/dataset_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
import mne
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from dataset.custom_dataset import MotorImageryDataset

mne.set_log_level('WARNING')

EXCLUDE_OUTLIERS = True
OUTLIER_SUBJECTS = [7, 8]

with open("config/dataset_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

sig_cfg = config['signal']
dl_cfg = config['dl']

SEP = "=" * 58


def print_stats(name: str, X: np.ndarray, y: np.ndarray, n_subjects: int,
                channels: list, sfreq: float, tmin: float, tmax: float) -> None:
    classes, counts = np.unique(y, return_counts=True)
    print(f"\n{SEP}")
    print(f"  {name}")
    print(SEP)
    print(f"  Subjects             : {n_subjects}")
    print(f"  Channels (used)      : {channels}")
    print(f"  Sampling rate        : {sfreq} Hz")
    print(f"  Epoch window         : {tmin} s to {tmax} s  ({X.shape[2]} samples)")
    print(f"  Total epochs         : {X.shape[0]}")
    for cls, cnt in zip(classes, counts):
        print(f"    {cls:<12}       : {cnt}  ({cnt / n_subjects:.1f} per subject avg)")
    print(f"  Input shape (X)      : {X.shape}  [epochs × channels × samples]")


# ---------------------------------------------------------------------------
# 1. Custom Motor Imagery Dataset
# ---------------------------------------------------------------------------
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

X_custom, y_custom, _ = custom_paradigm.get_data(custom_dataset)

print_stats(
    name="Custom Motor Imagery Dataset",
    X=X_custom,
    y=y_custom,
    n_subjects=len(custom_dataset.subject_list),
    channels=sig_cfg['channels'],
    sfreq=sig_cfg['sfreq'],
    tmin=dl_cfg['tmin'],
    tmax=dl_cfg['tmax'],
)

# ---------------------------------------------------------------------------
# 2. BCI Competition IV Dataset 2a  (BNCI2014_001)
# ---------------------------------------------------------------------------
print("\nLoading BCI IV-2a — will download ~200 MB on first run...")

bci2a_dataset = BNCI2014_001()

bci2a_paradigm = MotorImagery(
    events=['left_hand', 'right_hand'],
    n_classes=2,
    fmin=sig_cfg['filter']['l_freq'],
    fmax=sig_cfg['filter']['h_freq'],
    tmin=0.5,    # active MI window: 0.5–3.5 s post-cue (aligned with custom dataset)
    tmax=3.5,
    channels=['C3', 'C4', 'Cz'],
    resample=sig_cfg['sfreq'],
)

X_bci2a, y_bci2a, _ = bci2a_paradigm.get_data(bci2a_dataset)

print_stats(
    name="BCI Competition IV Dataset 2a  (BNCI2014_001)",
    X=X_bci2a,
    y=y_bci2a,
    n_subjects=len(bci2a_dataset.subject_list),
    channels=['C3', 'C4', 'Cz'],
    sfreq=sig_cfg['sfreq'],
    tmin=0.5,
    tmax=3.5,
)

print(f"\n{SEP}")
print("  Notes")
print(SEP)
print("  BCI IV-2a original sfreq : 250 Hz (resampled to 500 Hz)")
print("  BCI IV-2a full montage   : 22 EEG + 3 EOG channels")
print("  BCI IV-2a classes        : 4 total (left hand, right hand, feet, tongue)")
print("                             only left/right used here")
print("  Custom dataset outliers  : subjects 7 and 8 excluded (BCI illiteracy)")
print(SEP)
