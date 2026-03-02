"""
Epoch extraction using MOABB's MotorImagery paradigm.
Run from project root: python evaluation/extract_epochs.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mne
import numpy as np
from moabb.paradigms import MotorImagery

from dataset.custom_dataset import MotorImageryDataset

mne.set_log_level('WARNING')

# --- Init dataset ---
dataset = MotorImageryDataset(
    data_path="data/",
    config_path="config/dataset_config.yaml"
)

# --- Init paradigm ---
# Filtering already applied in dataset class, so fmin/fmax are redundant but harmless.
# tmin/tmax are taken from dataset.interval ([-3.5, 0.5]).
paradigm = MotorImagery(
    events=['left_hand', 'right_hand'],
    n_classes=2,
    fmin=8,
    fmax=30,
)

# --- Extract epochs for one subject ---
subject = 1
print(f"Extracting epochs for subject {subject} ({dataset._subject_list[subject - 1]})...")

X, y, metadata = paradigm.get_data(dataset, subjects=[subject])

print(f"\nX shape : {X.shape}  (n_epochs, n_channels, n_times)")
print(f"y shape : {y.shape}")
print(f"Classes : {np.unique(y)}")
print(f"Counts  : left_hand={np.sum(y == 'left_hand')}, right_hand={np.sum(y == 'right_hand')}")
print(f"\nMetadata:\n{metadata.head()}")
