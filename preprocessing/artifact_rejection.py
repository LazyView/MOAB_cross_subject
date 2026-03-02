"""
Epoch-level artifact rejection by peak-to-peak amplitude threshold.
"""

import numpy as np
import mne
from typing import Tuple


def reject_by_peak_to_peak(
    epochs: mne.Epochs,
    threshold_uv: float = 100.0
) -> Tuple[mne.Epochs, np.ndarray]:
    """
    Drop epochs where any channel exceeds peak-to-peak amplitude threshold.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs (data in Volts as per MNE convention).
    threshold_uv : float
        Peak-to-peak threshold in microvolts. Default: 100 μV.

    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs with artifacts removed.
    mask : np.ndarray of bool
        Boolean mask (True = kept) of shape (n_epochs,).
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    threshold_v = threshold_uv * 1e-6

    peak_to_peak = data.max(axis=2) - data.min(axis=2)  # (n_epochs, n_channels)
    mask = (peak_to_peak < threshold_v).all(axis=1)     # (n_epochs,)

    epochs_clean = epochs[mask]

    n_dropped = (~mask).sum()
    drop_pct = 100 * n_dropped / len(mask)
    print(f"Artifact rejection: dropped {n_dropped}/{len(mask)} epochs ({drop_pct:.1f}%)")

    return epochs_clean, mask
