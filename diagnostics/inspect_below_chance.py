"""
Visual inspection of below-chance subjects.

Saves plots to diagnostics/plots/<subject_id>/:
  - raw_signal.png     : raw timeseries (C3, C4, Cz) for each run
  - psd.png            : power spectral density per run
  - epoch_counts.png   : lh vs rh epoch counts
  - epoch_amplitudes.png : per-epoch peak-to-peak amplitude distribution

Run from project root:
    python diagnostics/inspect_below_chance.py

Edit SUSPECT_SUBJECT_NUMS to inspect different subjects.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mne
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless — saves to files instead of showing windows
import matplotlib.pyplot as plt

from moabb.paradigms import MotorImagery

from dataset.custom_dataset import MotorImageryDataset
from utils.file_utils import find_subject_files

mne.set_log_level('WARNING')

# ── Config ────────────────────────────────────────────────────────────────────
SUSPECT_SUBJECT_NUMS = [7, 8]   # MOABB integer subject numbers to inspect
CONFIG_PATH = "config/dataset_config.yaml"
PLOT_DIR = Path("diagnostics/plots")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# ── Init dataset & paradigm ───────────────────────────────────────────────────
dataset = MotorImageryDataset(data_path=config['data_path'], config_path=CONFIG_PATH)
paradigm = MotorImagery(
    events=['left_hand', 'right_hand'],
    n_classes=2,
    fmin=config['signal']['filter']['l_freq'],
    fmax=config['signal']['filter']['h_freq'],
    baseline=tuple(config['epoch']['baseline']),
)


def save_raw_signal(raws: dict, out_dir: Path) -> None:
    """Plot timeseries for each run (C3, C4, Cz stacked)."""
    n_runs = len(raws)
    fig, axes = plt.subplots(n_runs, 1, figsize=(18, 4 * n_runs), squeeze=False)
    fig.suptitle("Raw signal per run", fontsize=13)

    for ax, (run_name, raw) in zip(axes[:, 0], raws.items()):
        times = raw.times
        data = raw.get_data(units='uV')  # shape (3, n_times)
        ch_names = raw.ch_names

        for i, ch in enumerate(ch_names):
            offset = i * 150  # µV spacing between channels
            ax.plot(times, data[i] + offset, linewidth=0.4, label=ch)

        ax.set_title(f"Run: {run_name}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV, offset)")
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(times[0], times[-1])

    fig.tight_layout()
    fig.savefig(out_dir / "raw_signal.png", dpi=100)
    plt.close(fig)


def save_psd(raws: dict, out_dir: Path) -> None:
    """Plot PSD (0–50 Hz) for each run, one subplot per channel."""
    channels = config['signal']['channels']
    n_runs = len(raws)
    fig, axes = plt.subplots(len(channels), n_runs, figsize=(6 * n_runs, 4 * len(channels)), squeeze=False)
    fig.suptitle("PSD per run and channel", fontsize=13)

    for col, (run_name, raw) in enumerate(raws.items()):
        spectrum = raw.compute_psd(fmax=50, verbose=False)
        freqs = spectrum.freqs
        psds = spectrum.get_data() * 1e12  # V^2/Hz → µV^2/Hz

        for row, ch in enumerate(channels):
            ch_idx = raw.ch_names.index(ch)
            axes[row, col].semilogy(freqs, psds[ch_idx])
            axes[row, col].axvspan(8, 30, alpha=0.15, color='green', label='8–30 Hz')
            axes[row, col].set_title(f"{ch} | {run_name}", fontsize=9)
            axes[row, col].set_xlabel("Freq (Hz)")
            axes[row, col].set_ylabel("PSD (µV²/Hz)")
            axes[row, col].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "psd.png", dpi=100)
    plt.close(fig)


def save_epoch_counts(y: np.ndarray, out_dir: Path) -> None:
    """Bar chart: number of epochs per class."""
    labels, counts = np.unique(y, return_counts=True)
    label_names = [str(l) for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(label_names, counts, color=['steelblue', 'tomato'])
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.axhline(y=counts.min(), color='gray', linestyle='--', linewidth=0.8)
    ax.set_title("Epoch counts per class")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(out_dir / "epoch_counts.png", dpi=100)
    plt.close(fig)


def save_epoch_amplitudes(X: np.ndarray, y: np.ndarray, out_dir: Path) -> None:
    """Per-epoch peak-to-peak amplitude, coloured by class.

    X : (n_epochs, n_channels, n_times) — unit printed to stdout for verification.
    Threshold is set at mean + 3*std of ptp values (data-relative).
    """
    ptp = np.ptp(X, axis=2).max(axis=1)  # max p2p across channels, per epoch

    # Print actual scale so we can confirm units
    print(f"    X range: min={X.min():.4e}  max={X.max():.4e}  ptp median={np.median(ptp):.4e}")

    threshold = ptp.mean() + 3 * ptp.std()

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {'left_hand': 'steelblue', 'right_hand': 'tomato'}
    labels = np.unique(y)

    for lbl in labels:
        mask = y == lbl
        indices = np.where(mask)[0]
        ax.scatter(indices, ptp[mask], label=str(lbl), alpha=0.6, s=20,
                   color=colors.get(str(lbl), 'gray'))

    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1,
               label=f'mean+3σ = {threshold:.2e}')
    ax.set_title("Peak-to-peak amplitude per epoch (max across channels)")
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Peak-to-peak (data units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "epoch_amplitudes.png", dpi=100)
    plt.close(fig)


def print_annotation_summary(raws: dict) -> None:
    """Print event/annotation counts per run to stdout."""
    print("  Annotation summary:")
    for run_name, raw in raws.items():
        ann_counts: dict = {}
        for ann in raw.annotations:
            desc = ann['description']
            ann_counts[desc] = ann_counts.get(desc, 0) + 1
        print(f"    {run_name}: {ann_counts}")


# ── Main loop ─────────────────────────────────────────────────────────────────
for subj_num in SUSPECT_SUBJECT_NUMS:
    composite_id = dataset._subject_list[subj_num - 1]
    print(f"\n{'='*60}")
    print(f"Subject {subj_num}  →  composite ID: {composite_id}")
    print(f"{'='*60}")

    # Files on disk
    files = find_subject_files(config['data_path'], composite_id)
    print(f"  lh runs: {[str(f.file_path.name) for f in files['lh']]}")
    print(f"  rh runs: {[str(f.file_path.name) for f in files['rh']]}")

    # Load raw data for this subject (already preprocessed by dataset class)
    subject_data = dataset._get_single_subject_data(subj_num)
    raws = subject_data['0']  # session '0'

    print_annotation_summary(raws)

    # Extract epochs via paradigm — numpy output avoids baseline re-application crash
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subj_num])
    print(f"  Epochs shape: {X.shape}")
    labels, counts = np.unique(y, return_counts=True)
    print(f"  Class counts: { {str(l): int(c) for l, c in zip(labels, counts)} }")

    # Save plots
    out_dir = PLOT_DIR / composite_id
    out_dir.mkdir(parents=True, exist_ok=True)

    save_raw_signal(raws, out_dir)
    save_psd(raws, out_dir)
    save_epoch_counts(y, out_dir)
    save_epoch_amplitudes(X, y, out_dir)

    print(f"  Plots saved → {out_dir}/")

print("\nDone.")
