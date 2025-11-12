"""
Test script to verify custom MOABB dataset loading for subject 1.
"""

from dataset.custom_dataset import MotorImageryDataset
import mne

# Suppress MNE info messages for cleaner output
mne.set_log_level('WARNING')

print("=" * 60)
print("Testing Custom MOABB Dataset")
print("=" * 60)

# Initialize dataset
print("\n1. Initializing dataset...")
dataset = MotorImageryDataset(data_path="data/", config_path="config/dataset_config.yaml")

print(f"   Dataset code: {dataset.code}")
print(f"   Number of subjects: {len(dataset.subject_list)}")
print(f"   Event types: {list(dataset.event_id.keys())}")
print(f"   Epoch interval: {dataset.interval}")

# Load subject 1
subject_id = 1
print(f"\n2. Loading subject {subject_id}...")

try:
    sessions = dataset._get_single_subject_data(subject_id)

    print(f"   Sessions found: {list(sessions.keys())}")

    for session_name, runs in sessions.items():
        print(f"\n   Session: {session_name}")
        print(f"   Number of runs: {len(runs)}")

        for run_name, raw in runs.items():
            print(f"\n   Run: {run_name}")
            print(f"      Channels: {raw.ch_names}")
            print(f"      Sampling rate: {raw.info['sfreq']} Hz")
            print(f"      Duration: {raw.times[-1]:.2f} seconds")

            # Get annotations (events)
            annotations = raw.annotations
            print(f"      Annotations: {len(annotations)} events")

            # Count event types
            event_counts = {}
            for desc in annotations.description:
                event_counts[desc] = event_counts.get(desc, 0) + 1

            print(f"      Event breakdown:")
            for event_type, count in event_counts.items():
                print(f"         {event_type}: {count}")

    print("\n3. Testing epoch extraction...")
    # Try extracting epochs from first run
    first_session = list(sessions.keys())[0]
    first_run = list(sessions[first_session].keys())[0]
    raw = sessions[first_session][first_run]

    # Get events from annotations
    events, event_id = mne.events_from_annotations(raw)

    # Create epochs
    tmin = dataset.interval[0]
    tmax = dataset.interval[1]

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(-3.5, -3.0),
        preload=True,
        verbose=False
    )

    print(f"   Created {len(epochs)} epochs")
    print(f"   Epoch shape: {epochs.get_data().shape}")
    print(f"   (n_epochs, n_channels, n_times)")

    print("\n" + "=" * 60)
    print("SUCCESS: Dataset loaded correctly!")
    print("=" * 60)

except Exception as e:
    print("\n" + "=" * 60)
    print("ERROR during testing:")
    print("=" * 60)
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
