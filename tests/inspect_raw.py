"""
Quick inspection of raw channel names and event codes.
Run from project root: python tests/inspect_raw.py
"""

import mne
from pathlib import Path

mne.set_log_level('WARNING')

raw = mne.io.read_raw_brainvision("data/01_12_2020/1z01122020lh1.vhdr", preload=False)

print("Channels:")
for i, ch in enumerate(raw.ch_names, 1):
    print(f"  {i:2d}: {ch}")

print("\nEvents from annotations:")
events, event_id = mne.events_from_annotations(raw)
print(f"  event_id mapping: {event_id}")
print(f"  unique event codes in data: {sorted(set(events[:, 2].tolist()))}")
print(f"  total events: {len(events)}")
