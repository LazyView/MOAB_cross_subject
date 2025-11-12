# Dataset Structure Documentation

## Data Organization
- **Location:** `data/` folder
- **Structure:** Date-based folders (DD_MM_YYYY format)
- **Format:** BrainVision triplets (.eeg, .vhdr, .vmrk)
- **Filename pattern:** `<ID><gender><date><task><run>`
  - Example: `1z01122020lh1` = Subject 1, female, 01/12/2020, left hand, run 1
  - Tasks: `lh` (left hand), `rh` (right hand)

## Recording Parameters
- **Original sampling rate:** 1000 Hz (Mochura: 500 Hz)
- **Channels recorded:** 18 total (Fp1, F3, F4, C3, C4, P3, P4, F7, etc.)
- **Selected channels:** C3, C4, Cz (motor cortex)
- **Equipment:** BrainProducts V-AMP 16

## Experimental Paradigm
- **Session duration:** 10 minutes
- **Rest phase:** 10 seconds
  - Start: Event 1 (LED on)
  - Mid-rest: Event 2 (5s after LED on)
- **Movement phase:** 20 seconds
  - First movement: Event 5 (first force detection)
  - Subsequent movements: Event 4 (continued force detection)

## Event Markers
- **Event 1:** Start of rest phase (LED on)
- **Event 2:** Mid-rest phase (5s after rest start)
- **Event 4:** Subsequent movements in movement phase
- **Event 5:** First movement in movement phase (NOT present in all files!)

**Note:** Event 5 is missing from many files - use first Event 4 after Event 2 as movement onset

## Epoch Extraction Parameters
- **Time window:** -3.5s to +0.5s relative to event (4 seconds total)
- **Baseline:** -3.5s to -3.0s
- **Rest epochs:** Around Event 2 (mid-rest)
- **Movement epochs:** Around Event 5 (or first Event 4 after Event 2)
- **Sampling after resample:** 500 Hz → 2000 samples per epoch

## Preprocessing Pipeline (from master thesis)
1. **Concatenate runs:** Combine multiple runs per subject
2. **Resample:** Downsample to 500 Hz
3. **Select channels:** C3, C4, Cz only
4. **Epoch extraction:** -3.5 to +0.5s around events
5. **Baseline correction:** Using -3.5 to -3.0s
6. **Bandpass filter:** 8-30 Hz (alpha + beta bands)
7. **Artifact rejection:** Peak-to-peak > 100 μV threshold
8. **Epoch selection:** Balanced selection of rest/movement epochs (complex logic)

## Class Labels
**Binary classification:**
- Rest (label 2): ~1808 epochs
- Movement (label 5): ~1807 epochs

**Multiclass classification:**
- Rest (label 2): 916 epochs
- Right hand movement (label 5): 942 epochs
- Left hand movement (label 6): 860 epochs
- Note: Filename determines hand (lh/rh), not event code

## Dataset Statistics
- **Total subjects:** 29
- **Total binary epochs (after preprocessing):** 3615
- **Total multiclass epochs (after preprocessing):** 2718
- **Artifact rejection rate:** ~26.46%
- **Train/test split:** 80/20

## Notes for MOABB Implementation
1. File naming determines MI class (lh vs rh), not event codes
2. Event 5 missing in many files - fallback to Event 4 needed
3. Complex epoch selection logic to balance classes
4. Multiple runs per subject need concatenation
5. Channels limited to C3, C4, Cz (not all 18 channels)
