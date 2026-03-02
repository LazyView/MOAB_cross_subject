"""
Custom MOABB dataset for BrainVision Motor Imagery data.
"""

import mne
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Union
from moabb.datasets.base import BaseDataset

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.file_utils import find_subject_files, get_all_subjects


class MotorImageryDataset(BaseDataset):
    """
    Custom MOABB dataset for Motor Imagery BrainVision data.

    Parameters
    ----------
    data_path : str
        Path to the data directory containing BrainVision files
    config_path : str, optional
        Path to configuration YAML file
    """

    def __init__(self, data_path: str = "data/", config_path: str = "config/dataset_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_dir = data_path
        self._subject_list = get_all_subjects(data_path)

        # 2-class MI: left hand vs right hand
        event_id = {
            'left_hand': self.config['classes']['left_hand'],
            'right_hand': self.config['classes']['right_hand']
        }

        super().__init__(
            subjects=list(range(1, len(self._subject_list) + 1)),
            sessions_per_subject=1,
            events=event_id,
            # Code must match or be expandable from class name for MOABB validation
            code='MotorImageryDataset',
            interval=[self.config['epoch']['tmin'], self.config['epoch']['tmax']],
            paradigm='imagery',
            doi=''
        )

    def _get_single_subject_data(self, subject: int) -> Dict[str, Dict[str, mne.io.Raw]]:
        """
        Return data for a single subject in MOABB format.

        Returns
        -------
        dict : {session_name: {run_name: Raw}}
        """
        subject_id = self._subject_list[subject - 1]
        files = find_subject_files(self.data_dir, subject_id)

        # Session name must start with an integer (MOABB requirement)
        session_name = '0'
        sessions = {session_name: {}}

        for file_info in files['lh']:
            run_name = f'run_lh_{file_info.run}'
            sessions[session_name][run_name] = self._load_and_prepare_raw(file_info.file_path, task='lh')

        for file_info in files['rh']:
            run_name = f'run_rh_{file_info.run}'
            sessions[session_name][run_name] = self._load_and_prepare_raw(file_info.file_path, task='rh')

        return sessions

    def _load_and_prepare_raw(self, file_path: Path, task: str) -> mne.io.Raw:
        """
        Load BrainVision file, select channels, resample, filter, and set task annotations.

        Parameters
        ----------
        file_path : Path
            Path to .vhdr file
        task : str
            'lh' or 'rh'

        Returns
        -------
        mne.io.Raw
        """
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)

        raw.pick(self.config['signal']['channels'])
        raw.resample(self.config['signal']['sfreq'], verbose=False)
        raw.filter(
            l_freq=self.config['signal']['filter']['l_freq'],
            h_freq=self.config['signal']['filter']['h_freq'],
            verbose=False
        )

        events, event_id = mne.events_from_annotations(raw)
        new_annotations = self._create_task_annotations(raw, events, event_id, task)
        raw.set_annotations(new_annotations)

        return raw

    def _create_task_annotations(
        self,
        raw: mne.io.Raw,
        events: np.ndarray,
        event_id: dict,
        task: str
    ) -> mne.Annotations:
        """
        Map raw event codes to 'left_hand' / 'right_hand' / 'rest' annotations.

        Logic:
        - Event 2 → rest
        - Event 5 → movement onset (if present)
        - First Event 4 after Event 2 → movement onset (fallback when Event 5 missing)
        """
        sfreq = raw.info['sfreq']

        rest_mid_code = self.config['events']['rest_mid']
        movement_first_code = self.config['events']['movement_first']
        movement_cont_code = self.config['events']['movement_cont']

        onset_times = []
        durations = []
        descriptions = []

        last_rest_idx = -1
        used_movement_indices = set()

        for i, event in enumerate(events):
            event_code = event[2]
            event_time = event[0] / sfreq

            if event_code == rest_mid_code:
                onset_times.append(event_time)
                durations.append(0)
                descriptions.append('rest')
                last_rest_idx = i

            elif event_code == movement_first_code:
                onset_times.append(event_time)
                durations.append(0)
                descriptions.append('left_hand' if task == 'lh' else 'right_hand')
                used_movement_indices.add(i)

            elif event_code == movement_cont_code and last_rest_idx >= 0:
                if i not in used_movement_indices:
                    # Use first Event 4 after last rest only if no Event 5 followed that rest
                    has_event_5 = any(
                        events[j][2] == movement_first_code
                        for j in range(last_rest_idx + 1, i)
                    )
                    if not has_event_5:
                        onset_times.append(event_time)
                        durations.append(0)
                        descriptions.append('left_hand' if task == 'lh' else 'right_hand')
                        used_movement_indices.add(i)

        return mne.Annotations(
            onset=onset_times,
            duration=durations,
            description=descriptions,
            orig_time=raw.info['meas_date']
        )

    def data_path(
        self,
        subject: int,
        path: Union[str, Path, None] = None,
        force_update: bool = False,
        update_path: bool = None,
        verbose: Union[bool, str, int, None] = None
    ) -> List[str]:
        """Required by MOABB. Returns list of .vhdr file paths for the subject."""
        subject_id = self._subject_list[subject - 1]
        files = find_subject_files(self.data_dir, subject_id)

        return [
            str(file_info.file_path)
            for task in ['lh', 'rh']
            for file_info in files[task]
        ]
