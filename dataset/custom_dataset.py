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

    This dataset contains Motor Imagery recordings from rehabilitation robot experiments.
    Subjects performed left hand and right hand motor imagery tasks.

    Parameters
    ----------
    data_path : str
        Path to the data directory containing BrainVision files
    config_path : str, optional
        Path to configuration YAML file
    """

    def __init__(self, data_path: str = "data/", config_path: str = "config/dataset_config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_path = data_path

        # Get list of all subjects
        self._subject_list = get_all_subjects(data_path)

        # Define event codes for MOABB
        # We'll use Event 2 for rest and Event 5 (or 4) for movement
        event_id = {
            'rest': self.config['classes']['rest'],
            'left_hand': self.config['classes']['left_hand'],
            'right_hand': self.config['classes']['right_hand']
        }

        # Initialize BaseDataset
        super().__init__(
            subjects=list(range(1, len(self._subject_list) + 1)),
            sessions_per_subject=1,
            events=event_id,
            code='CustomMI',
            interval=[self.config['epoch']['tmin'], self.config['epoch']['tmax']],
            paradigm='imagery',
            doi=''
        )

    def _get_single_subject_data(self, subject: int) -> Dict[str, Dict[str, mne.io.Raw]]:
        """
        Return data for a single subject in MOABB format.

        Parameters
        ----------
        subject : int
            Subject number (1-indexed)

        Returns
        -------
        dict
            Dictionary with session keys, each containing dict with run keys and Raw objects
            Format: {session_name: {run_name: Raw}}
        """
        # Get subject ID
        subject_id = self._subject_list[subject - 1]

        # Find all files for this subject
        files = find_subject_files(self.data_path, subject_id)

        # Organize data by session and run
        sessions = {}
        session_name = 'session_1'
        sessions[session_name] = {}

        # Process left hand files
        for idx, file_info in enumerate(files['lh']):
            run_name = f'run_lh_{file_info.run}'
            raw = self._load_and_prepare_raw(file_info.file_path, task='lh')
            sessions[session_name][run_name] = raw

        # Process right hand files
        for idx, file_info in enumerate(files['rh']):
            run_name = f'run_rh_{file_info.run}'
            raw = self._load_and_prepare_raw(file_info.file_path, task='rh')
            sessions[session_name][run_name] = raw

        return sessions

    def _load_and_prepare_raw(self, file_path: Path, task: str) -> mne.io.Raw:
        """
        Load BrainVision file and prepare it for MOABB processing.

        Parameters
        ----------
        file_path : Path
            Path to .vhdr file
        task : str
            Task type: 'lh' or 'rh'

        Returns
        -------
        mne.io.Raw
            Prepared Raw object with updated annotations
        """
        # Load raw data
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)

        # Get events from annotations
        events, event_id = mne.events_from_annotations(raw)

        # Prepare new annotations based on task
        new_annotations = self._create_task_annotations(raw, events, event_id, task)

        # Set new annotations
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
        Create annotations with proper class labels based on task.

        This implements the epoch selection logic from the master thesis:
        - Rest epochs: Event 2 (mid-rest)
        - Movement epochs: Event 5, or first Event 4 after Event 2 if Event 5 missing

        Parameters
        ----------
        raw : mne.io.Raw
            Raw data object
        events : np.ndarray
            Events array
        event_id : dict
            Event ID mapping
        task : str
            Task type: 'lh' or 'rh'

        Returns
        -------
        mne.Annotations
            New annotations with task-specific labels
        """
        sfreq = raw.info['sfreq']

        # Get event codes
        rest_mid_code = self.config['events']['rest_mid']
        movement_first_code = self.config['events']['movement_first']
        movement_cont_code = self.config['events']['movement_cont']

        # Reverse event_id mapping (description -> code)
        code_to_name = {v: k for k, v in event_id.items()}

        # Collect annotations
        onset_times = []
        durations = []
        descriptions = []

        # Track last rest event index to find movement events
        last_rest_idx = -1
        used_movement_indices = set()

        for i, event in enumerate(events):
            event_code = event[2]
            event_sample = event[0]
            event_time = event_sample / sfreq

            # Map to event names
            if event_code == rest_mid_code:
                # Add rest annotation
                onset_times.append(event_time)
                durations.append(0)
                descriptions.append('rest')
                last_rest_idx = i

            elif event_code == movement_first_code:
                # Event 5 exists - use it for movement
                onset_times.append(event_time)
                durations.append(0)
                # Label based on task (filename)
                movement_label = 'left_hand' if task == 'lh' else 'right_hand'
                descriptions.append(movement_label)
                used_movement_indices.add(i)

            elif event_code == movement_cont_code and last_rest_idx >= 0:
                # Event 4 - only use first one after last rest if Event 5 wasn't found
                # Check if this is the first Event 4 after the last rest
                if i not in used_movement_indices:
                    # Check if there's already an Event 5 between last_rest and this Event 4
                    has_event_5_between = False
                    for j in range(last_rest_idx + 1, i):
                        if events[j][2] == movement_first_code:
                            has_event_5_between = True
                            break

                    if not has_event_5_between:
                        # This is the first movement marker after rest, use it
                        onset_times.append(event_time)
                        durations.append(0)
                        movement_label = 'left_hand' if task == 'lh' else 'right_hand'
                        descriptions.append(movement_label)
                        used_movement_indices.add(i)

        # Create annotations
        annotations = mne.Annotations(
            onset=onset_times,
            duration=durations,
            description=descriptions,
            orig_time=raw.info['meas_date']
        )

        return annotations

    def data_path(
        self,
        subject: int,
        path: Union[str, Path, None] = None,
        force_update: bool = False,
        update_path: bool = None,
        verbose: Union[bool, str, int, None] = None
    ) -> List[str]:
        """
        Required by MOABB. Returns list of file paths for the subject.

        Parameters
        ----------
        subject : int
            Subject number (1-indexed)
        path : str, optional
            Not used (data already local)
        force_update : bool
            Not used (data already local)
        update_path : bool, optional
            Not used
        verbose : bool, str, int, optional
            Verbosity level

        Returns
        -------
        list
            List of file paths
        """
        subject_id = self._subject_list[subject - 1]
        files = find_subject_files(self.data_path, subject_id)

        file_paths = []
        for task in ['lh', 'rh']:
            for file_info in files[task]:
                file_paths.append(str(file_info.file_path))

        return file_paths
