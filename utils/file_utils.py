"""
Utility functions for parsing and managing BrainVision files.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FileInfo:
    """Information parsed from filename."""
    subject_id: str
    gender: str
    date: str
    task: str  # 'lh' or 'rh'
    run: str
    file_path: Path

    @property
    def is_left_hand(self) -> bool:
        return self.task == 'lh'

    @property
    def is_right_hand(self) -> bool:
        return self.task == 'rh'

    @property
    def composite_id(self) -> str:
        """Globally unique subject ID: date_subject_id (e.g., '20201201_1')."""
        return f"{self.date}_{self.subject_id}"


def parse_mochura_filename(filepath: Path) -> Optional[FileInfo]:
    """
    Parse Mochura format filename: <ID><gender><date><task><run>
    Example: 1z01122020lh1

    Parameters
    ----------
    filepath : Path
        Path to the .vhdr file

    Returns
    -------
    FileInfo or None
        Parsed file information, or None if format doesn't match
    """
    pattern = r'^(\d+)([mz])(\d{8})(lh|rh)(\d+)$'
    filename = filepath.stem

    match = re.match(pattern, filename)
    if match:
        subject_id, gender, date, task, run = match.groups()
        return FileInfo(
            subject_id=subject_id,
            gender=gender,
            date=date,
            task=task,
            run=run,
            file_path=filepath
        )
    return None


def parse_saleh_filename(filepath: Path) -> Optional[FileInfo]:
    """
    Parse Saleh format filename: HR_<date>_<ID>_<haptic>_<side>
    Example: HR_01122020_1_bez_vibratoru_s_haptikou_leva

    Parameters
    ----------
    filepath : Path
        Path to the .vhdr file

    Returns
    -------
    FileInfo or None
        Parsed file information, or None if format doesn't match
    """
    pattern = r'^HR_(\d{8})_(\d+)_([^_]+(?:_[^_]+)*)_(leva|prava)$'
    filename = filepath.stem

    match = re.match(pattern, filename)
    if match:
        date, subject_id, haptic, side = match.groups()
        # Map Czech side names to lh/rh
        task = 'lh' if side == 'leva' else 'rh'
        return FileInfo(
            subject_id=f"saleh_{subject_id}",  # Prefix to distinguish from Mochura subjects
            gender='unknown',
            date=date,
            task=task,
            run='1',  # Saleh files don't have run numbers
            file_path=filepath
        )
    return None


def parse_filename(filepath: Path) -> Optional[FileInfo]:
    """
    Parse filename using either Mochura or Saleh format.

    Parameters
    ----------
    filepath : Path
        Path to the .vhdr file

    Returns
    -------
    FileInfo or None
        Parsed file information, or None if no format matches
    """
    # Try Mochura format first
    info = parse_mochura_filename(filepath)
    if info:
        return info

    # Try Saleh format
    info = parse_saleh_filename(filepath)
    if info:
        return info

    return None


def find_subject_files(data_path: str, subject_id: str) -> Dict[str, List[FileInfo]]:
    """
    Find all files for a given subject, organized by task (lh/rh).

    Parameters
    ----------
    data_path : str
        Path to data directory
    subject_id : str
        Composite subject ID (date_subject_id, e.g., '20201201_1')

    Returns
    -------
    dict
        Dictionary with keys 'lh' and 'rh', each containing list of FileInfo objects
    """
    data_dir = Path(data_path)
    files = {'lh': [], 'rh': []}

    # Search all .vhdr files recursively
    for vhdr_file in data_dir.rglob('*.vhdr'):
        file_info = parse_filename(vhdr_file)
        if file_info and file_info.composite_id == subject_id:
            files[file_info.task].append(file_info)

    # Sort by run number
    for task in ['lh', 'rh']:
        files[task].sort(key=lambda x: int(x.run))

    return files


def get_all_subjects(data_path: str) -> List[str]:
    """
    Get list of all unique subject IDs in the dataset.

    Uses composite IDs (date_subject_id) to ensure global uniqueness.

    Parameters
    ----------
    data_path : str
        Path to data directory

    Returns
    -------
    list
        Sorted list of unique composite subject IDs (e.g., '20201201_1')
    """
    data_dir = Path(data_path)
    subjects = set()

    for vhdr_file in data_dir.rglob('*.vhdr'):
        file_info = parse_filename(vhdr_file)
        if file_info:
            subjects.add(file_info.composite_id)

    return sorted(subjects)
