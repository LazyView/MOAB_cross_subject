# Session Log

## Session 2025-11-12

**Completed:**
- Analyzed INFO.txt documentation to understand dataset structure and preprocessing pipeline
- Created `DATASET_STRUCTURE.md` documenting all key parameters
- Implemented project structure (config/, dataset/, utils/)
- Created `config/dataset_config.yaml` with epoch and preprocessing parameters
- Implemented `utils/file_utils.py` for parsing Mochura and Saleh filename formats
- Implemented `dataset/custom_dataset.py` - custom MOABB dataset class
- implemented `tests/test_dataset.py` - tests correctness of `dataset/custom_dataset.py`
- implemented `tests/test_file_utils.py` - tests correctness of `utils/file_utils.py`
- Created test scripts and verified dataset loads correctly for subject 1
- Successfully loaded 4 runs (2 lh, 2 rh) with proper event annotations
- Epoch extraction working: (109, 18, 2001) shape

**Next steps:**
- Decide on subject identification strategy (date+ID or single date per ID)
- Reduce channels from 18 to 3 (C3, C4, Cz) as per preprocessing requirements
- Implement preprocessing pipeline (filtering 8-30Hz, artifact rejection 100μV, baseline correction)
- Decide when to apply preprocessing (in dataset class or via MOABB paradigm)
- Test with MOABB paradigm classes (MotorImagery paradigm)
- Implement cross-subject evaluation

**Issues/Notes:**
- Subject ID is only unique within measurement date, not globally
- Currently grouping files from different dates as same subject (ID "1" spans multiple dates)
- Need to either: (a) make subject_id include date, or (b) only use files from specific date per subject
- MOABB warning about class name abbreviation (non-critical)
- Master thesis concatenated runs - need to decide if we do this or keep separate
- Currently loading all 18 channels - need channel selection logic
