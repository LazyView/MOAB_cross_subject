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
- Master thesis concatenated runs - need to decide if we do this or keep separately
- Currently loading all 18 channels - need channel selection logic

## Session 2026-02-24

**Completed:**
- Confirmed subject ID strategy: composite `DDMMYYYY_ID` already implemented, 29 subjects detected
- Created `requirements.txt` with direct dependencies only (mne, moabb, numpy, pyyaml)
- Fixed `tests/test_dataset.py` and `tests/raw.py`: missing `sys.path.insert` and wrong class name
- Created `tests/inspect_raw.py` to inspect channel names and event codes
- Confirmed channels: C3(4), C4(5), Cz(14) present; channels 17/18 are unnamed reference/ground
- Confirmed event codes: S1=rest start, S2=rest mid, S4=movement (Event 5 absent in test file)
- Added channel selection `raw.pick(['C3','C4','Cz'])` in `_load_and_prepare_raw`
- Added resample to 500 Hz and bandpass 8-30 Hz in `_load_and_prepare_raw`
- Created `preprocessing/artifact_rejection.py` with `reject_by_peak_to_peak()`
- Decided 2-class MI (left_hand vs right_hand); removed rest from dataset event_id
- Created `evaluation/extract_epochs.py` using MOABB MotorImagery paradigm
- Fixed MOABB integration bugs: `self.data_path` shadowing method → renamed to `self.data_dir`; `code='MotorImageryDataset'`; session name `'session_1'` → `'0'`
- Epoch extraction via `paradigm.get_data()` working

**Next steps:**
- Apply artifact rejection after epoch extraction
- Implement cross-subject evaluation
- Integrate baseline correction (-3.5 to -3.0s)

**Issues/Notes:**
- Saleh format files (HR_) recorded at 1000 Hz; resample step handles this correctly
- Run names use `run_lh_1`, `run_rh_1` format — MOABB keeps runs separate (not concatenated)

## Session 2026-03-15

**Completed:**
- Created `evaluation/cross_subject_eval.py` with MOABB `CrossSubjectEvaluation` (LOSO)
- Added `evaluation.csp_components: 4` to `config/dataset_config.yaml`
- Fixed MOABB run name validation: renamed `run_lh_1` → `0lh1` style (must start with integer, no underscores)
- Fixed MOABB paradigm: `reject` is not a valid param — removed it
- Added baseline correction via paradigm `baseline=(-3.5, -3.0)`
- Added CSP+SVM and CSP+LR pipelines for comparison
- Results: CSP+LDA 0.674±0.218, CSP+LR 0.674±0.218, CSP+SVM 0.631±0.214 (29 subjects)

**Next steps:**
- Save results to CSV for thesis reporting
- Investigate below-chance subjects (7: 0.365, 8: 0.174) — possible data/label issue
- Consider SVM hyperparameter tuning (C, gamma) or drop it
- Add results visualization (per-subject bar chart)

**Issues/Notes:**
- CSP+LDA and CSP+LR are identical — LR adds nothing, LDA is preferred
- CSP+SVM inconsistent: better for subjects 7/8, much worse for 3/15 — needs tuning before thesis use
- Artifact rejection not integrated into MOABB eval loop (paradigm doesn't support `reject` param in this version)

## Session 2026-03-29

**Completed:**
- Created `docs/evaluation_approaches.md` documenting LOSO, CSP, LDA, LR, SVM — rationale, pros/cons, per-pipeline results

**Next steps:**
- Save results to CSV for thesis reporting
- Investigate below-chance subjects (7: 0.365, 8: 0.174)
- Add per-subject bar chart visualization
- SVM hyperparameter tuning (C, gamma) if included in thesis

**Issues/Notes:**
- No code changes this session

## Session 2026-03-29 (continued)

**Completed:**
- Created `docs/evaluation_approaches.md`
- Created `diagnostics/inspect_below_chance.py` — visual inspection script for suspect subjects (headless, saves PNGs to `diagnostics/plots/<composite_id>/`)
- Fixed bugs in diagnostic: `spectrum.get_data(units=...)` not supported → manual scaling; `ndarray.ptp` removed in NumPy 2.0 → `np.ptp()`; `return_epochs=True` baseline crash → use numpy output
- Confirmed MOABB returns data in µV (not V) from X range printout
- Added CSV export to `evaluation/cross_subject_eval.py` → saves to `results/cross_subject_eval.csv`
- Re-ran full evaluation to get per-subject scores

**Findings — below-chance subjects:**
- Subject 7 (03042023_4, score 0.365) and Subject 8 (03042023_5, score 0.174)
- Both have: balanced classes (21/21, 20/21), normal EEG amplitude (±70/±50 µV), clean PSDs, no artifacts
- Subjects 1–3 from the same recording session (03042023) score 0.893, 0.963, 0.978 → session-level issue ruled out
- Root cause: individual atypical ERD/ERS lateralization (BCI illiteracy), not data or code issues

**Next steps:**
- Add per-subject bar chart visualization
- Consider SVM hyperparameter tuning or drop it from thesis

## Session 2026-03-29 (continued 2)

**Completed:**
- Configured LaTeX Workshop in VS Code: pdflatex → biber → pdflatex×2 recipe, disabled latexmk clean
- Installed MiKTeX + GTAmerica fonts via MiKTeX Console
- Created `LaTeX_thesis/sablona/thesis.tex` — FASThesis template with correct metadata (english, bc, kiv)
- Created `LaTeX_thesis/sablona/thesis.bib` — empty bibliography file
- Thesis compiles successfully with proper fonts and layout

**Next steps:**
- Start writing thesis content — Theoretical Background chapter first (material.tex has extracted notes)
- Add per-subject bar chart visualization
- Consider SVM hyperparameter tuning or drop from thesis

## Session 2026-04-05

**Completed:**
- Decided on DL extension scope: EEGNet (braindecode), BCI IV-2a as second dataset, transfer learning Variant B (pretrain + fine-tune). Decisions recorded in `docs/design_decisions.md`.
- Epoch alignment decision: custom dataset uses -3.5 to -0.5 s (active MI before onset); BCI IV-2a uses +0.5 to +3.5 s (active MI after cue) — both 3 s × 500 Hz = 1501 samples.
- Added `dl` section to `config/dataset_config.yaml` (tmin, tmax, n_times, max_epochs, lr, batch_size).
- Created `evaluation/dl_cross_subject_eval.py` — EEGNet LOSO eval via braindecode + MOABB.
- Recreated venv (original path was broken after folder rename); set up `venv_linux` in WSL with CUDA torch + braindecode.
- Fixed braindecode API changes: `EEGNetv4` → `EEGNet`, `n_classes` → `n_outputs`, `input_window_samples` → `n_times`.
- Task 1 result: EEGNet 0.510 ± 0.094 on custom dataset (27 subjects, LOSO) vs CSP+LDA baseline 0.703.
- Created `diagnostics/dataset_analysis.py` — prints stats for both datasets.
- Task 2: confirmed actual trial counts — custom ~54/class/subject (slight imbalance: 51.5 lh vs 56.0 rh), BCI IV-2a 144/class/subject.
- BCI IV-2a returns 1502 samples after resampling (751 × 2); custom returns 1501. Fix: trim to 1501 in transfer learning pipeline. Added `n_times: 1501` to config.
- Thesis additions: `\section{BCI Competition IV Dataset 2a}` with paradigm description and comparison table (corrected numbers); updated preprocessing section intro; added `brunner2008bci` and `lawhern2018eegnet` to `thesis.bib`.

**Next steps:**
- Task 3: transfer learning — pretrain EEGNet on BCI IV-2a, fine-tune on custom dataset
  - Variant A (direct transfer, no fine-tune) as intermediate baseline
  - Variant B (pretrain + fine-tune) as main result
  - Compare: custom-only (0.510) vs direct transfer vs fine-tuned
- Task 4: BCI limitations subsection in Discussion chapter

**Issues/Notes:**
- EEGNet underperforms CSP+LDA on custom dataset alone (0.510 vs 0.703) — expected given small dataset + 3 channels; motivates transfer learning
- BCI IV-2a 1-sample mismatch (1502 vs 1501) requires trimming in transfer learning pipeline
- WSL venv is `venv_linux`; Windows venv is `venv` (both exist, use WSL for running)
- Transfer learning results: custom-only 0.510±0.094, direct transfer 0.519±0.080, fine-tuned 0.514±0.068 — all near chance; std reduction is the main measurable benefit
- Findings saved to `docs/transfer_learning_findings.md`
- Task 4: Discussion chapter written — results interpretation, pipeline limitations, BCI practical limitations (artifacts, BCI illiteracy, non-stationarity, deployment constraints)
- Fixed stale epoch count in thesis (840 → 1391 lh / 1511 rh); updated preprocessing table with both epoch windows
- Thesis compiles cleanly

**Next steps:**
- Write Methods chapter (CSP, classifiers, evaluation strategy — currently \blindtext)
- Write Results chapter (tables/figures from CSV results)
- Write Conclusion chapter
- Add per-subject bar chart visualization
- Consider adding EEGNet section to Methods chapter

## Session 2026-04-09

**Completed:**
- Moved `\section{Practical Limitations of BCI Systems}` from Discussion → Theoretical Background (after Cross-Subject Evaluation, before MOABB Framework)
- Updated section intro to frame limitations as foundational motivation; added thesis-response sentences to Signal Artifacts and Inter-Subject Variability subsections
- Discussion chapter now contains only: Interpretation of Results, Limitations of the Implemented Pipeline

**Next steps:**
- Continue writing remaining thesis chapters (Methods, Results, Conclusion)
