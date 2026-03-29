"""
Cross-subject evaluation using MOABB CrossSubjectEvaluation.
Pipelines: CSP+LDA, CSP+SVM, CSP+LR
Run from project root: python evaluation/cross_subject_eval.py

Results are saved to results/cross_subject_eval.csv
Set EXCLUDE_OUTLIERS = True to exclude atypical subjects (7, 8).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mne
import yaml
import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import MotorImagery

from dataset.custom_dataset import MotorImageryDataset

mne.set_log_level('WARNING')

# --- Options ---
EXCLUDE_OUTLIERS = True
OUTLIER_SUBJECTS = [7, 8]  # atypical ERD/ERS lateralization, confirmed via diagnostics

# --- Load config ---
with open("config/dataset_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# --- Init dataset ---
dataset = MotorImageryDataset(
    data_path=config['data_path'],
    config_path="config/dataset_config.yaml"
)

# --- Init paradigm ---
paradigm = MotorImagery(
    events=['left_hand', 'right_hand'],
    n_classes=2,
    fmin=config['signal']['filter']['l_freq'],
    fmax=config['signal']['filter']['h_freq'],
    baseline=tuple(config['epoch']['baseline']),
)

# --- Build pipelines ---
n_csp = config['evaluation']['csp_components']

pipelines = {
    'CSP+LDA': Pipeline([
        ('csp', CSP(n_components=n_csp, reg=None, log=True)),
        ('lda', LinearDiscriminantAnalysis()),
    ]),
    'CSP+SVM': Pipeline([
        ('csp', CSP(n_components=n_csp, reg=None, log=True)),
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale')),
    ]),
    'CSP+LR': Pipeline([
        ('csp', CSP(n_components=n_csp, reg=None, log=True)),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000)),
    ]),
}

# --- Optionally exclude outlier subjects ---
if EXCLUDE_OUTLIERS:
    dataset.subject_list = [s for s in dataset.subject_list if s not in OUTLIER_SUBJECTS]
    print(f"Excluded subjects: {OUTLIER_SUBJECTS} → {len(dataset.subject_list)} subjects remaining")

# --- Run cross-subject evaluation (leave-one-subject-out) ---
evaluation = CrossSubjectEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    overwrite=True,
)

results: pd.DataFrame = evaluation.process(pipelines)

# --- Report results ---
pivot = results.pivot(index='subject', columns='pipeline', values='score')
print("\n=== Cross-Subject Evaluation Results ===")
print(pivot.to_string())

print("\n=== Mean ± Std per Pipeline ===")
for pipeline_name in pipelines:
    scores = results[results['pipeline'] == pipeline_name]['score']
    print(f"  {pipeline_name:<10} {scores.mean():.3f} ± {scores.std():.3f}")

print(f"\nChance level  : 0.500")

# --- Save results ---
suffix = "_excluded_outliers" if EXCLUDE_OUTLIERS else ""
out_path = RESULTS_DIR / f"cross_subject_eval{suffix}.csv"
results.to_csv(out_path, index=False)
print(f"\nResults saved → {out_path}")
