"""
Cross-subject evaluation using MOABB CrossSubjectEvaluation.
Pipelines: CSP+LDA, CSP+SVM, CSP+LR
Run from project root: python evaluation/cross_subject_eval.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mne
import yaml
import pandas as pd
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
