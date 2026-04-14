"""
CSP-based pipeline factories.

All return paradigm_overrides={} so MOABB uses its default epoch window.

Configurable params (via pipeline.yaml):
    n_components : int   Number of CSP spatial filters (default: evaluation.csp_components)
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP


def build_csp_lda(config: dict, params: dict) -> tuple[Pipeline, dict]:
    n_csp = params.get('n_components', config['evaluation']['csp_components'])
    pipeline = Pipeline([
        ('csp', CSP(n_components=n_csp, reg=None, log=True)),
        ('lda', LinearDiscriminantAnalysis()),
    ])
    return pipeline, {}


def build_csp_lr(config: dict, params: dict) -> tuple[Pipeline, dict]:
    n_csp = params.get('n_components', config['evaluation']['csp_components'])
    pipeline = Pipeline([
        ('csp', CSP(n_components=n_csp, reg=None, log=True)),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000)),
    ])
    return pipeline, {}


def build_csp_svm(config: dict, params: dict) -> tuple[Pipeline, dict]:
    n_csp = params.get('n_components', config['evaluation']['csp_components'])
    pipeline = Pipeline([
        ('csp', CSP(n_components=n_csp, reg=None, log=True)),
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale')),
    ])
    return pipeline, {}
