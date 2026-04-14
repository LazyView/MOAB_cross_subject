"""
Method registry for run_pipeline.py.

Each entry maps a string key (used in pipeline.yaml) to a factory function.

Factory contract:
    build_<method>(config: dict, params: dict) -> tuple[sklearn.Pipeline, dict]
    - config    : full dataset_config.yaml dict (read-only)
    - params    : per-method overrides from pipeline.yaml (may be {})
    - return[0] : cloneable sklearn Pipeline ready for MOABB
    - return[1] : paradigm kwarg overrides; {} = use MOABB defaults (CSP window);
                  non-empty sets tmin/tmax for DL methods (active MI window)

To add a new method:
    1. Create methods/mymethod.py with build_mymethod(config, params) -> (Pipeline, dict)
    2. Import it below and add one line to REGISTRY.
    run_pipeline.py never needs to change.
"""

from methods.csp    import build_csp_lda, build_csp_lr, build_csp_svm
from methods.eegnet import build_eegnet
from methods.shallow import build_shallow_convnet

REGISTRY: dict[str, callable] = {
    "CSP+LDA":        build_csp_lda,
    "CSP+LR":         build_csp_lr,
    "CSP+SVM":        build_csp_svm,
    "EEGNet":         build_eegnet,
    "ShallowConvNet": build_shallow_convnet,
}


def get_pipeline(name: str, config: dict, params: dict = {}) -> tuple:
    """
    Resolve a method name to its pipeline and paradigm overrides.

    Parameters
    ----------
    name   : str   Registry key matching an entry in REGISTRY.
    config : dict  Full dataset_config.yaml content.
    params : dict  Per-method hyperparameter overrides from pipeline.yaml.

    Returns
    -------
    (sklearn.Pipeline, dict)  Pipeline and paradigm kwarg overrides.

    Raises
    ------
    KeyError if name is not in REGISTRY.
    """
    if name not in REGISTRY:
        valid = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(
            f"Method '{name}' not found in registry. Valid keys: {valid}"
        )
    return REGISTRY[name](config, params)
