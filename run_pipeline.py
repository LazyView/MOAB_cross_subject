"""
Registry-based pipeline runner for BCI Motor Imagery cross-subject evaluation.

Usage:
    python run_pipeline.py
    python run_pipeline.py --config config/pipeline.yaml

Reads config/pipeline.yaml to select datasets, methods, and per-method params.
Reads config/dataset_config.yaml for signal, epoch, and model parameters.
Results saved to results/<filename>[_excluded_outliers].csv.

Note: evaluation/transfer_learning.py is intentionally excluded — it uses a
bespoke two-dataset LOSO loop that cannot be expressed as a MOABB pipeline.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import importlib

import mne
import yaml
import pandas as pd
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import MotorImagery

from methods import get_pipeline
from evaluation.transfer_learning import run as run_transfer_learning

mne.set_log_level('WARNING')


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_configs(pipeline_config_path: str) -> tuple[dict, dict]:
    with open(pipeline_config_path, 'r') as f:
        pipeline_cfg = yaml.safe_load(f)
    with open("config/dataset_config.yaml", 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    return pipeline_cfg, dataset_cfg


# ---------------------------------------------------------------------------
# Methods parsing
# ---------------------------------------------------------------------------

def parse_methods(raw_list: list) -> list[dict]:
    """
    Normalise mixed method list to uniform dicts.

    Accepts:
        - "CSP+LDA"                            (plain string, no params)
        - {name: EEGNet, params: {lr: 0.001}}  (dict with overrides)

    Returns list of {'name': str, 'params': dict}.
    """
    result = []
    for entry in raw_list:
        if isinstance(entry, str):
            result.append({'name': entry, 'params': {}})
        else:
            result.append({'name': entry['name'], 'params': entry.get('params', {})})
    return result


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def parse_datasets(raw_list: list) -> list[dict]:
    """
    Normalise dataset list entries to uniform dicts.

    Each entry must be a dict with keys: name, class, init_params, paradigm.
    Returns list of {'name': str, 'class': str, 'init_params': dict, 'paradigm': dict}.
    """
    result = []
    for entry in raw_list:
        if not isinstance(entry, dict) or 'class' not in entry:
            raise ValueError(
                f"Dataset entry '{entry}' is missing required 'class' key. "
                f"Each dataset must specify 'name', 'class', 'init_params', and 'paradigm'."
            )
        result.append({
            'name':        entry['name'],
            'class':       entry['class'],
            'init_params': entry.get('init_params', {}),
            'paradigm':    entry.get('paradigm', {}),
        })
    return result


def _instantiate(class_path: str, init_params: dict):
    """Import and instantiate a class from a dotted path string."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls    = getattr(module, class_name)
    return cls(**init_params)


def build_datasets(pipeline_cfg: dict, dataset_cfg: dict) -> list[dict]:
    """
    Instantiate datasets from pipeline.yaml using importlib.

    Returns list of {'name': str, 'dataset': MOABB dataset, 'paradigm': dict}.
    The 'paradigm' dict contains dataset-specific overrides that take precedence
    over method paradigm overrides in build_paradigm().

    To add a new dataset: add an entry to pipeline.yaml — no code changes needed.
    """
    eval_cfg = pipeline_cfg['evaluation']
    entries  = []

    for ds in parse_datasets(pipeline_cfg['datasets']):
        try:
            dataset = _instantiate(ds['class'], ds['init_params'])
        except Exception as exc:
            print(f"[dataset] ERROR instantiating '{ds['name']}' ({ds['class']}): {exc} — skipping.")
            continue


        entries.append({'name': ds['name'], 'dataset': dataset, 'paradigm': ds['paradigm']})

    if not entries:
        raise RuntimeError("No valid datasets found. Check the 'datasets' key in pipeline.yaml.")
    return entries


# ---------------------------------------------------------------------------
# Paradigm building
# ---------------------------------------------------------------------------

def build_paradigm(dataset_cfg: dict, method_overrides: dict, dataset_overrides: dict) -> MotorImagery:
    """
    Build a MotorImagery paradigm by merging base params, method overrides, and dataset overrides.

    Merge order (later wins):
        base → method_overrides → dataset_overrides

    This means dataset-specific params (e.g. bci2a tmin/tmax, channels, resample)
    always take precedence over method defaults, which is the correct behaviour for
    any dataset with its own paradigm window or channel configuration.
    """
    sig_cfg = dataset_cfg['signal']
    base = dict(
        events=['left_hand', 'right_hand'],
        n_classes=2,
        fmin=sig_cfg['filter']['l_freq'],
        fmax=sig_cfg['filter']['h_freq'],
        baseline=tuple(dataset_cfg['epoch']['baseline']),
    )
    base.update(method_overrides)
    base.update(dataset_overrides)
    return MotorImagery(**base)


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def print_results(all_results: pd.DataFrame) -> None:
    print("\n=== Pipeline Results (per subject) ===")
    try:
        pivot = all_results.pivot_table(index='subject', columns='pipeline', values='score')
        print(pivot.to_string())
    except Exception:
        print(all_results.to_string(index=False))

    print("\n=== Summary ===")
    for pipeline_name in all_results['pipeline'].unique():
        scores = all_results[all_results['pipeline'] == pipeline_name]['score']
        print(f"  {pipeline_name:<20} {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"\n  Chance level: 0.500")


def save_results(all_results: pd.DataFrame, pipeline_cfg: dict) -> None:
    results_dir = Path(pipeline_cfg['output']['results_dir'])
    results_dir.mkdir(exist_ok=True)

    stem     = pipeline_cfg['output']['filename']
    out_path = results_dir / f"{stem}.csv"
    all_results.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(pipeline_config_path: str = "config/pipeline.yaml") -> None:
    pipeline_cfg, dataset_cfg = load_configs(pipeline_config_path)
    methods   = parse_methods(pipeline_cfg['methods'])
    overwrite = pipeline_cfg['evaluation'].get('overwrite', True)

    print(f"Methods  : {[m['name'] for m in methods]}")
    print(f"Datasets : {[d['name'] for d in parse_datasets(pipeline_cfg['datasets'])]}")

    ds_entries = build_datasets(pipeline_cfg, dataset_cfg)

    all_results: list[pd.DataFrame] = []

    for method in methods:
        name   = method['name']
        params = method['params']

        try:
            sklearn_pipeline, method_paradigm = get_pipeline(name, dataset_cfg, params)
        except KeyError as exc:
            print(f"\nERROR: {exc} — skipping '{name}'.")
            continue

        for ds_entry in ds_entries:
            print(f"\n{'='*60}")
            print(f"Running: {name} on {ds_entry['name']}" + (f"  params={params}" if params else ""))
            print(f"{'='*60}")

            # Dataset paradigm overrides method paradigm (dataset always wins)
            paradigm   = build_paradigm(dataset_cfg, method_paradigm, ds_entry['paradigm'])
            evaluation = CrossSubjectEvaluation(
                paradigm=paradigm,
                datasets=[ds_entry['dataset']],
                overwrite=overwrite,
            )

            try:
                results: pd.DataFrame = evaluation.process({name: sklearn_pipeline})
            except Exception as exc:
                print(f"  ERROR: {exc}")
                continue

            scores = results[results['pipeline'] == name]['score']
            print(f"  {name} [{ds_entry['name']}]: {scores.mean():.3f} ± {scores.std():.3f}")
            all_results.append(results)

    if not all_results:
        print("\nNo results collected — check errors above.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    print_results(combined)
    save_results(combined, pipeline_cfg)

    # --- Optional: transfer learning ---
    tl_cfg = pipeline_cfg.get('transfer_learning', {})
    if tl_cfg.get('enabled', False):
        run_transfer_learning(dataset_cfg, pipeline_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Registry-based BCI Motor Imagery pipeline runner"
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline config (default: config/pipeline.yaml)"
    )
    args = parser.parse_args()
    main(args.config)
