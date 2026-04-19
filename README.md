# Cross-Subject Motor Imagery BCI Pipeline

A configurable pipeline for cross-subject Motor Imagery EEG classification using [MOABB](https://moabb.neurotechx.com/) and [MNE](https://mne.tools/). Developed as part of a Bachelor's thesis at the University of West Bohemia.

## What it does

Evaluates BCI decoding methods on EEG motor imagery data using **leave-one-subject-out (LOSO)** cross-validation. Any combination of datasets and classification methods can be selected via a single config file — no code changes required.

**Implemented methods:**

| Method | Type |
|---|---|
| CSP + LDA | Classical |
| CSP + LR | Classical |
| CSP + SVM | Classical |
| EEGNet | Deep Learning |
| ShallowConvNet | Deep Learning |

**Supported datasets:**
- Custom BrainVision dataset (recorded in-house, 29 subjects, left/right hand MI)
- BCI Competition IV Dataset 2a (`BNCI2014_001` via MOABB, 9 subjects)
- Any other MOABB dataset — add it via config only

---

## Project structure

```
config/
  pipeline.yaml          # top-level run config — edit this to control what runs
  dataset_config.yaml    # signal, epoch, and model parameters

dataset/
  custom_dataset.py      # custom MOABB dataset class for BrainVision data

evaluation/
  transfer_learning.py   # EEGNet pretrain (BCI IV-2a) → fine-tune (custom dataset)

methods/
  __init__.py            # method registry
  csp.py                 # CSP+LDA, CSP+LR, CSP+SVM factories
  eegnet.py              # EEGNet factory
  shallow.py             # ShallowConvNet factory

diagnostics/
  dataset_analysis.py        # dataset statistics (trial counts, class balance)
  inspect_below_chance.py    # per-subject EEG plots for diagnosing low-accuracy subjects

utils/
  file_utils.py          # BrainVision filename parsing helpers

tests/
  test_dataset.py
  test_file_utils.py

run_pipeline.py          # entry point
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the BrainVision data in `data/` (date-based folder structure as recorded).

---

## Configuration files

The project uses two config files with distinct responsibilities:

| File | Purpose |
|---|---|
| `config/pipeline.yaml` | **What to run** — datasets, methods, hyperparameter overrides, transfer learning toggle |
| `config/dataset_config.yaml` | **How to process the signal** — sampling rate, channels, filter, epoch window, event codes, default model hyperparameters |

`dataset_config.yaml` is tied to the recording setup and protocol — it rarely changes. Its values are used as defaults throughout the pipeline (e.g. `dl.max_epochs` is the default that `pipeline.yaml` can override per method).

`pipeline.yaml` is what you edit when experimenting. It never duplicates keys from `dataset_config.yaml` — it only overrides them where needed.

---

## Running

```bash
python run_pipeline.py
```

Results are saved to `results/pipeline_results.csv`.

To use a different config file:

```bash
python run_pipeline.py --config path/to/pipeline.yaml
```

---

## Configuration — `config/pipeline.yaml`

This is the only file you need to edit for day-to-day use.

### Selecting methods

List any combination of registered methods. Plain name uses defaults; add a `params:` block to override hyperparameters.

```yaml
methods:
  - name: CSP+LDA              # uses defaults from dataset_config.yaml

  - name: EEGNet
    params:
      max_epochs: 200          # override specific params
      lr: 0.0005

  # - name: CSP+LR            # commented out = not run
  # - name: CSP+SVM
  # - name: ShallowConvNet
```

**Configurable params per method:**

| Method | Params |
|---|---|
| CSP+LDA / CSP+LR / CSP+SVM | `n_components` |
| EEGNet / ShallowConvNet | `max_epochs`, `lr`, `batch_size` |

### Selecting datasets

Each dataset entry specifies the Python class to instantiate, its constructor arguments, and any dataset-specific paradigm parameters.

```yaml
datasets:
  - name: custom
    class: dataset.custom_dataset.MotorImageryDataset
    init_params:
      data_path: "data/"
      config_path: "config/dataset_config.yaml"
    paradigm: {}               # no paradigm override — method defaults apply

  - name: bci2a
    class: moabb.datasets.BNCI2014_001
    init_params: {}
    paradigm:
      tmin: 0.5
      tmax: 3.5
      channels: ["C3", "C4", "Cz"]
      resample: 500
```

The `paradigm:` block sets dataset-specific parameters that override method defaults. This is important for datasets with different epoch windows or channel configurations.

### Transfer learning

```yaml
transfer_learning:
  enabled: false   # set to true to run EEGNet pretrain (BCI IV-2a) → fine-tune (custom)
```

Requires EEGNet to have been run first (uses its results as the custom-only baseline). Results saved to `results/transfer_learning.csv`.

---

## Adding a new method

**Step 1** — Create `methods/mymethod.py`:

```python
from sklearn.pipeline import Pipeline

def build_mymethod(config: dict, params: dict) -> tuple[Pipeline, dict]:
    # config  : full dataset_config.yaml
    # params  : overrides from pipeline.yaml (may be {})
    # return  : (sklearn Pipeline, paradigm kwarg overrides)

    pipeline = Pipeline([...])
    return pipeline, {}        # {} = use dataset's paradigm window
```

**Step 2** — Register it in `methods/__init__.py`:

```python
from methods.mymethod import build_mymethod   # add import

REGISTRY = {
    ...
    "MyMethod": build_mymethod,               # add entry
}
```

**Step 3** — Add it to `config/pipeline.yaml`:

```yaml
methods:
  - name: MyMethod
```

That's it. `run_pipeline.py` never needs to change.

---

## Adding a new dataset

No code changes required for standard MOABB datasets. Add an entry to `config/pipeline.yaml`:

```yaml
datasets:
  - name: mydataset
    class: moabb.datasets.SomeDataset    # dotted import path
    init_params: {}                       # constructor keyword arguments
    paradigm:
      tmin: 0.0
      tmax: 3.0
      channels: ["C3", "C4", "Cz"]
      resample: 250
```

For a **custom dataset class**, implement a MOABB-compatible `BaseDataset` subclass (see `dataset/custom_dataset.py` as reference), place it in `dataset/`, and point `class:` at it:

```yaml
- name: mycustomdataset
  class: dataset.my_custom_dataset.MyDataset
  init_params:
    data_path: "data/mydata/"
```
