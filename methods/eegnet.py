"""
EEGNet pipeline factory (braindecode).

Returns paradigm_overrides with the active MI epoch window (dl.tmin / dl.tmax).

Configurable params (via pipeline.yaml):
    max_epochs  : int    Training epochs   (default: dl.max_epochs)
    lr          : float  Learning rate     (default: dl.lr)
    batch_size  : int    Mini-batch size   (default: dl.batch_size)
"""

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from braindecode.models import EEGNet
from braindecode.classifier import EEGClassifier


def build_eegnet(config: dict, params: dict) -> tuple[Pipeline, dict]:
    dl_cfg  = config['dl']
    sig_cfg = config['signal']

    n_chans    = len(sig_cfg['channels'])
    # MNE Epochs include both endpoints: round((tmax - tmin) * sfreq) + 1
    n_times    = int(round((dl_cfg['tmax'] - dl_cfg['tmin']) * sig_cfg['sfreq'])) + 1
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_epochs = params.get('max_epochs', dl_cfg['max_epochs'])
    lr         = params.get('lr',         dl_cfg['lr'])
    batch_size = params.get('batch_size', dl_cfg['batch_size'])

    clf = EEGClassifier(
        module=EEGNet,
        module__n_chans=n_chans,
        module__n_outputs=2,
        module__n_times=n_times,
        module__final_conv_length='auto',
        max_epochs=max_epochs,
        lr=lr,
        batch_size=batch_size,
        train_split=None,
        device=device,
        verbose=0,
    )

    def cast_to_float32(X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)

    pipeline = Pipeline([
        ('cast', FunctionTransformer(cast_to_float32)),
        ('eegnet', clf),
    ])

    paradigm_overrides = {'tmin': dl_cfg['tmin'], 'tmax': dl_cfg['tmax']}
    return pipeline, paradigm_overrides
