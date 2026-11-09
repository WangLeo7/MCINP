"""POD-linear: the training-output POD basis with a linear map, selected on validation data.

The method is closed form, so it has no network and no training loop. Both dimensions fit the basis
and the map on training outputs only, which is why the fitting lives here rather than in the runtime.
"""
from pathlib import Path
from . import model_1d, model_2d
PARAMETRIC = False
def build(dimension, config, normalization=None):
    """Closed-form method: there is no module to construct."""
    return None
def fit(dimension, config, train, validation, path):
    """Fit the basis and the linear map on training rows only and return the selection record."""
    if int(dimension) == 2:
        return model_2d.fit_pod2(train, validation, config['pod_ranks'], config['pod_ridges'], path)
    return model_1d.fit_pod1(train, config, path)
def predictor(dimension, config, out_dir):
    """Load a fitted artifact and return the predictor used by evaluation."""
    if int(dimension) == 2:
        return model_2d.pod2_predictor(Path(out_dir) / 'pod.npz')
    return model_1d.pod1_predictor(Path(out_dir) / 'pod.npz')
__all__ = ['PARAMETRIC', 'build', 'fit', 'predictor']
