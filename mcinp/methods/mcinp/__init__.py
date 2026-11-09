"""MCINP: a learned modewise scalar that corrects each interaction mode."""
from .model_1d import MCINP1D
from .model_2d import MCINP2D
PARAMETRIC = True
_MODELS = {1: MCINP1D, 2: MCINP2D}
def build(dimension, config, normalization=None):
    """Construct the MCINP model; the two dimensions differ in their feature set, not in the propagator."""
    return _MODELS[int(dimension)](config['N'], config['B'], config['L'], config['width'], config['equation'])
__all__ = ['MCINP1D', 'MCINP2D', 'PARAMETRIC', 'build']
