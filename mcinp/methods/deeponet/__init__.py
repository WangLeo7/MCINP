"""DeepONet: the branch/trunk operator baseline, one implementation per dimension."""
from . import model_1d, model_2d
from .model_1d import DeepONet1D
from .model_2d import DeepONet2D, fit_normalization, sensor_vector
PARAMETRIC = True
def build(dimension, config, normalization=None):
    """Construct the DeepONet for ``dimension``.

    The 2-D branch reads 12290 numbers, so it needs the standardisation statistics fitted on the
    training split; passing them explicitly keeps that dependency visible instead of implicit.
    """
    if int(dimension) == 2:
        if normalization is None:
            raise ValueError('the 2-D DeepONet requires the training-feature normalization sidecar')
        return DeepONet2D(normalization, config.get('don_width', 128), config.get('don_latent', 64))
    return DeepONet1D(config['N'], config.get('don_width', 64))
__all__ = ['DeepONet1D', 'DeepONet2D', 'fit_normalization', 'sensor_vector', 'PARAMETRIC', 'build']
