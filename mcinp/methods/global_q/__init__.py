"""Global-Q: the ablation that replaces the learned field with one learned real scalar."""
from .model import GlobalQ
PARAMETRIC = True
def build(dimension, config, normalization=None):
    """Construct the Global-Q model for either dimension."""
    return GlobalQ(config['N'], config['B'], config['L'], config['equation'])
__all__ = ['GlobalQ', 'PARAMETRIC', 'build']
