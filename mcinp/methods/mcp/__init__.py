"""MCP: the deterministic mean-coefficient baseline with no interaction correction."""
from .model import MCP
PARAMETRIC = False
def build(dimension, config, normalization=None):
    """Construct the MCP model for either dimension."""
    return MCP(config['N'], config['B'], config['L'], config['equation'])
__all__ = ['MCP', 'PARAMETRIC', 'build']
