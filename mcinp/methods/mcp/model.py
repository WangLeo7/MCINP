"""MCP-only in both dimensions.

The carrier advances the mean-coefficient evolution over the whole interval in one exponential step and
adds no interaction correction, so the method has no parameters and no dimension-specific code. It is
the reference against which the correction methods have to earn their improvement.
"""
import torch
from torch import nn
from mcinp.methods._core import propagate
class MCP(nn.Module):
    """Mean-coefficient evolution; the propagator performs a single unconditional step."""
    correction_kind = 'none'
    def __init__(self, n, bandwidth, steps, equation='signed'):
        super().__init__()
        if equation != 'signed':
            raise ValueError('this release implements the signed directional equation only')
        self.n, self.bandwidth, self.steps, self.equation = n, bandwidth, steps, equation
    def forward(self, u, ds, a, t):
        return propagate(self, u, ds, a, t)
__all__ = ['MCP']
