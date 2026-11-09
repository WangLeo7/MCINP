"""Global-Q in both dimensions.

Global-Q keeps the interaction term but replaces MCINP's modewise scalar field with a single learned
real scalar. There is no dimension-specific code: the propagator derives the grid, the multipliers and
the band from the rank of the input, so one class covers both dimensions and the ablation isolates
exactly one design choice, the parameterisation of q.
"""
import torch
from torch import nn
from mcinp.methods._core import propagate
class GlobalQ(nn.Module):
    """One learned scalar multiplying the interaction action."""
    correction_kind = 'scalar'
    def __init__(self, n, bandwidth, steps, equation='signed'):
        super().__init__()
        if equation != 'signed':
            raise ValueError('this release implements the signed directional equation only')
        self.n, self.bandwidth, self.steps, self.equation = n, bandwidth, steps, equation
        self.q = nn.Parameter(torch.ones(()))
    def forward(self, u, ds, a, t):
        return propagate(self, u, ds, a, t)
__all__ = ['GlobalQ']
