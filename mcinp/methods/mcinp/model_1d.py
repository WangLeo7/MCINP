"""One-dimensional MCINP.

The correction network reads eight local features of the current state and the local operator action,
and returns one real scalar per sample. Everything else, including the signed multipliers, the
interaction band and the recurrence, belongs to the shared propagator in ``mcinp.methods._core``.
"""
import torch
from torch import nn
from mcinp.methods._core import propagate
FEATURES = ['Re_w', 'Im_w', 'Re_Kw', 'Im_Kw', 'alpha', 'tau', 'k/B', 'masked_abs_k^alpha']
class MCINP1D(nn.Module):
    """Modewise interaction correction for the 1-D signed directional problem."""
    dim = 1
    correction_kind = 'modewise'
    def __init__(self, n, bandwidth, steps, width, equation='signed'):
        super().__init__()
        if equation != 'signed':
            raise ValueError('this release implements the signed directional equation only')
        self.n, self.bandwidth, self.steps, self.equation = n, bandwidth, steps, equation
        self.net = nn.Sequential(nn.Linear(len(FEATURES), width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
    def q_value(self, w, J, parts, ctx, expand):
        """Map the current carrier state and the interaction action to the scalar correction."""
        ks, lam, tau, means, shp, alpha = ctx['ks'], ctx['lam'], ctx['tau'], ctx['means'], ctx['shp'], ctx['alpha']
        features = [w.real, w.imag, J.real, J.imag, expand(alpha[:, 0]), expand(tau),
                    ks[0].expand_as(w.real) / self.bandwidth, lam / means[:, 0].reshape(shp)]
        return self.net(torch.stack(features, -1)).squeeze(-1)
    def forward(self, u, ds, a, t):
        return propagate(self, u, ds, a, t)
__all__ = ['MCINP1D', 'FEATURES']
