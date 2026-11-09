"""Two-dimensional MCINP.

The 2-D correction sees four extra features: the second coordinate's order and normalised wavenumber,
the combined wavenumber, and the two directional interaction terms. The correction is still one real
scalar per sample, and the recurrence, the band mask and the Hermitian projection come from the shared
propagator in ``mcinp.methods._core``.
"""
import torch
from torch import nn
from mcinp.methods._core import propagate
FEATURES = ['Re_w', 'Im_w', 'Re_Kw', 'Im_Kw', 'alpha_x', 'alpha_y', 'tau', 'kx/B', 'ky/B', 'norm_k/B',
            'Re_Kxw', 'Re_Kyw', 'mean_Dx', 'mean_Dy']
class MCINP2D(nn.Module):
    """Modewise interaction correction for the 2-D directional (non-radial) problem."""
    dim = 2
    correction_kind = 'modewise'
    def __init__(self, n, bandwidth, steps, width, equation='signed'):
        super().__init__()
        if equation != 'signed':
            raise ValueError('this release implements the signed directional equation only')
        self.n, self.bandwidth, self.steps, self.equation = n, bandwidth, steps, equation
        self.net = nn.Sequential(nn.Linear(len(FEATURES), width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
    def q_value(self, w, J, parts, ctx, expand):
        """Map the current carrier state, both directional actions and the geometry to one scalar."""
        ks, lam, tau, means, shp, alpha = ctx['ks'], ctx['lam'], ctx['tau'], ctx['means'], ctx['shp'], ctx['alpha']
        features = [w.real, w.imag, J.real, J.imag, expand(alpha[:, 0]), expand(alpha[:, 1]), expand(tau),
                    ks[0].expand_as(w.real) / self.bandwidth, ks[1].expand_as(w.real) / self.bandwidth,
                    torch.sqrt(sum(component * component for component in ks)).expand_as(w.real) / self.bandwidth,
                    parts[0].real, parts[1].real, expand(means[:, 0]), expand(means[:, 1])]
        return self.net(torch.stack(features, -1)).squeeze(-1)
    def forward(self, u, ds, a, t):
        return propagate(self, u, ds, a, t)
__all__ = ['MCINP2D', 'FEATURES']
