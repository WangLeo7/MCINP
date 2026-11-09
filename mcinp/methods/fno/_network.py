"""The shared FNO wrapper around the official NeuralOperator implementation.

The architecture is stated explicitly rather than left to defaults: the spectral convolution is the
canonical real-FFT convolution from ``mcinp.methods.fno.canonical_spectral``, the inputs are the state, the
per-coordinate coefficients, the per-coordinate orders and the query time, and the layer count is
chosen from the spatial dimension.
"""
import torch
from torch import nn
class FourierNeuralOperator(nn.Module):
    """Official FNO with the paper's explicit architecture choices."""
    def __init__(self, dim, n, width, modes, layers):
        super().__init__()
        from neuralop.models import FNO

        from .canonical_spectral import CanonicalSpectralConv
        self.dim = dim
        self.net = FNO(n_modes=(modes,) * dim, hidden_channels=width, in_channels=2 * dim + 2, out_channels=1,
                       n_layers=layers, positional_embedding='grid', non_linearity=nn.GELU(),
                       channel_mlp_skip='soft-gating', fno_skip='linear', domain_padding=None,
                       conv_module=CanonicalSpectralConv)
    def forward(self, u, ds, a, t):
        shp = (len(u),) + (1,) * self.dim
        fields = [u] + [ds[:, j] for j in range(self.dim)] + [a[:, j].reshape(shp).expand_as(u) for j in range(self.dim)] + [t.reshape(shp).expand_as(u)]
        out = self.net(torch.stack(fields, 1)).squeeze(1)
        return torch.where((t == 0).reshape(shp), u, out)
__all__ = ['FourierNeuralOperator']
