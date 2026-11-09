"""One-dimensional DeepONet.

The branch reads the sensor-averaged state, the sensor-averaged coefficient and the fractional order;
the trunk reads the periodic coordinate and the query time. The latent inner product is the original
branch/trunk form, with fan-scaled initialisation.
"""
import torch
from torch import nn
class DeepONet1D(nn.Module):
    """Branch/trunk DeepONet for the 1-D problem."""
    def __init__(self, n, width=64):
        super().__init__()
        self.n = n
        def mlp(i):
            return nn.Sequential(nn.Linear(i, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))
        self.branch = mlp(2 * n + 1)
        self.trunk = mlp(3)
        self.bias = nn.Parameter(torch.zeros(1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                std = (2 / (m.in_features + m.out_features)) ** .5
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
                nn.init.zeros_(m.bias)
    def forward(self, u, ds, a, t):
        """Evaluate the operator; the sensor subsampling maps a finer grid onto the training sensors."""
        b, n = u.shape
        x = 2 * torch.pi * torch.arange(n, device=u.device, dtype=u.dtype) / n
        x = x.expand(b, -1)
        if n % self.n:
            raise ValueError('DeepONet periodic sensor mapping requires divisible resolution')
        us = u[:, ::n // self.n]
        d = ds[:, 0, ::n // self.n]
        branch = self.branch(torch.cat((us, d, a), -1))
        q = torch.stack((x.sin(), x.cos(), t[:, None].expand_as(x)), -1)
        out = torch.einsum('bl,bnl->bn', branch, self.trunk(q)) + self.bias
        return torch.where((t == 0)[:, None], u, out)
__all__ = ['DeepONet1D']
