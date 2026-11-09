"""Two-dimensional DeepONet.

The branch reads 12290 numbers: the state and both coefficients subsampled on the nested N64 sensor
grid, followed by the two fractional orders. Those features are standardised with statistics fitted on
the training split only, which is what keeps the very wide first layer well conditioned. At N128 the
trunk is queried directly on the finer coordinates while the branch keeps using the nested N64 sensors.
The latent inner product is the original branch/trunk form, and no carrier term or residual is added.
"""
import hashlib, json, math, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)
def sensor_vector(u, d, a, n=64):
    """Assemble the branch input as u0, Dx, Dy on the nested N64 sensors followed by the two orders."""
    size = u.shape[-1]
    if size not in [64, 128] or size % n:
        raise ValueError('fixed N64 sensors require N64/N128')
    stride = size // n
    return torch.cat((u[:, ::stride, ::stride].flatten(1), d[:, :, ::stride, ::stride].flatten(1), a), 1)
def fit_normalization(inputs, path, train_split):
    """Fit (or reload) the training-feature mean and scale; the statistics never see validation or test data."""
    root = Path(inputs) / 'data' / train_split
    receipt_hash = _sha(root / 'complete.json')
    if Path(path).exists():
        norm = _load_json(path)
        if norm['train_split_sha256'] != receipt_hash:
            raise ValueError('normalization train identity conflict')
        return norm
    # One instance contributes one row: values are not repeated over time, and no other split is read.
    cases = _load_json(root / 'cases.json')
    xs = []
    for row in cases:
        z = np.load(root / row['file'], allow_pickle=False)
        xs.append(np.concatenate((z['u0'].reshape(-1), z['D'].reshape(-1), z['alpha'])).astype(np.float64))
    x = np.stack(xs)
    center = x.mean(0)
    scale = np.maximum(x.std(0), 1e-6)
    norm = dict(schema='train-feature-standardization-v1', N_sensors=64, features=12290,
                order=['u0', 'D_x', 'D_y', 'alpha_x', 'alpha_y'], instances=len(xs), mean=center.tolist(),
                scale=scale.tolist(), scale_floor=1e-6, fit_split=train_split, train_split_sha256=receipt_hash,
                data_complete_sha256=_sha(Path(inputs) / 'data' / 'complete.json'), statistics_dtype='float64',
                model_dtype='float32', test_used=False)
    _write_json(path, norm)
    return norm
class DeepONet2D(nn.Module):
    """Branch/trunk DeepONet for the 2-D problem, with standardised branch inputs."""
    def __init__(self, norm, width=128, latent=64):
        super().__init__()
        self.n = 64
        self.register_buffer('input_mean', torch.tensor(norm['mean'], dtype=torch.float32))
        self.register_buffer('input_scale', torch.tensor(norm['scale'], dtype=torch.float32))
        if self.input_mean.shape != (12290,) or self.input_scale.shape != (12290,) or not torch.isfinite(self.input_mean).all() or not torch.isfinite(self.input_scale).all() or self.input_scale.min() <= 0:
            raise ValueError('normalization schema')
        def mlp(i):
            return nn.Sequential(nn.Linear(i, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, latent))
        self.branch = mlp(12290)
        self.trunk = mlp(5)
        self.bias = nn.Parameter(torch.zeros(()))
        # Fan-scaled first layer for the standardised inputs, and a smaller gain on the latent layer.
        for network in [self.branch, self.trunk]:
            for j in [0, 2, 4]:
                gain = math.sqrt(.5) if j == 0 else .5 if j == 4 else 1.
                nn.init.xavier_uniform_(network[j].weight, gain=gain)
                nn.init.zeros_(network[j].bias)
    def latent(self, u, d, a):
        x = (sensor_vector(u, d, a) - self.input_mean) / self.input_scale
        return self.branch(x)
    def query_features(self, u, t):
        b, n, _ = u.shape
        x = 2 * torch.pi * torch.arange(n, device=u.device, dtype=u.dtype) / n
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        q = torch.stack((xx.sin(), xx.cos(), yy.sin(), yy.cos()), -1).expand(b, -1, -1, -1)
        return torch.cat((q, t[:, None, None, None].expand(b, n, n, 1)), -1)
    def forward(self, u, d, a, t):
        out = torch.einsum('bl,bxyl->bxy', self.latent(u, d, a), self.trunk(self.query_features(u, t))) + self.bias
        return torch.where((t == 0)[:, None, None], u, out)
__all__ = ['DeepONet2D', 'fit_normalization', 'sensor_vector']
