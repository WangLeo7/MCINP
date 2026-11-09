"""One-dimensional POD-linear.

The rank and the ridge are fixed by the paper (16 and 1e-6), the basis comes from the training outputs
only, and the coefficients are the closed-form ridge solution. There is no selection step and no test
data is read, which is why ``selection`` below says so explicitly in the record it returns.
"""
import numpy as np
from scipy.signal import resample
from mcinp.runtime.common import save_npz
RANK = 16
RIDGE = 1e-6
def features_1d(u, d, alpha, t, n):
    """Feature map: resampled input fields, the fractional order, the query time and a bias column."""
    if u.shape[-1] != n:
        u = resample(u, n, axis=-1)
        d = resample(d, n, axis=-1)
    return np.c_[u, d[:, 0], alpha, np.asarray(t).reshape(-1), np.ones(len(u))]
def fit_pod1(train, config, path):
    """Fit the output mean, basis and ridge map on training rows only."""
    u, d, alpha, t, y = [item.numpy().astype(float) for item in train]
    n = config['N']
    mean = y.mean(0)
    _, _, vh = np.linalg.svd(y - mean, full_matrices=False)
    basis = vh[:min(RANK, len(vh))].T
    features = features_1d(u, d, alpha, t, n)
    weights = np.linalg.solve(features.T @ features + RIDGE * np.eye(features.shape[1]), features.T @ ((y - mean) @ basis))
    save_npz(path, mean=mean, basis=basis, weights=weights)
    return dict(fit_split='train', rank=int(basis.shape[1]), ridge=RIDGE,
                selection='fixed architecture, no test', optimizer='NA closed form')
def pod1_predictor(path):
    """Load a fitted artifact and return a predictor on the native or a finer grid."""
    z = np.load(path, allow_pickle=False)
    n = int(z['mean'].shape[0])
    def predict(u, d, alpha, time):
        if time == 0:
            return u.astype(np.float32)
        features = features_1d(u[None], d[None], alpha[None], [time], n)
        value = z['mean'] + features @ z['weights'] @ z['basis'].T
        return (value[0] if len(u) == n else resample(value[0], len(u))).astype(np.float32)
    return predict
__all__ = ['RANK', 'RIDGE', 'features_1d', 'fit_pod1', 'pod1_predictor']
