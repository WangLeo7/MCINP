"""Two-dimensional POD-linear baseline fitted on training outputs only."""
import numpy as np
from scipy.signal import resample
from mcinp.runtime.common import save_npz


def periodic_resample(array, n, axes=(-2, -1)):
    for axis in axes:
        if array.shape[axis] != n:
            array = resample(array, n, axis=axis)
    return array


def input_features(u, d, alpha, sensor=8):
    fields = [periodic_resample(u, sensor).reshape(len(u), -1),
              periodic_resample(d, sensor).reshape(len(u), -1), alpha]
    return np.concatenate(fields, axis=1)


def design(u, d, alpha, times, center, scale, time_center, time_scale):
    x = (input_features(u, d, alpha) - center) / scale
    tau = (np.asarray(times).reshape(-1, 1) - time_center) / time_scale
    return np.concatenate([np.concatenate((x, np.ones((len(x), 1))), axis=1) * tau**degree
                           for degree in range(3)], axis=1)


def fit_pod2(train, validation, ranks, ridges, path):
    """Fit the output mean/basis, feature normalization, and map from train rows; select with validation."""
    u, d, a, t, y = [item.numpy().astype(np.float64) for item in train]
    vu, vd, va, vt, vy = [item.numpy().astype(np.float64) for item in validation]
    xraw = input_features(u, d, a)
    center = xraw.mean(axis=0)
    scale = np.maximum(xraw.std(axis=0), 1e-6)
    time_center = float(t.mean())
    time_scale = max(float(t.std()), 1e-6)
    x = design(u, d, a, t, center, scale, time_center, time_scale)
    vx = design(vu, vd, va, vt, center, scale, time_center, time_scale)
    yflat = y.reshape(len(y), -1)
    output_mean = yflat.mean(axis=0)
    yc = yflat - output_mean

    # This deterministic randomized SVD sees training output rows only.
    max_rank = min(max(ranks), len(yc) - 1, yc.shape[1])
    rng = np.random.default_rng(20261001)
    omega = rng.normal(size=(yc.shape[1], min(max_rank + 8, min(yc.shape))))
    q, _ = np.linalg.qr(yc @ omega)
    for _ in range(2):
        q, _ = np.linalg.qr(yc @ (yc.T @ q))
    _, _, vh = np.linalg.svd(q.T @ yc, full_matrices=False)
    basis = vh[:max_rank].T

    gram = x.T @ x / len(x)
    eigenvalues, vectors = np.linalg.eigh(gram)
    rhs = x.T @ (yc @ basis) / len(x)
    candidates = []
    best = None
    for ridge in ridges:
        weights_all = vectors @ ((vectors.T @ rhs) / (eigenvalues[:, None] + ridge))
        for rank in ranks:
            if rank > max_rank:
                continue
            prediction = output_mean + (vx @ weights_all[:, :rank]) @ basis[:, :rank].T
            loss = float(np.mean((prediction - vy.reshape(len(vy), -1))**2))
            candidates.append(dict(rank=int(rank), ridge=float(ridge), validation_mse=loss))
            if best is None or loss < best[0]:
                best = (loss, int(rank), float(ridge), weights_all[:, :rank].copy())
    if best is None:
        raise ValueError('No valid POD rank/ridge candidates')
    _, rank, ridge, weights = best
    save_npz(path, mean=output_mean, basis=basis[:, :rank], weights=weights,
             center=center, scale=scale, time_center=np.array(time_center),
             time_scale=np.array(time_scale), N=np.array(y.shape[-1]))
    return dict(rank=rank, ridge=ridge, validation_mse=best[0], candidates=candidates,
                basis_fit='training outputs only; deterministic randomized SVD',
                normalization_fit='training instances only', time_polynomial_degree=2,
                train_examples=len(y), test_used=False,
                basis_relative_residual=float(np.linalg.norm(yc-(yc@basis[:,:rank])@basis[:,:rank].T)
                                              / max(np.linalg.norm(yc), 1e-12)))


def pod2_predictor(path):
    """Load the complete readable NPZ basis/map/normalization and return a strict predictor."""
    z = np.load(path, allow_pickle=False)
    required = {'mean', 'basis', 'weights', 'center', 'scale', 'time_center', 'time_scale', 'N'}
    if set(z.files) != required or int(z['N']) != 64:
        raise ValueError('POD artifact keys or fit grid do not match the paper contract')
    rank = z['basis'].shape[1]
    shapes = {'mean': (4096,), 'basis': (4096, rank), 'weights': (585, rank),
              'center': (194,), 'scale': (194,), 'time_center': (), 'time_scale': ()}
    for key, shape in shapes.items():
        if z[key].shape != shape or z[key].dtype != np.float64 or not np.isfinite(z[key]).all():
            raise ValueError(f'Invalid POD artifact array: {key}')
    if np.min(z['scale']) <= 0 or float(z['time_scale']) <= 0:
        raise ValueError('POD scales must be positive')

    def predict(u, d, alpha, time):
        if time == 0:
            return u.astype(np.float32)
        x = design(u[None], d[None], alpha[None], [time], z['center'], z['scale'],
                   float(z['time_center']), float(z['time_scale']))
        field = (z['mean'] + (x @ z['weights']) @ z['basis'].T).reshape(64, 64)
        return periodic_resample(field, u.shape[-1]).astype(np.float32)
    return predict
