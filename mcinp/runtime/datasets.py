"""Editable inputs for periodic directional diffusion."""
import numpy as np
def grid(n, dimension):
    x = 2 * np.pi * np.arange(n) / n
    return np.meshgrid(*([x] * dimension), indexing="ij")
def periodic_gaussian(n, center, sigma, amplitude=1.0, offset=0.0):
    """Sample a Gaussian of the wrapped distance to the supplied center."""
    center = np.atleast_1d(center)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), center.shape)
    if center.size not in (1, 2) or np.any(sigma <= 0):
        raise ValueError("Supply a 1D or 2D center and positive widths.")
    distance = sum((((x - c + np.pi) % (2 * np.pi) - np.pi) / w)**2
                   for x, c, w in zip(grid(n, center.size), center, sigma))
    return offset + amplitude * np.exp(-0.5 * distance)
def random_fourier(rng, n, wavevectors, decay=2.0, scale=1.0):
    """Sample real Fourier modes with Gaussian coefficients."""
    modes = np.asarray(wavevectors)
    if modes.ndim == 1:
        modes = modes[:, None]
    coordinates = grid(n, modes.shape[1])
    field = np.zeros_like(coordinates[0])
    for mode in modes:
        norm = np.linalg.norm(mode)
        if norm == 0:
            raise ValueError("Set the constant mode separately.")
        phase = sum(k * x for k, x in zip(mode, coordinates))
        cosine, sine = rng.normal(0, scale / norm**decay, size=2)
        field += cosine * np.cos(phase) + sine * np.sin(phase)
    return field


def gaussian_noise(field, rng, level, axes=None):
    """Add Gaussian noise scaled by the field's standard deviation."""
    return field + level * field.std(axis=axes, keepdims=True) * rng.normal(size=field.shape)


def sample_id_1d(rng, n):
    """Draw one central 1D case in the original random-stream order."""
    alpha = rng.uniform(1.15, 1.85)
    amplitude = rng.uniform(0.7, 1.3)
    modes = range(1, 9)
    u0 = rng.normal(0, 0.25 * amplitude) + random_fourier(rng, n, modes, 2.0, amplitude)
    field = random_fourier(rng, n, modes, 1.5)
    field = (field - field.mean()) / (field.std() + 1e-12)
    base = rng.uniform(1.0, 1.6)
    contrast = rng.uniform(0.15, 0.35)
    return u0, np.array([base * np.exp(contrast * field / 2)]), np.array([alpha])


def sample_id_2d(seed, index, n):
    """Draw one central 2D case, with the same coefficients at either grid size."""
    central_offset = sum(map(ord, "central"))
    rng = np.random.default_rng(seed + 1009 * index + central_offset)
    alpha = 1.4 + 0.4 * rng.random(2)
    coefficient_modes = []
    initial_modes = []
    for k in range(1, 5):
        scale = (0.16 if k <= 2 else 0.08) / k**0.7
        coefficient_modes.append((k, 1 + k % 3, scale * rng.normal() * 0.16,
                                  scale * rng.normal() * 0.16))
    for k in range(1, 5):
        scale = (0.5 if k <= 2 else 0.22) / k**0.8
        initial_modes.append((k, 1 + (k + 1) % 3, scale * rng.normal(), scale * rng.normal()))

    x, y = grid(n, 2)
    u0 = np.full((n, n), 0.15)
    dx = np.full_like(u0, 1.25)
    dy = dx.copy()
    for k, ell, p, q in initial_modes:
        u0 += p * np.cos(k * x + ell * y) + q * np.sin(ell * x - k * y)
    for k, ell, p, q in coefficient_modes:
        dx += p * np.cos(k * x + ell * y) + q * np.sin(ell * x - k * y)
        dy += p * np.sin(k * x - ell * y) + q * np.cos(ell * x + k * y)
    return u0, np.array([np.maximum(dx, 0.2), np.maximum(dy, 0.2)]), alpha


def scenario_plan(config):
    test_n = config["test_N"] or config["N"]
    if config["data_seed"] is not None:
        seeds = np.random.SeedSequence(config["data_seed"]).generate_state(3).tolist()
    elif config["dimension"] == 1:
        seeds = (101, 202, 700 if test_n == 128 else 303)
    else:
        seeds = (20260908, 20260909, 20260910)

    splits = (
        ("train", "train_count", config["N"]),
        ("validation", "validation_count", config["N"]),
        ("test", "test_count", test_n),
    )
    return [dict(split=role, role=role, count=config[key], N=n, data_seed=seed)
            for (role, key, n), seed in zip(splits, seeds)]


def f32_exact(value):
    return np.asarray(value, dtype=np.float32).astype(np.float64)


def make_case(dimension, row, index, fields):
    u0, coefficients, alpha = map(f32_exact, fields)
    # For a noise test, modify only the observed fields and keep D positive.
    observed_u, observed_d = u0.copy(), coefficients.copy()
    # For an order switch, specify its time and post-switch orders for the test split.
    switch, alpha_after = -1.0, alpha.copy()
    return dict(u0=f32_exact(observed_u), D=f32_exact(observed_d), alpha=alpha,
                truth_u0=u0, truth_D=coefficients, truth_alpha=alpha,
                alpha_after=f32_exact(alpha_after), switch=np.array(switch),
                identity=dict(dimension=dimension, **row, index=index))


def iter_cases(dimension, row):
    """Stream cases so the 1D split uses a single sequential random generator."""
    n = row["N"]
    seed = row["data_seed"]
    if dimension == 1:
        rng = np.random.default_rng(seed)
        for index in range(row["count"]):
            yield make_case(dimension, row, index, sample_id_1d(rng, n))
    else:
        for index in range(row["count"]):
            yield make_case(dimension, row, index, sample_id_2d(seed, index, n))
