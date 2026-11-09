"""Load one editable configuration and validate the numerical inputs."""
from math import isfinite
from pathlib import Path

from .methods import METHODS
from .runtime.common import load

DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "settings.json"
DIMENSION_DEFAULTS = {
    1: dict(B=16, modes=16, batch_size=16, optimizer="AdamW", weight_decay=1e-4, shuffle=True,
            don_width=64, don_latent=64, don_learning_rate=1e-3, pod_ranks=[16], pod_ridges=[1e-6],
            minimum_epochs=1, min_relative_improvement=0.0),
    2: dict(B=8, modes=8, batch_size=4, optimizer="Adam", weight_decay=0.0, shuffle=False,
            don_width=128, don_latent=64, don_learning_rate=1e-4, pod_ranks=[8, 16, 32], pod_ridges=[1e-6, 1e-3, 0.1],
            minimum_epochs=100, min_relative_improvement=0.001),
}

def load_config(path, method=None, dimension=None, seed=None, device=None):
    config = load(DEFAULTS_PATH)
    config.update(load(path))
    if dimension is not None:
        config["dimension"] = dimension
    if seed is not None:
        config["seed"] = seed
    if device is not None:
        config["device"] = device
    identifiers = {name.casefold(): key for key, (name, _) in METHODS.items()}
    identifiers.update({key: key for key in METHODS})
    name = (method or config["method"]).casefold()
    if name not in identifiers:
        raise ValueError("Unknown method. Choose MCINP, Global-Q, MCP, FNO, DeepONet or POD-linear.")
    config["method"] = identifiers[name]
    config["equation"] = "signed"
    if config["method"] in ("mcp", "pod_linear"):
        config["seed"] = 0
    dimension = config["dimension"]
    if dimension not in DIMENSION_DEFAULTS:
        raise ValueError("Use dimension 1 or 2.")
    for key, value in DIMENSION_DEFAULTS[dimension].items():
        config.setdefault(key, value)
    n = config["N"]
    if config["method"] == "deeponet":
        config["learning_rate"] = config["don_learning_rate"]
    test_n = n if config["test_N"] is None else config["test_N"]
    grids = (n, test_n)
    if any(type(g) is not int or g < 4 or g % 2 for g in grids):
        raise ValueError("Training and test grid sizes must be even integers of at least 4.")
    for key in ("B", "L", "epochs", "batch_size", "train_count", "validation_count", "test_count"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["B"] >= min(grids) // 2:
        raise ValueError("B must be smaller than half of every scenario grid size.")
    data_seed = config["data_seed"]
    if data_seed is not None and (type(data_seed) is not int or data_seed < 0):
        raise ValueError("data_seed must be null or a nonnegative integer.")
    times = config["times"]
    if len(times) < 2 or times[0] != 0 or any(not isfinite(t) for t in times) or any(
            right <= left for left, right in zip(times, times[1:])
    ):
        raise ValueError("List strictly increasing finite observation times starting at zero.")
    if dimension == 2 and config["method"] in ("deeponet", "pod_linear") and n != 64:
        raise ValueError("The 2D DeepONet and POD-linear baseline implementations use N=64 for training.")
    if dimension == 2 and config["method"] == "deeponet" and test_n not in (64, 128):
        raise ValueError("The 2D DeepONet evaluates on N=64 or N=128.")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda.")
    if config["optimizer"] not in ("Adam", "AdamW"):
        raise ValueError("optimizer must be Adam or AdamW.")
    return config
