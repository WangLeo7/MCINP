"""Names and model builders for the methods in the manuscript."""
from . import deeponet, fno, global_q, mcinp, mcp, pod_linear

METHODS = {
    "mcinp": ("MCINP", mcinp),
    "global_q": ("Global-Q", global_q),
    "mcp": ("MCP", mcp),
    "fno": ("FNO", fno),
    "deeponet": ("DeepONet", deeponet),
    "pod_linear": ("POD-linear", pod_linear),
}


def build(config, normalization=None):
    module = METHODS[config["method"]][1]
    return module.build(config["dimension"], config, normalization=normalization)
