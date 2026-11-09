"""FNO with four layers in one dimension and two in two dimensions."""
from ._network import FourierNeuralOperator


def build(dimension, config, normalization=None):
    layers = {1: 4, 2: 2}[dimension]
    return FourierNeuralOperator(dimension, config["N"], config["width"], config["modes"], layers)
