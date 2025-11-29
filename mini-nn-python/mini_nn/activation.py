"""Activation functions for neural networks."""

import numpy as np
from enum import Enum, auto
from .tensor import Tensor


class ActivationType(Enum):
    """Types of activation functions."""
    LINEAR = auto()
    RELU = auto()
    LEAKY_RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    SOFTMAX = auto()


class Activation:
    """Activation function implementations."""

    @staticmethod
    def apply(x: Tensor, activation_type: ActivationType) -> Tensor:
        """Apply activation function to tensor."""
        if activation_type == ActivationType.LINEAR:
            return x.copy()
        elif activation_type == ActivationType.RELU:
            return x.apply(lambda a: np.maximum(0, a))
        elif activation_type == ActivationType.LEAKY_RELU:
            return x.apply(lambda a: np.where(a >= 0, a, 0.01 * a))
        elif activation_type == ActivationType.SIGMOID:
            return x.apply(lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500))))
        elif activation_type == ActivationType.TANH:
            return x.apply(np.tanh)
        elif activation_type == ActivationType.SOFTMAX:
            return Activation._softmax(x)
        else:
            return x.copy()

    @staticmethod
    def derivative(x: Tensor, output: Tensor, activation_type: ActivationType) -> Tensor:
        """Compute derivative of activation function."""
        if activation_type == ActivationType.LINEAR:
            return Tensor.ones(x.rows, x.cols)
        elif activation_type == ActivationType.RELU:
            return x.apply(lambda a: (a > 0).astype(float))
        elif activation_type == ActivationType.LEAKY_RELU:
            return x.apply(lambda a: np.where(a >= 0, 1.0, 0.01))
        elif activation_type == ActivationType.SIGMOID:
            return output.apply(lambda a: a * (1.0 - a))
        elif activation_type == ActivationType.TANH:
            return output.apply(lambda a: 1.0 - a * a)
        elif activation_type == ActivationType.SOFTMAX:
            # Softmax derivative is complex; handled in loss gradient
            return Tensor.ones(x.rows, x.cols)
        else:
            return Tensor.ones(x.rows, x.cols)

    @staticmethod
    def _softmax(x: Tensor) -> Tensor:
        """Apply softmax activation."""
        data = x.numpy()
        # Subtract max for numerical stability
        shifted = data - np.max(data, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return Tensor(exp_vals / np.sum(exp_vals, axis=1, keepdims=True))
