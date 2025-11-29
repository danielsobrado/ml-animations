"""Loss functions for neural networks."""

import numpy as np
from enum import Enum, auto
from .tensor import Tensor


class LossType(Enum):
    """Types of loss functions."""
    MSE = auto()
    BINARY_CROSS_ENTROPY = auto()
    CROSS_ENTROPY = auto()


class Loss:
    """Loss function implementations."""

    EPSILON = 1e-7

    @staticmethod
    def compute(predictions: Tensor, targets: Tensor, loss_type: LossType) -> float:
        """Compute loss value."""
        pred = predictions.numpy()
        targ = targets.numpy()
        n = pred.shape[0]

        if loss_type == LossType.MSE:
            return float(np.sum((pred - targ) ** 2) / (2 * n))

        elif loss_type == LossType.BINARY_CROSS_ENTROPY:
            p = np.clip(pred, Loss.EPSILON, 1 - Loss.EPSILON)
            return float(-np.sum(targ * np.log(p) + (1 - targ) * np.log(1 - p)) / n)

        elif loss_type == LossType.CROSS_ENTROPY:
            p = np.clip(pred, Loss.EPSILON, 1 - Loss.EPSILON)
            return float(-np.sum(targ * np.log(p)) / n)

        return 0.0

    @staticmethod
    def gradient(predictions: Tensor, targets: Tensor, loss_type: LossType) -> Tensor:
        """Compute gradient of loss with respect to predictions."""
        pred = predictions.numpy()
        targ = targets.numpy()
        n = pred.shape[0]

        if loss_type == LossType.MSE:
            return Tensor((pred - targ) / n)

        elif loss_type == LossType.BINARY_CROSS_ENTROPY:
            p = np.clip(pred, Loss.EPSILON, 1 - Loss.EPSILON)
            return Tensor((-targ / p + (1 - targ) / (1 - p)) / n)

        elif loss_type == LossType.CROSS_ENTROPY:
            # For softmax + cross-entropy, gradient simplifies to (pred - target)
            return Tensor((pred - targ) / n)

        return Tensor.zeros(pred.shape[0], pred.shape[1])
