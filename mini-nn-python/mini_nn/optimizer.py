"""Optimizers for neural network training."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict
from .tensor import Tensor
from .layer import DenseLayer


class Optimizer(ABC):
    """Base class for optimizers."""

    @abstractmethod
    def step(self, layers: List[DenseLayer]):
        """Update layer parameters based on gradients."""
        pass


class SGDOptimizer(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate: float = 0.01):
        """Initialize SGD optimizer."""
        self.learning_rate = learning_rate

    def step(self, layers: List[DenseLayer]):
        """Update parameters using SGD."""
        for layer in layers:
            weights, bias = layer.get_parameters()
            weight_grad, bias_grad = layer.get_gradients()
            
            # W = W - lr * dW
            new_weights = weights.sub(weight_grad.scale(self.learning_rate))
            # b = b - lr * db
            new_bias = bias.sub(bias_grad.scale(self.learning_rate))
            
            layer.set_parameters(new_weights, new_bias)


class MomentumOptimizer(Optimizer):
    """SGD with Momentum optimizer."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """Initialize Momentum optimizer."""
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w: Dict[int, Tensor] = {}
        self.velocity_b: Dict[int, Tensor] = {}

    def step(self, layers: List[DenseLayer]):
        """Update parameters using Momentum."""
        for i, layer in enumerate(layers):
            weights, bias = layer.get_parameters()
            weight_grad, bias_grad = layer.get_gradients()
            
            # Initialize velocity if needed
            if i not in self.velocity_w:
                self.velocity_w[i] = Tensor.zeros(*weights.shape)
                self.velocity_b[i] = Tensor.zeros(*bias.shape)
            
            # Update velocity
            vw = self.velocity_w[i].scale(self.momentum).sub(
                weight_grad.scale(self.learning_rate)
            )
            vb = self.velocity_b[i].scale(self.momentum).sub(
                bias_grad.scale(self.learning_rate)
            )
            
            self.velocity_w[i] = vw
            self.velocity_b[i] = vb
            
            # Update parameters
            new_weights = weights.add(vw)
            new_bias = bias.add(vb)
            
            layer.set_parameters(new_weights, new_bias)


class AdamOptimizer(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """Initialize Adam optimizer."""
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
        # First and second moment estimates
        self.m_w: Dict[int, Tensor] = {}
        self.v_w: Dict[int, Tensor] = {}
        self.m_b: Dict[int, Tensor] = {}
        self.v_b: Dict[int, Tensor] = {}

    def step(self, layers: List[DenseLayer]):
        """Update parameters using Adam."""
        self.t += 1
        
        for i, layer in enumerate(layers):
            weights, bias = layer.get_parameters()
            weight_grad, bias_grad = layer.get_gradients()
            
            # Initialize moments if needed
            if i not in self.m_w:
                self.m_w[i] = Tensor.zeros(*weights.shape)
                self.v_w[i] = Tensor.zeros(*weights.shape)
                self.m_b[i] = Tensor.zeros(*bias.shape)
                self.v_b[i] = Tensor.zeros(*bias.shape)
            
            # Update weights
            new_weights = self._update_param(
                weights, weight_grad, self.m_w[i], self.v_w[i]
            )
            
            # Update bias
            new_bias = self._update_param(
                bias, bias_grad, self.m_b[i], self.v_b[i]
            )
            
            layer.set_parameters(new_weights, new_bias)

    def _update_param(
        self, param: Tensor, grad: Tensor, m: Tensor, v: Tensor
    ) -> Tensor:
        """Update a single parameter using Adam."""
        g = grad.numpy()
        m_data = m.numpy()
        v_data = v.numpy()
        
        # Update biased first moment estimate
        m_data[:] = self.beta1 * m_data + (1 - self.beta1) * g
        
        # Update biased second raw moment estimate
        v_data[:] = self.beta2 * v_data + (1 - self.beta2) * g * g
        
        # Compute bias-corrected estimates
        m_hat = m_data / (1 - self.beta1 ** self.t)
        v_hat = v_data / (1 - self.beta2 ** self.t)
        
        # Update parameter
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return Tensor(param.numpy() - update)
