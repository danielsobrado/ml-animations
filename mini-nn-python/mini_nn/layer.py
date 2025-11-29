"""Layer implementations for neural networks."""

import numpy as np
from enum import Enum, auto
from typing import Optional
from .tensor import Tensor
from .activation import Activation, ActivationType


class InitType(Enum):
    """Weight initialization types."""
    XAVIER = auto()
    HE = auto()


class DenseLayer:
    """Dense (fully connected) layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationType = ActivationType.LINEAR,
        init_type: InitType = InitType.XAVIER,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize dense layer."""
        self.input_size = input_size
        self.output_size = output_size
        self.activation_type = activation
        
        if rng is None:
            rng = np.random.default_rng()
        
        # Initialize weights
        if init_type == InitType.HE:
            self.weights = Tensor.he(input_size, output_size, rng)
        else:
            self.weights = Tensor.xavier(input_size, output_size, rng)
        
        # Initialize bias to zeros
        self.bias = Tensor.zeros(1, output_size)
        
        # Cached values for backpropagation
        self.input: Optional[Tensor] = None
        self.pre_activation: Optional[Tensor] = None
        self.output: Optional[Tensor] = None
        
        # Gradients
        self.weight_grad = Tensor.zeros(input_size, output_size)
        self.bias_grad = Tensor.zeros(1, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        self.input = x
        # z = x @ W + b
        self.pre_activation = x.matmul(self.weights).add(self.bias)
        # a = activation(z)
        self.output = Activation.apply(self.pre_activation, self.activation_type)
        return self.output

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass."""
        # Compute activation gradient
        activation_grad = Activation.derivative(
            self.pre_activation, self.output, self.activation_type
        )
        delta = grad_output.mul(activation_grad)
        
        # Compute weight gradient: dW = x^T @ delta
        self.weight_grad = self.input.transpose().matmul(delta)
        
        # Compute bias gradient: db = sum(delta, axis=0)
        self.bias_grad = delta.sum_axis(0)
        
        # Compute gradient for previous layer: dx = delta @ W^T
        return delta.matmul(self.weights.transpose())

    def get_parameters(self) -> tuple:
        """Get (weights, bias)."""
        return self.weights, self.bias

    def get_gradients(self) -> tuple:
        """Get (weight_grad, bias_grad)."""
        return self.weight_grad, self.bias_grad

    def set_parameters(self, weights: Tensor, bias: Tensor):
        """Set weights and bias."""
        self.weights = weights
        self.bias = bias

    def parameter_count(self) -> int:
        """Return total number of parameters."""
        return self.input_size * self.output_size + self.output_size

    def __repr__(self) -> str:
        return f"Dense({self.input_size} -> {self.output_size}) + {self.activation_type.name}"
