"""Neural network implementation."""

import numpy as np
from typing import List, Optional
from .tensor import Tensor
from .layer import DenseLayer, InitType
from .activation import ActivationType


class Network:
    """Neural network with builder pattern."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        """Initialize network."""
        self.layers: List[DenseLayer] = []
        self.rng = rng if rng is not None else np.random.default_rng()

    def add_dense(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationType = ActivationType.LINEAR,
        init_type: InitType = InitType.XAVIER,
    ) -> "Network":
        """Add a dense layer."""
        layer = DenseLayer(input_size, output_size, activation, init_type, self.rng)
        self.layers.append(layer)
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers."""
        current = x
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def predict(self, x: Tensor) -> Tensor:
        """Make predictions (alias for forward)."""
        return self.forward(x)

    def backward(self, grad_output: Tensor):
        """Backward pass through all layers."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def get_layers(self) -> List[DenseLayer]:
        """Get list of dense layers."""
        return self.layers

    def parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(layer.parameter_count() for layer in self.layers)

    def summary(self):
        """Print network summary."""
        print("Network Architecture:")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i + 1}: {layer}")
        print(f"  Total parameters: {self.parameter_count()}")
