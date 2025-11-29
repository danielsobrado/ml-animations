"""
Tensor operations for diffusion models.

Provides a simple wrapper around NumPy arrays with convenience methods
for 4D image tensors [batch, channels, height, width].
"""

import numpy as np
from typing import Union, Tuple, Optional


class Tensor:
    """4D tensor wrapper with diffusion-specific operations."""
    
    def __init__(self, data: np.ndarray):
        """Create tensor from numpy array."""
        self.data = data.astype(np.float64)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self.data.ndim
    
    def dim(self, i: int) -> int:
        """Get size of dimension i."""
        return self.data.shape[i]
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self.data.size
    
    # Factory methods
    @staticmethod
    def zeros(*shape: int) -> 'Tensor':
        """Create tensor filled with zeros."""
        return Tensor(np.zeros(shape))
    
    @staticmethod
    def ones(*shape: int) -> 'Tensor':
        """Create tensor filled with ones."""
        return Tensor(np.ones(shape))
    
    @staticmethod
    def randn(*shape: int, rng: Optional[np.random.Generator] = None) -> 'Tensor':
        """Create tensor with random normal values."""
        if rng is None:
            rng = np.random.default_rng()
        return Tensor(rng.standard_normal(shape))
    
    @staticmethod
    def from_shape(shape: Tuple[int, ...], value: float = 0.0) -> 'Tensor':
        """Create tensor with given shape filled with value."""
        return Tensor(np.full(shape, value))
    
    @staticmethod
    def xavier(fan_in: int, fan_out: int, rng: Optional[np.random.Generator] = None) -> 'Tensor':
        """Xavier/Glorot initialization."""
        if rng is None:
            rng = np.random.default_rng()
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return Tensor(rng.standard_normal((fan_in, fan_out)) * std)
    
    @staticmethod
    def kaiming(fan_in: int, fan_out: int, rng: Optional[np.random.Generator] = None) -> 'Tensor':
        """Kaiming/He initialization."""
        if rng is None:
            rng = np.random.default_rng()
        std = np.sqrt(2.0 / fan_in)
        return Tensor(rng.standard_normal((fan_in, fan_out)) * std)
    
    # Arithmetic operations
    def add(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise addition."""
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return Tensor(self.data + other)
    
    def sub(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise subtraction."""
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        return Tensor(self.data - other)
    
    def mul(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise multiplication."""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)
    
    def div(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise division."""
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)
    
    def neg(self) -> 'Tensor':
        """Negate tensor."""
        return Tensor(-self.data)
    
    # Operator overloads
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        return self.add(other)
    
    def __radd__(self, other: float) -> 'Tensor':
        return self.add(other)
    
    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        return self.sub(other)
    
    def __rsub__(self, other: float) -> 'Tensor':
        return Tensor(other - self.data)
    
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        return self.mul(other)
    
    def __rmul__(self, other: float) -> 'Tensor':
        return self.mul(other)
    
    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        return self.div(other)
    
    def __neg__(self) -> 'Tensor':
        return self.neg()
    
    # Matrix operations
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        return Tensor(self.data @ other.data)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return self.matmul(other)
    
    def transpose(self, *axes: int) -> 'Tensor':
        """Transpose tensor."""
        if len(axes) == 0:
            return Tensor(self.data.T)
        return Tensor(np.transpose(self.data, axes))
    
    @property
    def T(self) -> 'Tensor':
        """Shorthand for transpose."""
        return self.transpose()
    
    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor."""
        return Tensor(self.data.reshape(shape))
    
    def flatten(self) -> 'Tensor':
        """Flatten to 1D."""
        return Tensor(self.data.flatten())
    
    # Activation functions
    def relu(self) -> 'Tensor':
        """ReLU activation."""
        return Tensor(np.maximum(self.data, 0))
    
    def sigmoid(self) -> 'Tensor':
        """Sigmoid activation."""
        return Tensor(1.0 / (1.0 + np.exp(-self.data)))
    
    def tanh(self) -> 'Tensor':
        """Tanh activation."""
        return Tensor(np.tanh(self.data))
    
    def gelu(self) -> 'Tensor':
        """GELU activation (approximation)."""
        return Tensor(0.5 * self.data * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data ** 3)
        )))
    
    def silu(self) -> 'Tensor':
        """SiLU/Swish activation."""
        return Tensor(self.data / (1.0 + np.exp(-self.data)))
    
    def softmax(self, axis: int = -1) -> 'Tensor':
        """Softmax along axis."""
        exp_x = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        return Tensor(exp_x / np.sum(exp_x, axis=axis, keepdims=True))
    
    # Statistics
    def mean(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Compute mean."""
        result = np.mean(self.data, axis=axis)
        return float(result) if axis is None else Tensor(result)
    
    def std(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Compute standard deviation."""
        result = np.std(self.data, axis=axis)
        return float(result) if axis is None else Tensor(result)
    
    def var(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Compute variance."""
        result = np.var(self.data, axis=axis)
        return float(result) if axis is None else Tensor(result)
    
    def sum(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Compute sum."""
        result = np.sum(self.data, axis=axis)
        return float(result) if axis is None else Tensor(result)
    
    def max(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Compute max."""
        result = np.max(self.data, axis=axis)
        return float(result) if axis is None else Tensor(result)
    
    def min(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Compute min."""
        result = np.min(self.data, axis=axis)
        return float(result) if axis is None else Tensor(result)
    
    # Indexing
    def __getitem__(self, idx):
        """Get item or slice."""
        result = self.data[idx]
        if isinstance(result, np.ndarray):
            return Tensor(result)
        return float(result)
    
    def __setitem__(self, idx, value):
        """Set item or slice."""
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value
    
    # Utility
    def clone(self) -> 'Tensor':
        """Create a copy of the tensor."""
        return Tensor(self.data.copy())
    
    def numpy(self) -> np.ndarray:
        """Get underlying numpy array."""
        return self.data
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape})"
    
    def __str__(self) -> str:
        return str(self.data)
