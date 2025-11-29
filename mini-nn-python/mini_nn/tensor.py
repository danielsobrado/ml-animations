"""Tensor operations for neural networks."""

import numpy as np
from typing import Tuple, Optional, Callable


class Tensor:
    """A simple 2D tensor wrapper around numpy arrays."""

    def __init__(self, data: np.ndarray):
        """Initialize tensor from numpy array."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.data = data.astype(np.float64)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (rows, cols) shape."""
        return self.data.shape

    @property
    def rows(self) -> int:
        """Return number of rows."""
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        """Return number of columns."""
        return self.data.shape[1]

    # Static factory methods
    @staticmethod
    def zeros(rows: int, cols: int) -> "Tensor":
        """Create a tensor filled with zeros."""
        return Tensor(np.zeros((rows, cols)))

    @staticmethod
    def ones(rows: int, cols: int) -> "Tensor":
        """Create a tensor filled with ones."""
        return Tensor(np.ones((rows, cols)))

    @staticmethod
    def xavier(rows: int, cols: int, rng: Optional[np.random.Generator] = None) -> "Tensor":
        """Create a tensor with Xavier initialization."""
        if rng is None:
            rng = np.random.default_rng()
        scale = np.sqrt(2.0 / (rows + cols))
        return Tensor(rng.normal(0, scale, (rows, cols)))

    @staticmethod
    def he(rows: int, cols: int, rng: Optional[np.random.Generator] = None) -> "Tensor":
        """Create a tensor with He initialization."""
        if rng is None:
            rng = np.random.default_rng()
        scale = np.sqrt(2.0 / rows)
        return Tensor(rng.normal(0, scale, (rows, cols)))

    @staticmethod
    def from_array(arr: np.ndarray) -> "Tensor":
        """Create tensor from numpy array."""
        return Tensor(arr)

    # Element access
    def __getitem__(self, key):
        """Get element or slice."""
        result = self.data[key]
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                result = result.reshape(1, -1)
            return Tensor(result)
        return result

    def __setitem__(self, key, value):
        """Set element or slice."""
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value

    def get(self, i: int, j: int) -> float:
        """Get element at (i, j)."""
        return self.data[i, j]

    def set(self, i: int, j: int, value: float):
        """Set element at (i, j)."""
        self.data[i, j] = value

    def get_row(self, i: int) -> np.ndarray:
        """Get row i as 1D array."""
        return self.data[i].copy()

    # Matrix operations
    def matmul(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        return Tensor(self.data @ other.data)

    def transpose(self) -> "Tensor":
        """Transpose the tensor."""
        return Tensor(self.data.T)

    @property
    def T(self) -> "Tensor":
        """Transpose property."""
        return self.transpose()

    def add(self, other: "Tensor") -> "Tensor":
        """Element-wise addition with broadcasting."""
        return Tensor(self.data + other.data)

    def sub(self, other: "Tensor") -> "Tensor":
        """Element-wise subtraction."""
        return Tensor(self.data - other.data)

    def mul(self, other: "Tensor") -> "Tensor":
        """Element-wise multiplication."""
        return Tensor(self.data * other.data)

    def div(self, other: "Tensor") -> "Tensor":
        """Element-wise division."""
        return Tensor(self.data / other.data)

    def scale(self, scalar: float) -> "Tensor":
        """Scale by scalar."""
        return Tensor(self.data * scalar)

    def apply(self, func: Callable[[np.ndarray], np.ndarray]) -> "Tensor":
        """Apply function element-wise."""
        return Tensor(func(self.data))

    def sum(self) -> float:
        """Sum all elements."""
        return float(np.sum(self.data))

    def sum_axis(self, axis: int) -> "Tensor":
        """Sum along axis."""
        return Tensor(np.sum(self.data, axis=axis, keepdims=True))

    def mean(self) -> float:
        """Mean of all elements."""
        return float(np.mean(self.data))

    def slice_rows(self, start: int, end: int) -> "Tensor":
        """Slice rows from start to end (exclusive)."""
        return Tensor(self.data[start:end].copy())

    def copy(self) -> "Tensor":
        """Create a copy of this tensor."""
        return Tensor(self.data.copy())

    # In-place operations
    def add_inplace(self, other: "Tensor"):
        """In-place addition."""
        self.data += other.data

    def scale_inplace(self, scalar: float):
        """In-place scaling."""
        self.data *= scalar

    def fill(self, value: float):
        """Fill with value."""
        self.data.fill(value)

    # Numpy interop
    def numpy(self) -> np.ndarray:
        """Return underlying numpy array."""
        return self.data

    def __repr__(self) -> str:
        return f"Tensor({self.rows}, {self.cols})\n{self.data}"

    def __add__(self, other: "Tensor") -> "Tensor":
        return self.add(other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self.sub(other)

    def __mul__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return self.mul(other)
        return self.scale(other)

    def __rmul__(self, other) -> "Tensor":
        return self.scale(other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def __neg__(self) -> "Tensor":
        return self.scale(-1)
