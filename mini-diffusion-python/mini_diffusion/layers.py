"""
Neural network layers for diffusion models.

Provides Linear, Conv2d, and GroupNorm layers.
"""

import numpy as np
from typing import Optional

from .tensor import Tensor


class Linear:
    """Fully connected layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            rng: Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Kaiming initialization
        std = np.sqrt(2.0 / in_features)
        self.weight = Tensor(rng.standard_normal((in_features, out_features)) * std)
        self.bias = Tensor.zeros(out_features)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, in_features] or [batch, ..., in_features]
            
        Returns:
            Output tensor [batch, out_features] or [batch, ..., out_features]
        """
        out = x @ self.weight
        return out + self.bias
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameter_count(self) -> int:
        """Get number of parameters."""
        return self.in_features * self.out_features + self.out_features


class Conv2d:
    """2D Convolution layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize Conv2d layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Zero-padding added to both sides
            rng: Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Kaiming initialization
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.weight = Tensor(
            rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)) * std
        )
        self.bias = Tensor.zeros(out_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using im2col for efficient convolution.
        
        Args:
            x: Input tensor [batch, in_channels, height, width]
            
        Returns:
            Output tensor [batch, out_channels, out_height, out_width]
        """
        batch, in_ch, in_h, in_w = x.shape
        
        # Compute output dimensions
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Pad input if necessary
        if self.padding > 0:
            padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            padded = x.data
        
        # Naive convolution (efficient enough for educational purposes)
        output = np.zeros((batch, self.out_channels, out_h, out_w))
        
        for b in range(batch):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        
                        # Extract patch and compute convolution
                        patch = padded[
                            b, :,
                            h_start:h_start + self.kernel_size,
                            w_start:w_start + self.kernel_size
                        ]
                        output[b, oc, oh, ow] = (
                            np.sum(patch * self.weight.data[oc]) + self.bias.data[oc]
                        )
        
        return Tensor(output)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameter_count(self) -> int:
        """Get number of parameters."""
        return (
            self.out_channels * self.in_channels * self.kernel_size * self.kernel_size
            + self.out_channels
        )


class GroupNorm:
    """Group Normalization layer."""
    
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5
    ):
        """
        Initialize GroupNorm layer.
        
        Args:
            num_groups: Number of groups to divide channels into
            num_channels: Number of channels
            eps: Small constant for numerical stability
        """
        self.num_groups = min(num_groups, num_channels)
        self.num_channels = num_channels
        self.eps = eps
        
        self.gamma = Tensor.ones(num_channels)
        self.beta = Tensor.zeros(num_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Normalized tensor
        """
        batch, channels, height, width = x.shape
        channels_per_group = channels // self.num_groups
        
        # Reshape for group normalization
        x_reshaped = x.data.reshape(batch, self.num_groups, channels_per_group, height, width)
        
        # Compute mean and variance per group
        mean = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
        var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
        
        # Normalize
        x_norm = (x_reshaped - mean) / np.sqrt(var + self.eps)
        
        # Reshape back
        x_norm = x_norm.reshape(batch, channels, height, width)
        
        # Apply affine transform
        gamma = self.gamma.data.reshape(1, channels, 1, 1)
        beta = self.beta.data.reshape(1, channels, 1, 1)
        
        return Tensor(gamma * x_norm + beta)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameter_count(self) -> int:
        """Get number of parameters."""
        return 2 * self.num_channels
