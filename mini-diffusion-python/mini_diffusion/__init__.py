"""
Mini-Diffusion: A minimal Python implementation of diffusion models.

This package provides educational implementations of:
- Tensor operations for 4D image data
- Noise schedulers (linear, cosine, quadratic)
- Neural network layers (Linear, Conv2d, GroupNorm)
- U-Net architecture with residual blocks
- DDPM and DDIM sampling
"""

from .tensor import Tensor
from .scheduler import NoiseScheduler, ScheduleType
from .layers import Linear, Conv2d, GroupNorm
from .unet import ResBlock, Downsample, Upsample, UNet
from .sampler import Sampler, SamplerType

__version__ = "0.1.0"
__all__ = [
    "Tensor",
    "NoiseScheduler",
    "ScheduleType",
    "Linear",
    "Conv2d", 
    "GroupNorm",
    "ResBlock",
    "Downsample",
    "Upsample",
    "UNet",
    "Sampler",
    "SamplerType",
]
