"""
U-Net architecture for diffusion models.

Implements encoder-decoder with skip connections and time embeddings.
"""

import numpy as np
from typing import Optional

from .tensor import Tensor
from .layers import Linear, Conv2d, GroupNorm


def silu(x: Tensor) -> Tensor:
    """SiLU/Swish activation function."""
    return x.silu()


def timestep_embedding(timestep: int, dim: int, batch: int) -> Tensor:
    """
    Create sinusoidal timestep embedding.
    
    Args:
        timestep: Current timestep
        dim: Embedding dimension
        batch: Batch size
        
    Returns:
        Embedding tensor [batch, dim]
    """
    half_dim = dim // 2
    log_max = np.log(10000.0)
    
    emb = np.zeros((batch, dim))
    for i in range(half_dim):
        freq = np.exp(-log_max * i / half_dim)
        angle = timestep * freq
        emb[:, i] = np.sin(angle)
        emb[:, i + half_dim] = np.cos(angle)
    
    return Tensor(emb)


class ResBlock:
    """Residual block with time embedding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize ResBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_embed_dim: Dimension of time embedding
            rng: Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = GroupNorm(32, in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1, rng)
        self.norm2 = GroupNorm(32, out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1, rng)
        self.time_proj = Linear(time_embed_dim, out_channels, rng)
        
        # Shortcut for channel mismatch
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, 1, 1, 0, rng)
        else:
            self.shortcut = None
    
    def forward(self, x: Tensor, time_embed: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, in_channels, height, width]
            time_embed: Time embedding [batch, time_embed_dim]
            
        Returns:
            Output tensor [batch, out_channels, height, width]
        """
        residual = self.shortcut(x) if self.shortcut else x
        
        # First conv block
        h = self.norm1(x)
        h = silu(h)
        h = self.conv1(h)
        
        # Add time embedding (project and broadcast)
        time_proj = self.time_proj(time_embed)  # [batch, out_channels]
        time_proj = Tensor(time_proj.data.reshape(time_proj.shape[0], time_proj.shape[1], 1, 1))
        h = h + time_proj
        
        # Second conv block
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)
        
        return h + residual
    
    def __call__(self, x: Tensor, time_embed: Tensor) -> Tensor:
        return self.forward(x, time_embed)
    
    def parameter_count(self) -> int:
        """Get number of parameters."""
        count = self.norm1.parameter_count() + self.conv1.parameter_count()
        count += self.norm2.parameter_count() + self.conv2.parameter_count()
        count += self.time_proj.parameter_count()
        if self.shortcut:
            count += self.shortcut.parameter_count()
        return count


class Downsample:
    """Downsample block using strided convolution."""
    
    def __init__(
        self,
        channels: int,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize Downsample.
        
        Args:
            channels: Number of channels
            rng: Random number generator
        """
        self.conv = Conv2d(channels, channels, 3, 2, 1, rng)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass - halves spatial dimensions."""
        return self.conv(x)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameter_count(self) -> int:
        """Get number of parameters."""
        return self.conv.parameter_count()


class Upsample:
    """Upsample block using nearest-neighbor + convolution."""
    
    def __init__(
        self,
        channels: int,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize Upsample.
        
        Args:
            channels: Number of channels
            rng: Random number generator
        """
        self.conv = Conv2d(channels, channels, 3, 1, 1, rng)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass - doubles spatial dimensions."""
        batch, channels, height, width = x.shape
        
        # Nearest neighbor 2x upsampling
        upsampled = np.zeros((batch, channels, height * 2, width * 2))
        upsampled[:, :, 0::2, 0::2] = x.data
        upsampled[:, :, 0::2, 1::2] = x.data
        upsampled[:, :, 1::2, 0::2] = x.data
        upsampled[:, :, 1::2, 1::2] = x.data
        
        return self.conv(Tensor(upsampled))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameter_count(self) -> int:
        """Get number of parameters."""
        return self.conv.parameter_count()


class UNet:
    """U-Net architecture for noise prediction in diffusion models."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            model_channels: Base channel count
            rng: Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.time_embed_dim = model_channels * 4
        
        # Input projection
        self.input_conv = Conv2d(in_channels, model_channels, 3, 1, 1, rng)
        
        # Time embedding MLP
        self.time_embed1 = Linear(model_channels, self.time_embed_dim, rng)
        self.time_embed2 = Linear(self.time_embed_dim, self.time_embed_dim, rng)
        
        # Encoder
        self.encoder_res1 = ResBlock(model_channels, model_channels, self.time_embed_dim, rng)
        self.down1 = Downsample(model_channels, rng)
        self.encoder_res2 = ResBlock(model_channels, model_channels * 2, self.time_embed_dim, rng)
        self.down2 = Downsample(model_channels * 2, rng)
        
        # Middle
        self.mid_res1 = ResBlock(model_channels * 2, model_channels * 2, self.time_embed_dim, rng)
        self.mid_res2 = ResBlock(model_channels * 2, model_channels * 2, self.time_embed_dim, rng)
        
        # Decoder (with skip connections)
        self.up1 = Upsample(model_channels * 2, rng)
        self.decoder_res1 = ResBlock(model_channels * 4, model_channels, self.time_embed_dim, rng)
        self.up2 = Upsample(model_channels, rng)
        self.decoder_res2 = ResBlock(model_channels * 2, model_channels, self.time_embed_dim, rng)
        
        # Output
        self.out_norm = GroupNorm(32, model_channels)
        self.out_conv = Conv2d(model_channels, out_channels, 3, 1, 1, rng)
    
    def forward(self, x: Tensor, timestep: int) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, in_channels, height, width]
            timestep: Current timestep
            
        Returns:
            Predicted noise [batch, out_channels, height, width]
        """
        batch = x.dim(0)
        
        # Time embedding
        t = timestep_embedding(timestep, self.model_channels, batch)
        t = self.time_embed1(t)
        t = silu(t)
        t = self.time_embed2(t)
        
        # Input projection
        h = self.input_conv(x)
        
        # Encoder with skip connections
        h1 = self.encoder_res1(h, t)      # [batch, model_channels, H, W]
        h2 = self.down1(h1)                # [batch, model_channels, H/2, W/2]
        h3 = self.encoder_res2(h2, t)     # [batch, model_channels*2, H/2, W/2]
        h4 = self.down2(h3)                # [batch, model_channels*2, H/4, W/4]
        
        # Middle
        mid = self.mid_res1(h4, t)
        mid = self.mid_res2(mid, t)
        
        # Decoder with skip connections
        up = self.up1(mid)                 # [batch, model_channels*2, H/2, W/2]
        up = self._concat(up, h3)          # [batch, model_channels*4, H/2, W/2]
        up = self.decoder_res1(up, t)      # [batch, model_channels, H/2, W/2]
        
        up = self.up2(up)                  # [batch, model_channels, H, W]
        up = self._concat(up, h1)          # [batch, model_channels*2, H, W]
        up = self.decoder_res2(up, t)      # [batch, model_channels, H, W]
        
        # Output
        out = self.out_norm(up)
        out = silu(out)
        out = self.out_conv(out)
        
        return out
    
    def _concat(self, a: Tensor, b: Tensor) -> Tensor:
        """Concatenate tensors along channel dimension."""
        return Tensor(np.concatenate([a.data, b.data], axis=1))
    
    def __call__(self, x: Tensor, timestep: int) -> Tensor:
        return self.forward(x, timestep)
    
    def parameter_count(self) -> int:
        """Get total number of parameters."""
        count = self.input_conv.parameter_count()
        count += self.time_embed1.parameter_count() + self.time_embed2.parameter_count()
        count += self.encoder_res1.parameter_count() + self.down1.parameter_count()
        count += self.encoder_res2.parameter_count() + self.down2.parameter_count()
        count += self.mid_res1.parameter_count() + self.mid_res2.parameter_count()
        count += self.up1.parameter_count() + self.decoder_res1.parameter_count()
        count += self.up2.parameter_count() + self.decoder_res2.parameter_count()
        count += self.out_norm.parameter_count() + self.out_conv.parameter_count()
        return count
