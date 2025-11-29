"""
Sampler for diffusion models.

Implements DDPM and DDIM sampling algorithms.
"""

import numpy as np
from enum import Enum
from typing import Callable, Optional, Tuple

from .tensor import Tensor
from .scheduler import NoiseScheduler
from .unet import UNet


class SamplerType(Enum):
    """Type of sampler."""
    DDPM = "ddpm"
    DDIM = "ddim"


class Sampler:
    """Sampler for generating images from noise."""
    
    def __init__(
        self,
        scheduler: NoiseScheduler,
        num_inference_steps: int = 50,
        sampler_type: SamplerType = SamplerType.DDPM
    ):
        """
        Initialize sampler.
        
        Args:
            scheduler: Noise scheduler
            num_inference_steps: Number of denoising steps
            sampler_type: Type of sampling (DDPM or DDIM)
        """
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.sampler_type = sampler_type
        
        # Get timesteps
        self.timesteps = scheduler.get_timesteps(num_inference_steps)
    
    def sample(
        self,
        model: UNet,
        shape: Tuple[int, ...],
        rng: Optional[np.random.Generator] = None,
        callback: Optional[Callable[[int, int, int, Tensor], None]] = None
    ) -> Tensor:
        """
        Sample from pure noise using the model.
        
        Args:
            model: U-Net model for noise prediction
            shape: Shape of output [batch, channels, height, width]
            rng: Random number generator
            callback: Optional progress callback (step, total, timestep, current)
            
        Returns:
            Generated sample
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Start with pure noise
        x = Tensor.randn(*shape, rng=rng)
        
        # Iterate through timesteps
        for i, t in enumerate(self.timesteps):
            t = int(t)
            prev_t = int(self.timesteps[i + 1]) if i < len(self.timesteps) - 1 else -1
            
            # Predict noise
            noise_pred = model(x, t)
            
            # Step
            if self.sampler_type == SamplerType.DDPM:
                x = self.scheduler.step(x, noise_pred, t, rng)
            else:  # DDIM
                x = self.scheduler.step_ddim(x, noise_pred, t, prev_t)
            
            if callback:
                callback(i + 1, len(self.timesteps), t, x)
        
        return x
    
    @staticmethod
    def ddpm(
        scheduler: NoiseScheduler,
        num_inference_steps: int = 50
    ) -> 'Sampler':
        """Create DDPM sampler."""
        return Sampler(scheduler, num_inference_steps, SamplerType.DDPM)
    
    @staticmethod
    def ddim(
        scheduler: NoiseScheduler,
        num_inference_steps: int = 50
    ) -> 'Sampler':
        """Create DDIM sampler."""
        return Sampler(scheduler, num_inference_steps, SamplerType.DDIM)
