"""
Noise scheduler for diffusion models.

Implements linear, cosine, and quadratic beta schedules.
"""

import numpy as np
from enum import Enum
from typing import Optional

from .tensor import Tensor


class ScheduleType(Enum):
    """Type of noise schedule."""
    LINEAR = "linear"
    COSINE = "cosine"
    QUADRATIC = "quadratic"


class NoiseScheduler:
    """
    Noise scheduler for diffusion models.
    
    Implements the forward diffusion process q(x_t | x_0) and
    reverse process step predictions.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: ScheduleType = ScheduleType.LINEAR,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """
        Initialize noise scheduler.
        
        Args:
            num_timesteps: Total number of diffusion steps
            schedule_type: Type of beta schedule
            beta_start: Starting beta value
            beta_end: Ending beta value
        """
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        # Generate beta schedule
        if schedule_type == ScheduleType.LINEAR:
            self.betas = np.linspace(beta_start, beta_end, num_timesteps)
        
        elif schedule_type == ScheduleType.COSINE:
            s = 0.008
            steps = np.arange(num_timesteps + 1)
            alphas_cumprod = np.cos((steps / num_timesteps + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = np.clip(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], 0, 0.999)
        
        elif schedule_type == ScheduleType.QUADRATIC:
            self.betas = np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), num_timesteps) ** 2
        
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Compute alphas and cumulative products
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        
        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance for q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.maximum(self.posterior_variance, 1e-20)
        )
        
        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    @classmethod
    def linear(cls, num_timesteps: int = 1000) -> 'NoiseScheduler':
        """Create scheduler with linear beta schedule."""
        return cls(num_timesteps, ScheduleType.LINEAR)
    
    @classmethod
    def cosine(cls, num_timesteps: int = 1000) -> 'NoiseScheduler':
        """Create scheduler with cosine beta schedule."""
        return cls(num_timesteps, ScheduleType.COSINE)
    
    @classmethod
    def quadratic(cls, num_timesteps: int = 1000) -> 'NoiseScheduler':
        """Create scheduler with quadratic beta schedule."""
        return cls(num_timesteps, ScheduleType.QUADRATIC)
    
    def add_noise(
        self,
        sample: Tensor,
        noise: Tensor,
        timestep: int
    ) -> Tensor:
        """
        Add noise to sample at given timestep.
        
        q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        
        Args:
            sample: Original sample x_0
            noise: Gaussian noise
            timestep: Current timestep
            
        Returns:
            Noisy sample x_t
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep]
        
        return sample * sqrt_alpha + noise * sqrt_one_minus_alpha
    
    def step(
        self,
        sample: Tensor,
        noise_pred: Tensor,
        timestep: int,
        rng: Optional[np.random.Generator] = None
    ) -> Tensor:
        """
        DDPM step: predict x_{t-1} from x_t and predicted noise.
        
        Args:
            sample: Current noisy sample x_t
            noise_pred: Predicted noise epsilon_theta(x_t, t)
            timestep: Current timestep
            rng: Random number generator
            
        Returns:
            Denoised sample x_{t-1}
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Predict x_0 from x_t and noise prediction
        x0_pred = (
            sample * self.sqrt_recip_alphas_cumprod[timestep]
            - noise_pred * self.sqrt_recipm1_alphas_cumprod[timestep]
        )
        
        # Compute posterior mean
        posterior_mean = (
            x0_pred * self.posterior_mean_coef1[timestep]
            + sample * self.posterior_mean_coef2[timestep]
        )
        
        # Add noise if not the last step
        if timestep > 0:
            noise = Tensor.randn(*sample.shape, rng=rng)
            variance = np.sqrt(self.posterior_variance[timestep])
            return posterior_mean + noise * variance
        
        return posterior_mean
    
    def step_ddim(
        self,
        sample: Tensor,
        noise_pred: Tensor,
        timestep: int,
        prev_timestep: int,
        eta: float = 0.0
    ) -> Tensor:
        """
        DDIM step: deterministic or stochastic sampling.
        
        Args:
            sample: Current noisy sample x_t
            noise_pred: Predicted noise epsilon_theta(x_t, t)
            timestep: Current timestep
            prev_timestep: Previous timestep
            eta: Noise level (0 for deterministic DDIM)
            
        Returns:
            Denoised sample x_{t-1}
        """
        alpha_cumprod = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else 1.0
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timestep]
        
        # Predict x_0
        x0_pred = (sample - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
        
        # DDIM formula
        sqrt_alpha_cumprod_prev = np.sqrt(alpha_cumprod_prev)
        sqrt_one_minus_alpha_cumprod_prev = np.sqrt(1.0 - alpha_cumprod_prev)
        
        return x0_pred * sqrt_alpha_cumprod_prev + noise_pred * sqrt_one_minus_alpha_cumprod_prev
    
    def get_timesteps(self, num_inference_steps: int) -> np.ndarray:
        """Get evenly spaced timesteps for inference."""
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = np.arange(0, num_inference_steps) * step_ratio
        timesteps = self.num_timesteps - 1 - timesteps
        return timesteps.astype(np.int64)
