#!/usr/bin/env python3
"""
Mini-Diffusion Demo

Demonstrates the diffusion model components:
- Tensor operations
- Noise scheduler
- U-Net architecture
- DDPM and DDIM sampling
"""

import time
import numpy as np

from mini_diffusion import (
    Tensor,
    NoiseScheduler,
    ScheduleType,
    Linear,
    Conv2d,
    GroupNorm,
    UNet,
    Sampler,
    SamplerType,
)


def demo_tensor_ops():
    """Demonstrate tensor operations."""
    print("--- Tensor Operations ---")
    
    rng = np.random.default_rng(42)
    
    # Create tensors
    zeros = Tensor.zeros(2, 3, 4, 4)
    print(f"zeros shape: {zeros.shape}")
    
    ones = Tensor.ones(2, 3, 4, 4)
    print(f"ones shape: {ones.shape}")
    
    randn = Tensor.randn(2, 3, 4, 4, rng=rng)
    print(f"randn mean: {randn.mean():.4f}, std: {randn.std():.4f}")
    
    # Arithmetic
    sum_tensor = zeros + ones
    print(f"zeros + ones mean: {sum_tensor.mean():.4f}")
    
    scaled = randn * 2.0
    print(f"randn * 2 mean: {scaled.mean():.4f}, std: {scaled.std():.4f}")
    
    # Activations
    x = Tensor.randn(1, 1, 2, 2, rng=rng)
    print(f"Original: {x.data.flatten()}")
    print(f"ReLU: {x.relu().data.flatten()}")
    print(f"SiLU: {x.silu().data.flatten()}")
    
    print()


def demo_noise_scheduler():
    """Demonstrate noise scheduler."""
    print("--- Noise Scheduler ---")
    
    rng = np.random.default_rng(42)
    
    # Create schedulers
    linear = NoiseScheduler.linear(1000)
    cosine = NoiseScheduler.cosine(1000)
    quadratic = NoiseScheduler.quadratic(1000)
    
    print("Linear schedule:")
    print(f"  beta[0]={linear.betas[0]:.6f}, beta[500]={linear.betas[500]:.6f}, beta[999]={linear.betas[999]:.6f}")
    print(f"  alpha_cumprod[0]={linear.alphas_cumprod[0]:.6f}, "
          f"alpha_cumprod[500]={linear.alphas_cumprod[500]:.6f}, "
          f"alpha_cumprod[999]={linear.alphas_cumprod[999]:.6f}")
    
    print("Cosine schedule:")
    print(f"  alpha_cumprod[0]={cosine.alphas_cumprod[0]:.6f}, "
          f"alpha_cumprod[500]={cosine.alphas_cumprod[500]:.6f}, "
          f"alpha_cumprod[999]={cosine.alphas_cumprod[999]:.6f}")
    
    print("Quadratic schedule:")
    print(f"  alpha_cumprod[0]={quadratic.alphas_cumprod[0]:.6f}, "
          f"alpha_cumprod[500]={quadratic.alphas_cumprod[500]:.6f}, "
          f"alpha_cumprod[999]={quadratic.alphas_cumprod[999]:.6f}")
    
    # Demo add_noise
    sample = Tensor.ones(1, 3, 8, 8)
    noise = Tensor.randn(1, 3, 8, 8, rng=rng)
    
    print("\nAdding noise at different timesteps:")
    for t in [0, 250, 500, 750, 999]:
        noisy = linear.add_noise(sample, noise, t)
        print(f"  t={t}: mean={noisy.mean():.4f}, std={noisy.std():.4f}")
    
    print()


def demo_unet():
    """Demonstrate U-Net architecture."""
    print("--- U-Net Architecture ---")
    
    rng = np.random.default_rng(42)
    
    # Create U-Net
    in_channels = 3
    out_channels = 3
    model_channels = 32
    
    unet = UNet(in_channels, out_channels, model_channels, rng)
    print(f"U-Net created: in={in_channels}, out={out_channels}, model={model_channels}")
    print(f"Total parameters: {unet.parameter_count():,}")
    
    # Forward pass
    x = Tensor.randn(1, in_channels, 32, 32, rng=rng)
    print(f"Input shape: {x.shape}")
    
    start_time = time.time()
    out = unet(x, 500)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"Output shape: {out.shape}")
    print(f"Forward pass time: {elapsed:.0f} ms")
    print(f"Output mean: {out.mean():.4f}, std: {out.std():.4f}")
    
    print()


def demo_sampling():
    """Demonstrate sampling."""
    print("--- Sampling ---")
    
    rng = np.random.default_rng(42)
    
    # Create scheduler and model
    scheduler = NoiseScheduler.linear(1000)
    model = UNet(3, 3, 32, rng)
    
    # Create sampler
    num_steps = 10  # Few steps for demo
    ddpm_sampler = Sampler.ddpm(scheduler, num_steps)
    ddim_sampler = Sampler.ddim(scheduler, num_steps)
    
    print(f"Sampler with {num_steps} inference steps")
    print(f"Timesteps: {list(ddpm_sampler.timesteps)}")
    
    shape = (1, 3, 16, 16)  # Small for demo
    
    # DDPM sampling
    print("\nDDPM Sampling:")
    start_time = time.time()
    
    def progress_callback(step, total, t, current):
        print(f"  Step {step}/{total} (t={t}): mean={current.mean():.4f}")
    
    ddpm_sample = ddpm_sampler.sample(model, shape, rng, progress_callback)
    elapsed = (time.time() - start_time) * 1000
    print(f"DDPM sampling time: {elapsed:.0f} ms")
    print(f"Final sample mean: {ddpm_sample.mean():.4f}, std: {ddpm_sample.std():.4f}")
    
    # DDIM sampling
    print("\nDDIM Sampling:")
    start_time = time.time()
    ddim_sample = ddim_sampler.sample(model, shape, rng, progress_callback)
    elapsed = (time.time() - start_time) * 1000
    print(f"DDIM sampling time: {elapsed:.0f} ms")
    print(f"Final sample mean: {ddim_sample.mean():.4f}, std: {ddim_sample.std():.4f}")


def main():
    """Run all demos."""
    print("=== Mini-Diffusion Python Demo ===\n")
    
    demo_tensor_ops()
    demo_noise_scheduler()
    demo_unet()
    demo_sampling()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
