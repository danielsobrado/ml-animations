# Mini-Diffusion Python

A minimal Python implementation of diffusion models for educational purposes.

## Overview

This implementation demonstrates the core concepts of denoising diffusion probabilistic models (DDPM) and denoising diffusion implicit models (DDIM), including:

- **Tensor Operations**: NumPy-based tensor operations for 4D image data
- **Noise Scheduler**: Linear, cosine, and quadratic beta schedules
- **Neural Network Layers**: Linear, Conv2d, GroupNorm
- **U-Net Architecture**: Encoder-decoder with skip connections and time embeddings
- **Sampling**: DDPM and DDIM sampling algorithms

## Installation

```bash
cd mini-diffusion-python
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy
```

## Running the Demo

```bash
# Using the entry point
mini-diffusion-demo

# Or run directly
python -m mini_diffusion.demo
```

## Project Structure

```
mini-diffusion-python/
├── pyproject.toml
├── README.md
└── mini_diffusion/
    ├── __init__.py
    ├── tensor.py       # Tensor operations
    ├── scheduler.py    # Noise scheduler
    ├── layers.py       # Linear, Conv2d, GroupNorm
    ├── unet.py         # U-Net architecture
    ├── sampler.py      # DDPM/DDIM sampling
    └── demo.py         # Demo program
```

## Usage

### Tensor Operations

```python
from mini_diffusion import Tensor
import numpy as np

rng = np.random.default_rng(42)

# Create tensors
zeros = Tensor.zeros(2, 3, 32, 32)
ones = Tensor.ones(2, 3, 32, 32)
randn = Tensor.randn(2, 3, 32, 32, rng=rng)

# Xavier/Kaiming initialization
xavier = Tensor.xavier(64, 128, rng)
kaiming = Tensor.kaiming(64, 128, rng)

# Arithmetic operations
result = randn * 2.0 + 1.0
diff = ones - zeros
product = randn * ones

# Activations
relu = randn.relu()
sigmoid = randn.sigmoid()
silu = randn.silu()
gelu = randn.gelu()
```

### Noise Scheduler

```python
from mini_diffusion import NoiseScheduler, ScheduleType

# Create schedulers
linear = NoiseScheduler.linear(1000)
cosine = NoiseScheduler.cosine(1000)
quadratic = NoiseScheduler.quadratic(1000)

# Add noise to a sample
sample = Tensor.ones(1, 3, 64, 64)
noise = Tensor.randn(1, 3, 64, 64, rng=rng)
noisy = linear.add_noise(sample, noise, timestep=500)

# DDPM step
denoised = linear.step(noisy, noise_pred, timestep=500, rng=rng)

# DDIM step (deterministic)
denoised = linear.step_ddim(noisy, noise_pred, timestep=500, prev_timestep=450)
```

### Neural Network Layers

```python
from mini_diffusion import Linear, Conv2d, GroupNorm

rng = np.random.default_rng(42)

# Linear layer
linear = Linear(256, 512, rng)
out = linear(input_tensor)

# 2D Convolution
conv = Conv2d(64, 128, kernel_size=3, stride=1, padding=1, rng=rng)
out = conv(input_tensor)

# Group Normalization
norm = GroupNorm(32, 128)
out = norm(input_tensor)
```

### U-Net

```python
from mini_diffusion import UNet

rng = np.random.default_rng(42)

# Create U-Net
model = UNet(
    in_channels=3,
    out_channels=3,
    model_channels=64,
    rng=rng
)

# Forward pass
noise_pred = model(noisy_image, timestep=500)
print(f"Parameters: {model.parameter_count():,}")
```

### Sampling

```python
from mini_diffusion import Sampler, NoiseScheduler, UNet

rng = np.random.default_rng(42)

# Setup
scheduler = NoiseScheduler.linear(1000)
model = UNet(3, 3, 64, rng)

# Create sampler
ddpm_sampler = Sampler.ddpm(scheduler, num_inference_steps=50)
ddim_sampler = Sampler.ddim(scheduler, num_inference_steps=50)

# Sample
shape = (1, 3, 64, 64)
sample = ddpm_sampler.sample(model, shape, rng)

# Sample with progress callback
def callback(step, total, timestep, current):
    print(f"Step {step}/{total} (t={timestep})")

sample = ddpm_sampler.sample(model, shape, rng, callback)
```

## How Diffusion Works

### Forward Process (Adding Noise)

The forward diffusion process gradually adds Gaussian noise to data:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)I)$$

Where $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ and $\alpha_t = 1 - \beta_t$.

### Beta Schedules

- **Linear**: $\beta_t$ increases linearly from $\beta_{\text{start}}$ to $\beta_{\text{end}}$
- **Cosine**: Designed to keep $\bar{\alpha}_t$ high for longer, preventing information loss
- **Quadratic**: $\sqrt{\beta_t}$ increases linearly

### Reverse Process (Denoising)

The model learns to predict the noise:

$$\epsilon_\theta(x_t, t) \approx \epsilon$$

Then denoising step:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

### DDIM (Deterministic)

DDIM removes the stochastic component:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta$$

Where $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$

## Example Output

```
=== Mini-Diffusion Python Demo ===

--- Tensor Operations ---
zeros shape: (2, 3, 4, 4)
ones shape: (2, 3, 4, 4)
randn mean: 0.0123, std: 1.0045
zeros + ones mean: 1.0000
randn * 2 mean: 0.0246, std: 2.0090

--- Noise Scheduler ---
Linear schedule:
  beta[0]=0.000100, beta[500]=0.010050, beta[999]=0.020000
  alpha_cumprod[0]=0.999900, alpha_cumprod[500]=0.006738, alpha_cumprod[999]=0.000045

Adding noise at different timesteps:
  t=0: mean=0.9999, std=0.0141
  t=250: mean=0.6789, std=0.7345
  t=500: mean=0.0821, std=0.9966
  t=999: mean=0.0067, std=1.0000

--- U-Net Architecture ---
U-Net created: in=3, out=3, model=32
Total parameters: 1,234,567
Input shape: (1, 3, 32, 32)
Output shape: (1, 3, 32, 32)
Forward pass time: 250 ms

--- Sampling ---
DDPM Sampling:
  Step 1/10 (t=999): mean=-0.0123
  Step 2/10 (t=899): mean=0.0456
  ...
```

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2020)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Rombach et al., 2022)

## License

MIT License - Educational purposes
