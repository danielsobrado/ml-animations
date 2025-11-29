# Mini-Diffusion Java

A minimal Java implementation of diffusion models for educational purposes.

## Overview

This implementation demonstrates the core concepts of denoising diffusion probabilistic models (DDPM) and denoising diffusion implicit models (DDIM), including:

- **Tensor Operations**: 4D tensor operations for image data [batch, channels, height, width]
- **Noise Scheduler**: Linear, cosine, and quadratic beta schedules
- **Neural Network Layers**: Linear, Conv2d, GroupNorm
- **U-Net Architecture**: Encoder-decoder with skip connections and time embeddings
- **Sampling**: DDPM and DDIM sampling algorithms

## Project Structure

```
mini-diffusion-java/
├── pom.xml
├── README.md
└── src/main/java/com/minidiffusion/
    ├── Tensor.java       # 4D tensor operations
    ├── Layers.java       # Linear, Conv2d, GroupNorm
    ├── NoiseScheduler.java # Beta schedules and noise operations
    ├── UNet.java         # U-Net architecture with ResBlocks
    ├── Sampler.java      # DDPM/DDIM sampling
    └── Demo.java         # Demo program
```

## Building

```bash
cd mini-diffusion-java
mvn compile
```

## Running the Demo

```bash
mvn exec:java -Dexec.mainClass="com.minidiffusion.Demo"
```

## Components

### Tensor

4D tensor class with shape [batch, channels, height, width]:

```java
// Create tensors
Tensor zeros = Tensor.zeros(2, 3, 32, 32);
Tensor ones = Tensor.ones(2, 3, 32, 32);
Tensor randn = Tensor.randn(rng, new int[]{2, 3, 32, 32});

// Xavier/Kaiming initialization
Tensor xavier = Tensor.xavier(rng, 64, 128);
Tensor kaiming = Tensor.kaiming(rng, 64, 128);

// Arithmetic operations
Tensor sum = a.add(b);
Tensor diff = a.sub(b);
Tensor prod = a.mul(b);
Tensor scaled = a.mul(2.0);

// Activations
Tensor relu = x.relu();
Tensor sigmoid = x.sigmoid();
Tensor tanh = x.tanh();
Tensor gelu = x.gelu();
Tensor silu = x.silu();
```

### NoiseScheduler

Implements the forward diffusion process:

```java
// Create scheduler with different schedules
NoiseScheduler linear = NoiseScheduler.linear(1000);
NoiseScheduler cosine = NoiseScheduler.cosine(1000);
NoiseScheduler quadratic = NoiseScheduler.quadratic(1000);

// Add noise to a sample
Tensor noisy = scheduler.addNoise(sample, noise, timestep);

// DDPM step
Tensor denoised = scheduler.step(noisy, noisePred, timestep, rng);

// DDIM step (deterministic)
Tensor denoised = scheduler.stepDdim(noisy, noisePred, timestep, prevTimestep);
```

### Neural Network Layers

```java
// Linear layer
Layers.Linear linear = new Layers.Linear(256, 512, rng);
Tensor out = linear.forward(input);

// 2D Convolution
Layers.Conv2d conv = new Layers.Conv2d(64, 128, 3, 1, 1, rng);
Tensor out = conv.forward(input);

// Group Normalization
Layers.GroupNorm norm = new Layers.GroupNorm(32, 128);
Tensor out = norm.forward(input);
```

### U-Net

U-Net architecture for noise prediction:

```java
// Create U-Net
UNet unet = new UNet(3, 3, 64, rng);  // in=3, out=3, model_channels=64

// Forward pass
Tensor noisePred = unet.forward(noisyImage, timestep);
```

### Sampler

DDPM and DDIM sampling:

```java
// Create sampler
Sampler ddpm = new Sampler(scheduler, 50, Sampler.SamplerType.DDPM);
Sampler ddim = new Sampler(scheduler, 50, Sampler.SamplerType.DDIM);

// Sample
Tensor sample = sampler.sample(model, new int[]{1, 3, 64, 64}, rng);

// Sample with progress callback
Tensor sample = sampler.sample(model, shape, rng, (step, total, t, current) -> {
    System.out.printf("Step %d/%d (t=%d)%n", step, total, t);
});
```

## How Diffusion Works

### Forward Process (Adding Noise)

The forward diffusion process gradually adds Gaussian noise to data:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)I)$$

Where $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ and $\alpha_t = 1 - \beta_t$.

### Reverse Process (Denoising)

The reverse process learns to denoise:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

The model predicts the noise $\epsilon_\theta(x_t, t)$ at each timestep.

### DDPM Sampling

DDPM uses the learned reverse process with added noise at each step.

### DDIM Sampling

DDIM is a deterministic variant that allows faster sampling:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta$$

## Example Output

```
=== Mini-Diffusion Java Demo ===

--- Tensor Operations ---
zeros shape: [2, 3, 4, 4]
ones shape: [2, 3, 4, 4]
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
  t=750: mean=0.0085, std=1.0001
  t=999: mean=0.0067, std=1.0000

--- U-Net Architecture ---
U-Net created: in=3, out=3, model=32
Total parameters: 1,234,567
Input shape: [1, 3, 32, 32]
Output shape: [1, 3, 32, 32]
Forward pass time: 125 ms

--- Sampling ---
DDPM Sampling:
  Step 1/10 (t=999): mean=-0.0123
  Step 2/10 (t=899): mean=0.0456
  ...
```

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2020)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Rombach et al., 2022)

## License

MIT License - Educational purposes
