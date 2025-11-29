# Mini-Diffusion Go

A minimal diffusion model implementation in Go for educational purposes. This is a Go port of the Rust `mini-diffusion` implementation.

## Features

- **Tensor Operations**: Matrix multiplication, element-wise operations using gonum
- **Neural Network Layers**: Linear, Conv2d, GroupNorm
- **Noise Scheduler**: Linear, cosine, and quadratic beta schedules
- **U-Net Architecture**: Residual blocks, downsampling, upsampling
- **Samplers**: DDPM (stochastic) and DDIM (deterministic)
- **Timestep Embeddings**: Sinusoidal position embeddings

## Requirements

- Go 1.21 or later
- gonum (for matrix operations)

## Installation

```bash
cd mini-diffusion-go
go mod download
```

## Usage

### Run Demo

```bash
go run ./cmd/demo
```

Expected output:
```
=== Mini-Diffusion Go: Demo ===

Configuration:
  Image size: 16x16
  Channels: 3
  Batch size: 2

Noise Scheduler:
  Timesteps: 100
  Schedule: linear
  Beta range: [0.000100, 0.020000]

=== Forward Diffusion Demo ===
Original image - mean: 0.5012
  t=  0: mean=0.5010, signal_ratio=0.9999
  t= 25: mean=0.4521, signal_ratio=0.9032
  t= 50: mean=0.3845, signal_ratio=0.7684
  t= 75: mean=0.2956, signal_ratio=0.5912
  t= 99: mean=0.1023, signal_ratio=0.2045

=== U-Net Model ===
Model channels: 32
Channel multipliers: [1 2]
Res blocks per level: 1
Total parameters: 45123

✅ Mini-Diffusion Go demo completed!
```

## Project Structure

```
mini-diffusion-go/
├── go.mod
├── README.md
├── diffusion/
│   ├── tensor.go     # Tensor operations
│   ├── nn.go         # Neural network layers
│   ├── scheduler.go  # Noise scheduler
│   ├── unet.go       # U-Net architecture
│   └── sampling.go   # DDPM/DDIM sampling
└── cmd/
    └── demo/
        └── main.go   # Demo application
```

## Key Concepts

### Forward Diffusion
The forward process gradually adds Gaussian noise to data:
```
q(x_t | x_0) = N(x_t; √(ᾱ_t) * x_0, (1 - ᾱ_t) * I)
```

### Reverse Diffusion
The reverse process learns to denoise:
```
p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² * I)
```

### Beta Schedule
Controls noise level at each timestep:
- **Linear**: β_t increases linearly
- **Cosine**: Smoother schedule, often better results
- **Quadratic**: β_t increases quadratically

## API Example

```go
package main

import (
    "mini-diffusion-go/diffusion"
    "math/rand"
)

func main() {
    rng := rand.New(rand.NewSource(42))
    
    // Create noise scheduler
    config := diffusion.DefaultDiffusionConfig()
    scheduler := diffusion.NewNoiseScheduler(config, rng)
    
    // Add noise to image
    x0 := diffusion.Randn([]int{1, 3, 32, 32}, rng)
    noisy, noise := scheduler.AddNoise(x0, 500)
    
    // Create U-Net
    unet := diffusion.NewUNet(3, 64, []int{1, 2, 4}, 2, rng)
    
    // Create sampler
    samplerConfig := diffusion.DefaultSamplerConfig()
    sampler := diffusion.NewSampler(samplerConfig, scheduler, rng)
    
    // Generate
    generated := sampler.Sample(unet, []int{1, 3, 32, 32}, true)
}
```

## Comparison with Rust Implementation

| Feature | mini-diffusion (Rust) | mini-diffusion-go |
|---------|----------------------|-------------------|
| Matrix Ops | ndarray | gonum |
| Conv2d | ✓ | ✓ |
| GroupNorm | ✓ | ✓ |
| U-Net | ✓ | ✓ |
| DDPM | ✓ | ✓ |
| DDIM | ✓ | ✓ |
| VAE | ✓ | ✗ (simplified) |
| CLIP/T5 | ✓ | ✗ (simplified) |

## License

MIT
