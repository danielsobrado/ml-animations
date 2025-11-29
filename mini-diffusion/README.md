# Mini Diffusion

A minimal diffusion model implementation in Rust, built from scratch for educational purposes.

This project accompanies a 7-part blog series explaining how diffusion models like Flux and Stable Diffusion work under the hood.

## 📚 Blog Series

1. [Part 1: Tensor Foundations](../blog/content/posts/diffusion-part1-tensors.md) - Building the data foundation
2. [Part 2: Neural Network Layers](../blog/content/posts/diffusion-part2-neural-networks.md) - Linear, Conv2d, Normalization
3. [Part 3: Understanding Noise](../blog/content/posts/diffusion-part3-noise.md) - The forward diffusion process
4. [Part 4: U-Net Architecture](../blog/content/posts/diffusion-part4-unet.md) - The noise prediction network
5. [Part 5: Training Loop](../blog/content/posts/diffusion-part5-training.md) - Loss functions and optimization
6. [Part 6: Sampling](../blog/content/posts/diffusion-part6-sampling.md) - DDPM and DDIM sampling
7. [Part 7: Text Conditioning](../blog/content/posts/diffusion-part7-conditioning.md) - Text-to-image generation

## 🏗️ Project Structure

```
mini-diffusion/
├── Cargo.toml           # Dependencies and project config
├── src/
│   ├── lib.rs           # Library exports
│   ├── tensor.rs        # Tensor implementation
│   ├── nn.rs            # Neural network layers
│   ├── diffusion.rs     # Noise schedules and forward process
│   ├── unet.rs          # U-Net architecture
│   ├── training.rs      # Training loop and optimizer
│   ├── sampling.rs      # DDPM/DDIM sampling
│   └── bin/
│       ├── train.rs     # Training binary
│       └── generate.rs  # Generation binary
```

## 🚀 Getting Started

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))

### Build

```bash
cd mini-diffusion
cargo build --release
```

### Run Tests

```bash
cargo test
```

### Run Training Demo

```bash
cargo run --bin train
```

Note: This demo shows the training structure. Real training requires autograd (automatic differentiation) which isn't implemented. For actual training, use a library like `burn-rs` or `candle`.

### Run Generation Demo

```bash
cargo run --bin generate
```

Note: With random weights, this produces colored noise. The structure is correct for generating images with trained weights.

## 🧩 Architecture Overview

### Tensors (`tensor.rs`)

Multi-dimensional arrays with:
- Creation (zeros, ones, random)
- Element-wise operations
- Math functions (sqrt, exp, ln)
- Activations (ReLU, SiLU, GELU)
- Matrix multiplication

### Neural Network Layers (`nn.rs`)

- `Linear` - Fully connected layers
- `Conv2d` - 2D convolution
- `GroupNorm` - Group normalization (stable for small batches)
- `LayerNorm` - Layer normalization
- `SelfAttention` - Self-attention mechanism

### Diffusion Process (`diffusion.rs`)

- Noise schedules (linear, cosine)
- Forward diffusion (adding noise)
- Timestep embeddings

### U-Net (`unet.rs`)

- `ResBlock` - Residual blocks with time conditioning
- `Downsample` - Spatial downsampling
- `Upsample` - Spatial upsampling
- `UNet` - Complete encoder-decoder with skip connections

### Training (`training.rs`)

- MSE loss for noise prediction
- Adam optimizer
- Learning rate scheduling (cosine, warmup)

### Sampling (`sampling.rs`)

- DDPM (stochastic, 1000 steps)
- DDIM (deterministic, ~50 steps)
- Image saving utilities

## ⚙️ Configuration

### Diffusion Config

```rust
DiffusionConfig {
    num_timesteps: 1000,    // Number of noise levels
    beta_start: 1e-4,       // Starting noise
    beta_end: 0.02,         // Ending noise
    schedule: "cosine",     // "linear" or "cosine"
}
```

### Training Config

```rust
TrainingConfig {
    learning_rate: 1e-4,
    batch_size: 4,
    num_epochs: 100,
    beta1: 0.9,             // Adam momentum
    beta2: 0.999,           // Adam RMSprop
}
```

### Sampler Config

```rust
SamplerConfig {
    num_steps: 50,          // DDIM steps
    guidance_scale: 1.0,    // CFG scale
    use_ddim: true,         // DDIM vs DDPM
    eta: 0.0,               // DDIM stochasticity
}
```

## 🔬 What's Not Included

This is an educational implementation. For production, you'd need:

1. **Automatic differentiation** - We show forward passes only
2. **GPU acceleration** - CPU only currently
3. **Pretrained text encoder** - No CLIP/T5
4. **VAE** - We work in pixel space, not latent space
5. **Optimized operations** - Naive implementations for clarity

## 📖 Learning Resources

- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [DDIM Paper](https://arxiv.org/abs/2010.02502) - Denoising Diffusion Implicit Models
- [Improved DDPM](https://arxiv.org/abs/2102.09672) - Better noise schedules
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) - Text conditioning
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) - Latent diffusion models

## 🤝 Contributing

This is an educational project. Feel free to:
- Improve documentation
- Add more tests
- Optimize implementations
- Fix bugs

## 📄 License

MIT License - see LICENSE file for details.
