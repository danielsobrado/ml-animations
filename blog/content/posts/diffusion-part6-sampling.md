---
title: "Building a Diffusion Model from Scratch in Rust - Part 6: Sampling"
date: 2024-11-20
draft: false
tags: ["diffusion-models", "rust", "sampling", "ddpm", "ddim", "deep-learning"]
categories: ["Diffusion Models from Scratch"]
series: ["Mini Diffusion in Rust"]
weight: 6
---

Training is done. Model weights are optimized. Now comes the fun part: generating images.

Sampling is the reverse of the diffusion process. Start with pure noise. Iteratively denoise. End with an image.

## The Reverse Process

Forward diffusion: add noise step by step.
Reverse: remove noise step by step.

At each step, the model predicts what noise is present, and we subtract (some of) it:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

Where:
- $\epsilon_\theta$ is our trained U-Net
- $z \sim \mathcal{N}(0, I)$ is fresh random noise
- $\sigma_t$ controls stochasticity

## Sampler Configuration

```rust
// src/sampling.rs
pub struct SamplerConfig {
    /// Number of sampling steps
    pub num_steps: usize,
    /// Classifier-free guidance scale
    pub guidance_scale: f32,
    /// Use DDIM (deterministic) or DDPM (stochastic)
    pub use_ddim: bool,
    /// DDIM eta parameter (0 = deterministic)
    pub eta: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            num_steps: 50,
            guidance_scale: 1.0,
            use_ddim: true,
            eta: 0.0,
        }
    }
}
```

50 steps with DDIM gives decent results. 1000 steps DDPM gives slightly better but takes 20x longer.

## The Sampler

```rust
use crate::tensor::Tensor;
use crate::diffusion::NoiseScheduler;
use crate::unet::UNet;
use indicatif::{ProgressBar, ProgressStyle};

pub struct Sampler {
    pub config: SamplerConfig,
    pub noise_scheduler: NoiseScheduler,
}

impl Sampler {
    pub fn new(config: SamplerConfig, noise_scheduler: NoiseScheduler) -> Self {
        Sampler { config, noise_scheduler }
    }

    pub fn sample(&self, model: &UNet, shape: &[usize]) -> Tensor {
        if self.config.use_ddim {
            self.sample_ddim(model, shape)
        } else {
            self.sample_ddpm(model, shape)
        }
    }
}
```

## DDPM Sampling

The original method. Stochastic. Add noise at each step.

```rust
impl Sampler {
    pub fn sample_ddpm(&self, model: &UNet, shape: &[usize]) -> Tensor {
        let batch_size = shape[0];
        let total_steps = self.noise_scheduler.config.num_timesteps;
        
        // Start from pure noise
        let mut x = Tensor::randn(shape);

        let pb = ProgressBar::new(total_steps as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} Sampling: [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap());

        // Go backwards: t = T-1 down to 0
        for t in (0..total_steps).rev() {
            let timesteps: Vec<usize> = vec![t; batch_size];
            
            // Predict noise at this timestep
            let predicted_noise = model.forward(&x, &timesteps);

            // Get schedule parameters
            let alpha_t = self.noise_scheduler.alphas[t];
            let alpha_bar_t = self.noise_scheduler.alphas_cumprod[t];
            let beta_t = self.noise_scheduler.betas[t];

            // Compute denoised mean
            let coef1 = 1.0 / alpha_t.sqrt();
            let coef2 = beta_t / (1.0 - alpha_bar_t).sqrt();
            let mean = x.sub(&predicted_noise.mul_scalar(coef2)).mul_scalar(coef1);

            // Add noise (except at final step)
            if t > 0 {
                let sigma = beta_t.sqrt();
                let noise = Tensor::randn(shape);
                x = mean.add(&noise.mul_scalar(sigma));
            } else {
                x = mean;
            }

            pb.inc(1);
        }

        pb.finish_with_message("Done!");

        // Clamp to valid image range
        x.clamp(-1.0, 1.0)
    }
}
```

1000 forward passes through the network. Slow but produces good samples.

## DDIM Sampling

Denoising Diffusion Implicit Models. The key insight: you can skip steps.

DDPM needs all 1000 steps because each depends on the previous. DDIM reformulates the process to allow jumps.

```rust
impl Sampler {
    pub fn sample_ddim(&self, model: &UNet, shape: &[usize]) -> Tensor {
        let batch_size = shape[0];
        let total_steps = self.noise_scheduler.config.num_timesteps;
        
        // Calculate which timesteps to use (evenly spaced)
        let step_size = total_steps / self.config.num_steps;
        let timestep_seq: Vec<usize> = (0..self.config.num_steps)
            .map(|i| (self.config.num_steps - 1 - i) * step_size)
            .collect();

        // Start from noise
        let mut x = Tensor::randn(shape);

        let pb = ProgressBar::new(self.config.num_steps as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} DDIM: [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap());

        for (i, &t) in timestep_seq.iter().enumerate() {
            let timesteps: Vec<usize> = vec![t; batch_size];
            
            // Predict noise
            let predicted_noise = model.forward(&x, &timesteps);

            // Get alpha values
            let alpha_bar_t = self.noise_scheduler.alphas_cumprod[t];
            let alpha_bar_prev = if i + 1 < timestep_seq.len() {
                self.noise_scheduler.alphas_cumprod[timestep_seq[i + 1]]
            } else {
                1.0
            };

            // DDIM update equation
            // 1. Predict x_0 from x_t and noise
            let pred_x0 = x.sub(&predicted_noise.mul_scalar((1.0 - alpha_bar_t).sqrt()))
                .div_scalar(alpha_bar_t.sqrt());

            // 2. Direction pointing to x_t
            let dir_xt = predicted_noise.mul_scalar((1.0 - alpha_bar_prev).sqrt());

            // 3. Optional noise (eta controls this)
            let sigma = self.config.eta * 
                ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) 
                    * (1.0 - alpha_bar_t / alpha_bar_prev)).sqrt();

            // 4. Compute x_{t-1}
            x = pred_x0.mul_scalar(alpha_bar_prev.sqrt()).add(&dir_xt);
            
            if sigma > 0.0 && i + 1 < timestep_seq.len() {
                let noise = Tensor::randn(shape);
                x = x.add(&noise.mul_scalar(sigma));
            }

            pb.inc(1);
        }

        pb.finish_with_message("Done!");
        x.clamp(-1.0, 1.0)
    }
}
```

With 50 steps instead of 1000, we get a 20x speedup. Quality is surprisingly close.

## The Eta Parameter

DDIM's eta controls stochasticity:
- `eta = 0`: Fully deterministic. Same noise seed = same image.
- `eta = 1`: Same as DDPM. Adds noise at each step.
- In between: Partial stochasticity.

For most applications, `eta = 0` works well. Deterministic is nice for reproducibility.

## Saving Images

Generated tensors need to become actual image files:

```rust
use image::{ImageBuffer, Rgb};

pub fn tensor_to_image(tensor: &Tensor) -> Vec<u8> {
    // Tensor is in [-1, 1], convert to [0, 255]
    tensor.as_slice()
        .iter()
        .map(|&v| ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8)
        .collect()
}

pub fn save_images(images: &Tensor, prefix: &str) -> std::io::Result<()> {
    let shape = images.shape();
    let batch = shape[0];
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    if channels != 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Expected 3 channels (RGB)",
        ));
    }

    let data = images.as_slice();

    for b in 0..batch {
        let mut img_buf: ImageBuffer<Rgb<u8>, Vec<u8>> = 
            ImageBuffer::new(width as u32, height as u32);

        for h in 0..height {
            for w in 0..width {
                // Extract RGB values
                let r_idx = b * channels * height * width + 0 * height * width + h * width + w;
                let g_idx = b * channels * height * width + 1 * height * width + h * width + w;
                let b_idx = b * channels * height * width + 2 * height * width + h * width + w;

                let r = ((data[r_idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                let g = ((data[g_idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                let blue = ((data[b_idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;

                img_buf.put_pixel(w as u32, h as u32, Rgb([r, g, blue]));
            }
        }

        let filename = format!("{}_{}.png", prefix, b);
        img_buf.save(&filename).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        println!("Saved: {}", filename);
    }

    Ok(())
}
```

## The Generation Binary

```rust
// src/bin/generate.rs
use mini_diffusion::{
    DiffusionConfig, NoiseScheduler, UNet,
    sampling::{Sampler, SamplerConfig, save_images},
};

fn main() {
    println!("Mini Diffusion - Generation");
    println!("===========================\n");

    // Configuration
    let diffusion_config = DiffusionConfig {
        num_timesteps: 1000,
        schedule: "cosine".to_string(),
        ..Default::default()
    };

    let sampler_config = SamplerConfig {
        num_steps: 50,
        use_ddim: true,
        eta: 0.0,
        ..Default::default()
    };

    // Create model (would load from checkpoint in real usage)
    let model = UNet::new(3, 64, 3);
    println!("Model parameters: {}", model.num_parameters());

    // Create sampler
    let scheduler = NoiseScheduler::new(diffusion_config);
    let sampler = Sampler::new(sampler_config, scheduler);

    // Generate
    println!("\nGenerating 4 images at 32x32...");
    let images = sampler.sample(&model, &[4, 3, 32, 32]);

    // Save
    save_images(&images, "generated").unwrap();
    println!("\nDone!");
}
```

Note: Without training, you'll get colored noise. The structure is correct, the weights are random.

## Classifier-Free Guidance

Modern models like Stable Diffusion and Flux use classifier-free guidance for better prompt following:

$$\tilde\epsilon = \epsilon_\theta(\emptyset) + s \cdot (\epsilon_\theta(c) - \epsilon_\theta(\emptyset))$$

Where:
- $\epsilon_\theta(c)$ is noise predicted with conditioning (text prompt)
- $\epsilon_\theta(\emptyset)$ is noise predicted unconditionally
- $s$ is guidance scale (typically 7-15)

Higher guidance = follows prompt more strictly, but less diversity.

```rust
impl Sampler {
    pub fn sample_with_guidance(
        &self,
        model: &UNet,
        shape: &[usize],
        condition: &Tensor,
    ) -> Tensor {
        // Would need model that accepts conditioning
        // Run model twice: with condition and without
        // Combine predictions based on guidance_scale
        
        // Not implemented in our mini version
        self.sample(model, shape)
    }
}
```

Full implementation needs a conditioned U-Net, which we cover in Part 7.

## DDPM vs DDIM Comparison

| Aspect | DDPM | DDIM |
|--------|------|------|
| Steps | 1000 | 50-100 |
| Speed | Slow | Fast |
| Quality | Slightly better | Nearly as good |
| Determinism | Stochastic | Can be deterministic |
| Diversity | More | Less (if eta=0) |

For production: DDIM with eta=0. For best quality: DDPM or DDIM with more steps.

## Advanced Samplers

Modern systems use even fancier samplers:

- **DPM-Solver**: 10-20 steps with good quality
- **UniPC**: Unified predictor-corrector
- **Euler**: Simple but effective
- **Heun**: Second-order, more accurate

They're all variations on the same theme: predict noise, step towards clean image.

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion::DiffusionConfig;

    #[test]
    fn test_sampler_creation() {
        let config = SamplerConfig::default();
        let scheduler = NoiseScheduler::new(DiffusionConfig::default());
        let sampler = Sampler::new(config, scheduler);
        
        assert!(sampler.config.use_ddim);
        assert_eq!(sampler.config.num_steps, 50);
    }

    #[test]
    fn test_tensor_to_image() {
        // All zeros (gray)
        let t = Tensor::zeros(&[3, 4, 4]);
        let img = tensor_to_image(&t);
        assert!(img.iter().all(|&v| (v as i32 - 127).abs() <= 1));

        // All ones (white)
        let t = Tensor::ones(&[3, 4, 4]);
        let img = tensor_to_image(&t);
        assert!(img.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_sample_output_shape() {
        let mut config = SamplerConfig::default();
        config.num_steps = 2;  // Fast test
        
        let scheduler = NoiseScheduler::new(DiffusionConfig {
            num_timesteps: 10,
            ..Default::default()
        });
        
        let sampler = Sampler::new(config, scheduler);
        let model = UNet::new(3, 16, 3);
        
        let output = sampler.sample(&model, &[1, 3, 16, 16]);
        assert_eq!(output.shape(), &[1, 3, 16, 16]);
        
        // Output should be clamped
        let min = output.as_slice().iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = output.as_slice().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        assert!(min >= -1.0);
        assert!(max <= 1.0);
    }
}
```

## What We Built

- DDPM sampling (original, stochastic)
- DDIM sampling (fast, deterministic)
- Image saving utilities
- Generation binary

With a trained model, this would produce actual images. Without training, it produces structured noise.

## Next Up

In [Part 7: Text Conditioning](/posts/diffusion-part7-conditioning/), we add text prompts:
- Text encoders
- Cross-attention
- Classifier-free guidance

The full code is in the `mini-diffusion` folder: [View on GitHub](https://github.com/danielsobrado/ml-animations/tree/main/mini-diffusion)

---

**Series Navigation:**
- [Part 1: Tensor Foundations](/posts/diffusion-part1-tensors/)
- [Part 2: Neural Network Layers](/posts/diffusion-part2-neural-networks/)
- [Part 3: Understanding Noise](/posts/diffusion-part3-noise/)
- [Part 4: U-Net Architecture](/posts/diffusion-part4-unet/)
- [Part 5: Training Loop](/posts/diffusion-part5-training/)
- **Part 6: Sampling** (you are here)
- [Part 7: Text Conditioning](/posts/diffusion-part7-conditioning/)
