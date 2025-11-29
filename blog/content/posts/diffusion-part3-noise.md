---
title: "Building a Diffusion Model from Scratch in Rust - Part 3: Understanding Noise"
date: 2024-11-17
draft: false
tags: ["diffusion-models", "rust", "gaussian-noise", "deep-learning", "probability"]
categories: ["Diffusion Models from Scratch"]
series: ["Mini Diffusion in Rust"]
weight: 3
---

Here's where diffusion models get interesting. And maybe a bit confusing.

The core idea: destroy an image by gradually adding noise. Then train a neural network to reverse that process. Once trained, you can start from pure noise and generate new images.

Sounds simple. The math is elegant. Let's dig in.

## The Forward Process

Start with a clean image $x_0$. Add a little noise to get $x_1$. Add more to get $x_2$. Keep going until $x_T$ is pure Gaussian noise.

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

That's saying: $x_t$ is a Gaussian distribution centered at a scaled-down version of $x_{t-1}$, with variance $\beta_t$.

$\beta_t$ is small (like 0.0001 to 0.02). Each step adds just a tiny bit of noise.

## Why This Parametrization?

The clever trick: you can jump directly from $x_0$ to any $x_t$ without computing intermediate steps.

Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$.

Then:
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

Or in code terms:
$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon$$

Where $\epsilon \sim \mathcal{N}(0, I)$ is random noise.

This is huge for training. Sample a random timestep, jump directly there, predict the noise.

## Noise Schedules

How should $\beta_t$ change over time? This is the "schedule."

```rust
// src/diffusion.rs
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    pub num_timesteps: usize,
    pub beta_start: f32,
    pub beta_end: f32,
    pub schedule: String,  // "linear", "cosine", "quadratic"
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        DiffusionConfig {
            num_timesteps: 1000,
            beta_start: 1e-4,
            beta_end: 0.02,
            schedule: "linear".to_string(),
        }
    }
}
```

1000 timesteps is standard. More gives finer control but slower sampling.

## Linear Schedule

The original DDPM paper used linear:

```rust
fn get_betas_linear(config: &DiffusionConfig) -> Vec<f32> {
    let t = config.num_timesteps;
    (0..t)
        .map(|i| {
            config.beta_start + 
            (config.beta_end - config.beta_start) * (i as f32) / (t as f32 - 1.0)
        })
        .collect()
}
```

Simple but not optimal. Too much noise added early, not enough structure preserved.

## Cosine Schedule

Improved schedule from "Improved DDPM" paper:

```rust
fn get_betas_cosine(config: &DiffusionConfig) -> Vec<f32> {
    let t = config.num_timesteps;
    let s = 0.008;  // small offset
    let max_beta = 0.999;
    
    (0..t)
        .map(|i| {
            let t1 = i as f32 / t as f32;
            let t2 = (i + 1) as f32 / t as f32;
            
            // Alpha_bar follows cosine curve
            let f = |x: f32| ((x + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos().powi(2);
            let alpha_bar_t1 = f(t1);
            let alpha_bar_t2 = f(t2);
            
            (1.0 - alpha_bar_t2 / alpha_bar_t1).min(max_beta)
        })
        .collect()
}
```

Cosine keeps more structure longer, then destroys it more aggressively at the end. Better image quality.

## The Noise Scheduler

Now let's build the complete scheduler:

```rust
#[derive(Debug, Clone)]
pub struct NoiseScheduler {
    pub config: DiffusionConfig,
    pub betas: Vec<f32>,
    pub alphas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
    pub sqrt_alphas_cumprod: Vec<f32>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,
}

impl NoiseScheduler {
    pub fn new(config: DiffusionConfig) -> Self {
        let betas = match config.schedule.as_str() {
            "linear" => get_betas_linear(&config),
            "cosine" => get_betas_cosine(&config),
            _ => panic!("Unknown schedule"),
        };
        
        // Alpha = 1 - beta
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        
        // Cumulative product
        let mut alphas_cumprod = Vec::with_capacity(config.num_timesteps);
        let mut cumprod = 1.0;
        for &alpha in &alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }
        
        // Precompute sqrt values (used constantly)
        let sqrt_alphas_cumprod: Vec<f32> = 
            alphas_cumprod.iter().map(|a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f32> = 
            alphas_cumprod.iter().map(|a| (1.0 - a).sqrt()).collect();

        NoiseScheduler {
            config,
            betas,
            alphas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        }
    }
}
```

We precompute everything. These values get used thousands of times during training.

## Adding Noise

The core operation. Go from clean image to noisy image at timestep t:

```rust
impl NoiseScheduler {
    /// Add noise to x0 at timestep t
    /// Returns: (noisy_image, noise)
    pub fn add_noise(&self, x0: &Tensor, t: usize) -> (Tensor, Tensor) {
        let noise = Tensor::randn(x0.shape());
        let noisy = self.add_noise_with(x0, &noise, t);
        (noisy, noise)
    }

    /// Add specific noise to x0 at timestep t
    pub fn add_noise_with(&self, x0: &Tensor, noise: &Tensor, t: usize) -> Tensor {
        let sqrt_alpha = self.sqrt_alphas_cumprod[t];
        let sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t];
        
        // x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x0.mul_scalar(sqrt_alpha)
            .add(&noise.mul_scalar(sqrt_one_minus_alpha))
    }
}
```

At t=0: almost all signal, tiny bit of noise.
At t=999: almost all noise, tiny bit of signal.

## Visualizing the Process

Let's trace what happens to alpha_bar over time:

```rust
#[test]
fn visualize_schedule() {
    let scheduler = NoiseScheduler::new(DiffusionConfig::default());
    
    println!("t=0:   alpha_bar={:.4}", scheduler.alphas_cumprod[0]);
    println!("t=100: alpha_bar={:.4}", scheduler.alphas_cumprod[100]);
    println!("t=500: alpha_bar={:.4}", scheduler.alphas_cumprod[500]);
    println!("t=900: alpha_bar={:.4}", scheduler.alphas_cumprod[900]);
    println!("t=999: alpha_bar={:.4}", scheduler.alphas_cumprod[999]);
}
```

For cosine schedule, you'll see:
- t=0: ~0.9999 (almost all signal)
- t=500: ~0.5 (half and half)
- t=999: ~0.0001 (almost all noise)

## Timestep Embeddings

The neural network needs to know what timestep it's at. We encode timesteps using sinusoidal embeddings (same idea as transformer positional encodings):

```rust
pub fn get_timestep_embedding(timesteps: &[usize], embedding_dim: usize) -> Tensor {
    let half_dim = embedding_dim / 2;
    let emb_factor = -(10000.0_f32.ln()) / (half_dim as f32 - 1.0);
    
    let mut embeddings = Tensor::zeros(&[timesteps.len(), embedding_dim]);
    let emb_data = embeddings.as_mut_slice();
    
    for (batch_idx, &t) in timesteps.iter().enumerate() {
        for i in 0..half_dim {
            let freq = (i as f32 * emb_factor).exp();
            let angle = t as f32 * freq;
            
            // First half: sin
            emb_data[batch_idx * embedding_dim + i] = angle.sin();
            // Second half: cos
            emb_data[batch_idx * embedding_dim + half_dim + i] = angle.cos();
        }
    }
    
    embeddings
}
```

Low frequencies change slowly (capture rough timestep). High frequencies change fast (capture fine timestep).

## Sampling Random Timesteps

During training, we sample random timesteps for each batch element:

```rust
impl NoiseScheduler {
    pub fn sample_timesteps(&self, batch_size: usize) -> Vec<usize> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..batch_size)
            .map(|_| rng.gen_range(0..self.config.num_timesteps))
            .collect()
    }
}
```

Uniform sampling. Some papers suggest importance sampling (more samples near tricky timesteps), but uniform works fine.

## The Reverse Process (Preview)

Training teaches the network to predict noise. At inference, we run the process backwards:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

The network predicts what noise was added, we subtract a portion of it, and step backwards.

```rust
impl NoiseScheduler {
    /// One denoising step: x_t -> x_{t-1}
    pub fn step(&self, x_t: &Tensor, predicted_noise: &Tensor, t: usize) -> Tensor {
        let alpha_t = self.alphas[t];
        let alpha_bar_t = self.alphas_cumprod[t];
        let beta_t = self.betas[t];
        
        // Compute mean
        let coef1 = 1.0 / alpha_t.sqrt();
        let coef2 = beta_t / self.sqrt_one_minus_alphas_cumprod[t];
        let mean = x_t.sub(&predicted_noise.mul_scalar(coef2)).mul_scalar(coef1);
        
        if t == 0 {
            // Final step: no noise
            mean
        } else {
            // Add some noise for stochasticity
            let sigma = beta_t.sqrt();
            let noise = Tensor::randn(x_t.shape());
            mean.add(&noise.mul_scalar(sigma))
        }
    }
}
```

More on this in Part 6 when we cover sampling.

## Why Does This Work?

The intuition:
1. Adding Gaussian noise is a simple, well-understood process
2. The reverse is also Gaussian (for small steps)
3. Neural networks are great at predicting patterns
4. The network learns what "typical noise" looks like at each level

It's like learning to restore old photos. Show the network millions of examples of "pristine -> degraded", it learns to reverse the degradation.

## Testing the Noise Scheduler

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_scheduler() {
        let scheduler = NoiseScheduler::new(DiffusionConfig::default());
        
        assert_eq!(scheduler.betas.len(), 1000);
        // Alpha_bar should decrease
        assert!(scheduler.alphas_cumprod[0] > scheduler.alphas_cumprod[999]);
    }

    #[test]
    fn test_add_noise() {
        let scheduler = NoiseScheduler::new(DiffusionConfig::default());
        let x = Tensor::ones(&[1, 3, 32, 32]);
        
        // Early timestep: mostly signal
        let (noisy_early, _) = scheduler.add_noise(&x, 10);
        assert!((noisy_early.mean() - 1.0).abs() < 0.5);
        
        // Late timestep: mostly noise
        let (noisy_late, _) = scheduler.add_noise(&x, 990);
        assert!(noisy_late.mean().abs() < 0.5);
    }

    #[test]
    fn test_timestep_embedding() {
        let t = vec![0, 100, 500, 999];
        let emb = get_timestep_embedding(&t, 128);
        assert_eq!(emb.shape(), &[4, 128]);
    }
}
```

## Key Takeaways

1. Forward diffusion destroys images by adding noise gradually
2. We can jump to any timestep directly: $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$
3. Cosine schedule usually works better than linear
4. Timestep embeddings tell the network "how noisy is this?"
5. The reverse process undoes the noise, one small step at a time

## Next Up

In [Part 4: U-Net Architecture](/posts/diffusion-part4-unet/), we build the neural network that predicts noise:
- Residual blocks with time conditioning
- Downsampling and upsampling paths
- Skip connections

The full code is in the `mini-diffusion` folder: [View on GitHub](https://github.com/danielsobrado/ml-animations/tree/main/mini-diffusion)

---

**Series Navigation:**
- [Part 1: Tensor Foundations](/posts/diffusion-part1-tensors/)
- [Part 2: Neural Network Layers](/posts/diffusion-part2-neural-networks/)
- **Part 3: Understanding Noise** (you are here)
- [Part 4: U-Net Architecture](/posts/diffusion-part4-unet/)
- [Part 5: Training Loop](/posts/diffusion-part5-training/)
- [Part 6: Sampling](/posts/diffusion-part6-sampling/)
- [Part 7: Text Conditioning](/posts/diffusion-part7-conditioning/)
