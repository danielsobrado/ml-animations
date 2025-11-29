---
title: "Building a Diffusion Model from Scratch in Rust - Part 5: Training"
date: 2024-11-19
draft: false
tags: ["diffusion-models", "rust", "training", "optimization", "deep-learning"]
categories: ["Diffusion Models from Scratch"]
series: ["Mini Diffusion in Rust"]
weight: 5
---

We have tensors, layers, and a U-Net. Now we make it learn.

The training objective for diffusion models is surprisingly simple: predict the noise. That's it.

## The Loss Function

Given a clean image $x_0$ and random timestep $t$:

1. Sample noise $\epsilon \sim \mathcal{N}(0, I)$
2. Create noisy image $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$
3. Predict noise $\hat\epsilon = \text{UNet}(x_t, t)$
4. Loss = MSE between $\epsilon$ and $\hat\epsilon$

$$L = \mathbb{E}_{x_0, \epsilon, t} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]$$

Simple Mean Squared Error. The network learns to recognize noise patterns.

```rust
// src/training.rs
pub fn mse_loss(predicted: &Tensor, target: &Tensor) -> f32 {
    let diff = predicted.sub(target);
    diff.square().mean()
}
```

## Training Configuration

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub save_every: usize,
    pub beta1: f32,   // Adam momentum
    pub beta2: f32,   // Adam RMSprop
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 4,
            num_epochs: 100,
            save_every: 10,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}
```

1e-4 learning rate is standard for diffusion models. Some use 2e-4 or 3e-4.

## Adam Optimizer

Adam combines momentum and adaptive learning rates. Works great for deep networks.

```rust
#[derive(Debug, Clone)]
struct AdamState {
    m: Tensor,  // First moment (momentum)
    v: Tensor,  // Second moment (adaptive lr)
}

pub struct Adam {
    pub config: TrainingConfig,
    states: Vec<AdamState>,
    pub step: usize,
}

impl Adam {
    pub fn new(config: TrainingConfig, param_shapes: &[&[usize]]) -> Self {
        let states = param_shapes
            .iter()
            .map(|shape| AdamState {
                m: Tensor::zeros(shape),
                v: Tensor::zeros(shape),
            })
            .collect();

        Adam {
            config,
            states,
            step: 0,
        }
    }
}
```

We track running averages per parameter.

## The Adam Update

```rust
impl Adam {
    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) {
        self.step += 1;
        let t = self.step as f32;

        // Bias correction (important early in training)
        let bias_correction1 = 1.0 - self.config.beta1.powf(t);
        let bias_correction2 = 1.0 - self.config.beta2.powf(t);

        for ((param, grad), state) in params.iter_mut()
            .zip(grads.iter())
            .zip(self.states.iter_mut()) 
        {
            // Update momentum: m = beta1 * m + (1 - beta1) * grad
            state.m = state.m
                .mul_scalar(self.config.beta1)
                .add(&grad.mul_scalar(1.0 - self.config.beta1));

            // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
            state.v = state.v
                .mul_scalar(self.config.beta2)
                .add(&grad.square().mul_scalar(1.0 - self.config.beta2));

            // Bias-corrected estimates
            let m_hat = state.m.div_scalar(bias_correction1);
            let v_hat = state.v.div_scalar(bias_correction2);

            // Update: param -= lr * m_hat / (sqrt(v_hat) + eps)
            let update = m_hat.div(&v_hat.sqrt().add_scalar(self.config.eps));
            **param = param.sub(&update.mul_scalar(self.config.learning_rate));

            // Optional: weight decay (AdamW style)
            if self.config.weight_decay > 0.0 {
                **param = param.mul_scalar(
                    1.0 - self.config.learning_rate * self.config.weight_decay
                );
            }
        }
    }
}
```

The bias correction matters early. Without it, the first updates would be too small.

## The Trainer

```rust
use crate::diffusion::NoiseScheduler;
use crate::unet::UNet;
use indicatif::{ProgressBar, ProgressStyle};

pub struct Trainer {
    pub config: TrainingConfig,
    pub noise_scheduler: NoiseScheduler,
}

impl Trainer {
    pub fn new(config: TrainingConfig, noise_scheduler: NoiseScheduler) -> Self {
        Trainer { config, noise_scheduler }
    }
}
```

## Single Training Step

Here's what happens for each batch:

```rust
impl Trainer {
    pub fn train_step(&self, model: &UNet, images: &Tensor) -> f32 {
        let batch_size = images.shape()[0];

        // 1. Sample random timesteps for each image
        let timesteps = self.noise_scheduler.sample_timesteps(batch_size);

        // 2. Add noise to each image
        let mut noisy_images = Vec::new();
        let mut noise_targets = Vec::new();
        
        for (i, &t) in timesteps.iter().enumerate() {
            // Extract single image from batch
            let single_img = self.extract_single(images, i);
            
            // Add noise at timestep t
            let (noisy, noise) = self.noise_scheduler.add_noise(&single_img, t);
            noisy_images.push(noisy);
            noise_targets.push(noise);
        }

        // 3. Stack back into batches
        let noisy_batch = self.stack_tensors(&noisy_images);
        let noise_batch = self.stack_tensors(&noise_targets);

        // 4. Predict noise
        let predicted_noise = model.forward(&noisy_batch, &timesteps);

        // 5. Compute MSE loss
        mse_loss(&predicted_noise, &noise_batch)
    }
}
```

The network sees noisy images and has to guess what noise was added. Different timesteps = different difficulty levels.

## Helper Functions

```rust
impl Trainer {
    fn extract_single(&self, batch: &Tensor, index: usize) -> Tensor {
        let shape = batch.shape();
        let single_size: usize = shape[1..].iter().product();
        let start = index * single_size;
        
        let mut single = Tensor::zeros(&shape[1..]);
        single.as_mut_slice().copy_from_slice(
            &batch.as_slice()[start..start + single_size]
        );
        single
    }

    fn stack_tensors(&self, tensors: &[Tensor]) -> Tensor {
        if tensors.is_empty() {
            return Tensor::zeros(&[0]);
        }

        let single_shape = tensors[0].shape();
        let batch_size = tensors.len();
        let mut new_shape = vec![batch_size];
        new_shape.extend_from_slice(single_shape);

        let mut result = Tensor::zeros(&new_shape);
        let single_size = tensors[0].numel();

        for (i, tensor) in tensors.iter().enumerate() {
            let start = i * single_size;
            result.as_mut_slice()[start..start + single_size]
                .copy_from_slice(tensor.as_slice());
        }

        result
    }
}
```

## The Missing Piece: Backpropagation

Here's the thing. Our implementation doesn't have automatic differentiation. PyTorch has autograd. We don't.

To actually train, you'd need to either:

1. **Implement autograd** (reverse-mode autodiff)
2. **Use a Rust ML library** like `burn`, `tch-rs`, or `candle`
3. **Compute gradients manually** (not practical for U-Net)

For educational purposes, we've shown the structure. For real training:

```rust
// Using burn-rs (hypothetical)
let loss = model.forward(&noisy_batch, &timesteps);
let grads = loss.backward();  // Autograd
optimizer.step(&grads);
```

## Learning Rate Scheduling

Learning rate should decay over training. Two popular approaches:

### Cosine Annealing

```rust
pub fn cosine_lr_schedule(
    current_step: usize,
    total_steps: usize,
    base_lr: f32,
    min_lr: f32,
) -> f32 {
    let progress = current_step as f32 / total_steps as f32;
    min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
}
```

Smooth decay following a cosine curve.

### Warmup + Cosine

```rust
pub fn warmup_cosine_schedule(
    current_step: usize,
    warmup_steps: usize,
    total_steps: usize,
    base_lr: f32,
) -> f32 {
    if current_step < warmup_steps {
        // Linear warmup
        base_lr * (current_step as f32 / warmup_steps as f32)
    } else {
        // Cosine decay
        let progress = (current_step - warmup_steps) as f32 
            / (total_steps - warmup_steps) as f32;
        0.5 * base_lr * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}
```

Start small, ramp up, then decay. Prevents early instability.

## Training Loop Structure

```rust
impl Trainer {
    pub fn train(&self, model: &UNet, dataset: &[Tensor], num_epochs: usize) {
        let batches_per_epoch = dataset.len() / self.config.batch_size;
        
        for epoch in 0..num_epochs {
            let pb = ProgressBar::new(batches_per_epoch as u64);
            let mut epoch_loss = 0.0;

            for batch_idx in 0..batches_per_epoch {
                // Get batch
                let start = batch_idx * self.config.batch_size;
                let end = start + self.config.batch_size;
                let batch = self.stack_tensors(
                    &dataset[start..end].to_vec()
                );

                // Forward pass + loss
                let loss = self.train_step(model, &batch);
                epoch_loss += loss;

                // Backward pass + optimizer step
                // (requires autograd - not implemented)

                pb.inc(1);
            }

            pb.finish();
            let avg_loss = epoch_loss / batches_per_epoch as f32;
            println!("Epoch {}/{}: loss = {:.6}", epoch + 1, num_epochs, avg_loss);
        }
    }
}
```

## What Good Training Looks Like

Watch these signals:

1. **Loss decreases smoothly** - No wild spikes
2. **Loss curve has diminishing returns** - Steep early, flat later
3. **Generated samples improve** - Check periodically

Typical loss values:
- Start: 0.5 - 1.0
- After convergence: 0.01 - 0.05

## Training Tips

**Batch size**: Larger is more stable, but needs more memory. 4-16 for small GPUs, 64-256 for big ones.

**Gradient clipping**: Prevents explosion
```rust
fn clip_gradients(grads: &mut [Tensor], max_norm: f32) {
    let total_norm: f32 = grads.iter()
        .map(|g| g.square().sum())
        .sum::<f32>()
        .sqrt();
    
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for grad in grads {
            *grad = grad.mul_scalar(scale);
        }
    }
}
```

**EMA weights**: Keep exponential moving average of weights for better samples
```rust
fn update_ema(ema_params: &mut [Tensor], params: &[Tensor], decay: f32) {
    for (ema, param) in ema_params.iter_mut().zip(params.iter()) {
        *ema = ema.mul_scalar(decay).add(&param.mul_scalar(1.0 - decay));
    }
}
```

## Testing Training Components

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let a = Tensor::ones(&[10]);
        let b = Tensor::zeros(&[10]);
        let loss = mse_loss(&a, &b);
        assert!((loss - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_adam() {
        let config = TrainingConfig::default();
        let shapes: Vec<&[usize]> = vec![&[10, 5], &[5]];
        let mut optimizer = Adam::new(config, &shapes);

        let mut p1 = Tensor::randn(&[10, 5]);
        let mut p2 = Tensor::randn(&[5]);
        let g1 = Tensor::randn(&[10, 5]);
        let g2 = Tensor::randn(&[5]);

        let before = p1.sum();
        optimizer.step(&mut [&mut p1, &mut p2], &[g1, g2]);
        let after = p1.sum();
        
        assert!((before - after).abs() > 0.0);
    }

    #[test]
    fn test_lr_schedule() {
        // Cosine should be at base_lr at start
        let lr = cosine_lr_schedule(0, 1000, 1e-3, 1e-5);
        assert!((lr - 1e-3).abs() < 1e-6);
        
        // And at min_lr at end
        let lr = cosine_lr_schedule(1000, 1000, 1e-3, 1e-5);
        assert!((lr - 1e-5).abs() < 1e-6);
    }
}
```

## Reality Check

Our code shows the structure. For actual training:

1. Use a real ML framework (burn, candle, tch-rs)
2. Use GPU acceleration
3. Load real images (CIFAR-10 is a good start)
4. Train for many epochs (hundreds to thousands)

The concepts are all here. The autograd is not.

## What We Covered

- MSE loss for noise prediction
- Adam optimizer with momentum and adaptive lr
- Learning rate scheduling (cosine, warmup)
- Training loop structure
- Practical tips (gradient clipping, EMA)

## Next Up

In [Part 6: Sampling](/posts/diffusion-part6-sampling/), we generate images:
- DDPM sampling (stochastic)
- DDIM sampling (deterministic, faster)
- Saving generated images

The full code is in the `mini-diffusion` folder: [View on GitHub](https://github.com/danielsobrado/ml-animations/tree/main/mini-diffusion)

---

**Series Navigation:**
- [Part 1: Tensor Foundations](/posts/diffusion-part1-tensors/)
- [Part 2: Neural Network Layers](/posts/diffusion-part2-neural-networks/)
- [Part 3: Understanding Noise](/posts/diffusion-part3-noise/)
- [Part 4: U-Net Architecture](/posts/diffusion-part4-unet/)
- **Part 5: Training Loop** (you are here)
- [Part 6: Sampling](/posts/diffusion-part6-sampling/)
- [Part 7: Text Conditioning](/posts/diffusion-part7-conditioning/)
