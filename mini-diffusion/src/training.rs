//! Training Loop - How to train the diffusion model
//!
//! Implements the training loop with MSE loss on predicted noise,
//! Adam optimizer, and learning rate scheduling.

use crate::tensor::Tensor;
use crate::diffusion::NoiseScheduler;
use crate::unet::UNet;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub save_every: usize,
    pub beta1: f32,  // Adam momentum
    pub beta2: f32,  // Adam RMSprop
    pub eps: f32,    // Adam epsilon
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

/// Adam optimizer state for a single parameter
#[derive(Debug, Clone)]
struct AdamState {
    m: Tensor,  // First moment (momentum)
    v: Tensor,  // Second moment (RMSprop)
}

/// Simple Adam optimizer
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

    /// Update parameters using computed gradients
    /// 
    /// Adam update rule:
    /// m_t = beta1 * m_{t-1} + (1 - beta1) * g
    /// v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    /// m_hat = m_t / (1 - beta1^t)
    /// v_hat = v_t / (1 - beta2^t)
    /// param = param - lr * m_hat / (sqrt(v_hat) + eps)
    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) {
        self.step += 1;
        let t = self.step as f32;

        let bias_correction1 = 1.0 - self.config.beta1.powf(t);
        let bias_correction2 = 1.0 - self.config.beta2.powf(t);

        for ((param, grad), state) in params.iter_mut().zip(grads.iter()).zip(self.states.iter_mut()) {
            // Update momentum
            state.m = state.m.mul_scalar(self.config.beta1)
                .add(&grad.mul_scalar(1.0 - self.config.beta1));

            // Update second moment
            state.v = state.v.mul_scalar(self.config.beta2)
                .add(&grad.square().mul_scalar(1.0 - self.config.beta2));

            // Bias correction
            let m_hat = state.m.div_scalar(bias_correction1);
            let v_hat = state.v.div_scalar(bias_correction2);

            // Update parameter
            let update = m_hat.div(&v_hat.sqrt().add_scalar(self.config.eps));
            **param = param.sub(&update.mul_scalar(self.config.learning_rate));

            // Weight decay (decoupled, like AdamW)
            if self.config.weight_decay > 0.0 {
                **param = param.mul_scalar(1.0 - self.config.learning_rate * self.config.weight_decay);
            }
        }
    }
}

/// Compute mean squared error loss
pub fn mse_loss(predicted: &Tensor, target: &Tensor) -> f32 {
    let diff = predicted.sub(target);
    diff.square().mean()
}

/// Compute gradient of MSE loss (d_loss / d_predicted)
pub fn mse_loss_grad(predicted: &Tensor, target: &Tensor) -> Tensor {
    // d/dx (x - y)^2 = 2(x - y)
    let n = predicted.numel() as f32;
    predicted.sub(target).mul_scalar(2.0 / n)
}

/// Trainer for the diffusion model
pub struct Trainer {
    pub config: TrainingConfig,
    pub noise_scheduler: NoiseScheduler,
}

impl Trainer {
    pub fn new(config: TrainingConfig, noise_scheduler: NoiseScheduler) -> Self {
        Trainer {
            config,
            noise_scheduler,
        }
    }

    /// Train one batch
    /// 
    /// Returns loss value
    /// 
    /// Note: This is a simplified training step. A full implementation would need
    /// proper backpropagation through the entire network, which requires autograd.
    /// For educational purposes, we show the forward pass and loss computation.
    pub fn train_step(&self, model: &UNet, images: &Tensor) -> f32 {
        let batch_size = images.shape()[0];

        // 1. Sample random timesteps
        let timesteps = self.noise_scheduler.sample_timesteps(batch_size);

        // 2. Add noise to images
        let mut noisy_images = Vec::new();
        let mut noise_targets = Vec::new();
        
        for (i, &t) in timesteps.iter().enumerate() {
            // Extract single image (simplified)
            let img_data = images.to_vec();
            let img_size = images.numel() / batch_size;
            let start = i * img_size;
            let end = start + img_size;
            
            let mut single_img = Tensor::zeros(&images.shape()[1..]);
            single_img.as_mut_slice().copy_from_slice(&img_data[start..end]);
            
            let (noisy, noise) = self.noise_scheduler.add_noise(&single_img, t);
            noisy_images.push(noisy);
            noise_targets.push(noise);
        }

        // Combine back into batch
        let noisy_batch = self.stack_tensors(&noisy_images);
        let noise_batch = self.stack_tensors(&noise_targets);

        // 3. Predict noise using U-Net
        let predicted_noise = model.forward(&noisy_batch, &timesteps);

        // 4. Compute MSE loss between predicted and actual noise
        mse_loss(&predicted_noise, &noise_batch)
    }

    /// Stack tensors into a batch
    fn stack_tensors(&self, tensors: &[Tensor]) -> Tensor {
        if tensors.is_empty() {
            return Tensor::zeros(&[0]);
        }

        let single_shape = tensors[0].shape();
        let batch_size = tensors.len();
        let mut new_shape = vec![batch_size];
        new_shape.extend_from_slice(single_shape);

        let mut result = Tensor::zeros(&new_shape);
        let result_data = result.as_mut_slice();
        let single_size = tensors[0].numel();

        for (i, tensor) in tensors.iter().enumerate() {
            let start = i * single_size;
            let tensor_data = tensor.to_contiguous();
            result_data[start..start + single_size].copy_from_slice(tensor_data.as_slice());
        }

        result
    }

    /// Full training loop (demonstration)
    pub fn train(&self, model: &UNet, dataset: &[Tensor], num_epochs: usize) {
        let total_batches = (dataset.len() + self.config.batch_size - 1) / self.config.batch_size;
        
        for epoch in 0..num_epochs {
            let pb = ProgressBar::new(total_batches as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap());

            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            for batch_start in (0..dataset.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(dataset.len());
                let batch: Vec<_> = dataset[batch_start..batch_end].iter().collect();
                
                // Stack into batch tensor
                let batch_tensor = self.stack_tensors(&batch.iter().map(|t| (*t).clone()).collect::<Vec<_>>());

                let loss = self.train_step(model, &batch_tensor);
                epoch_loss += loss;
                num_batches += 1;

                pb.inc(1);
            }

            pb.finish();
            let avg_loss = epoch_loss / num_batches as f32;
            println!("Epoch {}/{}: avg_loss = {:.6}", epoch + 1, num_epochs, avg_loss);
        }
    }
}

/// Learning rate schedule - cosine annealing
pub fn cosine_lr_schedule(current_step: usize, total_steps: usize, base_lr: f32, min_lr: f32) -> f32 {
    let progress = current_step as f32 / total_steps as f32;
    min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
}

/// Learning rate schedule - linear warmup then cosine decay
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
        let progress = (current_step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32;
        0.5 * base_lr * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion::DiffusionConfig;

    #[test]
    fn test_adam_optimizer() {
        let config = TrainingConfig::default();
        let shapes: Vec<&[usize]> = vec![&[10, 5], &[5]];
        let mut optimizer = Adam::new(config, &shapes);

        let mut param1 = Tensor::randn(&[10, 5]);
        let mut param2 = Tensor::randn(&[5]);
        let grad1 = Tensor::randn(&[10, 5]);
        let grad2 = Tensor::randn(&[5]);

        let initial_sum = param1.sum() + param2.sum();
        
        optimizer.step(
            &mut [&mut param1, &mut param2],
            &[grad1, grad2],
        );

        let final_sum = param1.sum() + param2.sum();
        assert!((initial_sum - final_sum).abs() > 0.0);
    }

    #[test]
    fn test_mse_loss() {
        let a = Tensor::ones(&[10]);
        let b = Tensor::zeros(&[10]);
        let loss = mse_loss(&a, &b);
        assert!((loss - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lr_schedules() {
        let lr = cosine_lr_schedule(50, 100, 1e-3, 1e-5);
        assert!(lr > 1e-5 && lr < 1e-3);

        let lr = warmup_cosine_schedule(5, 10, 100, 1e-3);
        assert!((lr - 5e-4).abs() < 1e-6);
    }

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::default();
        let scheduler = NoiseScheduler::new(DiffusionConfig::default());
        let trainer = Trainer::new(config, scheduler);
        assert_eq!(trainer.config.batch_size, 4);
    }
}
