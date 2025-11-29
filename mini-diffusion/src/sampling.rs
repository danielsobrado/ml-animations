//! Sampling - Generating images from noise
//!
//! Implements DDPM and DDIM sampling for image generation.

use crate::tensor::Tensor;
use crate::diffusion::NoiseScheduler;
use crate::unet::UNet;
use indicatif::{ProgressBar, ProgressStyle};

/// Sampler configuration
pub struct SamplerConfig {
    /// Number of sampling steps (can be less than training steps for DDIM)
    pub num_steps: usize,
    /// Guidance scale for classifier-free guidance (1.0 = no guidance)
    pub guidance_scale: f32,
    /// Whether to use DDIM (deterministic) vs DDPM (stochastic)
    pub use_ddim: bool,
    /// Eta for DDIM (0 = deterministic, 1 = DDPM-like)
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

/// Image sampler
pub struct Sampler {
    pub config: SamplerConfig,
    pub noise_scheduler: NoiseScheduler,
}

impl Sampler {
    pub fn new(config: SamplerConfig, noise_scheduler: NoiseScheduler) -> Self {
        Sampler {
            config,
            noise_scheduler,
        }
    }

    /// Generate images from random noise
    /// 
    /// Args:
    ///   model: The trained U-Net
    ///   shape: Output shape [batch, channels, height, width]
    /// 
    /// Returns:
    ///   Generated images
    pub fn sample(&self, model: &UNet, shape: &[usize]) -> Tensor {
        if self.config.use_ddim {
            self.sample_ddim(model, shape)
        } else {
            self.sample_ddpm(model, shape)
        }
    }

    /// DDPM sampling (stochastic, original method)
    /// 
    /// Iteratively denoise from x_T to x_0:
    /// x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_theta) + sigma_t * z
    pub fn sample_ddpm(&self, model: &UNet, shape: &[usize]) -> Tensor {
        let batch_size = shape[0];
        let total_steps = self.noise_scheduler.config.num_timesteps;
        
        // Start from pure noise
        let mut x = Tensor::randn(shape);

        let pb = ProgressBar::new(total_steps as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} Sampling: [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap());

        // Iterate from T-1 to 0
        for t in (0..total_steps).rev() {
            let timesteps: Vec<usize> = vec![t; batch_size];
            
            // Predict noise
            let predicted_noise = model.forward(&x, &timesteps);

            // Get scheduler parameters
            let alpha_t = self.noise_scheduler.alphas[t];
            let alpha_bar_t = self.noise_scheduler.alphas_cumprod[t];
            let beta_t = self.noise_scheduler.betas[t];

            // Compute mean
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

    /// DDIM sampling (deterministic, faster)
    /// 
    /// DDIM allows skipping steps while maintaining quality.
    /// With eta=0, it's fully deterministic.
    pub fn sample_ddim(&self, model: &UNet, shape: &[usize]) -> Tensor {
        let batch_size = shape[0];
        let total_steps = self.noise_scheduler.config.num_timesteps;
        
        // Calculate step indices (evenly spaced)
        let step_size = total_steps / self.config.num_steps;
        let timestep_seq: Vec<usize> = (0..self.config.num_steps)
            .map(|i| (self.config.num_steps - 1 - i) * step_size)
            .collect();

        // Start from pure noise
        let mut x = Tensor::randn(shape);

        let pb = ProgressBar::new(self.config.num_steps as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} DDIM Sampling: [{bar:40.cyan/blue}] {pos}/{len}")
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
                1.0 // Final step
            };

            // DDIM update
            // Predict x_0
            let pred_x0 = x.sub(&predicted_noise.mul_scalar((1.0 - alpha_bar_t).sqrt()))
                .div_scalar(alpha_bar_t.sqrt());

            // Direction pointing to x_t
            let dir_xt = predicted_noise.mul_scalar((1.0 - alpha_bar_prev).sqrt());

            // Compute sigma for stochasticity (eta controls this)
            let sigma = self.config.eta * 
                ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev)).sqrt();

            // Compute x_{t-1}
            x = pred_x0.mul_scalar(alpha_bar_prev.sqrt()).add(&dir_xt);
            
            // Add noise if eta > 0
            if sigma > 0.0 && i + 1 < timestep_seq.len() {
                let noise = Tensor::randn(shape);
                x = x.add(&noise.mul_scalar(sigma));
            }

            pb.inc(1);
        }

        pb.finish_with_message("Done!");

        x.clamp(-1.0, 1.0)
    }

    /// Sample with classifier-free guidance
    /// 
    /// Requires model to be trained with unconditional dropout.
    /// prediction = uncond + guidance_scale * (cond - uncond)
    pub fn sample_with_guidance(
        &self,
        model: &UNet,
        shape: &[usize],
        _condition: Option<&Tensor>,
    ) -> Tensor {
        // For now, just do regular sampling
        // Full CFG would require conditioning input to model
        self.sample(model, shape)
    }
}

/// Convert tensor to image (0-255 uint8)
pub fn tensor_to_image(tensor: &Tensor) -> Vec<u8> {
    // Assume tensor is in [-1, 1] range
    tensor.to_vec()
        .iter()
        .map(|&v| ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8)
        .collect()
}

/// Save generated images to disk
pub fn save_images(images: &Tensor, prefix: &str) -> std::io::Result<()> {
    use image::{ImageBuffer, Rgb};
    
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

    let data = images.to_vec();

    for b in 0..batch {
        let mut img_buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width as u32, height as u32);

        for h in 0..height {
            for w in 0..width {
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
        let tensor = Tensor::zeros(&[3, 32, 32]).add_scalar(-1.0); // All black
        let img = tensor_to_image(&tensor);
        assert!(img.iter().all(|&v| v == 0));

        let tensor = Tensor::ones(&[3, 32, 32]); // All white
        let img = tensor_to_image(&tensor);
        assert!(img.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_sample_shape() {
        // This test is slow, so we use minimal settings
        let mut config = SamplerConfig::default();
        config.num_steps = 2;
        
        let scheduler = NoiseScheduler::new(DiffusionConfig {
            num_timesteps: 10,
            ..Default::default()
        });
        
        let sampler = Sampler::new(config, scheduler);
        let model = UNet::new(3, 16, 3);
        
        let output = sampler.sample(&model, &[1, 3, 16, 16]);
        assert_eq!(output.shape(), &[1, 3, 16, 16]);
    }
}
