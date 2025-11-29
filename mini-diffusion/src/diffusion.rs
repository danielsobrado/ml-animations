//! Diffusion Process - The heart of diffusion models
//!
//! Implements the forward diffusion (adding noise) and provides utilities
//! for the reverse process (removing noise).

use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Configuration for the diffusion process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionConfig {
    /// Number of diffusion timesteps
    pub num_timesteps: usize,
    /// Starting noise level (beta_start)
    pub beta_start: f32,
    /// Ending noise level (beta_end)
    pub beta_end: f32,
    /// Schedule type: "linear", "cosine", "quadratic"
    pub schedule: String,
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

/// Noise Scheduler - handles adding noise at different timesteps
#[derive(Debug, Clone)]
pub struct NoiseScheduler {
    pub config: DiffusionConfig,
    /// Noise schedule: beta_t for each timestep
    pub betas: Vec<f32>,
    /// Alpha = 1 - beta
    pub alphas: Vec<f32>,
    /// Cumulative product of alphas: alpha_bar_t = prod(alpha_1...alpha_t)
    pub alphas_cumprod: Vec<f32>,
    /// sqrt(alpha_bar_t)
    pub sqrt_alphas_cumprod: Vec<f32>,
    /// sqrt(1 - alpha_bar_t)
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,
}

impl NoiseScheduler {
    /// Create a new noise scheduler with given config
    pub fn new(config: DiffusionConfig) -> Self {
        let betas = Self::get_betas(&config);
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        
        // Compute cumulative product of alphas
        let mut alphas_cumprod = Vec::with_capacity(config.num_timesteps);
        let mut cumprod = 1.0;
        for &alpha in &alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }
        
        let sqrt_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|a| a.sqrt()).collect();
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

    /// Generate beta schedule based on config
    fn get_betas(config: &DiffusionConfig) -> Vec<f32> {
        let t = config.num_timesteps;
        
        match config.schedule.as_str() {
            "linear" => {
                // Linear schedule from beta_start to beta_end
                (0..t)
                    .map(|i| {
                        config.beta_start + (config.beta_end - config.beta_start) * (i as f32) / (t as f32 - 1.0)
                    })
                    .collect()
            }
            "cosine" => {
                // Cosine schedule - smoother, often works better
                let s = 0.008; // offset to prevent beta from being too small at t=0
                let max_beta = 0.999;
                
                (0..t)
                    .map(|i| {
                        let t1 = i as f32 / t as f32;
                        let t2 = (i + 1) as f32 / t as f32;
                        let alpha_bar_t1 = ((t1 + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos().powi(2);
                        let alpha_bar_t2 = ((t2 + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos().powi(2);
                        (1.0 - alpha_bar_t2 / alpha_bar_t1).min(max_beta)
                    })
                    .collect()
            }
            "quadratic" => {
                // Quadratic schedule
                (0..t)
                    .map(|i| {
                        let fraction = i as f32 / (t as f32 - 1.0);
                        config.beta_start + (config.beta_end - config.beta_start) * fraction * fraction
                    })
                    .collect()
            }
            _ => panic!("Unknown schedule: {}", config.schedule),
        }
    }

    /// Add noise to data at timestep t (forward diffusion)
    /// 
    /// q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    /// 
    /// Returns: (noisy_data, noise)
    pub fn add_noise(&self, x0: &Tensor, t: usize) -> (Tensor, Tensor) {
        let noise = Tensor::randn(x0.shape());
        let noisy = self.add_noise_with(&x0, &noise, t);
        (noisy, noise)
    }

    /// Add specific noise to data at timestep t
    pub fn add_noise_with(&self, x0: &Tensor, noise: &Tensor, t: usize) -> Tensor {
        let sqrt_alpha = self.sqrt_alphas_cumprod[t];
        let sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t];
        
        // x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x0.mul_scalar(sqrt_alpha).add(&noise.mul_scalar(sqrt_one_minus_alpha))
    }

    /// Sample random timesteps for training batch
    pub fn sample_timesteps(&self, batch_size: usize) -> Vec<usize> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..batch_size)
            .map(|_| rng.gen_range(0..self.config.num_timesteps))
            .collect()
    }

    /// Compute single denoising step (for inference)
    /// 
    /// Given x_t and predicted noise, compute x_{t-1}
    pub fn step(&self, x_t: &Tensor, predicted_noise: &Tensor, t: usize) -> Tensor {
        let beta_t = self.betas[t];
        let alpha_t = self.alphas[t];
        let _alpha_bar_t = self.alphas_cumprod[t];
        
        // x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * predicted_noise) + sigma_t * z
        let coef1 = 1.0 / alpha_t.sqrt();
        let coef2 = beta_t / self.sqrt_one_minus_alphas_cumprod[t];
        
        // Calculate mean (used for understanding, kept for documentation)
        let mean = x_t.sub(&predicted_noise.mul_scalar(coef2)).mul_scalar(coef1);
        
        if t == 0 {
            // No noise at final step
            mean
        } else {
            // Add noise proportional to beta_t
            let sigma_t = beta_t.sqrt();
            let noise = Tensor::randn(x_t.shape());
            mean.add(&noise.mul_scalar(sigma_t))
        }
    }

    /// Compute posterior mean and variance for DDPM sampling
    pub fn get_posterior_params(&self, t: usize) -> (f32, f32, f32) {
        let beta_t = self.betas[t];
        let alpha_t = self.alphas[t];
        let alpha_bar_t = self.alphas_cumprod[t];
        let alpha_bar_t_minus_1 = if t > 0 { self.alphas_cumprod[t - 1] } else { 1.0 };
        
        // Posterior variance
        let posterior_variance = beta_t * (1.0 - alpha_bar_t_minus_1) / (1.0 - alpha_bar_t);
        
        // Coefficients for mean
        let coef_x0 = (beta_t * alpha_bar_t_minus_1.sqrt()) / (1.0 - alpha_bar_t);
        let coef_xt = ((1.0 - alpha_bar_t_minus_1) * alpha_t.sqrt()) / (1.0 - alpha_bar_t);
        
        (posterior_variance, coef_x0, coef_xt)
    }
}

/// Get timestep embedding for conditioning the model
/// 
/// Uses sinusoidal embeddings similar to Transformer positional encodings
pub fn get_timestep_embedding(timesteps: &[usize], embedding_dim: usize) -> Tensor {
    let half_dim = embedding_dim / 2;
    let emb_factor = -(10000.0_f32.ln()) / (half_dim as f32 - 1.0);
    
    let mut embeddings = Tensor::zeros(&[timesteps.len(), embedding_dim]);
    let emb_data = embeddings.as_mut_slice();
    
    for (batch_idx, &t) in timesteps.iter().enumerate() {
        for i in 0..half_dim {
            let freq = (i as f32 * emb_factor).exp();
            let angle = t as f32 * freq;
            emb_data[batch_idx * embedding_dim + i] = angle.sin();
            emb_data[batch_idx * embedding_dim + half_dim + i] = angle.cos();
        }
    }
    
    embeddings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_scheduler_creation() {
        let config = DiffusionConfig::default();
        let scheduler = NoiseScheduler::new(config);
        
        assert_eq!(scheduler.betas.len(), 1000);
        assert_eq!(scheduler.alphas_cumprod.len(), 1000);
        
        // Alpha_bar should decrease over time
        assert!(scheduler.alphas_cumprod[0] > scheduler.alphas_cumprod[999]);
    }

    #[test]
    fn test_add_noise() {
        let scheduler = NoiseScheduler::new(DiffusionConfig::default());
        let x = Tensor::ones(&[1, 3, 32, 32]);
        
        // At t=0, should be mostly original image
        let (noisy, _) = scheduler.add_noise(&x, 0);
        assert!((noisy.mean() - 1.0).abs() < 0.5);
        
        // At t=999, should be mostly noise
        let (noisy, _) = scheduler.add_noise(&x, 999);
        assert!(noisy.mean().abs() < 1.0);
    }

    #[test]
    fn test_timestep_embedding() {
        let timesteps = vec![0, 100, 500, 999];
        let embeddings = get_timestep_embedding(&timesteps, 128);
        
        assert_eq!(embeddings.shape(), &[4, 128]);
    }

    #[test]
    fn test_cosine_schedule() {
        let config = DiffusionConfig {
            schedule: "cosine".to_string(),
            ..Default::default()
        };
        let scheduler = NoiseScheduler::new(config);
        
        // Cosine schedule should be smoother
        assert!(scheduler.alphas_cumprod[0] > 0.99);
    }
}
