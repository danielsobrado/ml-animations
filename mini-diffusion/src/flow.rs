//! Flow Matching and Euler Scheduler
//!
//! Flow Matching is a newer approach to diffusion that models the process as an ODE
//! (ordinary differential equation) rather than an SDE (stochastic differential equation).
//!
//! ## Why Flow Matching?
//!
//! Traditional diffusion (DDPM) adds noise gradually and learns to reverse it.
//! Flow matching instead learns the velocity field that transforms noise to data.
//!
//! Benefits:
//! - Simpler training objective (velocity prediction vs noise prediction)
//! - Can use any ODE solver (deterministic)
//! - Fewer sampling steps needed
//! - More stable training
//!
//! ## The Flow Matching Framework
//!
//! We define a continuous path from noise (t=0) to data (t=1):
//!   x_t = (1 - t) * noise + t * data     (linear interpolation)
//!
//! The velocity (or "flow") is:
//!   v_t = d(x_t)/dt = data - noise
//!
//! We train a neural network to predict this velocity given x_t and t.

use crate::tensor::Tensor;

/// Logit-Normal Distribution for Timestep Sampling
///
/// Instead of uniform sampling t ~ U(0,1), SD3 uses logit-normal:
///   u ~ N(mean, std)
///   t = sigmoid(u)
///
/// This focuses training on intermediate timesteps where learning is hardest.
/// Early (t≈0) and late (t≈1) timesteps are easier.
pub struct LogitNormalSampler {
    /// Mean of the normal distribution (in logit space)
    pub mean: f32,
    /// Standard deviation of the normal distribution
    pub std: f32,
}

impl LogitNormalSampler {
    pub fn new(mean: f32, std: f32) -> Self {
        LogitNormalSampler { mean, std }
    }
    
    /// Default configuration (SD3 uses mean=0, std=1)
    pub fn default() -> Self {
        Self::new(0.0, 1.0)
    }
    
    /// Sample a single timestep
    pub fn sample(&self) -> f32 {
        // Sample from standard normal
        let u: f32 = Self::sample_normal();
        // Transform: u * std + mean
        let logit = u * self.std + self.mean;
        // Sigmoid to get t in (0, 1)
        1.0 / (1.0 + (-logit).exp())
    }
    
    /// Sample batch of timesteps
    pub fn sample_batch(&self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.sample()).collect()
    }
    
    /// Box-Muller transform for normal sampling
    fn sample_normal() -> f32 {
        use std::f32::consts::PI;
        let u1: f32 = rand::random::<f32>().max(1e-10);
        let u2: f32 = rand::random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Sigma (noise level) Schedule for Flow Matching
///
/// Maps timestep t to sigma (noise scale).
/// Different schedules work better for different applications.
pub enum SigmaSchedule {
    /// Linear: sigma = 1 - t (simple, SD3-style)
    Linear,
    /// Cosine: smoother noise schedule
    Cosine,
    /// Exponential: sigma = exp(-t * scale)
    Exponential { scale: f32 },
    /// Karras schedule (used in EDM)
    Karras { sigma_min: f32, sigma_max: f32, rho: f32 },
}

impl SigmaSchedule {
    /// Get sigma at timestep t (where t goes from 0 to 1, or 0 to num_steps)
    pub fn sigma(&self, t: f32) -> f32 {
        match self {
            SigmaSchedule::Linear => 1.0 - t,
            SigmaSchedule::Cosine => {
                use std::f32::consts::PI;
                (t * PI / 2.0).cos()
            }
            SigmaSchedule::Exponential { scale } => (-t * scale).exp(),
            SigmaSchedule::Karras { sigma_min, sigma_max, rho } => {
                let inv_rho = 1.0 / rho;
                (sigma_max.powf(inv_rho) + t * (sigma_min.powf(inv_rho) - sigma_max.powf(inv_rho))).powf(*rho)
            }
        }
    }
    
    /// Get sigmas for a given number of steps
    pub fn get_sigmas(&self, num_steps: usize) -> Vec<f32> {
        (0..=num_steps)
            .map(|i| {
                let t = i as f32 / num_steps as f32;
                self.sigma(t)
            })
            .collect()
    }
}

/// Flow Matching Noise Scheduler
///
/// Implements the core flow matching interpolation:
///   x_t = (1 - t) * noise + t * data
///
/// Or with sigma notation:
///   x_t = sigma_t * noise + (1 - sigma_t) * data
pub struct FlowMatchingScheduler {
    /// Sigma schedule
    pub schedule: SigmaSchedule,
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Precomputed sigmas for inference
    pub sigmas: Vec<f32>,
    /// Timestep sampler for training
    pub timestep_sampler: LogitNormalSampler,
}

impl FlowMatchingScheduler {
    pub fn new(num_inference_steps: usize) -> Self {
        let schedule = SigmaSchedule::Linear;
        let sigmas = schedule.get_sigmas(num_inference_steps);
        
        FlowMatchingScheduler {
            schedule,
            num_inference_steps,
            sigmas,
            timestep_sampler: LogitNormalSampler::default(),
        }
    }
    
    pub fn with_schedule(num_inference_steps: usize, schedule: SigmaSchedule) -> Self {
        let sigmas = schedule.get_sigmas(num_inference_steps);
        
        FlowMatchingScheduler {
            schedule,
            num_inference_steps,
            sigmas,
            timestep_sampler: LogitNormalSampler::default(),
        }
    }
    
    /// Add noise to data at timestep t
    /// 
    /// x_t = sigma * noise + (1 - sigma) * data
    pub fn add_noise(&self, data: &Tensor, noise: &Tensor, sigma: f32) -> Tensor {
        let scaled_noise = noise.mul_scalar(sigma);
        let scaled_data = data.mul_scalar(1.0 - sigma);
        scaled_noise.add(&scaled_data)
    }
    
    /// Compute velocity target for training
    /// 
    /// In flow matching, the target is the velocity:
    /// v = (data - x_t) / (1 - sigma)  or simply  v = data - noise
    pub fn get_velocity(&self, data: &Tensor, noise: &Tensor) -> Tensor {
        data.sub(noise)
    }
    
    /// Sample timesteps for training
    pub fn sample_timesteps(&self, batch_size: usize) -> Vec<f32> {
        self.timestep_sampler.sample_batch(batch_size)
    }
    
    /// Get sigma at step index (for inference)
    pub fn sigma_at_step(&self, step: usize) -> f32 {
        self.sigmas[step.min(self.sigmas.len() - 1)]
    }
}

/// Euler ODE Solver
///
/// The simplest ODE solver. Given:
///   dx/dt = velocity(x, t)
///
/// Euler step:
///   x_{t+dt} = x_t + dt * velocity(x_t, t)
///
/// For flow matching, we step from noise (t=0) to data (t=1).
/// The model predicts velocity, and we integrate using Euler steps.
pub struct EulerSolver {
    /// Number of steps
    pub num_steps: usize,
    /// Scheduler for sigma values
    pub scheduler: FlowMatchingScheduler,
}

impl EulerSolver {
    pub fn new(num_steps: usize) -> Self {
        EulerSolver {
            num_steps,
            scheduler: FlowMatchingScheduler::new(num_steps),
        }
    }
    
    /// Perform one Euler step
    /// 
    /// x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * velocity
    /// 
    /// Note: sigma decreases (1 → 0), so sigma_{i+1} < sigma_i,
    /// making (sigma_{i+1} - sigma_i) negative, which is correct
    /// because velocity points from noise to data.
    pub fn step(&self, x: &Tensor, velocity: &Tensor, step_idx: usize) -> Tensor {
        let sigma_curr = self.scheduler.sigma_at_step(step_idx);
        let sigma_next = self.scheduler.sigma_at_step(step_idx + 1);
        
        let dt = sigma_next - sigma_curr;
        
        // Euler: x_new = x + dt * v
        let dx = velocity.mul_scalar(dt);
        x.add(&dx)
    }
    
    /// Get current timestep for model input
    /// 
    /// The model needs to know "where" in the process we are.
    /// We pass sigma directly or convert to a different scale.
    pub fn get_timestep(&self, step_idx: usize) -> f32 {
        self.scheduler.sigma_at_step(step_idx)
    }
}

/// Euler-Maruyama Solver (with noise)
///
/// Adds stochasticity to Euler solver for potentially better results.
/// Used in some diffusion implementations for diversity.
///
/// x_{i+1} = x_i + dt * velocity + sqrt(2 * dt) * noise
pub struct EulerMaruyamaSolver {
    pub euler: EulerSolver,
    /// Amount of noise to add (0 = deterministic Euler)
    pub eta: f32,
}

impl EulerMaruyamaSolver {
    pub fn new(num_steps: usize, eta: f32) -> Self {
        EulerMaruyamaSolver {
            euler: EulerSolver::new(num_steps),
            eta,
        }
    }
    
    /// Step with optional noise injection
    pub fn step(&self, x: &Tensor, velocity: &Tensor, step_idx: usize) -> Tensor {
        // First do deterministic Euler step
        let x_euler = self.euler.step(x, velocity, step_idx);
        
        if self.eta <= 0.0 {
            return x_euler;
        }
        
        // Add noise
        let sigma_curr = self.euler.scheduler.sigma_at_step(step_idx);
        let sigma_next = self.euler.scheduler.sigma_at_step(step_idx + 1);
        let dt = (sigma_curr - sigma_next).abs();
        
        let noise = Tensor::randn(x.shape());
        let noise_scale = self.eta * (2.0 * dt).sqrt();
        
        x_euler.add(&noise.mul_scalar(noise_scale))
    }
}

/// Heun's Method (Improved Euler)
///
/// Second-order ODE solver for better accuracy:
/// 
/// 1. Euler step to get prediction: x̃ = x + dt * v(x, t)
/// 2. Evaluate at prediction: ṽ = model(x̃, t+dt)
/// 3. Average: x_new = x + dt * (v + ṽ) / 2
///
/// Requires 2 model evaluations per step but is more accurate.
pub struct HeunSolver {
    pub num_steps: usize,
    pub scheduler: FlowMatchingScheduler,
}

impl HeunSolver {
    pub fn new(num_steps: usize) -> Self {
        HeunSolver {
            num_steps,
            scheduler: FlowMatchingScheduler::new(num_steps),
        }
    }
    
    /// First half of Heun step: get preliminary next state
    pub fn step_first_half(&self, x: &Tensor, velocity: &Tensor, step_idx: usize) -> Tensor {
        let sigma_curr = self.scheduler.sigma_at_step(step_idx);
        let sigma_next = self.scheduler.sigma_at_step(step_idx + 1);
        let dt = sigma_next - sigma_curr;
        
        x.add(&velocity.mul_scalar(dt))
    }
    
    /// Second half of Heun step: correct with average velocity
    pub fn step_second_half(
        &self, 
        x: &Tensor, 
        velocity_curr: &Tensor, 
        velocity_next: &Tensor,
        step_idx: usize
    ) -> Tensor {
        let sigma_curr = self.scheduler.sigma_at_step(step_idx);
        let sigma_next = self.scheduler.sigma_at_step(step_idx + 1);
        let dt = sigma_next - sigma_curr;
        
        // Average velocity
        let avg_velocity = velocity_curr.add(velocity_next).mul_scalar(0.5);
        
        x.add(&avg_velocity.mul_scalar(dt))
    }
}

/// Timestep Embedding
///
/// Convert scalar timestep to embedding vector for the model.
/// Uses sinusoidal encoding similar to positional encoding.
pub fn timestep_embedding(timesteps: &[f32], dim: usize) -> Tensor {
    let half_dim = dim / 2;
    let max_period = 10000.0f32;
    
    let mut embeddings = Vec::with_capacity(timesteps.len() * dim);
    
    for &t in timesteps {
        for i in 0..half_dim {
            let freq = (-((i as f32) / half_dim as f32) * max_period.ln()).exp();
            embeddings.push((t * freq).sin());
            embeddings.push((t * freq).cos());
        }
    }
    
    Tensor::from_vec(embeddings, &[timesteps.len(), dim])
}

/// Compute training loss for flow matching
///
/// Unlike DDPM which predicts noise, flow matching predicts velocity.
/// Loss = MSE(predicted_velocity, target_velocity)
pub fn flow_matching_loss(predicted: &Tensor, target: &Tensor) -> f32 {
    let diff = predicted.sub(target);
    let diff_data = diff.to_vec();
    
    diff_data.iter().map(|&x| x * x).sum::<f32>() / diff_data.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_logit_normal() {
        let sampler = LogitNormalSampler::default();
        let samples: Vec<f32> = (0..100).map(|_| sampler.sample()).collect();
        
        // All samples should be in (0, 1)
        for &s in &samples {
            assert!(s > 0.0 && s < 1.0);
        }
    }
    
    #[test]
    fn test_sigma_schedules() {
        let linear = SigmaSchedule::Linear;
        assert!((linear.sigma(0.0) - 1.0).abs() < 0.01);
        assert!((linear.sigma(1.0) - 0.0).abs() < 0.01);
        assert!((linear.sigma(0.5) - 0.5).abs() < 0.01);
        
        let cosine = SigmaSchedule::Cosine;
        assert!((cosine.sigma(0.0) - 1.0).abs() < 0.01);
        assert!(cosine.sigma(1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_add_noise() {
        let scheduler = FlowMatchingScheduler::new(10);
        
        let data = Tensor::ones(&[2, 2]);
        let noise = Tensor::zeros(&[2, 2]);
        
        // At sigma=0, should get pure data
        let noisy = scheduler.add_noise(&data, &noise, 0.0);
        let noisy_vec = noisy.to_vec();
        assert!((noisy_vec[0] - 1.0).abs() < 0.01);
        
        // At sigma=1, should get pure noise
        let noisy = scheduler.add_noise(&data, &noise, 1.0);
        let noisy_vec = noisy.to_vec();
        assert!(noisy_vec[0].abs() < 0.01);
    }
    
    #[test]
    fn test_euler_step() {
        let solver = EulerSolver::new(10);
        
        // Simple test: constant velocity should move in that direction
        let x = Tensor::zeros(&[2]);
        let velocity = Tensor::ones(&[2]);
        
        let x_next = solver.step(&x, &velocity, 0);
        let x_vec = x_next.to_vec();
        
        // Should have moved in velocity direction
        // (exact value depends on sigma schedule)
        assert!(x_vec[0] != 0.0);
    }
    
    #[test]
    fn test_timestep_embedding() {
        let timesteps = vec![0.0, 0.5, 1.0];
        let embeddings = timestep_embedding(&timesteps, 16);
        
        assert_eq!(embeddings.shape(), &[3, 16]);
    }
}
