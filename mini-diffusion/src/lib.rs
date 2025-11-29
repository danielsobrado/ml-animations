//! Mini Diffusion - A minimal diffusion model from scratch in Rust
//!
//! This library implements a small diffusion model similar to Flux/Stable Diffusion,
//! built entirely from scratch for educational purposes.
//!
//! ## Architecture Overview
//!
//! This library implements SD3/Flux-style components:
//!
//! - **VAE**: Variational Autoencoder for latent space compression
//! - **Tokenizers**: BPE (for CLIP) and Unigram (for T5)
//! - **CLIP**: Text encoder with causal attention
//! - **T5**: Text encoder with relative position bias
//! - **DiT**: Diffusion Transformer (replaces U-Net)
//! - **Joint Attention**: Multi-modal attention for text+image
//! - **Flow Matching**: Modern training/sampling approach
//!
//! ## Basic Example
//!
//! ```ignore
//! use mini_diffusion::*;
//!
//! // Create models
//! let vae = vae::VAE::new(3, 4, 128, 4);
//! let dit = dit::DiT::new(4, 2, 256, 6, 8, 32);
//! let scheduler = flow::FlowMatchingScheduler::new(20);
//!
//! // Sample
//! let noise = Tensor::randn(&[1, 32, 32, 4]);
//! // ... inference loop with dit.forward() and solver.step()
//! let latent = noise; // after sampling
//! let image = vae.decode(&latent);
//! ```

// Core modules
pub mod tensor;
pub mod nn;

// Original DDPM-style modules
pub mod diffusion;
pub mod unet;
pub mod training;
pub mod sampling;

// SD3/Flux-style modules
pub mod vae;
pub mod tokenizer;
pub mod clip;
pub mod t5;
pub mod flow;
pub mod joint_attention;
pub mod dit;

// Re-exports for convenience
pub use tensor::Tensor;
pub use nn::{Linear, Conv2d, GroupNorm, LayerNorm, Activation};
pub use diffusion::{NoiseScheduler, DiffusionConfig};
pub use unet::UNet;
pub use training::Trainer;
pub use sampling::Sampler;

// SD3-style re-exports
pub use vae::VAE;
pub use tokenizer::{BPETokenizer, UnigramTokenizer};
pub use clip::CLIPTextEncoder;
pub use t5::T5TextEncoder;
pub use flow::{FlowMatchingScheduler, EulerSolver};
pub use joint_attention::JointAttention;
pub use dit::{DiT, MMDiT};
