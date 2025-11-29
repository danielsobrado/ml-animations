//! DiT - Diffusion Transformer
//!
//! DiT replaces U-Net with a Transformer architecture for diffusion.
//! Key innovations:
//!
//! 1. **Patch Embedding**: Convert image to patches like ViT
//! 2. **AdaLN (Adaptive LayerNorm)**: Condition on timestep via learned scale/shift
//! 3. **Joint Attention**: For multi-modal (text+image) conditioning
//!
//! ## Why Transformer over U-Net?
//!
//! - Better scaling properties (more compute = better results)
//! - Easier to parallelize training
//! - Natural fit for multi-modal inputs
//! - SD3 and Flux use DiT variants
//!
//! ## MMDiT Architecture
//!
//! SD3 uses MMDiT (Multi-Modal DiT):
//! - Two streams: text and image
//! - Joint attention lets them interact
//! - Each has its own MLP but shared attention

use crate::tensor::Tensor;
use crate::nn::Linear;
use crate::clip::{LayerNorm, gelu, MLP};
use crate::joint_attention::JointAttention;

/// Patch Embedding
///
/// Like ViT, we convert the image/latent into patches:
/// 1. Split into non-overlapping patches
/// 2. Flatten each patch
/// 3. Linear projection to embedding dimension
///
/// For SD3 latents (H×W×4), we use 2×2 patches, giving H/2 × W/2 tokens.
pub struct PatchEmbed {
    /// Linear projection
    pub proj: Linear,
    /// Patch size
    pub patch_size: usize,
    /// Number of input channels
    pub in_channels: usize,
    /// Embedding dimension
    pub embed_dim: usize,
}

impl PatchEmbed {
    pub fn new(patch_size: usize, in_channels: usize, embed_dim: usize) -> Self {
        let patch_dim = patch_size * patch_size * in_channels;
        
        PatchEmbed {
            proj: Linear::new(patch_dim, embed_dim),
            patch_size,
            in_channels,
            embed_dim,
        }
    }
    
    /// Convert image to patch embeddings
    ///
    /// Input: [batch, height, width, channels]
    /// Output: [batch, num_patches, embed_dim]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let (batch, height, width, channels) = match shape {
            [b, h, w, c] => (*b, *h, *w, *c),
            _ => panic!("Expected [B, H, W, C] input"),
        };
        
        let num_patches_h = height / self.patch_size;
        let num_patches_w = width / self.patch_size;
        let num_patches = num_patches_h * num_patches_w;
        let patch_dim = self.patch_size * self.patch_size * channels;
        
        let x_data = x.to_vec();
        let mut patches = vec![0.0f32; batch * num_patches * patch_dim];
        
        // Extract patches
        for b in 0..batch {
            for ph in 0..num_patches_h {
                for pw in 0..num_patches_w {
                    let patch_idx = ph * num_patches_w + pw;
                    
                    for i in 0..self.patch_size {
                        for j in 0..self.patch_size {
                            for c in 0..channels {
                                let src_h = ph * self.patch_size + i;
                                let src_w = pw * self.patch_size + j;
                                let src_idx = ((b * height + src_h) * width + src_w) * channels + c;
                                
                                let dst_offset = (i * self.patch_size + j) * channels + c;
                                let dst_idx = (b * num_patches + patch_idx) * patch_dim + dst_offset;
                                
                                patches[dst_idx] = x_data[src_idx];
                            }
                        }
                    }
                }
            }
        }
        
        // Project patches
        let patches_tensor = Tensor::from_vec(patches, &[batch * num_patches, patch_dim]);
        let embedded = self.proj.forward(&patches_tensor);
        
        // Reshape to [batch, num_patches, embed_dim]
        let embed_data = embedded.to_vec();
        Tensor::from_vec(embed_data, &[batch, num_patches, self.embed_dim])
    }
    
    /// Reverse: convert patch embeddings back to image
    pub fn unpatchify(&self, x: &Tensor, height: usize, width: usize) -> Tensor {
        let shape = x.shape();
        let (batch, num_patches, _) = match shape {
            [b, n, d] => (*b, *n, *d),
            _ => panic!("Expected [B, N, D] input"),
        };
        
        let num_patches_h = height / self.patch_size;
        let num_patches_w = width / self.patch_size;
        
        // This would require a reverse projection - for now return zeros
        let output_size = batch * height * width * self.in_channels;
        Tensor::zeros(&[batch, height, width, self.in_channels])
    }
}

/// Adaptive Layer Normalization (AdaLN)
///
/// Standard LayerNorm: y = (x - mean) / std * γ + β
/// AdaLN: y = (x - mean) / std * (1 + scale) + shift
///
/// Where scale and shift are predicted from the conditioning (timestep + optional class).
/// This lets the model modulate features based on the timestep.
pub struct AdaLNModulation {
    /// Projects conditioning to scale and shift
    pub linear: Linear,
    /// Feature dimension
    pub dim: usize,
}

impl AdaLNModulation {
    pub fn new(cond_dim: usize, dim: usize) -> Self {
        // Output 2 * dim for scale and shift
        AdaLNModulation {
            linear: Linear::new(cond_dim, dim * 2),
            dim,
        }
    }
    
    /// Get scale and shift from conditioning
    pub fn forward(&self, c: &Tensor) -> (Tensor, Tensor) {
        let out = self.linear.forward(c);
        let out_data = out.to_vec();
        
        let shape = c.shape();
        let batch = shape[0];
        
        let mut scale_data = Vec::with_capacity(batch * self.dim);
        let mut shift_data = Vec::with_capacity(batch * self.dim);
        
        for b in 0..batch {
            for d in 0..self.dim {
                scale_data.push(out_data[b * self.dim * 2 + d]);
                shift_data.push(out_data[b * self.dim * 2 + self.dim + d]);
            }
        }
        
        (
            Tensor::from_vec(scale_data, &[batch, self.dim]),
            Tensor::from_vec(shift_data, &[batch, self.dim]),
        )
    }
}

/// AdaLN-Zero
///
/// Variant used in DiT: initializes the final linear layer to zero,
/// so the block initially acts as identity (residual only).
/// This helps with training stability.
pub struct AdaLNZero {
    pub norm: LayerNorm,
    pub modulation: AdaLNModulation,
}

impl AdaLNZero {
    pub fn new(dim: usize, cond_dim: usize) -> Self {
        AdaLNZero {
            norm: LayerNorm::new(dim),
            modulation: AdaLNModulation::new(cond_dim, dim),
        }
    }
    
    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Tensor {
        let normalized = self.norm.forward(x);
        let (scale, shift) = self.modulation.forward(c);
        
        // Apply: (1 + scale) * normalized + shift
        let norm_data = normalized.to_vec();
        let scale_data = scale.to_vec();
        let shift_data = shift.to_vec();
        
        let shape = x.shape();
        let (batch, seq_len, dim) = match shape {
            [b, s, d] => (*b, *s, *d),
            [s, d] => (1, *s, *d),
            _ => panic!("Expected 2D or 3D input"),
        };
        
        let mut output = vec![0.0f32; norm_data.len()];
        
        for b in 0..batch {
            for s in 0..seq_len {
                for d in 0..dim {
                    let idx = (b * seq_len + s) * dim + d;
                    let scale_val = if d < scale_data.len() / batch { 
                        scale_data[b * dim + d] 
                    } else { 
                        0.0 
                    };
                    let shift_val = if d < shift_data.len() / batch { 
                        shift_data[b * dim + d] 
                    } else { 
                        0.0 
                    };
                    output[idx] = (1.0 + scale_val) * norm_data[idx] + shift_val;
                }
            }
        }
        
        Tensor::from_vec(output, shape)
    }
}

/// SiLU (Swish) activation
pub fn silu(x: &Tensor) -> Tensor {
    let data = x.to_vec();
    let result: Vec<f32> = data.iter()
        .map(|&val| val / (1.0 + (-val).exp()))
        .collect();
    Tensor::from_vec(result, x.shape())
}

/// Timestep Embedding MLP
///
/// Converts scalar timestep to embedding, then through MLP.
pub struct TimestepEmbedder {
    pub mlp: MLP,
    pub embed_dim: usize,
}

impl TimestepEmbedder {
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        TimestepEmbedder {
            mlp: MLP::new(embed_dim, hidden_dim),
            embed_dim,
        }
    }
    
    /// Embed timesteps using sinusoidal encoding + MLP
    pub fn forward(&self, timesteps: &[f32]) -> Tensor {
        // Sinusoidal embedding
        let embeddings = crate::flow::timestep_embedding(timesteps, self.embed_dim);
        
        // Through MLP with SiLU
        let h = self.mlp.fc1.forward(&embeddings);
        let h = silu(&h);
        self.mlp.fc2.forward(&h)
    }
}

/// DiT Block (Single Stream)
///
/// For basic DiT without multi-modal:
/// 1. AdaLN
/// 2. Self-attention
/// 3. AdaLN
/// 4. MLP
pub struct DiTBlock {
    pub norm1: AdaLNZero,
    pub attn: crate::clip::CausalSelfAttention,
    pub norm2: AdaLNZero,
    pub mlp: MLP,
    pub dim: usize,
}

impl DiTBlock {
    pub fn new(dim: usize, num_heads: usize, cond_dim: usize) -> Self {
        DiTBlock {
            norm1: AdaLNZero::new(dim, cond_dim),
            attn: crate::clip::CausalSelfAttention::new(dim, num_heads),
            norm2: AdaLNZero::new(dim, cond_dim),
            mlp: MLP::new(dim, dim * 4),
            dim,
        }
    }
    
    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Tensor {
        // Self-attention with AdaLN
        let h = self.norm1.forward(x, c);
        let h = self.attn.forward(&h);
        let x = x.add(&h);
        
        // MLP with AdaLN
        let h = self.norm2.forward(&x, c);
        let h = self.mlp.forward(&h);
        x.add(&h)
    }
}

/// MMDiT Block (Multi-Modal)
///
/// SD3-style block with joint attention for text and image:
/// 1. Separate AdaLN for each modality
/// 2. Joint attention (text and image attend to each other)
/// 3. Separate MLP for each modality
pub struct MMDiTBlock {
    /// Text stream
    pub text_norm1: AdaLNZero,
    pub text_norm2: LayerNorm,
    pub text_mlp: MLP,
    /// Image stream
    pub img_norm1: AdaLNZero,
    pub img_norm2: LayerNorm,
    pub img_mlp: MLP,
    /// Shared attention
    pub joint_attn: JointAttention,
    pub dim: usize,
}

impl MMDiTBlock {
    pub fn new(dim: usize, num_heads: usize, cond_dim: usize) -> Self {
        MMDiTBlock {
            text_norm1: AdaLNZero::new(dim, cond_dim),
            text_norm2: LayerNorm::new(dim),
            text_mlp: MLP::new(dim, dim * 4),
            img_norm1: AdaLNZero::new(dim, cond_dim),
            img_norm2: LayerNorm::new(dim),
            img_mlp: MLP::new(dim, dim * 4),
            joint_attn: JointAttention::new(dim, num_heads, false),
            dim,
        }
    }
    
    /// Forward pass
    ///
    /// Args:
    ///   text: text embeddings
    ///   img: image/latent patch embeddings
    ///   c: conditioning (timestep embedding)
    pub fn forward(&self, text: &Tensor, img: &Tensor, c: &Tensor) -> (Tensor, Tensor) {
        // Pre-norm with conditioning
        let text_normed = self.text_norm1.forward(text, c);
        let img_normed = self.img_norm1.forward(img, c);
        
        // Joint attention
        let (text_attn, img_attn) = self.joint_attn.forward(&text_normed, &img_normed);
        
        // Residual
        let text = text.add(&text_attn);
        let img = img.add(&img_attn);
        
        // MLP
        let text_h = self.text_norm2.forward(&text);
        let text_h = self.text_mlp.forward(&text_h);
        let text = text.add(&text_h);
        
        let img_h = self.img_norm2.forward(&img);
        let img_h = self.img_mlp.forward(&img_h);
        let img = img.add(&img_h);
        
        (text, img)
    }
}

/// Final Layer for DiT
///
/// Projects from hidden dim back to patch output.
pub struct FinalLayer {
    pub norm: AdaLNZero,
    pub proj: Linear,
}

impl FinalLayer {
    pub fn new(dim: usize, cond_dim: usize, patch_size: usize, out_channels: usize) -> Self {
        let out_dim = patch_size * patch_size * out_channels;
        FinalLayer {
            norm: AdaLNZero::new(dim, cond_dim),
            proj: Linear::new(dim, out_dim),
        }
    }
    
    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Tensor {
        let h = self.norm.forward(x, c);
        self.proj.forward(&h)
    }
}

/// Full DiT Model
///
/// Single-stream DiT for unconditional or class-conditional generation.
pub struct DiT {
    pub patch_embed: PatchEmbed,
    pub time_embed: TimestepEmbedder,
    pub blocks: Vec<DiTBlock>,
    pub final_layer: FinalLayer,
    pub pos_embed: Tensor, // Learnable position embeddings
    pub dim: usize,
    pub patch_size: usize,
    pub in_channels: usize,
}

impl DiT {
    pub fn new(
        in_channels: usize,
        patch_size: usize,
        dim: usize,
        num_layers: usize,
        num_heads: usize,
        input_size: usize, // Spatial dimension (e.g., 32 for 32x32 latent)
    ) -> Self {
        let num_patches = (input_size / patch_size).pow(2);
        
        DiT {
            patch_embed: PatchEmbed::new(patch_size, in_channels, dim),
            time_embed: TimestepEmbedder::new(dim, dim * 4),
            blocks: (0..num_layers)
                .map(|_| DiTBlock::new(dim, num_heads, dim))
                .collect(),
            final_layer: FinalLayer::new(dim, dim, patch_size, in_channels),
            pos_embed: Tensor::randn(&[1, num_patches, dim]).mul_scalar(0.02),
            dim,
            patch_size,
            in_channels,
        }
    }
    
    /// Forward pass
    ///
    /// Args:
    ///   x: [batch, height, width, channels] - noisy latent
    ///   t: timesteps
    ///
    /// Returns:
    ///   velocity prediction (same shape as x)
    pub fn forward(&self, x: &Tensor, timesteps: &[f32]) -> Tensor {
        let shape = x.shape();
        let (batch, height, width, _channels) = match shape {
            [b, h, w, c] => (*b, *h, *w, *c),
            _ => panic!("Expected [B, H, W, C] input"),
        };
        
        // Embed patches
        let mut h = self.patch_embed.forward(x);
        
        // Add position embeddings (broadcast to batch)
        let pos_data = self.pos_embed.to_vec();
        let h_data = h.to_vec();
        let h_shape = h.shape();
        let num_patches = h_shape[1];
        
        let mut h_with_pos = h_data.clone();
        for b in 0..batch {
            for p in 0..num_patches {
                for d in 0..self.dim {
                    let idx = (b * num_patches + p) * self.dim + d;
                    let pos_idx = p * self.dim + d;
                    h_with_pos[idx] += pos_data[pos_idx];
                }
            }
        }
        h = Tensor::from_vec(h_with_pos, h.shape());
        
        // Timestep embedding
        let c = self.time_embed.forward(timesteps);
        
        // DiT blocks
        for block in &self.blocks {
            h = block.forward(&h, &c);
        }
        
        // Final layer
        h = self.final_layer.forward(&h, &c);
        
        // Unpatchify back to image shape
        self.unpatchify(&h, height, width)
    }
    
    fn unpatchify(&self, x: &Tensor, height: usize, width: usize) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let num_patches = shape[1];
        let patch_dim = self.patch_size * self.patch_size * self.in_channels;
        
        let num_patches_h = height / self.patch_size;
        let num_patches_w = width / self.patch_size;
        
        let x_data = x.to_vec();
        let mut output = vec![0.0f32; batch * height * width * self.in_channels];
        
        for b in 0..batch {
            for ph in 0..num_patches_h {
                for pw in 0..num_patches_w {
                    let patch_idx = ph * num_patches_w + pw;
                    
                    for i in 0..self.patch_size {
                        for j in 0..self.patch_size {
                            for c in 0..self.in_channels {
                                let dst_h = ph * self.patch_size + i;
                                let dst_w = pw * self.patch_size + j;
                                let dst_idx = ((b * height + dst_h) * width + dst_w) * self.in_channels + c;
                                
                                let src_offset = (i * self.patch_size + j) * self.in_channels + c;
                                let src_idx = (b * num_patches + patch_idx) * patch_dim + src_offset;
                                
                                if src_idx < x_data.len() {
                                    output[dst_idx] = x_data[src_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Tensor::from_vec(output, &[batch, height, width, self.in_channels])
    }
}

/// MMDiT Model (Multi-Modal DiT)
///
/// SD3-style architecture with:
/// - Separate patch embedding for image
/// - Text embeddings from CLIP/T5
/// - Joint attention blocks
pub struct MMDiT {
    pub patch_embed: PatchEmbed,
    pub time_embed: TimestepEmbedder,
    pub text_proj: Linear, // Project text embeddings to model dim
    pub blocks: Vec<MMDiTBlock>,
    pub final_layer: FinalLayer,
    pub pos_embed: Tensor,
    pub dim: usize,
    pub patch_size: usize,
    pub in_channels: usize,
}

impl MMDiT {
    pub fn new(
        in_channels: usize,
        patch_size: usize,
        dim: usize,
        num_layers: usize,
        num_heads: usize,
        text_dim: usize, // Dimension of text embeddings
        input_size: usize,
    ) -> Self {
        let num_patches = (input_size / patch_size).pow(2);
        
        MMDiT {
            patch_embed: PatchEmbed::new(patch_size, in_channels, dim),
            time_embed: TimestepEmbedder::new(dim, dim * 4),
            text_proj: Linear::new(text_dim, dim),
            blocks: (0..num_layers)
                .map(|_| MMDiTBlock::new(dim, num_heads, dim))
                .collect(),
            final_layer: FinalLayer::new(dim, dim, patch_size, in_channels),
            pos_embed: Tensor::randn(&[1, num_patches, dim]).mul_scalar(0.02),
            dim,
            patch_size,
            in_channels,
        }
    }
    
    /// Forward pass
    ///
    /// Args:
    ///   x: noisy latent [batch, height, width, channels]
    ///   text: text embeddings [batch, seq_len, text_dim]
    ///   timesteps: current timesteps
    pub fn forward(&self, x: &Tensor, text: &Tensor, timesteps: &[f32]) -> Tensor {
        let shape = x.shape();
        let (batch, height, width, _) = match shape {
            [b, h, w, c] => (*b, *h, *w, *c),
            _ => panic!("Expected [B, H, W, C] input"),
        };
        
        // Embed patches
        let mut img = self.patch_embed.forward(x);
        let img_shape = img.shape();
        let num_patches = img_shape[1];
        
        // Add position embeddings
        let pos_data = self.pos_embed.to_vec();
        let img_data = img.to_vec();
        let mut img_with_pos = img_data.clone();
        for b in 0..batch {
            for p in 0..num_patches {
                for d in 0..self.dim {
                    let idx = (b * num_patches + p) * self.dim + d;
                    let pos_idx = p * self.dim + d;
                    img_with_pos[idx] += pos_data[pos_idx];
                }
            }
        }
        img = Tensor::from_vec(img_with_pos, img.shape());
        
        // Project text to model dimension
        let mut text_h = self.text_proj.forward(text);
        
        // Timestep embedding
        let c = self.time_embed.forward(timesteps);
        
        // MMDiT blocks
        for block in &self.blocks {
            let (text_out, img_out) = block.forward(&text_h, &img, &c);
            text_h = text_out;
            img = img_out;
        }
        
        // Final layer (only for image)
        let h = self.final_layer.forward(&img, &c);
        
        // Unpatchify
        self.unpatchify(&h, height, width)
    }
    
    fn unpatchify(&self, x: &Tensor, height: usize, width: usize) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let num_patches = shape[1];
        let patch_dim = self.patch_size * self.patch_size * self.in_channels;
        
        let num_patches_h = height / self.patch_size;
        let num_patches_w = width / self.patch_size;
        
        let x_data = x.to_vec();
        let mut output = vec![0.0f32; batch * height * width * self.in_channels];
        
        for b in 0..batch {
            for ph in 0..num_patches_h {
                for pw in 0..num_patches_w {
                    let patch_idx = ph * num_patches_w + pw;
                    
                    for i in 0..self.patch_size {
                        for j in 0..self.patch_size {
                            for c in 0..self.in_channels {
                                let dst_h = ph * self.patch_size + i;
                                let dst_w = pw * self.patch_size + j;
                                let dst_idx = ((b * height + dst_h) * width + dst_w) * self.in_channels + c;
                                
                                let src_offset = (i * self.patch_size + j) * self.in_channels + c;
                                let src_idx = (b * num_patches + patch_idx) * patch_dim + src_offset;
                                
                                if src_idx < x_data.len() {
                                    output[dst_idx] = x_data[src_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Tensor::from_vec(output, &[batch, height, width, self.in_channels])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_patch_embed() {
        let embed = PatchEmbed::new(2, 4, 64);
        let x = Tensor::randn(&[1, 8, 8, 4]); // 8x8 image with 4 channels
        let patches = embed.forward(&x);
        
        // 8/2 * 8/2 = 16 patches
        assert_eq!(patches.shape(), &[1, 16, 64]);
    }
    
    #[test]
    fn test_dit_block() {
        let block = DiTBlock::new(64, 4, 64);
        let x = Tensor::randn(&[1, 16, 64]);
        let c = Tensor::randn(&[1, 64]);
        
        let y = block.forward(&x, &c);
        assert_eq!(y.shape(), &[1, 16, 64]);
    }
    
    #[test]
    fn test_dit() {
        let dit = DiT::new(4, 2, 64, 2, 4, 8);
        
        let x = Tensor::randn(&[1, 8, 8, 4]);
        let t = vec![0.5];
        
        let output = dit.forward(&x, &t);
        assert_eq!(output.shape(), &[1, 8, 8, 4]);
    }
    
    #[test]
    fn test_mmdit_block() {
        let block = MMDiTBlock::new(64, 4, 64);
        
        let text = Tensor::randn(&[1, 8, 64]);
        let img = Tensor::randn(&[1, 16, 64]);
        let c = Tensor::randn(&[1, 64]);
        
        let (text_out, img_out) = block.forward(&text, &img, &c);
        
        assert_eq!(text_out.shape(), &[1, 8, 64]);
        assert_eq!(img_out.shape(), &[1, 16, 64]);
    }
}
