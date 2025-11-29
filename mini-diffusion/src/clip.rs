//! CLIP Text Encoder
//!
//! CLIP (Contrastive Language-Image Pre-training) learns to align text and
//! image embeddings in a shared space. For diffusion models, we use the text
//! encoder to convert prompts into embeddings that guide image generation.
//!
//! ## Architecture
//!
//! CLIP's text encoder is a Transformer:
//! 1. Token embeddings + positional embeddings
//! 2. Multiple transformer blocks (self-attention + MLP)
//! 3. Final layer norm
//! 4. Take the [EOS] token embedding as the sentence representation
//!
//! SD3 uses the pooled output from CLIP-L and CLIP-G models.

use crate::tensor::Tensor;
use crate::nn::Linear;

/// Layer Normalization
/// 
/// Normalizes across the feature dimension: y = (x - mean) / sqrt(var + eps) * gamma + beta
/// 
/// Unlike BatchNorm, LayerNorm normalizes each sample independently,
/// making it suitable for variable-length sequences in transformers.
pub struct LayerNorm {
    pub gamma: Tensor, // Scale parameter
    pub beta: Tensor,  // Shift parameter
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(features: usize) -> Self {
        LayerNorm {
            gamma: Tensor::ones(&[features]),
            beta: Tensor::zeros(&[features]),
            eps: 1e-5,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let features = shape[shape.len() - 1];
        
        // Compute mean and variance along last dimension
        let x_data = x.to_vec();
        let mut result = vec![0.0f32; x_data.len()];
        
        let num_vectors = x_data.len() / features;
        
        for i in 0..num_vectors {
            let start = i * features;
            let end = start + features;
            let slice = &x_data[start..end];
            
            // Mean
            let mean: f32 = slice.iter().sum::<f32>() / features as f32;
            
            // Variance
            let var: f32 = slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / features as f32;
            
            let std = (var + self.eps).sqrt();
            
            // Normalize and apply scale/shift
            let gamma = self.gamma.to_vec();
            let beta = self.beta.to_vec();
            
            for j in 0..features {
                result[start + j] = (slice[j] - mean) / std * gamma[j] + beta[j];
            }
        }
        
        Tensor::from_vec(result, shape)
    }
}

/// GELU activation function
/// 
/// Gaussian Error Linear Unit: x * Phi(x) where Phi is the CDF of standard normal.
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// 
/// GELU is smoother than ReLU and used in most modern transformers.
pub fn gelu(x: &Tensor) -> Tensor {
    let data = x.to_vec();
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    
    let result: Vec<f32> = data.iter().map(|&val| {
        let inner = sqrt_2_over_pi * (val + 0.044715 * val.powi(3));
        0.5 * val * (1.0 + inner.tanh())
    }).collect();
    
    Tensor::from_vec(result, x.shape())
}

/// Quick GELU (faster approximation)
/// 
/// x * sigmoid(1.702 * x)
pub fn quick_gelu(x: &Tensor) -> Tensor {
    let data = x.to_vec();
    
    let result: Vec<f32> = data.iter().map(|&val| {
        val * (1.0 / (1.0 + (-1.702 * val).exp()))
    }).collect();
    
    Tensor::from_vec(result, x.shape())
}

/// MLP (Feed-Forward Network) in Transformer
/// 
/// Two linear layers with GELU activation:
/// output = Linear2(GELU(Linear1(x)))
/// 
/// Usually expands dimension by 4x then projects back.
pub struct MLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl MLP {
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        MLP {
            fc1: Linear::new(dim, hidden_dim),
            fc2: Linear::new(hidden_dim, dim),
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.fc1.forward(x);
        let h = gelu(&h);
        self.fc2.forward(&h)
    }
}

/// Causal Self-Attention for CLIP
/// 
/// CLIP uses causal (left-to-right) attention in the text encoder,
/// meaning each token can only attend to previous tokens.
/// This is implemented via an attention mask.
pub struct CausalSelfAttention {
    pub qkv: Linear,      // Combined Q, K, V projection
    pub proj: Linear,     // Output projection
    pub num_heads: usize,
    pub head_dim: usize,
}

impl CausalSelfAttention {
    pub fn new(dim: usize, num_heads: usize) -> Self {
        let head_dim = dim / num_heads;
        CausalSelfAttention {
            qkv: Linear::new(dim, dim * 3),
            proj: Linear::new(dim, dim),
            num_heads,
            head_dim,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let (batch_size, seq_len, dim) = match shape {
            [b, s, d] => (*b, *s, *d),
            [s, d] => (1, *s, *d),
            _ => panic!("Expected 2D or 3D input"),
        };
        
        // Project to Q, K, V
        let qkv = self.qkv.forward(x);
        let qkv_data = qkv.to_vec();
        
        // Split into Q, K, V
        let total_dim = dim * 3;
        let mut q_data = Vec::with_capacity(batch_size * seq_len * dim);
        let mut k_data = Vec::with_capacity(batch_size * seq_len * dim);
        let mut v_data = Vec::with_capacity(batch_size * seq_len * dim);
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                let idx = (b * seq_len + s) * total_dim;
                q_data.extend_from_slice(&qkv_data[idx..idx + dim]);
                k_data.extend_from_slice(&qkv_data[idx + dim..idx + 2 * dim]);
                v_data.extend_from_slice(&qkv_data[idx + 2 * dim..idx + 3 * dim]);
            }
        }
        
        let q = Tensor::from_vec(q_data, &[batch_size, seq_len, dim]);
        let k = Tensor::from_vec(k_data, &[batch_size, seq_len, dim]);
        let v = Tensor::from_vec(v_data, &[batch_size, seq_len, dim]);
        
        // Compute attention scores
        let scale = (self.head_dim as f32).sqrt();
        
        // For simplicity, compute full attention then apply causal mask
        // scores[i,j] = Q[i] · K[j] / scale
        let mut scores = vec![0.0f32; batch_size * seq_len * seq_len];
        let q_vec = q.to_vec();
        let k_vec = k.to_vec();
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    // Causal mask: only attend to j <= i
                    if j > i {
                        scores[(b * seq_len + i) * seq_len + j] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..dim {
                            let q_idx = (b * seq_len + i) * dim + d;
                            let k_idx = (b * seq_len + j) * dim + d;
                            dot += q_vec[q_idx] * k_vec[k_idx];
                        }
                        scores[(b * seq_len + i) * seq_len + j] = dot / scale;
                    }
                }
            }
        }
        
        // Softmax over last dimension
        for b in 0..batch_size {
            for i in 0..seq_len {
                let start = (b * seq_len + i) * seq_len;
                let end = start + seq_len;
                let row = &mut scores[start..end];
                
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
                
                for val in row.iter_mut() {
                    *val = (*val - max).exp() / exp_sum;
                }
            }
        }
        
        // Apply attention to values
        let v_vec = v.to_vec();
        let mut output = vec![0.0f32; batch_size * seq_len * dim];
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                for d in 0..dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq_len {
                        let attn = scores[(b * seq_len + i) * seq_len + j];
                        let v_val = v_vec[(b * seq_len + j) * dim + d];
                        sum += attn * v_val;
                    }
                    output[(b * seq_len + i) * dim + d] = sum;
                }
            }
        }
        
        let attended = Tensor::from_vec(output, &[batch_size, seq_len, dim]);
        
        // Final projection
        self.proj.forward(&attended)
    }
}

/// Transformer Block for CLIP
/// 
/// Pre-norm architecture: LayerNorm before attention and MLP
/// x = x + Attention(LayerNorm(x))
/// x = x + MLP(LayerNorm(x))
pub struct CLIPTransformerBlock {
    pub ln1: LayerNorm,
    pub attn: CausalSelfAttention,
    pub ln2: LayerNorm,
    pub mlp: MLP,
}

impl CLIPTransformerBlock {
    pub fn new(dim: usize, num_heads: usize, mlp_ratio: usize) -> Self {
        CLIPTransformerBlock {
            ln1: LayerNorm::new(dim),
            attn: CausalSelfAttention::new(dim, num_heads),
            ln2: LayerNorm::new(dim),
            mlp: MLP::new(dim, dim * mlp_ratio),
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Self-attention with residual
        let h = self.ln1.forward(x);
        let h = self.attn.forward(&h);
        let x = x.add(&h);
        
        // MLP with residual
        let h = self.ln2.forward(&x);
        let h = self.mlp.forward(&h);
        x.add(&h)
    }
}

/// CLIP Text Encoder
/// 
/// Takes tokenized text and produces embeddings that can condition image generation.
/// 
/// ## Usage in Diffusion
/// 
/// 1. Tokenize prompt: "a photo of a cat" → [49406, 320, 1125, ...]
/// 2. Embed tokens and add positions
/// 3. Pass through transformer blocks
/// 4. Extract pooled representation (EOS token) or sequence embeddings
pub struct CLIPTextEncoder {
    /// Token embeddings
    pub token_embedding: Tensor,
    /// Positional embeddings (learned, not sinusoidal)
    pub position_embedding: Tensor,
    /// Transformer blocks
    pub blocks: Vec<CLIPTransformerBlock>,
    /// Final layer norm
    pub final_layer_norm: LayerNorm,
    /// Output projection (optional, for pooled output)
    pub text_projection: Option<Linear>,
    /// Embedding dimension
    pub dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl CLIPTextEncoder {
    /// Create a new CLIP text encoder
    /// 
    /// Common configurations:
    /// - CLIP-L: dim=768, heads=12, layers=12, vocab=49408, max_len=77
    /// - CLIP-G: dim=1280, heads=20, layers=32, vocab=49408, max_len=77
    pub fn new(
        vocab_size: usize,
        dim: usize,
        num_heads: usize,
        num_layers: usize,
        max_seq_len: usize,
        projection_dim: Option<usize>,
    ) -> Self {
        // Initialize embeddings with small random values
        let token_embedding = Tensor::randn(&[vocab_size, dim]).mul_scalar(0.02);
        let position_embedding = Tensor::randn(&[max_seq_len, dim]).mul_scalar(0.01);
        
        // Create transformer blocks
        let blocks: Vec<_> = (0..num_layers)
            .map(|_| CLIPTransformerBlock::new(dim, num_heads, 4))
            .collect();
        
        let final_layer_norm = LayerNorm::new(dim);
        
        let text_projection = projection_dim.map(|proj_dim| Linear::new(dim, proj_dim));
        
        CLIPTextEncoder {
            token_embedding,
            position_embedding,
            blocks,
            final_layer_norm,
            text_projection,
            dim,
            max_seq_len,
            vocab_size,
        }
    }
    
    /// CLIP-L configuration (Large)
    pub fn clip_l() -> Self {
        Self::new(49408, 768, 12, 12, 77, Some(768))
    }
    
    /// CLIP-G configuration (Giant) - used in SD3
    pub fn clip_g() -> Self {
        Self::new(49408, 1280, 20, 32, 77, Some(1280))
    }
    
    /// Get token embeddings from indices
    fn embed_tokens(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        let mut output = vec![0.0f32; seq_len * self.dim];
        
        let embed_data = self.token_embedding.to_vec();
        
        for (i, &token_id) in token_ids.iter().enumerate() {
            let token_id = (token_id as usize).min(self.vocab_size - 1);
            let start = token_id * self.dim;
            let end = start + self.dim;
            
            for (j, &val) in embed_data[start..end].iter().enumerate() {
                output[i * self.dim + j] = val;
            }
        }
        
        Tensor::from_vec(output, &[seq_len, self.dim])
    }
    
    /// Add positional embeddings
    fn add_positions(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let seq_len = shape[shape.len() - 2];
        
        let pos_data = self.position_embedding.to_vec();
        let x_data = x.to_vec();
        
        let mut output = x_data.clone();
        
        // Add position embeddings (broadcast over batch if present)
        let num_vectors = x_data.len() / self.dim;
        
        for i in 0..num_vectors {
            let pos = i % seq_len;
            for d in 0..self.dim {
                output[i * self.dim + d] += pos_data[pos * self.dim + d];
            }
        }
        
        Tensor::from_vec(output, shape)
    }
    
    /// Forward pass
    /// 
    /// Returns both:
    /// - `hidden_states`: [batch, seq_len, dim] - full sequence embeddings
    /// - `pooled_output`: [batch, projection_dim] - sentence-level embedding
    pub fn forward(&self, token_ids: &[u32]) -> (Tensor, Tensor) {
        // Get token embeddings
        let mut hidden_states = self.embed_tokens(token_ids);
        
        // Add positional embeddings
        hidden_states = self.add_positions(&hidden_states);
        
        // Pass through transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states);
        }
        
        // Final layer norm
        hidden_states = self.final_layer_norm.forward(&hidden_states);
        
        // Get pooled output (EOS token embedding)
        // EOS is typically at the position after the last real token
        let seq_len = token_ids.len();
        let eos_pos = token_ids.iter()
            .rposition(|&id| id == 1 || id == 49407) // EOS tokens
            .unwrap_or(seq_len - 1);
        
        let hidden_data = hidden_states.to_vec();
        let pooled_data: Vec<f32> = hidden_data[eos_pos * self.dim..(eos_pos + 1) * self.dim].to_vec();
        
        let pooled = Tensor::from_vec(pooled_data.clone(), &[1, self.dim]);
        
        // Apply text projection if present
        let pooled_output = match &self.text_projection {
            Some(proj) => proj.forward(&pooled),
            None => pooled,
        };
        
        (hidden_states, pooled_output)
    }
    
    /// Encode batch of texts
    pub fn forward_batch(&self, all_token_ids: &[Vec<u32>]) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut all_hidden = Vec::new();
        let mut all_pooled = Vec::new();
        
        for token_ids in all_token_ids {
            let (hidden, pooled) = self.forward(token_ids);
            all_hidden.push(hidden);
            all_pooled.push(pooled);
        }
        
        (all_hidden, all_pooled)
    }
}

/// Encode text using CLIP tokenizer and encoder
/// 
/// This is the main interface for text-to-embedding.
pub fn encode_text(
    text: &str,
    tokenizer: &crate::tokenizer::BPETokenizer,
    encoder: &CLIPTextEncoder,
) -> (Tensor, Tensor) {
    let token_ids = tokenizer.encode(text);
    encoder.forward(&token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(4);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let y = ln.forward(&x);
        
        // Output should be normalized
        let y_vec = y.to_vec();
        let mean: f32 = y_vec.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 0.1); // Approximately zero mean
    }
    
    #[test]
    fn test_gelu() {
        let x = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]);
        let y = gelu(&x);
        let y_vec = y.to_vec();
        
        // GELU(0) ≈ 0
        assert!(y_vec[1].abs() < 0.01);
        // GELU(1) ≈ 0.841
        assert!((y_vec[2] - 0.841).abs() < 0.01);
    }
    
    #[test]
    fn test_clip_encoder_shape() {
        // Small encoder for testing
        let encoder = CLIPTextEncoder::new(1000, 64, 4, 2, 16, Some(64));
        
        let token_ids = vec![0, 100, 200, 1]; // start, tokens, end
        let (hidden, pooled) = encoder.forward(&token_ids);
        
        // Hidden states shape: [seq_len, dim]
        assert_eq!(hidden.shape(), &[4, 64]);
        
        // Pooled output shape: [1, projection_dim]
        assert_eq!(pooled.shape(), &[1, 64]);
    }
}
