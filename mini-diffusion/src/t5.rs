//! T5 Text Encoder
//!
//! T5 (Text-to-Text Transfer Transformer) is an encoder-decoder model.
//! For diffusion, we use only the encoder to get text embeddings.
//!
//! ## Key Differences from CLIP
//!
//! 1. **Bidirectional**: T5 encoder uses full attention (not causal)
//! 2. **Relative Position Bias**: Instead of absolute position embeddings
//! 3. **No Pooling**: Uses full sequence embeddings (no CLS/EOS pooling)
//! 4. **Unigram Tokenizer**: Uses SentencePiece instead of BPE
//!
//! SD3 uses T5-XXL encoder for detailed text understanding.

use crate::tensor::Tensor;
use crate::nn::Linear;
use crate::clip::LayerNorm;

/// T5 uses RMSNorm instead of LayerNorm
/// 
/// RMSNorm: y = x / RMS(x) * gamma
/// where RMS(x) = sqrt(mean(x^2))
/// 
/// Simpler than LayerNorm (no mean subtraction, no beta).
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize) -> Self {
        RMSNorm {
            weight: Tensor::ones(&[dim]),
            eps: 1e-6,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let dim = shape[shape.len() - 1];
        let x_data = x.to_vec();
        let weight = self.weight.to_vec();
        
        let mut result = vec![0.0f32; x_data.len()];
        let num_vectors = x_data.len() / dim;
        
        for i in 0..num_vectors {
            let start = i * dim;
            let slice = &x_data[start..start + dim];
            
            // Compute RMS
            let sq_mean: f32 = slice.iter().map(|&x| x * x).sum::<f32>() / dim as f32;
            let rms = (sq_mean + self.eps).sqrt();
            
            // Normalize and scale
            for j in 0..dim {
                result[start + j] = slice[j] / rms * weight[j];
            }
        }
        
        Tensor::from_vec(result, shape)
    }
}

/// Relative Position Bias for T5
///
/// T5 uses relative position embeddings instead of absolute.
/// The bias depends only on the relative distance between positions.
/// 
/// Benefits:
/// - Better generalization to different sequence lengths
/// - Translation invariance
/// - More efficient (shared across layers)
pub struct RelativePositionBias {
    /// Bias values per relative position bucket per head
    pub bias: Tensor,
    pub num_heads: usize,
    pub num_buckets: usize,
    pub max_distance: usize,
}

impl RelativePositionBias {
    pub fn new(num_heads: usize, num_buckets: usize, max_distance: usize) -> Self {
        let bias = Tensor::randn(&[num_heads, num_buckets]).mul_scalar(0.02);
        
        RelativePositionBias {
            bias,
            num_heads,
            num_buckets,
            max_distance,
        }
    }
    
    /// Compute bucket index for a relative position
    /// 
    /// T5 uses logarithmic bucketing for distant positions:
    /// - Buckets 0..num_buckets/2: exact positions for small distances
    /// - Buckets num_buckets/2..num_buckets: log-spaced for large distances
    fn relative_position_bucket(&self, relative_position: i32, bidirectional: bool) -> usize {
        let mut rel_pos = relative_position;
        let num_buckets = self.num_buckets as i32;
        let max_exact = num_buckets / 2;
        
        let mut bucket = 0i32;
        
        if bidirectional {
            // Half buckets for positive, half for negative
            if rel_pos > 0 {
                bucket = max_exact;
            }
            rel_pos = rel_pos.abs();
        } else {
            rel_pos = (-rel_pos).max(0);
        }
        
        // Small distances: exact bucket
        if rel_pos < max_exact {
            return (bucket + rel_pos) as usize;
        }
        
        // Large distances: log-spaced buckets
        let rel_pos_float = rel_pos as f32;
        let max_exact_float = max_exact as f32;
        let max_distance_float = self.max_distance as f32;
        
        let log_ratio = (rel_pos_float / max_exact_float).ln() 
            / (max_distance_float / max_exact_float).ln();
        let bucket_offset = (log_ratio * (max_exact_float - 1.0)) as i32;
        
        (bucket + max_exact + bucket_offset.min(max_exact - 1)) as usize
    }
    
    /// Compute bias matrix for given query and key lengths
    pub fn compute_bias(&self, query_len: usize, key_len: usize) -> Tensor {
        let bias_data = self.bias.to_vec();
        let mut output = vec![0.0f32; self.num_heads * query_len * key_len];
        
        for i in 0..query_len {
            for j in 0..key_len {
                let rel_pos = j as i32 - i as i32;
                let bucket = self.relative_position_bucket(rel_pos, true);
                let bucket = bucket.min(self.num_buckets - 1);
                
                for h in 0..self.num_heads {
                    let bias_val = bias_data[h * self.num_buckets + bucket];
                    output[(h * query_len + i) * key_len + j] = bias_val;
                }
            }
        }
        
        Tensor::from_vec(output, &[self.num_heads, query_len, key_len])
    }
}

/// T5 Attention
/// 
/// Self-attention with relative position bias, no absolute positions.
pub struct T5Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub relative_attention_bias: Option<RelativePositionBias>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl T5Attention {
    pub fn new(dim: usize, num_heads: usize, has_relative_bias: bool) -> Self {
        let head_dim = dim / num_heads;
        
        T5Attention {
            q: Linear::new(dim, dim),
            k: Linear::new(dim, dim),
            v: Linear::new(dim, dim),
            o: Linear::new(dim, dim),
            relative_attention_bias: if has_relative_bias {
                Some(RelativePositionBias::new(num_heads, 32, 128))
            } else {
                None
            },
            num_heads,
            head_dim,
        }
    }
    
    pub fn forward(&self, x: &Tensor, position_bias: Option<&Tensor>) -> Tensor {
        let shape = x.shape();
        let (batch_size, seq_len, dim) = match shape {
            [b, s, d] => (*b, *s, *d),
            [s, d] => (1, *s, *d),
            _ => panic!("Expected 2D or 3D input"),
        };
        
        // Project Q, K, V
        let q = self.q.forward(x);
        let k = self.k.forward(x);
        let v = self.v.forward(x);
        
        // Compute attention scores
        let scale = (self.head_dim as f32).sqrt();
        let q_vec = q.to_vec();
        let k_vec = k.to_vec();
        let v_vec = v.to_vec();
        
        let mut scores = vec![0.0f32; batch_size * self.num_heads * seq_len * seq_len];
        
        // Reshape and compute attention per head
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut dot = 0.0f32;
                        for d in 0..self.head_dim {
                            let q_idx = (b * seq_len + i) * dim + h * self.head_dim + d;
                            let k_idx = (b * seq_len + j) * dim + h * self.head_dim + d;
                            dot += q_vec[q_idx] * k_vec[k_idx];
                        }
                        let score_idx = ((b * self.num_heads + h) * seq_len + i) * seq_len + j;
                        scores[score_idx] = dot / scale;
                    }
                }
            }
        }
        
        // Add position bias
        // First, compute bias if not provided
        let computed_bias = if position_bias.is_none() {
            self.relative_attention_bias.as_ref()
                .map(|rpb| rpb.compute_bias(seq_len, seq_len))
        } else {
            None
        };
        
        let bias_tensor = position_bias.or(computed_bias.as_ref());
        
        if let Some(bias) = bias_tensor {
            let bias_vec = bias.to_vec();
            for b in 0..batch_size {
                for h in 0..self.num_heads {
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            let score_idx = ((b * self.num_heads + h) * seq_len + i) * seq_len + j;
                            let bias_idx = (h * seq_len + i) * seq_len + j;
                            if bias_idx < bias_vec.len() {
                                scores[score_idx] += bias_vec[bias_idx];
                            }
                        }
                    }
                }
            }
        }
        
        // Softmax
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    let start = ((b * self.num_heads + h) * seq_len + i) * seq_len;
                    let row = &mut scores[start..start + seq_len];
                    
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
                    
                    for val in row.iter_mut() {
                        *val = (*val - max).exp() / exp_sum;
                    }
                }
            }
        }
        
        // Apply attention to values
        let mut output = vec![0.0f32; batch_size * seq_len * dim];
        
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for d in 0..self.head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..seq_len {
                            let attn_idx = ((b * self.num_heads + h) * seq_len + i) * seq_len + j;
                            let v_idx = (b * seq_len + j) * dim + h * self.head_dim + d;
                            sum += scores[attn_idx] * v_vec[v_idx];
                        }
                        let out_idx = (b * seq_len + i) * dim + h * self.head_dim + d;
                        output[out_idx] = sum;
                    }
                }
            }
        }
        
        let attended = Tensor::from_vec(output, &[batch_size, seq_len, dim]);
        
        // Output projection
        self.o.forward(&attended)
    }
}

/// T5 Feed-Forward Network
/// 
/// Uses gated linear unit (GLU) variant: GeGLU
/// output = (Linear1(x) * GELU(Gate(x))) · Linear2
pub struct T5DenseGatedActDense {
    pub wi_0: Linear, // Gate
    pub wi_1: Linear, // Up projection
    pub wo: Linear,   // Down projection
}

impl T5DenseGatedActDense {
    pub fn new(dim: usize, ff_dim: usize) -> Self {
        T5DenseGatedActDense {
            wi_0: Linear::new(dim, ff_dim),
            wi_1: Linear::new(dim, ff_dim),
            wo: Linear::new(ff_dim, dim),
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.wi_0.forward(x);
        let up = self.wi_1.forward(x);
        
        // GeGLU: gelu(gate) * up
        let gate = crate::clip::gelu(&gate);
        let hidden = gate.mul(&up);
        
        self.wo.forward(&hidden)
    }
}

/// T5 Encoder Block
/// 
/// Pre-norm architecture with:
/// 1. Self-attention with relative position bias
/// 2. Feed-forward with GeGLU
pub struct T5EncoderBlock {
    pub layer_norm1: RMSNorm,
    pub self_attention: T5Attention,
    pub layer_norm2: RMSNorm,
    pub ff: T5DenseGatedActDense,
}

impl T5EncoderBlock {
    pub fn new(dim: usize, num_heads: usize, ff_dim: usize, has_relative_bias: bool) -> Self {
        T5EncoderBlock {
            layer_norm1: RMSNorm::new(dim),
            self_attention: T5Attention::new(dim, num_heads, has_relative_bias),
            layer_norm2: RMSNorm::new(dim),
            ff: T5DenseGatedActDense::new(dim, ff_dim),
        }
    }
    
    pub fn forward(&self, x: &Tensor, position_bias: Option<&Tensor>) -> Tensor {
        // Self-attention
        let h = self.layer_norm1.forward(x);
        let h = self.self_attention.forward(&h, position_bias);
        let x = x.add(&h);
        
        // Feed-forward
        let h = self.layer_norm2.forward(&x);
        let h = self.ff.forward(&h);
        x.add(&h)
    }
}

/// T5 Text Encoder
/// 
/// Encoder-only version of T5 for text embeddings.
/// 
/// ## Architecture Details
/// 
/// - Uses RMSNorm instead of LayerNorm
/// - Relative position bias (shared across layers)
/// - GeGLU activation in feed-forward
/// - No pooling: returns full sequence
///
/// ## Configurations
/// 
/// - T5-Small: dim=512, heads=8, layers=6, ff=2048
/// - T5-Base: dim=768, heads=12, layers=12, ff=3072
/// - T5-Large: dim=1024, heads=16, layers=24, ff=4096
/// - T5-XL: dim=2048, heads=32, layers=24, ff=5120
/// - T5-XXL: dim=4096, heads=64, layers=24, ff=10240 (used in SD3)
pub struct T5TextEncoder {
    /// Token embeddings
    pub token_embedding: Tensor,
    /// Encoder blocks
    pub blocks: Vec<T5EncoderBlock>,
    /// Final layer norm
    pub final_norm: RMSNorm,
    /// Embedding dimension
    pub dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl T5TextEncoder {
    pub fn new(
        vocab_size: usize,
        dim: usize,
        num_heads: usize,
        num_layers: usize,
        ff_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let token_embedding = Tensor::randn(&[vocab_size, dim]).mul_scalar(0.02);
        
        let blocks: Vec<_> = (0..num_layers)
            .map(|i| T5EncoderBlock::new(dim, num_heads, ff_dim, i == 0))
            .collect();
        
        let final_norm = RMSNorm::new(dim);
        
        T5TextEncoder {
            token_embedding,
            blocks,
            final_norm,
            dim,
            max_seq_len,
            vocab_size,
        }
    }
    
    /// T5-XXL configuration (used in SD3)
    pub fn t5_xxl() -> Self {
        Self::new(32128, 4096, 64, 24, 10240, 512)
    }
    
    /// Small configuration for testing
    pub fn t5_small() -> Self {
        Self::new(32128, 512, 8, 6, 2048, 512)
    }
    
    /// Get token embeddings
    fn embed_tokens(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        let mut output = vec![0.0f32; seq_len * self.dim];
        
        let embed_data = self.token_embedding.to_vec();
        
        for (i, &token_id) in token_ids.iter().enumerate() {
            let token_id = (token_id as usize).min(self.vocab_size - 1);
            let start = token_id * self.dim;
            
            for j in 0..self.dim {
                output[i * self.dim + j] = embed_data[start + j];
            }
        }
        
        Tensor::from_vec(output, &[seq_len, self.dim])
    }
    
    /// Forward pass
    /// 
    /// Returns sequence embeddings [seq_len, dim].
    /// Unlike CLIP, T5 doesn't pool - all token embeddings are used.
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        // Get token embeddings (no position embeddings - using relative bias)
        let mut hidden_states = self.embed_tokens(token_ids);
        
        // Compute position bias once (shared across layers)
        let position_bias = self.blocks[0].self_attention
            .relative_attention_bias
            .as_ref()
            .map(|rpb| rpb.compute_bias(token_ids.len(), token_ids.len()));
        
        // Pass through encoder blocks
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, position_bias.as_ref());
        }
        
        // Final normalization
        self.final_norm.forward(&hidden_states)
    }
    
    /// Encode batch of texts
    pub fn forward_batch(&self, all_token_ids: &[Vec<u32>]) -> Vec<Tensor> {
        all_token_ids.iter()
            .map(|ids| self.forward(ids))
            .collect()
    }
}

/// Encode text using T5 tokenizer and encoder
pub fn encode_text(
    text: &str,
    tokenizer: &crate::tokenizer::UnigramTokenizer,
    encoder: &T5TextEncoder,
) -> Tensor {
    let token_ids = tokenizer.encode(text);
    encoder.forward(&token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rms_norm() {
        let norm = RMSNorm::new(4);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let y = norm.forward(&x);
        
        let y_vec = y.to_vec();
        // RMS-normalized, should have similar magnitude
        let rms: f32 = (y_vec.iter().map(|&x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.5);
    }
    
    #[test]
    fn test_relative_position_bias() {
        let rpb = RelativePositionBias::new(4, 32, 128);
        let bias = rpb.compute_bias(8, 8);
        
        assert_eq!(bias.shape(), &[4, 8, 8]);
    }
    
    #[test]
    fn test_t5_encoder_shape() {
        let encoder = T5TextEncoder::new(1000, 64, 4, 2, 256, 32);
        
        let token_ids = vec![1, 100, 200, 300, 2];
        let output = encoder.forward(&token_ids);
        
        // Output shape: [seq_len, dim]
        assert_eq!(output.shape(), &[5, 64]);
    }
}
