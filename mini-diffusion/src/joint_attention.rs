//! Joint Attention for Multi-Modal Diffusion
//!
//! Joint Attention is a key innovation in SD3/Flux-style models.
//! It allows text and image tokens to attend to each other directly,
//! enabling better text-image alignment than cross-attention.
//!
//! ## Traditional Cross-Attention
//!
//! In SD 1.x/2.x, text conditions image via cross-attention:
//!   - Image tokens are Query
//!   - Text tokens are Key and Value
//!   - One-way: image attends to text, not vice versa
//!
//! ## Joint Attention (MMDiT style)
//!
//! SD3 concatenates text and image tokens and does full self-attention:
//!   - [text_tokens, image_tokens] attend to [text_tokens, image_tokens]
//!   - Bidirectional: text can also condition on image
//!   - More expressive but more compute
//!
//! The trick is using separate projections for text vs image modalities.

use crate::tensor::Tensor;
use crate::nn::Linear;
use crate::clip::LayerNorm;

/// QKV Projection for a single modality
///
/// Each modality (text, image) has its own projection weights,
/// but they share the attention computation.
pub struct ModalityQKV {
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
}

impl ModalityQKV {
    pub fn new(dim: usize) -> Self {
        ModalityQKV {
            to_q: Linear::new(dim, dim),
            to_k: Linear::new(dim, dim),
            to_v: Linear::new(dim, dim),
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        (
            self.to_q.forward(x),
            self.to_k.forward(x),
            self.to_v.forward(x),
        )
    }
}

/// RoPE (Rotary Position Embeddings)
///
/// RoPE encodes position through rotation:
///   q' = rotate(q, pos)
///   k' = rotate(k, pos)
///   
/// The dot product q' · k' then encodes relative position.
///
/// Benefits over absolute position embeddings:
/// - Better extrapolation to longer sequences
/// - Relative position naturally encoded
/// - Used in many modern LLMs and SD3/Flux
pub struct RotaryPositionEmbedding {
    pub dim: usize,
    pub max_seq_len: usize,
    /// Precomputed sin values
    pub sin_cache: Tensor,
    /// Precomputed cos values
    pub cos_cache: Tensor,
}

impl RotaryPositionEmbedding {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let head_dim = dim;
        let theta_base = 10000.0f32;
        
        // Compute frequencies
        let freqs: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / theta_base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        
        // Compute sin/cos for all positions
        let mut sin_cache = Vec::with_capacity(max_seq_len * head_dim);
        let mut cos_cache = Vec::with_capacity(max_seq_len * head_dim);
        
        for pos in 0..max_seq_len {
            for freq in &freqs {
                let angle = pos as f32 * freq;
                sin_cache.push(angle.sin());
                sin_cache.push(angle.sin());
                cos_cache.push(angle.cos());
                cos_cache.push(angle.cos());
            }
        }
        
        RotaryPositionEmbedding {
            dim,
            max_seq_len,
            sin_cache: Tensor::from_vec(sin_cache, &[max_seq_len, head_dim]),
            cos_cache: Tensor::from_vec(cos_cache, &[max_seq_len, head_dim]),
        }
    }
    
    /// Apply rotary embedding to query/key
    pub fn apply(&self, x: &Tensor, offset: usize) -> Tensor {
        let shape = x.shape();
        let seq_len = shape[shape.len() - 2];
        let head_dim = shape[shape.len() - 1];
        
        let x_data = x.to_vec();
        let sin_data = self.sin_cache.to_vec();
        let cos_data = self.cos_cache.to_vec();
        
        let mut output = x_data.clone();
        
        let num_vectors = x_data.len() / head_dim;
        
        for i in 0..num_vectors {
            let pos = (i % seq_len) + offset;
            if pos >= self.max_seq_len {
                continue;
            }
            
            // Apply rotation in pairs
            for j in (0..head_dim).step_by(2) {
                let sin_val = sin_data[pos * head_dim + j];
                let cos_val = cos_data[pos * head_dim + j];
                
                let idx0 = i * head_dim + j;
                let idx1 = i * head_dim + j + 1;
                
                let x0 = x_data[idx0];
                let x1 = x_data[idx1];
                
                // Rotation: [cos, -sin; sin, cos] * [x0; x1]
                output[idx0] = x0 * cos_val - x1 * sin_val;
                output[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
        
        Tensor::from_vec(output, shape)
    }
}

/// Joint Attention Layer
///
/// Processes text and image tokens together in a single attention operation.
///
/// Architecture:
/// 1. Separate QKV projections for text and image
/// 2. Concatenate: [text_Q, image_Q], [text_K, image_K], [text_V, image_V]
/// 3. Standard multi-head attention on concatenated sequences
/// 4. Split output back to text and image parts
/// 5. Separate output projections
pub struct JointAttention {
    /// Text modality projections
    pub text_qkv: ModalityQKV,
    pub text_proj: Linear,
    /// Image modality projections
    pub image_qkv: ModalityQKV,
    pub image_proj: Linear,
    /// Rotary embeddings (optional)
    pub rope: Option<RotaryPositionEmbedding>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Total dimension
    pub dim: usize,
}

impl JointAttention {
    pub fn new(dim: usize, num_heads: usize, use_rope: bool) -> Self {
        let head_dim = dim / num_heads;
        
        JointAttention {
            text_qkv: ModalityQKV::new(dim),
            text_proj: Linear::new(dim, dim),
            image_qkv: ModalityQKV::new(dim),
            image_proj: Linear::new(dim, dim),
            rope: if use_rope { 
                Some(RotaryPositionEmbedding::new(head_dim, 4096)) 
            } else { 
                None 
            },
            num_heads,
            head_dim,
            dim,
        }
    }
    
    /// Forward pass with joint attention
    ///
    /// Args:
    ///   text: [batch, text_len, dim] - text embeddings
    ///   image: [batch, img_len, dim] - image/latent embeddings
    ///
    /// Returns:
    ///   (text_out, image_out) - updated embeddings
    pub fn forward(&self, text: &Tensor, image: &Tensor) -> (Tensor, Tensor) {
        let text_shape = text.shape();
        let image_shape = image.shape();
        
        let (batch_size, text_len) = match text_shape {
            [b, s, _] => (*b, *s),
            [s, _] => (1, *s),
            _ => panic!("Expected 2D or 3D text input"),
        };
        
        let image_len = match image_shape {
            [_, s, _] => *s,
            [s, _] => *s,
            _ => panic!("Expected 2D or 3D image input"),
        };
        
        // Get Q, K, V for each modality
        let (text_q, text_k, text_v) = self.text_qkv.forward(text);
        let (image_q, image_k, image_v) = self.image_qkv.forward(image);
        
        // Apply RoPE if enabled
        let (text_q, text_k) = match &self.rope {
            Some(rope) => (rope.apply(&text_q, 0), rope.apply(&text_k, 0)),
            None => (text_q, text_k),
        };
        
        let (image_q, image_k) = match &self.rope {
            Some(rope) => (
                rope.apply(&image_q, text_len),
                rope.apply(&image_k, text_len),
            ),
            None => (image_q, image_k),
        };
        
        // Concatenate text and image tokens
        let total_len = text_len + image_len;
        
        let q = Self::concat_sequences(&text_q, &image_q, batch_size, text_len, image_len, self.dim);
        let k = Self::concat_sequences(&text_k, &image_k, batch_size, text_len, image_len, self.dim);
        let v = Self::concat_sequences(&text_v, &image_v, batch_size, text_len, image_len, self.dim);
        
        // Compute attention
        let attended = self.attention(&q, &k, &v, batch_size, total_len);
        
        // Split back into text and image
        let (text_out, image_out) = Self::split_sequences(
            &attended, batch_size, text_len, image_len, self.dim
        );
        
        // Apply output projections
        (
            self.text_proj.forward(&text_out),
            self.image_proj.forward(&image_out),
        )
    }
    
    fn concat_sequences(
        a: &Tensor,
        b: &Tensor,
        batch_size: usize,
        a_len: usize,
        b_len: usize,
        dim: usize,
    ) -> Tensor {
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        
        let total_len = a_len + b_len;
        let mut output = vec![0.0f32; batch_size * total_len * dim];
        
        for batch in 0..batch_size {
            // Copy a
            for i in 0..a_len {
                for d in 0..dim {
                    let src_idx = (batch * a_len + i) * dim + d;
                    let dst_idx = (batch * total_len + i) * dim + d;
                    if src_idx < a_data.len() {
                        output[dst_idx] = a_data[src_idx];
                    }
                }
            }
            // Copy b
            for i in 0..b_len {
                for d in 0..dim {
                    let src_idx = (batch * b_len + i) * dim + d;
                    let dst_idx = (batch * total_len + a_len + i) * dim + d;
                    if src_idx < b_data.len() {
                        output[dst_idx] = b_data[src_idx];
                    }
                }
            }
        }
        
        Tensor::from_vec(output, &[batch_size, total_len, dim])
    }
    
    fn split_sequences(
        x: &Tensor,
        batch_size: usize,
        a_len: usize,
        b_len: usize,
        dim: usize,
    ) -> (Tensor, Tensor) {
        let x_data = x.to_vec();
        let total_len = a_len + b_len;
        
        let mut a_data = vec![0.0f32; batch_size * a_len * dim];
        let mut b_data = vec![0.0f32; batch_size * b_len * dim];
        
        for batch in 0..batch_size {
            // Extract a
            for i in 0..a_len {
                for d in 0..dim {
                    let src_idx = (batch * total_len + i) * dim + d;
                    let dst_idx = (batch * a_len + i) * dim + d;
                    if src_idx < x_data.len() {
                        a_data[dst_idx] = x_data[src_idx];
                    }
                }
            }
            // Extract b
            for i in 0..b_len {
                for d in 0..dim {
                    let src_idx = (batch * total_len + a_len + i) * dim + d;
                    let dst_idx = (batch * b_len + i) * dim + d;
                    if src_idx < x_data.len() {
                        b_data[dst_idx] = x_data[src_idx];
                    }
                }
            }
        }
        
        (
            Tensor::from_vec(a_data, &[batch_size, a_len, dim]),
            Tensor::from_vec(b_data, &[batch_size, b_len, dim]),
        )
    }
    
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        let q_vec = q.to_vec();
        let k_vec = k.to_vec();
        let v_vec = v.to_vec();
        
        let scale = (self.head_dim as f32).sqrt();
        
        // Compute attention scores
        let mut scores = vec![0.0f32; batch_size * self.num_heads * seq_len * seq_len];
        
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut dot = 0.0f32;
                        for d in 0..self.head_dim {
                            let q_idx = (b * seq_len + i) * self.dim + h * self.head_dim + d;
                            let k_idx = (b * seq_len + j) * self.dim + h * self.head_dim + d;
                            if q_idx < q_vec.len() && k_idx < k_vec.len() {
                                dot += q_vec[q_idx] * k_vec[k_idx];
                            }
                        }
                        let score_idx = ((b * self.num_heads + h) * seq_len + i) * seq_len + j;
                        scores[score_idx] = dot / scale;
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
        let mut output = vec![0.0f32; batch_size * seq_len * self.dim];
        
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for d in 0..self.head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..seq_len {
                            let attn_idx = ((b * self.num_heads + h) * seq_len + i) * seq_len + j;
                            let v_idx = (b * seq_len + j) * self.dim + h * self.head_dim + d;
                            if v_idx < v_vec.len() {
                                sum += scores[attn_idx] * v_vec[v_idx];
                            }
                        }
                        let out_idx = (b * seq_len + i) * self.dim + h * self.head_dim + d;
                        output[out_idx] = sum;
                    }
                }
            }
        }
        
        Tensor::from_vec(output, &[batch_size, seq_len, self.dim])
    }
}

/// Cross-Attention (alternative to Joint Attention)
///
/// Traditional cross-attention for comparison:
/// - Image tokens attend to text tokens only
/// - Text provides Key and Value, image provides Query
pub struct CrossAttention {
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_out: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub dim: usize,
}

impl CrossAttention {
    pub fn new(dim: usize, context_dim: usize, num_heads: usize) -> Self {
        let head_dim = dim / num_heads;
        
        CrossAttention {
            to_q: Linear::new(dim, dim),
            to_k: Linear::new(context_dim, dim),
            to_v: Linear::new(context_dim, dim),
            to_out: Linear::new(dim, dim),
            num_heads,
            head_dim,
            dim,
        }
    }
    
    /// Forward pass
    ///
    /// Args:
    ///   x: [batch, img_len, dim] - image tokens (query)
    ///   context: [batch, text_len, context_dim] - text tokens (key/value)
    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Tensor {
        let x_shape = x.shape();
        let ctx_shape = context.shape();
        
        let (batch_size, x_len) = match x_shape {
            [b, s, _] => (*b, *s),
            [s, _] => (1, *s),
            _ => panic!("Expected 2D or 3D input"),
        };
        
        let ctx_len = match ctx_shape {
            [_, s, _] => *s,
            [s, _] => *s,
            _ => panic!("Expected 2D or 3D context"),
        };
        
        let q = self.to_q.forward(x);
        let k = self.to_k.forward(context);
        let v = self.to_v.forward(context);
        
        // Attention: Q from x, K/V from context
        let q_vec = q.to_vec();
        let k_vec = k.to_vec();
        let v_vec = v.to_vec();
        
        let scale = (self.head_dim as f32).sqrt();
        
        let mut scores = vec![0.0f32; batch_size * self.num_heads * x_len * ctx_len];
        
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..x_len {
                    for j in 0..ctx_len {
                        let mut dot = 0.0f32;
                        for d in 0..self.head_dim {
                            let q_idx = (b * x_len + i) * self.dim + h * self.head_dim + d;
                            let k_idx = (b * ctx_len + j) * self.dim + h * self.head_dim + d;
                            if q_idx < q_vec.len() && k_idx < k_vec.len() {
                                dot += q_vec[q_idx] * k_vec[k_idx];
                            }
                        }
                        let score_idx = ((b * self.num_heads + h) * x_len + i) * ctx_len + j;
                        scores[score_idx] = dot / scale;
                    }
                }
            }
        }
        
        // Softmax
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..x_len {
                    let start = ((b * self.num_heads + h) * x_len + i) * ctx_len;
                    let row = &mut scores[start..start + ctx_len];
                    
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
                    
                    for val in row.iter_mut() {
                        *val = (*val - max).exp() / exp_sum;
                    }
                }
            }
        }
        
        // Apply to values
        let mut output = vec![0.0f32; batch_size * x_len * self.dim];
        
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..x_len {
                    for d in 0..self.head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..ctx_len {
                            let attn_idx = ((b * self.num_heads + h) * x_len + i) * ctx_len + j;
                            let v_idx = (b * ctx_len + j) * self.dim + h * self.head_dim + d;
                            if v_idx < v_vec.len() {
                                sum += scores[attn_idx] * v_vec[v_idx];
                            }
                        }
                        let out_idx = (b * x_len + i) * self.dim + h * self.head_dim + d;
                        output[out_idx] = sum;
                    }
                }
            }
        }
        
        let attended = Tensor::from_vec(output, &[batch_size, x_len, self.dim]);
        self.to_out.forward(&attended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rope() {
        let rope = RotaryPositionEmbedding::new(64, 1024);
        let x = Tensor::randn(&[4, 64]); // 4 positions, 64 dim
        let y = rope.apply(&x, 0);
        
        assert_eq!(y.shape(), x.shape());
    }
    
    #[test]
    fn test_joint_attention() {
        let attn = JointAttention::new(64, 4, false);
        
        let text = Tensor::randn(&[1, 8, 64]); // 8 text tokens
        let image = Tensor::randn(&[1, 16, 64]); // 16 image tokens
        
        let (text_out, image_out) = attn.forward(&text, &image);
        
        assert_eq!(text_out.shape(), &[1, 8, 64]);
        assert_eq!(image_out.shape(), &[1, 16, 64]);
    }
    
    #[test]
    fn test_cross_attention() {
        let attn = CrossAttention::new(64, 128, 4);
        
        let x = Tensor::randn(&[1, 16, 64]);
        let context = Tensor::randn(&[1, 8, 128]);
        
        let output = attn.forward(&x, &context);
        
        assert_eq!(output.shape(), &[1, 16, 64]);
    }
}
