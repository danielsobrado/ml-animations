//! Neural Network Layers - Building blocks for our diffusion model
//!
//! Implements Linear, Conv2d, LayerNorm, GroupNorm and attention layers.

use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Activation function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
    None,
}

impl Activation {
    pub fn apply(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => x.relu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Tanh => x.tanh(),
            Activation::GELU => x.gelu(),
            Activation::SiLU => x.silu(),
            Activation::None => x.clone(),
        }
    }
}

/// Linear (fully connected) layer: y = xW + b
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor, // shape: [in_features, out_features]
    pub bias: Tensor,   // shape: [out_features]
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Create a new linear layer with Kaiming initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Linear {
            weight: Tensor::kaiming_init(in_features, out_features),
            bias: Tensor::zeros(&[out_features]),
            in_features,
            out_features,
        }
    }

    /// Forward pass: y = xW + b
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x shape: [batch, in_features]
        // weight shape: [in_features, out_features]
        // output shape: [batch, out_features]
        let out = x.matmul(&self.weight);
        
        // Broadcast bias to batch size
        let batch_size = x.shape()[0];
        let bias_broadcast = self.bias.broadcast(&[batch_size, self.out_features]);
        
        out.add(&bias_broadcast)
    }

    /// Get all parameters for optimization
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    /// Get mutable parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

/// 2D Convolution layer
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub weight: Tensor,  // shape: [out_channels, in_channels, kernel_h, kernel_w]
    pub bias: Tensor,    // shape: [out_channels]
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        // Kaiming initialization for conv layers
        let fan_in = in_channels * kernel_size * kernel_size;
        let std = (2.0 / fan_in as f32).sqrt();
        
        Conv2d {
            weight: Tensor::randn(&[out_channels, in_channels, kernel_size, kernel_size])
                .mul_scalar(std),
            bias: Tensor::zeros(&[out_channels]),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// Forward pass - naive implementation for clarity
    /// Input shape: [batch, channels, height, width]
    /// Output shape: [batch, out_channels, out_height, out_width]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.to_contiguous();
        let shape = x.shape();
        let batch = shape[0];
        let _in_c = shape[1];
        let in_h = shape[2];
        let in_w = shape[3];

        // Calculate output dimensions
        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut output = Tensor::zeros(&[batch, self.out_channels, out_h, out_w]);
        let out_data = output.as_mut_slice();
        let x_data = x.as_slice();
        let w_data = self.weight.as_slice();
        let b_data = self.bias.as_slice();

        // Naive convolution - O(n^6) but clear to understand
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = b_data[oc];
                        
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    let ih = oh * self.stride + kh;
                                    let iw = ow * self.stride + kw;
                                    
                                    // Handle padding
                                    let ih_padded = ih as isize - self.padding as isize;
                                    let iw_padded = iw as isize - self.padding as isize;
                                    
                                    if ih_padded >= 0 && ih_padded < in_h as isize
                                        && iw_padded >= 0 && iw_padded < in_w as isize
                                    {
                                        let x_idx = b * self.in_channels * in_h * in_w
                                            + ic * in_h * in_w
                                            + ih_padded as usize * in_w
                                            + iw_padded as usize;
                                        let w_idx = oc * self.in_channels * self.kernel_size * self.kernel_size
                                            + ic * self.kernel_size * self.kernel_size
                                            + kh * self.kernel_size
                                            + kw;
                                        sum += x_data[x_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }
                        
                        let out_idx = b * self.out_channels * out_h * out_w
                            + oc * out_h * out_w
                            + oh * out_w
                            + ow;
                        out_data[out_idx] = sum;
                    }
                }
            }
        }

        output
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

/// Layer Normalization - normalizes across features
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Tensor,  // scale parameter
    pub beta: Tensor,   // shift parameter
    pub eps: f32,
    pub normalized_shape: Vec<usize>,
}

impl LayerNorm {
    pub fn new(normalized_shape: &[usize]) -> Self {
        let size: usize = normalized_shape.iter().product();
        LayerNorm {
            gamma: Tensor::ones(&[size]),
            beta: Tensor::zeros(&[size]),
            eps: 1e-5,
            normalized_shape: normalized_shape.to_vec(),
        }
    }

    /// Normalize: y = gamma * (x - mean) / sqrt(var + eps) + beta
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mean = x.mean();
        let centered = x.add_scalar(-mean);
        let var = centered.square().mean();
        let normalized = centered.div_scalar((var + self.eps).sqrt());
        
        // Apply scale and shift
        let shape = x.shape();
        let gamma_broadcast = self.gamma.broadcast(shape);
        let beta_broadcast = self.beta.broadcast(shape);
        
        normalized.mul(&gamma_broadcast).add(&beta_broadcast)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }
}

/// Group Normalization - normalizes across groups of channels
/// Used extensively in diffusion models (better than BatchNorm for small batches)
#[derive(Debug, Clone)]
pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub gamma: Tensor,
    pub beta: Tensor,
    pub eps: f32,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        assert!(num_channels % num_groups == 0);
        GroupNorm {
            num_groups,
            num_channels,
            gamma: Tensor::ones(&[num_channels]),
            beta: Tensor::zeros(&[num_channels]),
            eps: 1e-5,
        }
    }

    /// Forward pass
    /// Input shape: [batch, channels, height, width]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.to_contiguous();
        let shape = x.shape();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        
        let channels_per_group = channels / self.num_groups;
        
        let mut output = Tensor::zeros(shape);
        let x_data = x.as_slice();
        let out_data = output.as_mut_slice();
        let gamma = self.gamma.as_slice();
        let beta = self.beta.as_slice();

        for b in 0..batch {
            for g in 0..self.num_groups {
                // Calculate mean and variance for this group
                let mut sum = 0.0;
                let mut sq_sum = 0.0;
                let count = (channels_per_group * height * width) as f32;
                
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for h in 0..height {
                        for w in 0..width {
                            let idx = b * channels * height * width
                                + channel_idx * height * width
                                + h * width + w;
                            let val = x_data[idx];
                            sum += val;
                            sq_sum += val * val;
                        }
                    }
                }
                
                let mean = sum / count;
                let var = sq_sum / count - mean * mean;
                let std = (var + self.eps).sqrt();
                
                // Normalize and apply scale/shift
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for h in 0..height {
                        for w in 0..width {
                            let idx = b * channels * height * width
                                + channel_idx * height * width
                                + h * width + w;
                            let normalized = (x_data[idx] - mean) / std;
                            out_data[idx] = gamma[channel_idx] * normalized + beta[channel_idx];
                        }
                    }
                }
            }
        }

        output
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }
}

/// Self-Attention mechanism
#[derive(Debug, Clone)]
pub struct SelfAttention {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl SelfAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(embed_dim % num_heads == 0);
        let head_dim = embed_dim / num_heads;
        
        SelfAttention {
            query: Linear::new(embed_dim, embed_dim),
            key: Linear::new(embed_dim, embed_dim),
            value: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Forward pass
    /// Input shape: [batch, seq_len, embed_dim]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];

        // Project to Q, K, V
        // For simplicity, we flatten batch and seq dimensions
        let x_flat = x.reshape(&[batch * seq_len, embed_dim]);
        
        let q = self.query.forward(&x_flat);
        let k = self.key.forward(&x_flat);
        let v = self.value.forward(&x_flat);

        // Reshape back and compute attention
        let q = q.reshape(&[batch, seq_len, embed_dim]);
        let k = k.reshape(&[batch, seq_len, embed_dim]);
        let v = v.reshape(&[batch, seq_len, embed_dim]);

        // Simplified: compute attention scores
        // In a full implementation, we'd split into heads
        let k_t = k.transpose();
        
        // For each batch element, compute Q @ K^T
        // This is simplified - real implementation needs proper batched matmul
        let q_2d = q.reshape(&[batch * seq_len, embed_dim]);
        let k_2d = k_t.reshape(&[embed_dim, batch * seq_len]);
        let scores = q_2d.matmul(&Tensor::new(k_2d.data.clone().into_shape(ndarray::IxDyn(&[embed_dim, seq_len])).unwrap()));
        
        // Scale and softmax
        let scores = scores.mul_scalar(self.scale);
        let attn = scores.softmax();
        
        // Apply attention to values
        let v_2d = v.reshape(&[batch * seq_len, embed_dim]);
        let out = attn.matmul(&v_2d);
        
        // Final projection
        let out_flat = out.reshape(&[batch * seq_len, embed_dim]);
        let out = self.out_proj.forward(&out_flat);
        
        out.reshape(&[batch, seq_len, embed_dim])
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.query.parameters();
        params.extend(self.key.parameters());
        params.extend(self.value.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let layer = Linear::new(10, 5);
        let x = Tensor::randn(&[2, 10]);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), &[2, 5]);
    }

    #[test]
    fn test_conv2d() {
        let conv = Conv2d::new(3, 16, 3, 1, 1);
        let x = Tensor::randn(&[1, 3, 32, 32]);
        let y = conv.forward(&x);
        assert_eq!(y.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_group_norm() {
        let gn = GroupNorm::new(4, 16);
        let x = Tensor::randn(&[2, 16, 8, 8]);
        let y = gn.forward(&x);
        assert_eq!(y.shape(), &[2, 16, 8, 8]);
    }
}
