---
title: "Building a Diffusion Model from Scratch in Rust - Part 2: Neural Network Layers"
date: 2024-11-16
draft: false
tags: ["diffusion-models", "rust", "neural-networks", "convolution", "deep-learning"]
categories: ["Diffusion Models from Scratch"]
series: ["Mini Diffusion in Rust"]
weight: 2
---

We have tensors. Now we need layers. The actual neural network components that transform data.

Diffusion models need specific layers: linear projections, 2D convolutions, normalization. Let's build them all from scratch.

## Linear Layers

The simplest neural network layer. Matrix multiply plus bias.

$$y = xW + b$$

Where:
- $x$ is input `[batch, in_features]`
- $W$ is weights `[in_features, out_features]`
- $b$ is bias `[out_features]`
- $y$ is output `[batch, out_features]`

```rust
// src/nn.rs
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor,  // [in_features, out_features]
    pub bias: Tensor,    // [out_features]
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Linear {
            // Kaiming init for ReLU/SiLU networks
            weight: Tensor::kaiming_init(in_features, out_features),
            bias: Tensor::zeros(&[out_features]),
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = x.matmul(&self.weight);
        
        // Broadcast bias to batch size
        let batch_size = x.shape()[0];
        let bias_broadcast = self.bias.broadcast(&[batch_size, self.out_features]);
        
        out.add(&bias_broadcast)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }
}
```

That's it. A linear layer is just matrix multiplication. The magic is in stacking them with nonlinearities.

## Activation Functions as an Enum

For flexibility, let's make activation functions selectable:

```rust
#[derive(Debug, Clone, Copy)]
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
```

Diffusion models use SiLU almost everywhere. It's smoother than ReLU and trains better.

## 2D Convolution

Convolutions are the backbone of image processing. They slide a small kernel across the image, computing dot products.

This is the most complex layer we'll build. Take your time with it.

```rust
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub weight: Tensor,  // [out_channels, in_channels, kernel_h, kernel_w]
    pub bias: Tensor,    // [out_channels]
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
        // Kaiming init for conv layers
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
}
```

The weight shape matters. `[out_channels, in_channels, kH, kW]` means:
- We have `out_channels` different filters
- Each filter looks at all `in_channels` input channels
- Each filter is `kH x kW` pixels

## The Convolution Forward Pass

Here's the naive implementation. It's O(n^6) but crystal clear:

```rust
impl Conv2d {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let in_h = shape[2];
        let in_w = shape[3];

        // Output size formula
        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut output = Tensor::zeros(&[batch, self.out_channels, out_h, out_w]);
        let out_data = output.as_mut_slice();
        let x_data = x.as_slice();
        let w_data = self.weight.as_slice();
        let b_data = self.bias.as_slice();

        // Six nested loops. Slow but readable.
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        // Start with bias
                        let mut sum = b_data[oc];
                        
                        // Convolve
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
                                        let w_idx = oc * self.in_channels 
                                            * self.kernel_size * self.kernel_size
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
}
```

In production, you'd use im2col or Winograd transforms. But this shows exactly what convolution does: slide the kernel, multiply, sum.

## Understanding Padding and Stride

**Padding** adds zeros around the input. With `padding=1` and `kernel_size=3`, the output is the same size as input.

**Stride** controls how far the kernel moves each step. `stride=2` halves the spatial dimensions.

Common patterns:
- `kernel=3, stride=1, padding=1`: same size output
- `kernel=3, stride=2, padding=1`: halve dimensions (downsampling)
- `kernel=4, stride=2, padding=1`: also halves, slightly different

## Group Normalization

BatchNorm struggles with small batches. GroupNorm doesn't care about batch size. That's why diffusion models use it.

The idea: split channels into groups, normalize within each group.

```rust
#[derive(Debug, Clone)]
pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub gamma: Tensor,  // scale
    pub beta: Tensor,   // shift
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
}
```

## GroupNorm Forward Pass

```rust
impl GroupNorm {
    pub fn forward(&self, x: &Tensor) -> Tensor {
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
                            out_data[idx] = gamma[channel_idx] * normalized 
                                + beta[channel_idx];
                        }
                    }
                }
            }
        }

        output
    }
}
```

Why group norm works: it normalizes across spatial dimensions and a subset of channels. Independent of batch size. Stable training.

## Layer Normalization

Simpler than GroupNorm. Normalizes across all features at each position.

```rust
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
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

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mean = x.mean();
        let centered = x.add_scalar(-mean);
        let var = centered.square().mean();
        let normalized = centered.div_scalar((var + self.eps).sqrt());
        
        let shape = x.shape();
        let gamma_broadcast = self.gamma.broadcast(shape);
        let beta_broadcast = self.beta.broadcast(shape);
        
        normalized.mul(&gamma_broadcast).add(&beta_broadcast)
    }
}
```

Layer norm is used in transformers and some attention blocks. Simpler math, sometimes faster.

## Self-Attention

Attention lets every position look at every other position. Essential for modern diffusion models.

```rust
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
}
```

The attention formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

That scale factor `1/sqrt(d_k)` keeps the dot products from getting too large before softmax.

## Testing the Layers

```rust
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
        // Same spatial size with padding=1, kernel=3, stride=1
        assert_eq!(y.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_conv2d_downsample() {
        let conv = Conv2d::new(16, 32, 3, 2, 1);
        let x = Tensor::randn(&[1, 16, 32, 32]);
        let y = conv.forward(&x);
        // Halved spatial size with stride=2
        assert_eq!(y.shape(), &[1, 32, 16, 16]);
    }

    #[test]
    fn test_group_norm() {
        let gn = GroupNorm::new(4, 16);  // 4 groups of 4 channels each
        let x = Tensor::randn(&[2, 16, 8, 8]);
        let y = gn.forward(&x);
        assert_eq!(y.shape(), &[2, 16, 8, 8]);
    }
}
```

## Performance Notes

Our convolution is slow. Like, really slow. That's fine for learning. For production:

1. Use im2col to convert conv to matrix multiply
2. Use BLAS libraries (OpenBLAS, Intel MKL)
3. Use GPU (CUDA, Metal, WebGPU)

But understanding the naive version matters. You know exactly what's happening.

## What We Built

- Linear layers (dense/fully connected)
- 2D Convolution with padding and stride
- Group Normalization
- Layer Normalization
- Self-Attention

These are the building blocks for U-Net. Which we'll build next.

## Next Up

In [Part 3: Understanding Noise](/posts/diffusion-part3-noise/), we'll dive into:
- What makes diffusion models different
- The forward diffusion process
- Noise schedules
- Why this approach works

The full code is in the `mini-diffusion` folder: [View on GitHub](https://github.com/danielsobrado/ml-animations/tree/main/mini-diffusion)

---

**Series Navigation:**
- [Part 1: Tensor Foundations](/posts/diffusion-part1-tensors/)
- **Part 2: Neural Network Layers** (you are here)
- [Part 3: Understanding Noise](/posts/diffusion-part3-noise/)
- [Part 4: U-Net Architecture](/posts/diffusion-part4-unet/)
- [Part 5: Training Loop](/posts/diffusion-part5-training/)
- [Part 6: Sampling](/posts/diffusion-part6-sampling/)
- [Part 7: Text Conditioning](/posts/diffusion-part7-conditioning/)
