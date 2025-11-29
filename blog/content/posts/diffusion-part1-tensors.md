---
title: "Building a Diffusion Model from Scratch in Rust - Part 1: Tensor Foundations"
date: 2024-11-15
draft: false
tags: ["diffusion-models", "rust", "tensors", "deep-learning", "from-scratch"]
categories: ["Diffusion Models from Scratch"]
series: ["Mini Diffusion in Rust"]
weight: 1
---

So you want to build a diffusion model. Not just use one. Build one. From nothing.

Most tutorials throw PyTorch at you and call it a day. But do you actually understand what's happening under the hood? Let's fix that. We're going to build a miniature version of Flux/Stable Diffusion from scratch in Rust. No ML frameworks. Just math and code.

This is part 1 of a series. We start with tensors. Because everything in deep learning is tensors.

## What is a Tensor Anyway?

A tensor is just a multi-dimensional array. That's it. Nothing fancy.

- 0D tensor: a scalar (just a number)
- 1D tensor: a vector `[1, 2, 3]`
- 2D tensor: a matrix (rows and columns)
- 3D tensor: a cube of numbers
- 4D tensor: what images look like in neural networks `[batch, channels, height, width]`

For diffusion models, we care mostly about 4D tensors. An image batch looks like `[batch_size, 3, 256, 256]` - batch of RGB images.

## Setting Up the Rust Project

Create a new project:

```bash
cargo new mini-diffusion
cd mini-diffusion
```

We'll use `ndarray` as our foundation. Not because we're lazy - because implementing BLAS from scratch would take months. We're building on top of it.

```toml
# Cargo.toml
[dependencies]
ndarray = { version = "0.15", features = ["rayon"] }
ndarray-rand = "0.14"
rand = "0.8"
rand_distr = "0.4"
```

## Our Tensor Struct

```rust
// src/tensor.rs
use ndarray::{Array, IxDyn};
use ndarray_rand::RandomExt;
use rand_distr::{Normal, Uniform};

/// A multi-dimensional tensor for neural network operations
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Array<f32, IxDyn>,
}
```

We use `IxDyn` for dynamic dimensions. Could be 1D, could be 4D. The struct doesn't care.

## Creation Methods

First thing you need: ways to create tensors.

```rust
impl Tensor {
    /// Create from an existing ndarray
    pub fn new(data: Array<f32, IxDyn>) -> Self {
        Tensor { data }
    }

    /// Zeros. Useful for accumulating results.
    pub fn zeros(shape: &[usize]) -> Self {
        Tensor {
            data: Array::zeros(IxDyn(shape)),
        }
    }

    /// Ones. Less common but sometimes needed.
    pub fn ones(shape: &[usize]) -> Self {
        Tensor {
            data: Array::ones(IxDyn(shape)),
        }
    }

    /// Random normal distribution. This is the big one.
    /// Diffusion models start from random noise.
    pub fn randn(shape: &[usize]) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        Tensor {
            data: Array::random(IxDyn(shape), normal),
        }
    }

    /// Uniform distribution between bounds
    pub fn rand_uniform(shape: &[usize], low: f32, high: f32) -> Self {
        let uniform = Uniform::new(low, high);
        Tensor {
            data: Array::random(IxDyn(shape), uniform),
        }
    }
}
```

That `randn` function? You'll use it constantly. Diffusion models live and breathe Gaussian noise.

## Weight Initialization

Neural networks are sensitive to initial weights. Too big and gradients explode. Too small and they vanish. Two popular schemes:

```rust
impl Tensor {
    /// Xavier/Glorot initialization
    /// Works well for sigmoid/tanh networks
    pub fn xavier_init(fan_in: usize, fan_out: usize) -> Self {
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        Self::rand_uniform(&[fan_in, fan_out], -limit, limit)
    }

    /// Kaiming/He initialization
    /// Better for ReLU networks (which we'll use)
    pub fn kaiming_init(fan_in: usize, fan_out: usize) -> Self {
        let std = (2.0 / fan_in as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        Tensor {
            data: Array::random(IxDyn(&[fan_in, fan_out]), normal),
        }
    }
}
```

Xavier keeps variance constant through layers. Kaiming accounts for ReLU killing half the values. Use Kaiming for modern architectures.

## Basic Shape Operations

You need to know your tensor's shape. And reshape it. A lot.

```rust
impl Tensor {
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        Tensor {
            data: self.data.clone().into_shape(IxDyn(new_shape)).unwrap(),
        }
    }

    pub fn transpose(&self) -> Self {
        let ndim = self.data.ndim();
        let mut axes: Vec<usize> = (0..ndim).collect();
        if ndim >= 2 {
            axes.swap(ndim - 1, ndim - 2);
        }
        Tensor {
            data: self.data.clone().permuted_axes(axes),
        }
    }
}
```

Reshaping doesn't copy data. It just reinterprets the same memory with different dimensions. Fast.

## Element-wise Operations

Most tensor ops are element-wise. Apply the same operation to every element.

```rust
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Self {
        Tensor { data: &self.data + &other.data }
    }

    pub fn add_scalar(&self, scalar: f32) -> Self {
        Tensor { data: &self.data + scalar }
    }

    pub fn sub(&self, other: &Tensor) -> Self {
        Tensor { data: &self.data - &other.data }
    }

    pub fn mul(&self, other: &Tensor) -> Self {
        Tensor { data: &self.data * &other.data }
    }

    pub fn mul_scalar(&self, scalar: f32) -> Self {
        Tensor { data: &self.data * scalar }
    }

    pub fn div(&self, other: &Tensor) -> Self {
        Tensor { data: &self.data / &other.data }
    }

    pub fn div_scalar(&self, scalar: f32) -> Self {
        Tensor { data: &self.data / scalar }
    }
}
```

Addition, subtraction, multiplication. The building blocks.

## Math Functions

Neural networks need more than arithmetic. Square roots, exponentials, logs.

```rust
impl Tensor {
    pub fn sqrt(&self) -> Self {
        Tensor { data: self.data.mapv(|x| x.sqrt()) }
    }

    pub fn square(&self) -> Self {
        Tensor { data: self.data.mapv(|x| x * x) }
    }

    pub fn exp(&self) -> Self {
        Tensor { data: self.data.mapv(|x| x.exp()) }
    }

    pub fn ln(&self) -> Self {
        Tensor { data: self.data.mapv(|x| x.ln()) }
    }

    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Tensor { data: self.data.mapv(|x| x.clamp(min, max)) }
    }
}
```

The `mapv` function applies a closure to every element. Clean and efficient.

## Reductions

Sum, mean, variance. You need these for loss functions and normalization.

```rust
impl Tensor {
    pub fn sum(&self) -> f32 {
        self.data.sum()
    }

    pub fn mean(&self) -> f32 {
        self.data.mean().unwrap()
    }

    pub fn sum_axis(&self, axis: usize) -> Self {
        Tensor {
            data: self.data.sum_axis(ndarray::Axis(axis)).into_dyn(),
        }
    }

    pub fn mean_axis(&self, axis: usize) -> Self {
        Tensor {
            data: self.data.mean_axis(ndarray::Axis(axis)).unwrap().into_dyn(),
        }
    }
}
```

These collapse dimensions. A `[batch, features]` tensor summed along axis 1 becomes `[batch]`.

## Activation Functions

The nonlinearities that make neural networks actually work.

```rust
impl Tensor {
    /// ReLU: max(0, x)
    /// Simple. Effective. Used everywhere.
    pub fn relu(&self) -> Self {
        Tensor { data: self.data.mapv(|x| x.max(0.0)) }
    }

    /// Sigmoid: 1 / (1 + exp(-x))
    /// Squashes to (0, 1). Good for probabilities.
    pub fn sigmoid(&self) -> Self {
        Tensor { data: self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())) }
    }

    /// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    /// Squashes to (-1, 1). Centered around zero.
    pub fn tanh(&self) -> Self {
        Tensor { data: self.data.mapv(|x| x.tanh()) }
    }

    /// GELU: used in transformers
    /// Smoother than ReLU
    pub fn gelu(&self) -> Self {
        Tensor {
            data: self.data.mapv(|x| {
                0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
            }),
        }
    }

    /// SiLU/Swish: x * sigmoid(x)
    /// What modern diffusion models use
    pub fn silu(&self) -> Self {
        Tensor { data: self.data.mapv(|x| x / (1.0 + (-x).exp())) }
    }
}
```

SiLU is the important one for us. Stable Diffusion and Flux both use it. It's smooth, differentiable, and just works well.

## Matrix Multiplication

The workhorse operation. Every layer does this.

```rust
impl Tensor {
    /// Matrix multiplication for 2D tensors
    pub fn matmul(&self, other: &Tensor) -> Self {
        let a = self.data.clone()
            .into_dimensionality::<ndarray::Ix2>().unwrap();
        let b = other.data.clone()
            .into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = a.dot(&b);
        Tensor { data: result.into_dyn() }
    }
}
```

For `[M, K] @ [K, N]`, you get `[M, N]`. Each element is a dot product of a row and column.

## Softmax

Turns arbitrary numbers into a probability distribution.

```rust
impl Tensor {
    pub fn softmax(&self) -> Self {
        // Subtract max for numerical stability
        let max_val = self.data.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_data = self.data.mapv(|x| (x - max_val).exp());
        let sum = exp_data.sum();
        Tensor { data: exp_data / sum }
    }
}
```

The max subtraction trick prevents overflow. Without it, `exp(1000)` would explode.

## Concatenation and Broadcasting

For U-Net skip connections and adding embeddings to features.

```rust
impl Tensor {
    pub fn concat(tensors: &[&Tensor], axis: usize) -> Self {
        let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let result = ndarray::concatenate(ndarray::Axis(axis), &views).unwrap();
        Tensor { data: result.into_dyn() }
    }

    pub fn broadcast(&self, shape: &[usize]) -> Self {
        let broadcast = self.data.broadcast(IxDyn(shape)).unwrap();
        Tensor { data: broadcast.to_owned() }
    }
}
```

Broadcasting lets you add a `[channels]` tensor to a `[batch, channels, H, W]` tensor without explicit loops.

## Testing

Always test your tensor operations.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_operations() {
        let a = Tensor::ones(&[2, 2]);
        let b = Tensor::ones(&[2, 2]).mul_scalar(2.0);
        let c = a.add(&b);
        assert_eq!(c.mean(), 3.0);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::ones(&[2, 3]);
        let b = Tensor::ones(&[3, 4]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 4]);
        // Each element should be 3 (dot product of three 1s)
        assert_eq!(c.mean(), 3.0);
    }
}
```

Run with `cargo test`. If these pass, your foundation is solid.

## What We Built

A tensor library with:
- Creation methods (zeros, ones, random)
- Weight initialization (Xavier, Kaiming)
- Element-wise operations
- Math functions
- Reductions
- Activation functions
- Matrix multiplication

This is the foundation. Everything else builds on top of it.

## Next Up

In [Part 2: Neural Network Layers](/posts/diffusion-part2-neural-networks/), we'll use these tensors to build:
- Linear layers
- Convolution layers
- Normalization layers
- The building blocks of U-Net

The full code is in the `mini-diffusion` folder: [View on GitHub](https://github.com/danielsobrado/ml-animations/tree/main/mini-diffusion)

---

**Series Navigation:**
- **Part 1: Tensor Foundations** (you are here)
- [Part 2: Neural Network Layers](/posts/diffusion-part2-neural-networks/)
- [Part 3: Understanding Noise](/posts/diffusion-part3-noise/)
- [Part 4: U-Net Architecture](/posts/diffusion-part4-unet/)
- [Part 5: Training Loop](/posts/diffusion-part5-training/)
- [Part 6: Sampling](/posts/diffusion-part6-sampling/)
- [Part 7: Text Conditioning](/posts/diffusion-part7-conditioning/)
