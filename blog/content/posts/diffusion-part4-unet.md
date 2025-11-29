---
title: "Building a Diffusion Model from Scratch in Rust - Part 4: U-Net Architecture"
date: 2024-11-18
draft: false
tags: ["diffusion-models", "rust", "unet", "architecture", "deep-learning"]
categories: ["Diffusion Models from Scratch"]
series: ["Mini Diffusion in Rust"]
weight: 4
---

Time to build the brain of our diffusion model. The U-Net.

Why U-Net? Because it looks like a U. Seriously. Input goes down one side (encoding), reaches a bottleneck, then goes up the other side (decoding). Skip connections bridge across.

For diffusion models, U-Net predicts the noise that was added to an image. Give it a noisy image and a timestep, it outputs what it thinks the noise looks like.

## The Big Picture

```
Input (noisy image + timestep)
    │
    ├─►[Conv]──────────────────────────────────►[Concat]──►[ResBlock]──►Output
    │                                              ▲
    ├─►[ResBlock]──►[Down]─────────────►[Up]──►[Concat]──►[ResBlock]
    │                    │                    ▲
    └─►[ResBlock]──►[Down]──►[ResBlock]──►[Up]
                          │            ▲
                          └──►[Mid]────┘
```

The encoder compresses spatial dimensions while increasing channels.
The decoder expands back, using skip connections to recover detail.

## Residual Blocks

The workhorse of modern architectures. Add the input back to the output:

$$\text{output} = \text{input} + F(\text{input})$$

For diffusion, we also inject the timestep:

```rust
// src/unet.rs
use crate::nn::{Conv2d, GroupNorm, Linear};
use crate::tensor::Tensor;
use crate::diffusion::get_timestep_embedding;

#[derive(Debug, Clone)]
pub struct ResBlock {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub norm1: GroupNorm,
    pub norm2: GroupNorm,
    pub time_mlp: Linear,
    pub skip_conv: Option<Conv2d>,  // If channels change
    pub channels: usize,
}

impl ResBlock {
    pub fn new(in_channels: usize, out_channels: usize, time_emb_dim: usize) -> Self {
        let skip_conv = if in_channels != out_channels {
            Some(Conv2d::new(in_channels, out_channels, 1, 1, 0))
        } else {
            None
        };

        let num_groups = 8.min(out_channels);  // GroupNorm groups
        
        ResBlock {
            conv1: Conv2d::new(in_channels, out_channels, 3, 1, 1),
            conv2: Conv2d::new(out_channels, out_channels, 3, 1, 1),
            norm1: GroupNorm::new(num_groups, out_channels),
            norm2: GroupNorm::new(num_groups, out_channels),
            time_mlp: Linear::new(time_emb_dim, out_channels),
            skip_conv,
            channels: out_channels,
        }
    }
}
```

The time embedding gets projected and added to the features. This is how the network knows the noise level.

## ResBlock Forward Pass

```rust
impl ResBlock {
    pub fn forward(&self, x: &Tensor, time_emb: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let height = shape[2];
        let width = shape[3];

        // Path 1: conv -> norm -> activation
        let h = self.conv1.forward(x);
        let h = self.norm1.forward(&h);
        let h = h.silu();  // SiLU activation

        // Add time embedding (broadcast to spatial dims)
        let time_proj = self.time_mlp.forward(time_emb);
        let time_proj = time_proj.silu();
        let time_proj = time_proj.reshape(&[batch, self.channels, 1, 1]);
        let time_proj = time_proj.broadcast(&[batch, self.channels, height, width]);
        let h = h.add(&time_proj);

        // Path 2: conv -> norm -> activation
        let h = self.conv2.forward(&h);
        let h = self.norm2.forward(&h);
        let h = h.silu();

        // Skip connection
        let skip = match &self.skip_conv {
            Some(conv) => conv.forward(x),
            None => x.clone(),
        };

        h.add(&skip)
    }
}
```

Notice SiLU everywhere. It's what Stable Diffusion uses. Smooth, works well.

## Downsampling

Reduce spatial dimensions by 2x:

```rust
#[derive(Debug, Clone)]
pub struct Downsample {
    pub conv: Conv2d,
}

impl Downsample {
    pub fn new(channels: usize) -> Self {
        // Stride 2 convolution halves dimensions
        Downsample {
            conv: Conv2d::new(channels, channels, 3, 2, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.conv.forward(x)
    }
}
```

Some implementations use average pooling. Strided conv works fine and is learnable.

## Upsampling

Increase spatial dimensions by 2x:

```rust
#[derive(Debug, Clone)]
pub struct Upsample {
    pub conv: Conv2d,
}

impl Upsample {
    pub fn new(channels: usize) -> Self {
        Upsample {
            conv: Conv2d::new(channels, channels, 3, 1, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        // Nearest neighbor upsampling
        let new_height = height * 2;
        let new_width = width * 2;
        
        let mut upsampled = Tensor::zeros(&[batch, channels, new_height, new_width]);
        let x_data = x.as_slice();
        let up_data = upsampled.as_mut_slice();

        // Copy each pixel to 2x2 block
        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let val = x_data[b * channels * height * width 
                            + c * height * width + h * width + w];
                        
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let nh = h * 2 + dh;
                                let nw = w * 2 + dw;
                                let idx = b * channels * new_height * new_width
                                    + c * new_height * new_width
                                    + nh * new_width + nw;
                                up_data[idx] = val;
                            }
                        }
                    }
                }
            }
        }

        // Smooth with conv
        self.conv.forward(&upsampled)
    }
}
```

Nearest neighbor is simple. Bilinear would be smoother. The conv after helps either way.

## The Complete U-Net

Here's our mini U-Net. Smaller than Stable Diffusion but same structure:

```rust
#[derive(Debug)]
pub struct UNet {
    // Time embedding MLP
    pub time_mlp1: Linear,
    pub time_mlp2: Linear,
    pub time_emb_dim: usize,
    
    // Initial convolution
    pub init_conv: Conv2d,
    
    // Encoder
    pub down1: ResBlock,
    pub down2: ResBlock,
    pub down3: ResBlock,
    pub pool1: Downsample,
    pub pool2: Downsample,
    
    // Bottleneck
    pub mid1: ResBlock,
    pub mid2: ResBlock,
    
    // Decoder
    pub up3: ResBlock,
    pub up2: ResBlock,
    pub up1: ResBlock,
    pub upsample2: Upsample,
    pub upsample1: Upsample,
    
    // Output
    pub final_norm: GroupNorm,
    pub final_conv: Conv2d,
    
    pub channels: [usize; 4],
}
```

Channel progression: start small, grow as we downsample, shrink as we upsample.

## U-Net Initialization

```rust
impl UNet {
    pub fn new(in_channels: usize, model_channels: usize, out_channels: usize) -> Self {
        let time_emb_dim = model_channels * 4;
        
        // Channel progression: 1x -> 2x -> 4x -> 4x
        let channels = [
            model_channels,
            model_channels * 2,
            model_channels * 4,
            model_channels * 4,
        ];

        UNet {
            // Time embedding
            time_mlp1: Linear::new(model_channels, time_emb_dim),
            time_mlp2: Linear::new(time_emb_dim, time_emb_dim),
            time_emb_dim,
            
            // Initial: 3 -> model_channels
            init_conv: Conv2d::new(in_channels, channels[0], 3, 1, 1),
            
            // Encoder
            down1: ResBlock::new(channels[0], channels[1], time_emb_dim),
            pool1: Downsample::new(channels[1]),
            down2: ResBlock::new(channels[1], channels[2], time_emb_dim),
            pool2: Downsample::new(channels[2]),
            down3: ResBlock::new(channels[2], channels[3], time_emb_dim),
            
            // Bottleneck
            mid1: ResBlock::new(channels[3], channels[3], time_emb_dim),
            mid2: ResBlock::new(channels[3], channels[3], time_emb_dim),
            
            // Decoder (doubled channels for skip connections)
            up3: ResBlock::new(channels[3] * 2, channels[2], time_emb_dim),
            upsample2: Upsample::new(channels[2]),
            up2: ResBlock::new(channels[2] * 2, channels[1], time_emb_dim),
            upsample1: Upsample::new(channels[1]),
            up1: ResBlock::new(channels[1] * 2, channels[0], time_emb_dim),
            
            // Output
            final_norm: GroupNorm::new(8.min(channels[0]), channels[0]),
            final_conv: Conv2d::new(channels[0], out_channels, 3, 1, 1),
            
            channels,
        }
    }
}
```

Notice the decoder ResBlocks take doubled channels. That's because we concatenate skip connections.

## The Forward Pass

```rust
impl UNet {
    pub fn forward(&self, x: &Tensor, timesteps: &[usize]) -> Tensor {
        // 1. Time embedding
        let t_emb = get_timestep_embedding(timesteps, self.channels[0]);
        let t_emb = self.time_mlp1.forward(&t_emb);
        let t_emb = t_emb.silu();
        let t_emb = self.time_mlp2.forward(&t_emb);
        
        // 2. Initial conv
        let h = self.init_conv.forward(x);
        
        // 3. Encoder (save activations for skip connections)
        let h1 = self.down1.forward(&h, &t_emb);
        let h = self.pool1.forward(&h1);
        
        let h2 = self.down2.forward(&h, &t_emb);
        let h = self.pool2.forward(&h2);
        
        let h3 = self.down3.forward(&h, &t_emb);
        
        // 4. Bottleneck
        let h = self.mid1.forward(&h3, &t_emb);
        let h = self.mid2.forward(&h, &t_emb);
        
        // 5. Decoder with skip connections
        let h = Tensor::concat(&[&h, &h3], 1);  // Concat along channels
        let h = self.up3.forward(&h, &t_emb);
        let h = self.upsample2.forward(&h);
        
        let h = Tensor::concat(&[&h, &h2], 1);
        let h = self.up2.forward(&h, &t_emb);
        let h = self.upsample1.forward(&h);
        
        let h = Tensor::concat(&[&h, &h1], 1);
        let h = self.up1.forward(&h, &t_emb);
        
        // 6. Output
        let h = self.final_norm.forward(&h);
        let h = h.silu();
        self.final_conv.forward(&h)
    }
}
```

Follow the data flow:
1. Timestep gets embedded into a vector
2. Image goes through initial conv
3. Down path: ResBlock -> Downsample, saving each output
4. Bottleneck processes the compressed representation
5. Up path: Concat with skip -> ResBlock -> Upsample
6. Final conv outputs predicted noise

## Parameter Count

```rust
impl UNet {
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}
```

For `model_channels=64`:
- About 5-10 million parameters

Stable Diffusion's U-Net is ~860 million. We're building a toy version.

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resblock() {
        let block = ResBlock::new(32, 64, 128);
        let x = Tensor::randn(&[2, 32, 16, 16]);
        let t = Tensor::randn(&[2, 128]);
        let y = block.forward(&x, &t);
        assert_eq!(y.shape(), &[2, 64, 16, 16]);
    }

    #[test]
    fn test_unet_shape() {
        let unet = UNet::new(3, 32, 3);
        let x = Tensor::randn(&[1, 3, 32, 32]);
        let timesteps = vec![100];
        let y = unet.forward(&x, &timesteps);
        
        // Output same shape as input
        assert_eq!(y.shape(), &[1, 3, 32, 32]);
        
        println!("Parameters: {}", unet.num_parameters());
    }
}
```

Input shape = Output shape. The network predicts noise with the same dimensions as the input image.

## Why Skip Connections?

Without them, the decoder has to reconstruct fine details from just the bottleneck. That's hard.

Skip connections give the decoder direct access to encoder features at each resolution. The encoder captures "what's there", the decoder figures out "what noise to remove".

High-res skips: fine details
Low-res skips: semantic information

## Attention (Optional)

Real diffusion models add self-attention at lower resolutions. We skipped it for simplicity, but here's the idea:

```rust
// In the middle block or low-res decoder blocks:
let h = self.mid1.forward(&h, &t_emb);
let h = self.attention.forward(&h);  // Self-attention
let h = self.mid2.forward(&h, &t_emb);
```

Attention lets distant pixels communicate. Important for coherent structure in larger images.

## Conditioning (Preview)

For text-to-image, you'd also inject text embeddings:

```rust
// Cross-attention in ResBlock or separate block
pub fn forward(&self, x: &Tensor, time_emb: &Tensor, context: &Tensor) -> Tensor {
    // ... existing code ...
    // Add cross-attention to text context
    let h = self.cross_attn.forward(&h, context);
    // ...
}
```

More on this in Part 7.

## Memory and Speed

Our U-Net is slow. Some optimizations for production:

1. **FlashAttention**: Faster attention with less memory
2. **Gradient checkpointing**: Trade compute for memory
3. **Mixed precision**: Use fp16 for most ops
4. **Fused kernels**: Combine operations

But for learning, clarity beats speed.

## What We Built

A complete U-Net with:
- Residual blocks with time conditioning
- Encoder path with downsampling
- Decoder path with upsampling  
- Skip connections between encoder and decoder
- GroupNorm for stable training

This predicts noise. Next, we train it.

## Next Up

In [Part 5: Training Loop](/posts/diffusion-part5-training/), we cover:
- The loss function (surprisingly simple)
- Adam optimizer
- Learning rate schedules
- Training dynamics

The full code is in the `mini-diffusion` folder: [View on GitHub](https://github.com/danielsobrado/ml-animations/tree/main/mini-diffusion)

---

**Series Navigation:**
- [Part 1: Tensor Foundations](/posts/diffusion-part1-tensors/)
- [Part 2: Neural Network Layers](/posts/diffusion-part2-neural-networks/)
- [Part 3: Understanding Noise](/posts/diffusion-part3-noise/)
- **Part 4: U-Net Architecture** (you are here)
- [Part 5: Training Loop](/posts/diffusion-part5-training/)
- [Part 6: Sampling](/posts/diffusion-part6-sampling/)
- [Part 7: Text Conditioning](/posts/diffusion-part7-conditioning/)
