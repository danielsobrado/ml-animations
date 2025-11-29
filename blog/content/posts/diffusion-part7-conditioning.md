---
title: "Building a Diffusion Model from Scratch in Rust - Part 7: Text Conditioning"
date: 2024-11-21
draft: false
tags: ["diffusion-models", "rust", "text-to-image", "conditioning", "deep-learning"]
categories: ["Diffusion Models from Scratch"]
series: ["Mini Diffusion in Rust"]
weight: 7
---

So far we've built unconditional generation. Start from noise, get an image. But what image? Random.

Text-to-image models like Stable Diffusion and Flux let you specify what you want. "A cat riding a bicycle." The model generates that.

This is conditioning. And it's where things get really interesting.

## The Big Picture

Text conditioning requires:
1. **Text encoder**: Convert text to embeddings
2. **Cross-attention**: Let the U-Net "look at" the text
3. **Classifier-free guidance**: Control how strongly to follow the prompt

Let's build each piece.

## Text Encoding

Real models use CLIP or T5 for text encoding. These are massive pretrained transformers.

For our mini version, we'll sketch the architecture:

```rust
// src/text_encoder.rs (conceptual)
pub struct TextEncoder {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub embedding: Tensor,  // [vocab_size, embed_dim]
    pub transformer_blocks: Vec<TransformerBlock>,
}

impl TextEncoder {
    pub fn encode(&self, token_ids: &[usize]) -> Tensor {
        // 1. Look up embeddings
        let mut hidden = self.lookup_embeddings(token_ids);
        
        // 2. Pass through transformer layers
        for block in &self.transformer_blocks {
            hidden = block.forward(&hidden);
        }
        
        // Output: [seq_len, embed_dim]
        hidden
    }
}
```

CLIP outputs 77 tokens of 768 dimensions. T5 can be larger.

## Cross-Attention

The key mechanism. Every U-Net layer can attend to the text.

Regular self-attention: Q, K, V all come from the same input.
Cross-attention: Q comes from image features, K and V come from text.

```rust
#[derive(Debug, Clone)]
pub struct CrossAttention {
    pub query: Linear,   // Projects image features to queries
    pub key: Linear,     // Projects text to keys
    pub value: Linear,   // Projects text to values
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl CrossAttention {
    pub fn new(query_dim: usize, context_dim: usize, num_heads: usize) -> Self {
        let head_dim = query_dim / num_heads;
        
        CrossAttention {
            query: Linear::new(query_dim, query_dim),
            key: Linear::new(context_dim, query_dim),
            value: Linear::new(context_dim, query_dim),
            out_proj: Linear::new(query_dim, query_dim),
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// x: image features [batch, seq_len, dim]
    /// context: text embeddings [batch, text_len, context_dim]
    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let dim = shape[2];
        let text_len = context.shape()[1];

        // Project to Q, K, V
        let x_flat = x.reshape(&[batch * seq_len, dim]);
        let ctx_flat = context.reshape(&[batch * text_len, context.shape()[2]]);
        
        let q = self.query.forward(&x_flat).reshape(&[batch, seq_len, dim]);
        let k = self.key.forward(&ctx_flat).reshape(&[batch, text_len, dim]);
        let v = self.value.forward(&ctx_flat).reshape(&[batch, text_len, dim]);

        // Compute attention scores
        // Q: [batch, img_seq, dim]
        // K: [batch, text_seq, dim]
        // scores: [batch, img_seq, text_seq]
        
        // Simplified: flatten batch
        let q_2d = q.reshape(&[batch * seq_len, dim]);
        let k_2d = k.reshape(&[batch * text_len, dim]).transpose();
        
        // Attention scores
        let scores = q_2d.matmul(&k_2d).mul_scalar(self.scale);
        let attn_weights = scores.softmax();
        
        // Apply to values
        let v_2d = v.reshape(&[batch * text_len, dim]);
        let attended = attn_weights.matmul(&v_2d);
        
        // Project output
        self.out_proj.forward(&attended).reshape(&[batch, seq_len, dim])
    }
}
```

Each image position can now look at any word in the prompt. "Cat" gets high attention on the furry regions. "Bicycle" gets attention on the wheels.

## Modified ResBlock with Cross-Attention

We need to inject cross-attention into our U-Net blocks:

```rust
#[derive(Debug, Clone)]
pub struct ConditionedResBlock {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub norm1: GroupNorm,
    pub norm2: GroupNorm,
    pub time_mlp: Linear,
    pub skip_conv: Option<Conv2d>,
    pub cross_attn: Option<CrossAttention>,  // New!
    pub channels: usize,
}

impl ConditionedResBlock {
    pub fn forward(
        &self, 
        x: &Tensor, 
        time_emb: &Tensor,
        context: Option<&Tensor>,  // Text embeddings
    ) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let height = shape[2];
        let width = shape[3];

        // Conv path
        let h = self.conv1.forward(x);
        let h = self.norm1.forward(&h);
        let h = h.silu();

        // Add time embedding
        let time_proj = self.time_mlp.forward(time_emb).silu();
        let time_proj = time_proj.reshape(&[batch, self.channels, 1, 1])
            .broadcast(&[batch, self.channels, height, width]);
        let h = h.add(&time_proj);

        // Cross-attention to text (if provided)
        if let (Some(attn), Some(ctx)) = (&self.cross_attn, context) {
            // Reshape to [batch, h*w, channels] for attention
            let h_seq = h.reshape(&[batch, self.channels, height * width])
                .transpose();  // [batch, h*w, channels]
            
            let attended = attn.forward(&h_seq, ctx);
            
            // Reshape back
            let h_attn = attended.transpose()
                .reshape(&[batch, self.channels, height, width]);
            let h = h.add(&h_attn);  // Residual connection
        }

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

## Classifier-Free Guidance

The magic sauce for prompt following. Train the model to work with and without conditioning:

During training:
- Sometimes drop the text (replace with null embedding)
- Model learns unconditional generation too

During inference:
- Run model twice: with prompt and without
- Combine predictions:

$$\tilde\epsilon = \epsilon_\theta(\emptyset) + s \cdot (\epsilon_\theta(c) - \epsilon_\theta(\emptyset))$$

```rust
impl Sampler {
    pub fn sample_cfg(
        &self,
        model: &ConditionedUNet,
        shape: &[usize],
        context: &Tensor,
        null_context: &Tensor,  // Learned "no prompt" embedding
    ) -> Tensor {
        let batch_size = shape[0];
        let mut x = Tensor::randn(shape);
        
        for t in (0..self.noise_scheduler.config.num_timesteps).rev() {
            let timesteps: Vec<usize> = vec![t; batch_size];
            
            // Predict with conditioning
            let noise_cond = model.forward(&x, &timesteps, Some(context));
            
            // Predict without conditioning
            let noise_uncond = model.forward(&x, &timesteps, Some(null_context));
            
            // Combine with guidance scale
            let guided_noise = noise_uncond.add(
                &noise_cond.sub(&noise_uncond).mul_scalar(self.config.guidance_scale)
            );
            
            // Denoise step
            x = self.denoise_step(&x, &guided_noise, t);
        }
        
        x.clamp(-1.0, 1.0)
    }
}
```

Higher guidance scale (7-15) = stronger prompt following
Lower (1-3) = more diversity, less literal

## The Null Embedding

For CFG to work, you need a "null" text embedding representing "no prompt":

```rust
impl TextEncoder {
    pub fn get_null_embedding(&self) -> Tensor {
        // Encode empty string or special token
        self.encode(&[0])  // Assuming 0 is padding/null token
    }
}
```

Or learn it as a parameter:

```rust
pub struct ConditionedUNet {
    // ... existing fields ...
    pub null_context: Tensor,  // Learnable [1, context_len, context_dim]
}
```

## Flux-Style Architecture Notes

Flux (and similar modern models) have some differences:

1. **DiT (Diffusion Transformer)**: Replace U-Net with pure transformer
2. **Joint attention**: Image and text attend to each other
3. **Rectified flow**: Different noise schedule
4. **Larger text encoders**: T5-XXL + CLIP

The concepts are the same. The scale is different.

```rust
// DiT block (conceptual)
pub struct DiTBlock {
    pub self_attn: SelfAttention,
    pub cross_attn: CrossAttention,
    pub mlp: MLP,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: LayerNorm,
    pub scale_shift: AdaLN,  // Adaptive layer norm for time
}

impl DiTBlock {
    pub fn forward(&self, x: &Tensor, time_emb: &Tensor, context: &Tensor) -> Tensor {
        // AdaLN modulates based on timestep
        let (scale1, shift1) = self.scale_shift.get_params(time_emb);
        
        let h = self.norm1.forward(x).mul(&scale1).add(&shift1);
        let h = x.add(&self.self_attn.forward(&h));
        
        let h = h.add(&self.cross_attn.forward(&self.norm2.forward(&h), context));
        
        let h = h.add(&self.mlp.forward(&self.norm3.forward(&h)));
        
        h
    }
}
```

## Training with Conditioning

Modify training to randomly drop conditioning:

```rust
impl Trainer {
    pub fn train_step_conditioned(
        &self,
        model: &ConditionedUNet,
        images: &Tensor,
        contexts: &Tensor,  // Text embeddings
        null_context: &Tensor,
    ) -> f32 {
        let batch_size = images.shape()[0];
        let timesteps = self.noise_scheduler.sample_timesteps(batch_size);
        
        // Randomly drop conditioning (10% of the time)
        let contexts = if rand::random::<f32>() < 0.1 {
            null_context.broadcast(&contexts.shape())
        } else {
            contexts.clone()
        };
        
        // Add noise
        let noisy_batch = ...;
        
        // Predict noise
        let predicted = model.forward(&noisy_batch, &timesteps, Some(&contexts));
        
        mse_loss(&predicted, &noise_batch)
    }
}
```

## Image-to-Image

Same architecture, different starting point:

```rust
pub fn img2img(
    &self,
    model: &ConditionedUNet,
    init_image: &Tensor,
    context: &Tensor,
    strength: f32,  // 0.0 = keep original, 1.0 = ignore original
) -> Tensor {
    let num_steps = (self.config.num_steps as f32 * strength) as usize;
    let start_timestep = ((1.0 - strength) * self.noise_scheduler.config.num_timesteps as f32) as usize;
    
    // Start from noised version of init_image
    let (x, _) = self.noise_scheduler.add_noise(init_image, start_timestep);
    
    // Denoise from start_timestep, not from T
    for t in (0..start_timestep).rev() {
        // ... normal denoising ...
    }
    
    x.clamp(-1.0, 1.0)
}
```

Low strength: minor modifications
High strength: basically ignores the input

## Inpainting

Mask out regions, regenerate them:

```rust
pub fn inpaint(
    &self,
    model: &ConditionedUNet,
    image: &Tensor,
    mask: &Tensor,  // 1 = regenerate, 0 = keep
    context: &Tensor,
) -> Tensor {
    let mut x = Tensor::randn(image.shape());
    
    for t in (0..self.noise_scheduler.config.num_timesteps).rev() {
        // Denoise step
        let denoised = self.denoise_step(&x, ...);
        
        // Replace unmasked regions with original (noised appropriately)
        let (noised_orig, _) = self.noise_scheduler.add_noise(image, t);
        
        // Blend: mask * denoised + (1 - mask) * noised_original
        x = denoised.mul(&mask).add(
            &noised_orig.mul(&Tensor::ones(mask.shape()).sub(&mask))
        );
    }
    
    x.clamp(-1.0, 1.0)
}
```

## Putting It Together

A complete text-to-image pipeline:

```rust
fn generate_from_text(prompt: &str) -> Tensor {
    // 1. Encode text
    let context = text_encoder.encode(tokenize(prompt));
    let null_context = text_encoder.get_null_embedding();
    
    // 2. Create sampler
    let sampler = Sampler::new(SamplerConfig {
        num_steps: 50,
        guidance_scale: 7.5,
        use_ddim: true,
        ..Default::default()
    }, noise_scheduler);
    
    // 3. Sample with classifier-free guidance
    sampler.sample_cfg(&model, &[1, 3, 512, 512], &context, &null_context)
}
```

## What We Didn't Build

A full text-to-image system needs:
- **Pretrained text encoder** (CLIP, T5)
- **Trained U-Net** on millions of image-text pairs
- **VAE** for latent diffusion (more efficient than pixel space)
- **Scheduler improvements** (DPM-Solver, etc.)

That's billions of parameters and massive compute. Our mini version shows the architecture.

## Summary

Text conditioning adds:
1. Text encoder to convert words to embeddings
2. Cross-attention so image features can look at text
3. Classifier-free guidance for controllable generation

The same principles apply to other conditions: class labels, images, audio, whatever.

## Series Complete!

We've built a diffusion model from scratch in Rust:

1. **Tensors**: The data foundation
2. **Layers**: Linear, Conv2d, Normalization
3. **Noise**: Forward diffusion process
4. **U-Net**: The noise prediction network
5. **Training**: Loss function and optimization
6. **Sampling**: DDPM and DDIM
7. **Conditioning**: Text-to-image

The full code is in the `mini-diffusion` folder: [View on GitHub](https://github.com/danielsobrado/ml-animations/tree/main/mini-diffusion)

For a production system, you'd use proper autograd, GPU acceleration, and pretrained components. But now you understand what's happening under the hood.

---

**Series Navigation:**
- [Part 1: Tensor Foundations](/posts/diffusion-part1-tensors/)
- [Part 2: Neural Network Layers](/posts/diffusion-part2-neural-networks/)
- [Part 3: Understanding Noise](/posts/diffusion-part3-noise/)
- [Part 4: U-Net Architecture](/posts/diffusion-part4-unet/)
- [Part 5: Training Loop](/posts/diffusion-part5-training/)
- [Part 6: Sampling](/posts/diffusion-part6-sampling/)
- **Part 7: Text Conditioning** (you are here)
