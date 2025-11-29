---
title: "Layer Normalization - keeping activations stable"
date: 2024-11-17
draft: false
tags: ["layer-norm", "normalization", "transformers", "neural-networks"]
categories: ["Neural Networks"]
---

Deep networks have unstable activations. Values explode or vanish as they pass through layers. Normalization fixes this. Layer norm is the flavor transformers use.

## The problem

Without normalization, activations can:
- Grow exponentially (exploding)
- Shrink to near-zero (vanishing)
- Shift distribution layer by layer

This makes training unstable. Learning rate that works for one layer fails for another.

## Layer Norm formula

For each sample, normalize across the feature dimension:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- μ = mean across features
- σ² = variance across features  
- γ, β = learnable scale and shift
- ε = small constant for stability

![Layer Normalization](https://danielsobrado.github.io/ml-animations/animation/layer-normalization)

See it visualized: [Layer Norm Animation](https://danielsobrado.github.io/ml-animations/animation/layer-normalization)

## Code

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
```

Or just:
```python
layer_norm = nn.LayerNorm(hidden_size)
output = layer_norm(x)
```

## Layer Norm vs Batch Norm

**Batch Norm:**
- Normalize across batch dimension
- Needs batch statistics
- Different behavior train vs eval
- Works great for CNNs

**Layer Norm:**
- Normalize across feature dimension
- Each sample independent
- Same behavior train and eval
- Works great for transformers

```
Batch Norm: normalize each feature across batch
           [sample1_feat1, sample2_feat1, ...] → normalize

Layer Norm: normalize each sample across features
           [sample1_feat1, sample1_feat2, ...] → normalize
```

## Why transformers use Layer Norm

Batch norm fails with:
- Variable sequence lengths (different pad amounts)
- Small batches
- Recurrent processing

Layer norm handles these because each sample normalized independently.

## Pre-norm vs Post-norm

**Post-norm (original transformer):**
```python
x = x + sublayer(x)
x = layer_norm(x)
```

**Pre-norm (GPT-2 style):**
```python
x = x + sublayer(layer_norm(x))
```

Pre-norm trains more stably. Most modern models use it.

## RMSNorm

Simplified version used in LLaMA:

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{n}\sum_i x_i^2 + \epsilon}}$$

Skip the mean subtraction. Just normalize by root mean square.

Slightly faster, works just as well usually.

```python
def rms_norm(x, gamma, eps=1e-5):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return gamma * x / rms
```

## Learnable parameters

Why γ and β?

After normalizing, all activations have mean 0, variance 1. Sometimes network needs different distribution.

γ and β let network learn optimal scale and shift per feature.

Initial values: γ = 1, β = 0 (identity initially)

## Where to put it

In transformer:
```python
# Attention block
x = x + attention(layer_norm(x))

# FFN block  
x = x + ffn(layer_norm(x))
```

Some architectures add final layer norm at the end.

## Debugging tip

If training explodes:
1. Check if layer norm is applied correctly
2. Lower learning rate
3. Add gradient clipping
4. Check initialization

Layer norm usually saves you from explosion but not always.

## Effect on gradients

Layer norm helps gradient flow:
- Reduces internal covariate shift
- Keeps activations in good range
- Gradients don't vanish/explode as easily

This is why deep transformers can train at all.

The animation shows how normalization stabilizes activations: [Layer Norm Animation](https://danielsobrado.github.io/ml-animations/animation/layer-normalization)

---

Related:
- [Transformer Architecture](/posts/transformer-architecture/)
- [Gradient Descent](/posts/gradient-descent/)
