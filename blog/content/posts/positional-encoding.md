---
title: "Positional Encoding - how transformers know word order"
date: 2024-11-28
draft: false
tags: ["transformers", "positional-encoding", "nlp"]
categories: ["Machine Learning"]
series: ["Understanding Attention"]
---

Here's something that confused me for a while. Attention mechanism treats all positions equally. It doesn't know that "cat" comes before "sat" in sentence.

But word order matters! "Dog bites man" vs "Man bites dog" are very different.

So how do transformers handle this? Positional encoding.

## The problem

Self-attention is permutation equivariant. Feed in tokens in different order, get outputs in that same different order. The attention operation itself has no concept of position.

This is actually a feature for parallelization. But we need to inject position info somehow.

## The solution: add position to embeddings

Simple idea: before feeding tokens to transformer, add position information to each token embedding.

```
input_with_position = token_embedding + positional_encoding
```

The positional encoding is a vector same dimension as embedding. One unique encoding per position.

## Sinusoidal encoding (original transformer)

The original "Attention Is All You Need" paper used sinusoidal functions:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

where pos is position, i is dimension index, d is embedding dimension

![Positional Encoding Visualization](https://danielsobrado.github.io/ml-animations/animation/positional-encoding)

Check out: [Positional Encoding Animation](https://danielsobrado.github.io/ml-animations/animation/positional-encoding)

## Why sinusoids?

Few reasons:
1. Unique pattern for each position
2. Can extrapolate to longer sequences than seen in training
3. Relative positions have consistent relationship (PE[pos+k] can be expressed as linear function of PE[pos])

The different frequencies let model learn both local and global position patterns.

## Visualizing it

In the animation you can see:
- How patterns change across positions
- Different frequencies in different dimensions
- The wavy structure

Low dimensions have high frequency (rapid oscillation). High dimensions have low frequency (slow changes). This gives model multi-scale position information.

## Learned vs fixed positional encoding

Original used fixed sinusoidal. Some models learn position embeddings instead.

BERT uses learned positional embeddings. Each position has trainable vector.

Tradeoff:
- Learned: can adapt to task but limited to training sequence length
- Fixed sinusoidal: generalizes to longer sequences but less flexible

## Relative positional encoding

Newer twist: instead of absolute positions, encode relative distances.

"How far apart are these two tokens?" instead of "token at position 5 and position 8"

Models like T5 and Transformer-XL use variants of this.

## What the animation shows

The visualization demonstrates:
- Sinusoidal patterns across positions
- How to read the encoding matrix
- Comparison of different position values

You can adjust parameters and see how patterns change. Helped me understand why certain frequency choices work.

[Interactive Demo](https://danielsobrado.github.io/ml-animations/animation/positional-encoding)

## Practical notes

Positional encoding is added (not concatenated). This means position info can be "overwritten" by strong semantic signals in some cases.

For sequences up to training length, either approach works. For longer sequences, sinusoidal tends to extrapolate better but it's not guaranteed.

Some recent work (ALiBi, RoPE) explores other encoding schemes. Rotary positional embedding in particular has become popular in LLMs.

## Quick implementation

```python
import torch
import math

def sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * 
        (-math.log(10000.0) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe
```

---

*Part of the [Understanding Attention](/series/understanding-attention/) series*

Next up: [Transformer Architecture](/posts/transformer-architecture/) - putting it all together
