---
title: "Transformer Architecture - the full picture"
date: 2024-11-28
draft: false
tags: ["transformers", "architecture", "deep-learning", "nlp"]
categories: ["Machine Learning"]
series: ["Understanding Attention"]
---

ok so we covered attention, self-attention, positional encoding. Time to see how they fit together in actual transformer.

The original "Attention Is All You Need" architecture. Still the foundation for GPT, BERT, etc.

## High level structure

Two main parts:
- Encoder: processes input, builds representations
- Decoder: generates output, attends to encoder output

Some models use both (original transformer, T5). Some use only encoder (BERT). Some use only decoder (GPT).

![Transformer Architecture](https://danielsobrado.github.io/ml-animations/animation/transformer)

Interactive visualization: [Transformer Animation](https://danielsobrado.github.io/ml-animations/animation/transformer)

## Encoder stack

Stack of N identical layers (original used N=6). Each layer has:

1. Multi-head self-attention
2. Add & Norm (residual connection + layer normalization)
3. Feed-forward network
4. Add & Norm again

```
Input 
  → Self-Attention 
  → Add & Norm 
  → FFN 
  → Add & Norm 
  → Output
```

The residual connections are crucial. Without them, deep networks don't train well. Adding input to output of each sublayer.

## Decoder stack

Similar to encoder but with extra attention layer:

1. Masked self-attention (can only attend to previous positions)
2. Add & Norm
3. Cross-attention (attends to encoder output)
4. Add & Norm
5. Feed-forward network
6. Add & Norm

The masking is important. During training, decoder sees full target but shouldn't peek at future tokens. Mask enforces this.

## Why layer normalization?

Normalizes activations across features (not batch). Helps training stability.

Placed after residual connection in original transformer. Some variants put it before (Pre-LN vs Post-LN).

See: [Layer Normalization Animation](https://danielsobrado.github.io/ml-animations/animation/layer-normalization)

## Feed-forward networks

Simple two-layer MLP applied to each position independently:

$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Usually inner dimension is 4x model dimension. So if d_model=512, inner dimension is 2048.

This is where lot of the "knowledge" gets stored in large language models.

## Putting it together

For translation task (original use case):

1. Tokenize source sentence
2. Add positional encoding
3. Pass through encoder stack
4. Start decoder with special START token
5. Decoder generates one token at a time
6. Each step: decoder attends to encoder output + previous decoder outputs
7. Continue until END token or max length

## The visualization

The animation shows:
- Data flow through layers
- Attention patterns at each stage
- How encoder and decoder interact
- Residual connections visually

Try: [Transformer Architecture Animation](https://danielsobrado.github.io/ml-animations/animation/transformer)

Seeing the full picture helped me understand why certain design choices were made.

## Modern variations

Original transformer spawned many variants:

**Encoder-only (BERT style):**
- Bidirectional (each position sees full context)
- Good for classification, NER, understanding tasks

**Decoder-only (GPT style):**
- Autoregressive (each position only sees past)
- Good for generation tasks
- Simpler architecture, easier to scale

**Encoder-decoder (T5, BART):**
- Full original architecture
- Good for seq2seq tasks like translation, summarization

## Scaling insights

Transformers scale well. More layers, wider dimensions, more heads generally improves performance. Up to a point.

But compute grows quadratically with sequence length (attention is O(n²)). Various techniques address this: sparse attention, linear attention, etc.

Memory also gets tricky. Activations for backprop eat lots of VRAM. Gradient checkpointing trades compute for memory.

## What surprised me

The simplicity. Once you understand each component, the full architecture isn't that complex. Stack of attention + feedforward, with residual connections and normalization.

The original paper is still worth reading. Pretty accessible actually.

---

*Part of the [Understanding Attention](/series/understanding-attention/) series*

Related:
- [BERT Architecture](/posts/bert/)
- [Layer Normalization](/posts/layer-normalization/)
