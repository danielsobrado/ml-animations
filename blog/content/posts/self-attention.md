---
title: "Self-Attention explained without the jargon"
date: 2024-11-28
draft: false
tags: ["transformers", "attention", "self-attention", "nlp"]
categories: ["Machine Learning"]
series: ["Understanding Attention"]
---

ok so in last post we covered attention basics. Now let's dig into self-attention which is what transformers actually use.

## Self vs regular attention

Regular attention: Query from one place, Keys/Values from another place
Self-attention: Query, Keys, AND Values all come from same sequence

This is the key insight. Each position in sequence can attend to every other position in same sequence. Including itself.

## Why self-attention matters

Consider sentence: "The cat sat on the mat because it was tired"

What does "it" refer to? Cat or mat?

Human brain instantly knows it's the cat. Self-attention lets model figure this out by letting "it" attend to all other words and learn that "cat" is most relevant.

![Self-Attention Demo](https://danielsobrado.github.io/ml-animations/animation/self-attention)

Try the interactive demo: [Self-Attention Animation](https://danielsobrado.github.io/ml-animations/animation/self-attention)

## How it works step by step

Take input sequence with n tokens. Each token is a vector.

1. Project each token to Q, K, V using learned weight matrices:
   - $Q = XW_Q$
   - $K = XW_K$  
   - $V = XW_V$

2. For each position, compute attention weights to all positions:
   - similarity = $QK^T / \sqrt{d_k}$
   - weights = softmax(similarity)

3. Output = weighted sum of Values

So each token gets to "look at" all other tokens and decide what's relevant.

## Visualization helps

In the animation, you can see:
- Query vector highlighted
- Keys it's comparing to
- Resulting attention weights (darker = higher weight)
- How output is computed

Watching the flow makes it click way faster than reading equations.

## The position problem

One thing that confused me: self-attention is permutation invariant. Meaning if you shuffle input tokens, outputs just shuffle same way. No notion of position.

This is why transformers need positional encoding (covered in separate post).

## Multi-head attention

Instead of one attention operation, do multiple in parallel with different learned projections. Then concatenate.

Why? Each head can learn to focus on different things:
- One head: syntactic relationships
- Another: semantic similarity
- Another: coreference (like our "it" example)

```python
# pseudocode
heads = []
for i in range(num_heads):
    Q = X @ W_Q[i]
    K = X @ W_K[i]
    V = X @ W_V[i]
    heads.append(attention(Q, K, V))
    
output = concat(heads) @ W_O
```

## Computational cost

Self-attention is O(n²) where n is sequence length. For each of n positions, computing attention to all n positions.

This is why long sequences are expensive. BERT handles 512 tokens. GPT models do 2048-8192 typically. Longer requires tricks like sparse attention.

Memory also scales O(n²) which often hits VRAM limits before compute limits.

## What I struggled with

The projection matrices $W_Q$, $W_K$, $W_V$ are learned. This means model learns what to attend to during training. Not hardcoded.

also the softmax gives distribution that sums to 1. So it's always attending to SOMETHING even if nothing is relevant. Some newer architectures address this.

## Try it yourself

Best way to understand: play with visualization

[Self-Attention Interactive Demo](https://danielsobrado.github.io/ml-animations/animation/self-attention)

Watch how changing input changes attention patterns. See multi-head in action.

---

*Part of the [Understanding Attention](/series/understanding-attention/) series*

Next: [Positional Encoding](/posts/positional-encoding/) - how transformers know word order
