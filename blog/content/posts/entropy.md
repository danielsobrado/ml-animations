---
title: "Entropy - measuring uncertainty"
date: 2024-11-02
draft: false
tags: ["entropy", "information-theory", "probability"]
categories: ["Probability & Statistics"]
---

How uncertain is a distribution? Entropy measures that. High entropy = hard to predict. Low entropy = predictable.

## Definition

$$H(X) = -\sum_x P(x) \log P(x)$$

For continuous distributions, use differential entropy with integral.

![Entropy](https://danielsobrado.github.io/ml-animations/animation/entropy)

Interactive demo: [Entropy Animation](https://danielsobrado.github.io/ml-animations/animation/entropy)

## Intuition

Consider distribution over weather: {sunny, rainy, cloudy, snowy}

**High entropy:** Each equally likely (25% each). Very unpredictable.
$$H = -4 \times 0.25 \log_2(0.25) = 2 \text{ bits}$$

**Low entropy:** Sunny 90%, others 3.33% each. Pretty predictable.
$$H \approx 0.61 \text{ bits}$$

**Zero entropy:** Sunny 100%. No uncertainty.
$$H = 0$$

## Properties

1. **Non-negative:** H(X) ≥ 0

2. **Maximum for uniform:** Among distributions over k outcomes, uniform has highest entropy (log k)

3. **Additive for independent:** H(X, Y) = H(X) + H(Y)

## Computing entropy

```python
import numpy as np
from scipy.stats import entropy

# From probability distribution
probs = [0.25, 0.25, 0.25, 0.25]
H = entropy(probs, base=2)  # in bits

# Manual calculation
def compute_entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]  # avoid log(0)
    return -np.sum(probs * np.log2(probs))
```

## Bits vs nats

Log base changes units:
- Base 2 → bits (computer science)
- Base e → nats (physics, ML)
- Base 10 → bans (rare)

Just constant factor: H_nats = H_bits × ln(2)

## In machine learning

**Decision trees:**
Split to maximize information gain (reduce entropy).

$$\text{IG}(S, A) = H(S) - \sum_v \frac{|S_v|}{|S|} H(S_v)$$

**Maximum entropy models:**
Among distributions satisfying constraints, choose highest entropy (least assuming).

**Language models:**
Perplexity = 2^H. Measures how "surprised" model is.

## Cross-entropy

Compare true distribution P with model Q:

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

Always H(P, Q) ≥ H(P). Equality when Q = P.

This is the loss function for classification!

More details: [Cross-Entropy post](/posts/cross-entropy/)

## KL divergence

Difference between distributions:

$$D_{KL}(P || Q) = H(P, Q) - H(P) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

Not symmetric! D(P||Q) ≠ D(Q||P)

Used in:
- VAE loss
- Information bottleneck
- Bayesian inference

## Conditional entropy

Entropy of Y given X:

$$H(Y|X) = -\sum_{x,y} P(x,y) \log P(y|x)$$

How much uncertainty remains in Y after observing X.

## Mutual information

How much knowing X tells you about Y:

$$I(X; Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)$$

Symmetric. Zero if independent.

Used for feature selection, representation learning.

## Entropy and compression

Shannon's theorem: can't compress below entropy bits on average.

High entropy data (random) = incompressible
Low entropy data (patterns) = compressible

ZIP files work because text has structure (low entropy relative to random bytes).

## Connection to thermodynamics

Same math, different context. Information entropy and thermodynamic entropy are related.

Maximum entropy = equilibrium = most disordered state.

The animation shows how entropy changes with distribution shape: [Entropy Animation](https://danielsobrado.github.io/ml-animations/animation/entropy)

---

Related:
- [Cross-Entropy Loss](/posts/cross-entropy/)
- [Probability Distributions](/posts/probability-distributions/)
- [Softmax outputs distributions](/posts/softmax/)
