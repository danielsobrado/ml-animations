---
title: "ReLU - why neural networks learned to use simple functions"
date: 2024-11-20
draft: false
tags: ["relu", "activation", "neural-networks", "deep-learning"]
categories: ["Neural Networks"]
---

Sigmoid was the standard. Then ReLU came along and made deep learning actually work. Such a simple function but it changed everything.

## What is ReLU?

Rectified Linear Unit. Just:

$$f(x) = \max(0, x)$$

Negative inputs → 0
Positive inputs → unchanged

```python
def relu(x):
    return max(0, x)

# or with numpy
def relu(x):
    return np.maximum(0, x)
```

That's it. Why is this good?

![ReLU Function](https://danielsobrado.github.io/ml-animations/animation/relu)

Visual explanation: [ReLU Animation](https://danielsobrado.github.io/ml-animations/animation/relu)

## The sigmoid problem

Before ReLU, networks used sigmoid:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Looks nice but has issues.

**Vanishing gradients**

Sigmoid derivative maxes at 0.25. Multiple layers multiply gradients together. 0.25^10 = 0.0000009. Signal dies.

Deep networks couldn't train. Gradients vanished before reaching early layers.

**Saturation**

Very negative or very positive inputs have gradient ≈ 0. Neurons get "stuck" and stop learning.

**Compute expensive**

Exponential is slow compared to max.

## ReLU solves these

**Gradient is 1 for positive inputs**

No matter how deep, positive signals pass through unchanged.

**No saturation on positive side**

Large positive values don't squash gradient.

**Stupid fast**

Just a comparison. No exponential.

## ReLU's problem - dying neurons

Negative side has gradient 0. If neuron always outputs negative, it stops learning entirely.

"Dead ReLU" - neuron that never activates.

Happens when:
- Learning rate too high
- Bad initialization
- Unlucky input distribution

Some networks end up with 20-40% dead neurons.

## Leaky ReLU

Small slope for negative values:

$$f(x) = \begin{cases} x & x > 0 \\ 0.01x & x \leq 0 \end{cases}$$

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

Dead neurons can recover. Still fast.

More on this: [Leaky ReLU Animation](https://danielsobrado.github.io/ml-animations/animation/leaky-relu)

## Other variants

**PReLU (Parametric ReLU)**

Like Leaky but alpha is learned per channel.

**ELU**
$$f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

Smooth, pushes mean activations toward zero.

**GELU (Gaussian Error Linear Unit)**

$$f(x) = x \cdot \Phi(x)$$

Where Φ is CDF of standard normal. Used in BERT, GPT.

**Swish/SiLU**
$$f(x) = x \cdot \sigma(x)$$

Google found it through automated search. Works well.

## Which one to use?

**For most cases:** ReLU or Leaky ReLU

**For transformers:** GELU

**For very deep networks:** Check if dead neurons are a problem, switch to Leaky if so

**Don't overthink it.** Difference is usually small.

## Code examples

PyTorch:
```python
import torch.nn as nn

# In Sequential
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

# Other activations
nn.LeakyReLU(0.01)
nn.ELU()
nn.GELU()
```

TensorFlow:
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10)
])
```

## Where activations go

After every linear layer (except usually the last one):
```
Input → Linear → ReLU → Linear → ReLU → Linear → Output
```

Last layer:
- Classification: Softmax
- Regression: None (linear)
- Binary: Sigmoid

## Historical note

ReLU wasn't new when deep learning took off. But Krizhevsky used it in AlexNet (2012) and showed it trained much faster than sigmoid/tanh.

Combined with better initialization and batch norm, deep networks finally worked.

The animation shows how ReLU shapes neural network learning: [ReLU Animation](https://danielsobrado.github.io/ml-animations/animation/relu)

---

Related:
- [Leaky ReLU variant](/posts/leaky-relu/)
- [Softmax for outputs](/posts/softmax/)
- [Layer Normalization](/posts/layer-normalization/)
