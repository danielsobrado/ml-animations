---
title: "Gradient Descent - how neural networks learn"
date: 2024-11-11
draft: false
tags: ["gradient-descent", "optimization", "backprop", "deep-learning"]
categories: ["Math Fundamentals"]
---

Neural networks learn by adjusting weights to reduce error. Gradient descent tells you which direction to adjust. It's the engine behind all of deep learning.

## The basic idea

You have a loss function L(w) that measures how wrong your predictions are.

Goal: find weights w that minimize L.

Gradient ∇L tells you which direction increases L fastest. Go opposite direction to decrease L.

$$w_{new} = w_{old} - \alpha \nabla L(w)$$

Where α is learning rate.

![Gradient Descent](https://danielsobrado.github.io/ml-animations/animation/gradient-descent)

Watch the optimization: [Gradient Descent Animation](https://danielsobrado.github.io/ml-animations/animation/gradient-descent)

## Why it works

Loss surface is like landscape. You're at some point. Gradient points uphill.

Move opposite direction = move downhill = lower loss.

Repeat until you reach a minimum.

## Learning rate matters

**Too small:** Takes forever to converge. Might get stuck.

**Too large:** Overshoots minimum. Loss oscillates or diverges.

**Just right:** Converges smoothly.

```python
# Typical ranges
lr = 1e-3   # common starting point
lr = 1e-4   # smaller, more stable
lr = 3e-4   # often works for Adam
```

## Types of gradient descent

**Batch (full) gradient descent:**
Compute gradient on entire dataset. Accurate but slow.

```python
for epoch in range(epochs):
    gradient = compute_gradient(all_data)
    weights -= lr * gradient
```

**Stochastic gradient descent (SGD):**
Compute gradient on single sample. Fast but noisy.

```python
for sample in dataset:
    gradient = compute_gradient(sample)
    weights -= lr * gradient
```

**Mini-batch gradient descent:**
Compute gradient on small batch. Best of both worlds.

```python
for batch in dataloader:  # batch_size = 32, 64, 128...
    gradient = compute_gradient(batch)
    weights -= lr * gradient
```

This is what everyone uses in practice.

## Momentum

SGD is noisy, oscillates. Momentum smooths it out.

Keep running average of gradients:
$$v_t = \beta v_{t-1} + \nabla L$$
$$w = w - \alpha v_t$$

Like a ball rolling downhill - it builds up speed in consistent directions.

```python
v = 0
for batch in dataloader:
    gradient = compute_gradient(batch)
    v = beta * v + gradient
    weights -= lr * v
```

## Adam optimizer

Adaptive learning rate per parameter. Most popular optimizer.

Combines momentum with adaptive scaling:
- Parameters with large gradients: smaller effective learning rate
- Parameters with small gradients: larger effective learning rate

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()
```

Adam usually "just works" but sometimes SGD+momentum generalizes better.

## Learning rate scheduling

Learning rate should often decrease during training.

**Step decay:** Reduce by factor every N epochs
```python
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```

**Cosine annealing:** Smooth decrease following cosine curve
```python
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

**Warmup:** Start small, increase, then decrease
```python
# Linear warmup for first 1000 steps
# Then decay
```

## Local minima and saddle points

Loss surface isn't simple bowl. Has:
- Local minima (not globally optimal)
- Saddle points (minimum in some directions, maximum in others)

Scary but in high dimensions, most "bad" critical points are saddle points. Noise from mini-batches helps escape them.

## Practical tips

1. **Start with Adam, lr=1e-3 or 3e-4**

2. **Watch training loss:**
   - Decreasing smoothly: good
   - Stuck: lr too small or other issues
   - Oscillating wildly: lr too large

3. **Use gradient clipping for stability:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

4. **Weight decay (L2 regularization):**
```python
optimizer = Adam(params, lr=1e-3, weight_decay=1e-4)
```

5. **Monitor validation loss for overfitting**

## Backpropagation

How do you actually compute gradients for neural networks?

Chain rule. Loss depends on output, output depends on weights through a chain of operations.

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdots \frac{\partial a_2}{\partial w_1}$$

Frameworks compute this automatically:
```python
loss.backward()  # computes all gradients
```

## Vanishing/exploding gradients

In deep networks, gradients multiply through many layers.

**Vanishing:** Gradients approach 0. Early layers don't learn.
**Exploding:** Gradients blow up. Training unstable.

Solutions:
- Better activation functions (ReLU)
- Skip connections (ResNet)
- Layer normalization
- Careful initialization

The animation shows how gradient descent navigates the loss landscape: [Gradient Descent Animation](https://danielsobrado.github.io/ml-animations/animation/gradient-descent)

---

Related:
- [Linear Regression as simple example](/posts/linear-regression/)
- [Layer Normalization for stability](/posts/layer-normalization/)
