---
title: "Cross-Entropy Loss - the classification loss function"
date: 2024-11-01
draft: false
tags: ["cross-entropy", "loss-function", "classification"]
categories: ["Probability & Statistics"]
---

Why do we use cross-entropy for classification? What does it actually measure? Understanding this helps understand why neural networks work.

## Definition

For true distribution P and predicted distribution Q:

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

In classification, P is one-hot (true label), Q is softmax output.

![Cross-Entropy](https://danielsobrado.github.io/ml-animations/animation/cross-entropy)

See the math: [Cross-Entropy Animation](https://danielsobrado.github.io/ml-animations/animation/cross-entropy)

## For classification

True label: class 2 (one-hot: [0, 0, 1, 0])
Prediction: [0.1, 0.2, 0.6, 0.1]

$$\text{Loss} = -\sum_i y_i \log(\hat{y}_i) = -\log(0.6) = 0.51$$

Only the true class matters! Others multiply by 0.

Equivalent to: $-\log(\text{predicted probability of true class})$

## Why it works

**High confidence, correct:** -log(0.99) = 0.01. Low loss.
**Low confidence, correct:** -log(0.1) = 2.3. Higher loss.
**Wrong answer:** -log(0.01) = 4.6. Very high loss.

Heavily penalizes confident wrong predictions.

## Binary cross-entropy

For binary classification with sigmoid output:

$$\text{BCE} = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$$

```python
import torch.nn.functional as F

# Binary
loss = F.binary_cross_entropy(predictions, targets)

# Or with logits (more stable)
loss = F.binary_cross_entropy_with_logits(logits, targets)
```

## Multi-class cross-entropy

$$\text{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

In PyTorch, `CrossEntropyLoss` expects raw logits (applies softmax internally):

```python
loss_fn = nn.CrossEntropyLoss()

# logits: (batch, num_classes), NOT softmax'd
# targets: (batch,) with class indices
loss = loss_fn(logits, targets)
```

## Why logits preferred

Computing softmax then log is numerically unstable:
- softmax can underflow to 0
- log(0) = -∞

Combining them (log-softmax) is stable:
$$\log\text{softmax}(z_i) = z_i - \log\sum_j e^{z_j}$$

```python
# Unstable
probs = F.softmax(logits, dim=-1)
loss = -torch.log(probs[target])

# Stable
loss = F.cross_entropy(logits, target)
```

## Connection to KL divergence

$$H(P, Q) = H(P) + D_{KL}(P || Q)$$

For one-hot P, H(P) = 0. So:
$$\text{Cross-entropy} = D_{KL}(P || Q)$$

Minimizing cross-entropy = minimizing KL divergence from true distribution.

## Label smoothing

Instead of hard one-hot [0, 0, 1, 0], use soft [0.025, 0.025, 0.925, 0.025].

Prevents overconfidence. Regularization effect.

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## Multi-label classification

Each class independent (can have multiple labels):

```python
# Use BCE with sigmoid, not cross-entropy with softmax
loss = F.binary_cross_entropy_with_logits(
    logits,  # (batch, num_classes)
    targets  # (batch, num_classes) with 0s and 1s
)
```

## Class imbalance

Some classes rare? Weight the loss:

```python
# Classes weighted inversely to frequency
weights = torch.tensor([0.1, 0.3, 1.0, 0.5])
loss_fn = nn.CrossEntropyLoss(weight=weights)
```

## Focal loss

For extreme imbalance. Down-weight easy examples:

$$\text{FL} = -(1-\hat{y})^\gamma \log(\hat{y})$$

Easy examples (high ŷ) contribute less.

```python
# Not in PyTorch by default, but easy to implement
def focal_loss(logits, targets, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()
```

## Gradient

For softmax + cross-entropy, gradient is beautifully simple:

$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

Predicted minus actual. Clean and intuitive.

## Common mistakes

1. **Applying softmax before CrossEntropyLoss** - it does it internally
2. **Using MSE for classification** - works but cross-entropy better
3. **Ignoring class imbalance** - leads to biased models
4. **Not using with_logits versions** - numerical issues

The animation shows how loss responds to predictions: [Cross-Entropy Animation](https://danielsobrado.github.io/ml-animations/animation/cross-entropy)

---

Related:
- [Softmax outputs](/posts/softmax/)
- [Entropy background](/posts/entropy/)
- [Probability Distributions](/posts/probability-distributions/)
