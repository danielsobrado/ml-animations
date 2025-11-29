---
title: "2D Convolutions - how neural networks see images"
date: 2024-11-15
draft: false
tags: ["conv2d", "cnn", "convolutional", "computer-vision"]
categories: ["Neural Networks"]
---

Images are grids of pixels. Fully connected layers would need billions of parameters. Convolutions exploit spatial structure - nearby pixels are related.

## What is convolution?

Slide a small filter (kernel) across the image. At each position, compute dot product between filter and image patch.

```
Image patch:     Filter:        Output:
1  2  3          1  0  1        1*1 + 2*0 + 3*1 +
4  5  6    *     0  1  0    =   4*0 + 5*1 + 6*0 +    = 15
7  8  9          1  0  1        7*1 + 8*0 + 9*1
```

The filter "detects" certain patterns - edges, textures, shapes.

![Conv2D Operation](https://danielsobrado.github.io/ml-animations/animation/conv2d)

Watch it work: [Conv2D Animation](https://danielsobrado.github.io/ml-animations/animation/conv2d)

## Key concepts

**Kernel size:** Typically 3x3, 5x5, 7x7. Larger = bigger receptive field but more parameters.

**Stride:** How many pixels to move between positions. Stride 2 halves spatial dimensions.

**Padding:** Add zeros around edges. "Same" padding keeps dimensions unchanged.

**Channels:** Input has channels (RGB = 3), output has as many as there are filters.

## The math

For input I of shape (H, W, C_in) and kernel K of shape (k, k, C_in, C_out):

$$O_{x,y,c} = \sum_{i=0}^{k-1}\sum_{j=0}^{k-1}\sum_{c'=0}^{C_{in}-1} I_{x+i, y+j, c'} \cdot K_{i,j,c',c}$$

## Why convolutions work

**Parameter sharing:** Same filter applied everywhere. Edge detector at top-left works at bottom-right too.

**Translation equivariance:** Shift input → output shifts same amount. Position doesn't matter for pattern detection.

**Local connectivity:** Each output depends only on small region. Matches image structure.

## Building a CNN

```python
import torch.nn as nn

model = nn.Sequential(
    # Conv block 1
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),  # 32x32 → 16x16
    
    # Conv block 2
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),  # 16x16 → 8x8
    
    # Flatten and classify
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 10)
)
```

## Pooling

Reduce spatial dimensions. Makes representation more compact and invariant.

**Max pooling:** Take maximum value in each window

**Average pooling:** Take average value

```python
nn.MaxPool2d(kernel_size=2, stride=2)  # halves dimensions
nn.AvgPool2d(kernel_size=2, stride=2)
```

Global pooling: pool entire spatial dimension to 1x1
```python
nn.AdaptiveAvgPool2d((1, 1))  # any input → 1x1 output
```

## Output size calculation

$$O = \frac{I - K + 2P}{S} + 1$$

Where:
- I = input size
- K = kernel size
- P = padding
- S = stride

Example: input 32, kernel 3, padding 1, stride 1:
$$O = \frac{32 - 3 + 2}{1} + 1 = 32$$

Same padding keeps size. Stride 2 halves it.

## Common architectures

**VGG:** Stack of 3x3 convs. Simple but effective.

**ResNet:** Add skip connections. Enables very deep networks.

**Inception:** Multiple kernel sizes in parallel.

**EfficientNet:** Balanced depth/width/resolution scaling.

## 1x1 Convolutions

Filter size 1x1. Seems useless but:
- Changes number of channels
- Adds nonlinearity (with activation)
- Reduces computation

Used in ResNet bottleneck blocks, Inception, etc.

```python
nn.Conv2d(256, 64, kernel_size=1)  # reduce channels
nn.Conv2d(64, 256, kernel_size=1)   # expand back
```

## Depthwise Separable Convolutions

Split regular conv into:
1. Depthwise: one filter per input channel
2. Pointwise: 1x1 conv to combine

Much fewer parameters. Used in MobileNet.

```python
# Regular conv: 3x3x128x256 = 294,912 params
# Depthwise separable: 3x3x128 + 1x1x128x256 = 34,048 params
nn.Sequential(
    nn.Conv2d(128, 128, 3, groups=128, padding=1),  # depthwise
    nn.Conv2d(128, 256, 1)  # pointwise
)
```

## Transposed Convolution

Going the other way - upsampling.

Used in segmentation, autoencoders, GANs.

```python
nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# doubles spatial dimensions
```

## Modern trends

Vision Transformers (ViT) challenge CNNs but convolutions still useful:
- ConvNeXt: modernized CNN competitive with ViT
- Hybrid models: conv stems with transformer bodies
- Efficient on edge devices

The animation shows exactly how sliding window works: [Conv2D Animation](https://danielsobrado.github.io/ml-animations/animation/conv2d)

---

Related:
- [Conv + ReLU combination](/posts/conv-relu/)
- [ReLU activation](/posts/relu/)
