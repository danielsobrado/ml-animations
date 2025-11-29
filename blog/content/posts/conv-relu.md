---
title: "Conv + ReLU - the building block of CNNs"
date: 2024-11-14
draft: false
tags: ["conv-relu", "cnn", "neural-networks"]
categories: ["Neural Networks"]
---

Convolution followed by ReLU. This pattern repeats hundreds of times in modern image networks. Simple but there's a reason it works.

## The pattern

```python
# Basic building block
conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
relu = nn.ReLU()

output = relu(conv(input))
```

Conv extracts features. ReLU adds nonlinearity.

![Conv ReLU Pattern](https://danielsobrado.github.io/ml-animations/animation/conv-relu)

See both operations: [Conv-ReLU Animation](https://danielsobrado.github.io/ml-animations/animation/conv-relu)

## Why ReLU after conv?

Convolution alone is linear. Stack of linear operations = one linear operation. No point going deep.

$$\text{Conv}_2(\text{Conv}_1(x)) = \text{Conv}_{combined}(x)$$

ReLU breaks linearity. Now deeper = more powerful.

## What actually happens

1. **Conv:** Weighted sum of local region. Detects patterns.
2. **ReLU:** Keeps positive activations, zeros negative ones.

The "feature map" after ReLU shows where patterns were detected.

Negative activation = "not this pattern"
Positive activation = "yes this pattern, this strongly"

## Adding batch norm

Modern networks add batch normalization:

```python
# Conv → BatchNorm → ReLU
nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU()
)
```

Or sometimes:
```python
# Conv → ReLU → BatchNorm
nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(128)
)
```

Both work. First is more common.

## In ResNet

ResNet uses this in blocks with skip connections:

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # skip connection
        out = F.relu(out)
        return out
```

Note: ReLU after addition, not before.

## Fused operations

Some frameworks fuse Conv+ReLU into single operation for efficiency:

```python
# CuDNN can fuse these internally
# PyTorch with torch.backends.cudnn.benchmark = True
```

Same math, faster execution.

## What features look like

Early layers: edges, colors, simple textures
Middle layers: parts, patterns, shapes
Late layers: objects, concepts

ReLU essentially creates "feature detectors" that fire (positive) or don't (zero).

## Alternatives to ReLU

Same pattern works with other activations:

```python
nn.Conv2d(...),
nn.LeakyReLU(0.1)  # handles dead neurons

nn.Conv2d(...),
nn.GELU()  # smoother, used in newer architectures

nn.Conv2d(...), 
nn.SiLU()  # Swish, self-gated
```

But ReLU still default for most CNN applications.

## Number of operations

For Conv2d with kernel k, input channels C_in, output channels C_out, spatial size H×W:

Operations per output pixel: k² × C_in (multiply-adds)
Total: k² × C_in × C_out × H × W

ReLU: one comparison per element = C_out × H × W

ReLU is basically free compared to conv.

## Debugging tips

Activation values too large → consider batch norm
Lots of zeros after ReLU → might have dead neurons
Activations all same → conv might not be learning

```python
# Quick check
activations = relu(conv(x))
print(f"Mean: {activations.mean():.3f}")
print(f"% zeros: {(activations == 0).float().mean():.1%}")
```

The animation shows feature extraction step by step: [Conv-ReLU Animation](https://danielsobrado.github.io/ml-animations/animation/conv-relu)

---

Related:
- [Conv2D in detail](/posts/conv2d/)
- [ReLU activation](/posts/relu/)
- [Leaky ReLU](/posts/leaky-relu/)
