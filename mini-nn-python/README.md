# Mini-NN Python

A minimal neural network library implemented in Python for educational purposes. This is a Python port of the Rust `mini-nn` implementation.

## Features

- **Tensor Operations**: Matrix multiplication, element-wise operations, transposition (using NumPy)
- **Activations**: Linear, ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- **Loss Functions**: MSE, Binary Cross-Entropy, Cross-Entropy
- **Layers**: Dense (fully connected) with Xavier/He initialization
- **Optimizers**: SGD, SGD with Momentum, Adam
- **Training**: Mini-batch gradient descent, validation split, early stopping

## Requirements

- Python 3.8 or later
- NumPy >= 1.20.0

## Installation

```bash
cd mini-nn-python
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy
```

## Usage

### XOR Demo

```bash
python demo_xor.py
```

Expected output:
```
=== Mini-NN Python: XOR Demo ===

Network Architecture:
  Layer 1: Dense(2 -> 8) + RELU
  Layer 2: Dense(8 -> 8) + RELU
  Layer 3: Dense(8 -> 1) + SIGMOID
  Total parameters: 105

Training...
Epoch  10: train_loss=0.2341, train_acc=92.50%, val_loss=0.2156, val_acc=94.00%
...

=== Final Evaluation on Pure XOR ===
Input: [0, 0] -> Expected: 0, Predicted: 0.0234 (0) ✓
Input: [0, 1] -> Expected: 1, Predicted: 0.9821 (1) ✓
Input: [1, 0] -> Expected: 1, Predicted: 0.9756 (1) ✓
Input: [1, 1] -> Expected: 0, Predicted: 0.0312 (0) ✓

Final Accuracy: 100.0% (4/4)
```

## Project Structure

```
mini-nn-python/
├── pyproject.toml
├── README.md
├── demo_xor.py
└── mini_nn/
    ├── __init__.py
    ├── tensor.py       # Tensor operations
    ├── activation.py   # Activation functions
    ├── loss.py         # Loss functions
    ├── layer.py        # Dense layer
    ├── optimizer.py    # SGD, Momentum, Adam
    ├── network.py      # Network builder
    └── trainer.py      # Training loop
```

## API Example

```python
import numpy as np
from mini_nn import (
    Tensor,
    Network,
    Trainer,
    TrainingConfig,
    AdamOptimizer,
    ActivationType,
    InitType,
    LossType,
)

# Set random seed
rng = np.random.default_rng(42)

# Create data
x = Tensor(np.random.randn(100, 2))
y = Tensor(np.random.randint(0, 2, (100, 1)).astype(float))

# Build network
network = (
    Network(rng)
    .add_dense(2, 8, ActivationType.RELU, InitType.HE)
    .add_dense(8, 1, ActivationType.SIGMOID, InitType.XAVIER)
)

# Create optimizer
optimizer = AdamOptimizer(learning_rate=0.01)

# Create trainer
config = TrainingConfig(
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    verbose=True,
    early_stop_patience=15,
)
trainer = Trainer(config, rng)

# Train
history = trainer.fit(network, x, y, LossType.BINARY_CROSS_ENTROPY, optimizer)

# Predict
predictions = network.predict(x)
```

## Comparison with Rust Implementation

| Feature | mini-nn (Rust) | mini-nn-python |
|---------|----------------|----------------|
| Matrix Ops | ndarray | NumPy |
| Activations | ✓ | ✓ |
| Optimizers | SGD, Momentum, Adam | SGD, Momentum, Adam |
| XOR Accuracy | 100% | 100% |

## License

MIT
