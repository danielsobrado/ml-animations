# Mini-NN Go

A minimal neural network library implemented in Go for educational purposes. This is a Go port of the Rust `mini-nn` implementation.

## Features

- **Tensor Operations**: Matrix multiplication, element-wise operations, transposition
- **Activations**: Linear, ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- **Loss Functions**: MSE, Binary Cross-Entropy, Cross-Entropy
- **Layers**: Dense (fully connected) with Xavier/He initialization
- **Optimizers**: SGD, SGD with Momentum, Adam
- **Training**: Mini-batch gradient descent, validation split, early stopping

## Requirements

- Go 1.21 or later
- gonum (for matrix operations)

## Installation

```bash
cd mini-nn-go
go mod download
```

## Usage

### XOR Demo

```bash
go run ./cmd/demo_xor
```

Expected output:
```
=== Mini-NN Go: XOR Demo ===

Network Architecture:
Layer 1: Dense(2 -> 8) + ReLU
Layer 2: Dense(8 -> 8) + ReLU
Layer 3: Dense(8 -> 1) + Sigmoid
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

### Titanic Survival Prediction

```bash
go run ./cmd/train_titanic
```

Note: Requires Titanic dataset CSV files in the `data/` directory or `../mini-nn/data/`.

## Project Structure

```
mini-nn-go/
├── go.mod
├── README.md
├── nn/
│   ├── tensor.go      # Tensor operations
│   ├── activation.go  # Activation functions
│   ├── loss.go        # Loss functions
│   ├── layer.go       # Layer interface and Dense
│   ├── optimizer.go   # SGD, Momentum, Adam
│   ├── network.go     # Network builder
│   ├── training.go    # Training loop
│   └── data.go        # Data loading utilities
└── cmd/
    ├── demo_xor/
    │   └── main.go    # XOR classification demo
    └── train_titanic/
        └── main.go    # Titanic benchmark
```

## API Example

```go
package main

import (
    "mini-nn-go/nn"
    "math/rand"
)

func main() {
    rng := rand.New(rand.NewSource(42))
    
    // Generate data
    x, y := nn.GenerateExpandedXORData(1000, rng)
    
    // Build network
    network := nn.NewNetworkBuilder().
        AddDense(2, 8, nn.ReLU, nn.HeInit).
        AddDense(8, 1, nn.Sigmoid, nn.XavierInit).
        Build()
    
    // Create optimizer and loss
    optimizer := nn.NewAdam(0.01, 0.9, 0.999, 1e-8)
    loss := nn.NewBinaryCrossEntropyLoss()
    
    // Train
    config := nn.TrainingConfig{
        Epochs:          100,
        BatchSize:       32,
        ValidationSplit: 0.2,
        Shuffle:         true,
        Verbose:         true,
    }
    trainer := nn.NewTrainer(config, rng)
    trainer.Fit(network, x, y, loss, optimizer)
    
    // Predict
    pred := network.Predict(x)
}
```

## Comparison with Rust Implementation

| Feature | mini-nn (Rust) | mini-nn-go |
|---------|----------------|------------|
| Matrix Ops | ndarray | gonum |
| Activations | ✓ | ✓ |
| Optimizers | SGD, Momentum, Adam | SGD, Momentum, Adam |
| XOR Accuracy | 100% | 100% |
| Titanic Accuracy | ~84% | ~84% |

## License

MIT
