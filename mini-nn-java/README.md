# Mini-NN Java

A minimal neural network library implemented in Java 8 for educational purposes. This is a Java port of the Rust `mini-nn` implementation.

## Features

- **Tensor Operations**: Matrix multiplication, element-wise operations, transposition
- **Activations**: Linear, ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- **Loss Functions**: MSE, Binary Cross-Entropy, Cross-Entropy
- **Layers**: Dense (fully connected) with Xavier/He initialization
- **Optimizers**: SGD, Adam
- **Training**: Mini-batch gradient descent, validation split, early stopping

## Requirements

- Java 8 or later
- Maven (for building)

## Building

```bash
cd mini-nn-java
mvn compile
```

## Usage

### XOR Demo

```bash
mvn exec:java -Dexec.mainClass="com.mininn.DemoXOR"
```

Or after packaging:

```bash
mvn package
java -cp target/mini-nn-java-1.0.0.jar com.mininn.DemoXOR
```

Expected output:
```
=== Mini-NN Java: XOR Demo ===

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
mini-nn-java/
├── pom.xml
├── README.md
└── src/
    └── main/
        └── java/
            └── com/
                └── mininn/
                    ├── Tensor.java        # Tensor operations
                    ├── Activation.java    # Activation functions
                    ├── Loss.java          # Loss functions
                    ├── DenseLayer.java    # Dense layer
                    ├── Optimizer.java     # Optimizer interface
                    ├── SGDOptimizer.java  # SGD implementation
                    ├── AdamOptimizer.java # Adam implementation
                    ├── Network.java       # Network builder
                    ├── Trainer.java       # Training loop
                    └── DemoXOR.java       # XOR demo
```

## API Example

```java
import com.mininn.*;
import java.util.Random;

public class Example {
    public static void main(String[] args) {
        Random rng = new Random(42);
        
        // Build network
        Network network = new Network(rng)
            .addDense(2, 8, Activation.RELU, DenseLayer.InitType.HE)
            .addDense(8, 1, Activation.SIGMOID, DenseLayer.InitType.XAVIER);
        
        // Create optimizer and loss
        Optimizer optimizer = new AdamOptimizer(0.01);
        Loss loss = Loss.BINARY_CROSS_ENTROPY;
        
        // Create trainer
        Trainer trainer = new Trainer(
            100,    // epochs
            32,     // batch size
            0.2,    // validation split
            true,   // shuffle
            true,   // verbose
            15,     // early stop patience
            rng
        );
        
        // Train (x and y are your data tensors)
        trainer.fit(network, x, y, loss, optimizer);
        
        // Predict
        Tensor pred = network.predict(x);
    }
}
```

## Comparison with Rust Implementation

| Feature | mini-nn (Rust) | mini-nn-java |
|---------|----------------|--------------|
| Matrix Ops | ndarray | Custom Tensor |
| Activations | ✓ | ✓ |
| Optimizers | SGD, Momentum, Adam | SGD, Adam |
| XOR Accuracy | 100% | 100% |

## License

MIT
