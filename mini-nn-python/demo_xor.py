#!/usr/bin/env python3
"""XOR classification demo using Mini-NN Python."""

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


def generate_xor_data():
    """Generate basic XOR data."""
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)
    return Tensor(x), Tensor(y)


def generate_expanded_xor_data(n: int, rng: np.random.Generator):
    """Generate expanded XOR data with noise."""
    x = np.zeros((n, 2))
    y = np.zeros((n, 1))
    
    for i in range(n):
        a = rng.integers(0, 2)
        b = rng.integers(0, 2)
        xor = a ^ b
        
        # Add small noise to inputs
        noise = 0.1
        x[i, 0] = a + rng.uniform(-noise/2, noise/2)
        x[i, 1] = b + rng.uniform(-noise/2, noise/2)
        y[i, 0] = xor
    
    return Tensor(x), Tensor(y)


def main():
    print("=== Mini-NN Python: XOR Demo ===")
    print()
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Generate expanded XOR data for training
    train_size = 1000
    x_train, y_train = generate_expanded_xor_data(train_size, rng)
    
    # Build network: 2 -> 8 -> 8 -> 1
    network = (
        Network(rng)
        .add_dense(2, 8, ActivationType.RELU, InitType.HE)
        .add_dense(8, 8, ActivationType.RELU, InitType.HE)
        .add_dense(8, 1, ActivationType.SIGMOID, InitType.XAVIER)
    )
    
    network.summary()
    print()
    
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
    print("Training...")
    history = trainer.fit(
        network, x_train, y_train, LossType.BINARY_CROSS_ENTROPY, optimizer
    )
    
    # Evaluate on original XOR patterns
    print()
    print("=== Final Evaluation on Pure XOR ===")
    
    x_test, y_test = generate_xor_data()
    
    correct = 0
    for i in range(4):
        input_tensor = x_test.slice_rows(i, i + 1)
        target = y_test.get(i, 0)
        pred = network.predict(input_tensor)
        pred_val = pred.get(0, 0)
        pred_class = 1 if pred_val >= 0.5 else 0
        target_class = int(target)
        
        status = "✓" if pred_class == target_class else "✗"
        if pred_class == target_class:
            correct += 1
        
        print(
            f"Input: [{input_tensor.get(0, 0):.0f}, {input_tensor.get(0, 1):.0f}] -> "
            f"Expected: {target_class}, Predicted: {pred_val:.4f} ({pred_class}) {status}"
        )
    
    accuracy = correct / 4.0 * 100
    print(f"\nFinal Accuracy: {accuracy:.1f}% ({correct}/4)")
    
    # Print final training stats
    if history.val_accuracy:
        final_val_acc = history.val_accuracy[-1] * 100
        print(f"Final Validation Accuracy: {final_val_acc:.1f}%")


if __name__ == "__main__":
    main()
