"""Training utilities for neural networks."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .tensor import Tensor
from .network import Network
from .loss import Loss, LossType
from .optimizer import Optimizer


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    shuffle: bool = True
    verbose: bool = True
    early_stop_patience: int = 10


@dataclass
class TrainingHistory:
    """Training history container."""
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)


class Trainer:
    """Neural network trainer."""

    def __init__(
        self,
        config: TrainingConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize trainer."""
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(
        self,
        network: Network,
        x: Tensor,
        y: Tensor,
        loss_type: LossType,
        optimizer: Optimizer,
    ) -> TrainingHistory:
        """Train the network."""
        history = TrainingHistory()
        
        n_samples = x.rows
        n_val = int(n_samples * self.config.validation_split)
        n_train = n_samples - n_val
        
        # Shuffle and split data
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        x_train = self._select_rows(x, train_idx)
        y_train = self._select_rows(y, train_idx)
        x_val = self._select_rows(x, val_idx)
        y_val = self._select_rows(y, val_idx)
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Shuffle training data
            if self.config.shuffle:
                perm = self.rng.permutation(n_train)
                x_train = self._select_rows(x_train, perm)
                y_train = self._select_rows(y_train, perm)
            
            # Training
            train_loss, train_acc = self._train_epoch(
                network, x_train, y_train, loss_type, optimizer
            )
            
            # Validation
            val_loss, val_acc = self._evaluate(network, x_val, y_val, loss_type)
            
            history.train_loss.append(train_loss)
            history.train_accuracy.append(train_acc)
            history.val_loss.append(val_loss)
            history.val_accuracy.append(val_acc)
            
            if self.config.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:3d}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc * 100:.2f}%, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc * 100:.2f}%"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stop_patience:
                    if self.config.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return history

    def _train_epoch(
        self,
        network: Network,
        x: Tensor,
        y: Tensor,
        loss_type: LossType,
        optimizer: Optimizer,
    ) -> tuple:
        """Run one training epoch."""
        n_samples = x.rows
        batch_size = self.config.batch_size
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_samples)
            
            x_batch = x.slice_rows(start, end)
            y_batch = y.slice_rows(start, end)
            
            # Forward pass
            pred = network.forward(x_batch)
            
            # Compute loss
            batch_loss = Loss.compute(pred, y_batch, loss_type)
            total_loss += batch_loss * (end - start)
            
            # Compute accuracy
            correct += self._compute_accuracy(pred, y_batch)
            total += end - start
            
            # Backward pass
            grad = Loss.gradient(pred, y_batch, loss_type)
            network.backward(grad)
            
            # Update weights
            optimizer.step(network.get_layers())
        
        return total_loss / n_samples, correct / total

    def _evaluate(
        self,
        network: Network,
        x: Tensor,
        y: Tensor,
        loss_type: LossType,
    ) -> tuple:
        """Evaluate network on data."""
        pred = network.forward(x)
        loss = Loss.compute(pred, y, loss_type)
        correct = self._compute_accuracy(pred, y)
        return loss, correct / x.rows

    def _compute_accuracy(self, pred: Tensor, target: Tensor) -> int:
        """Compute classification accuracy."""
        rows, cols = pred.shape
        correct = 0
        
        pred_data = pred.numpy()
        target_data = target.numpy()
        
        for i in range(rows):
            if cols == 1:
                # Binary classification
                pred_class = 1 if pred_data[i, 0] >= 0.5 else 0
                target_class = int(round(target_data[i, 0]))
                if pred_class == target_class:
                    correct += 1
            else:
                # Multi-class classification
                pred_argmax = np.argmax(pred_data[i])
                target_argmax = np.argmax(target_data[i])
                if pred_argmax == target_argmax:
                    correct += 1
        
        return correct

    def _select_rows(self, t: Tensor, indices: np.ndarray) -> Tensor:
        """Select rows by indices."""
        return Tensor(t.numpy()[indices])
