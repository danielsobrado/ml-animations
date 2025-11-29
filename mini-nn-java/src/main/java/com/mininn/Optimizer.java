package com.mininn;

/**
 * Interface for optimizers.
 */
public interface Optimizer {
    void step(DenseLayer[] layers);
}
