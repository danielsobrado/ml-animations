package com.mininn;

/**
 * Activation functions for neural networks.
 */
public enum Activation {
    LINEAR,
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    SOFTMAX;

    public Tensor apply(Tensor x) {
        switch (this) {
            case LINEAR:
                return x.copy();
            case RELU:
                return x.apply(v -> Math.max(0, v));
            case LEAKY_RELU:
                return x.apply(v -> v >= 0 ? v : 0.01 * v);
            case SIGMOID:
                return x.apply(v -> 1.0 / (1.0 + Math.exp(-clip(v))));
            case TANH:
                return x.apply(Math::tanh);
            case SOFTMAX:
                return applySoftmax(x);
            default:
                return x.copy();
        }
    }

    public Tensor derivative(Tensor x, Tensor output) {
        switch (this) {
            case LINEAR:
                return Tensor.ones(x.getRows(), x.getCols());
            case RELU:
                return x.apply(v -> v > 0 ? 1.0 : 0.0);
            case LEAKY_RELU:
                return x.apply(v -> v >= 0 ? 1.0 : 0.01);
            case SIGMOID:
                return output.apply(v -> v * (1.0 - v));
            case TANH:
                return output.apply(v -> 1.0 - v * v);
            case SOFTMAX:
                // Softmax derivative is complex; handled in loss gradient
                return Tensor.ones(x.getRows(), x.getCols());
            default:
                return Tensor.ones(x.getRows(), x.getCols());
        }
    }

    private static Tensor applySoftmax(Tensor x) {
        int rows = x.getRows();
        int cols = x.getCols();
        Tensor result = new Tensor(rows, cols);

        for (int i = 0; i < rows; i++) {
            // Find max for numerical stability
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) {
                max = Math.max(max, x.get(i, j));
            }

            // Compute exp and sum
            double sum = 0;
            double[] exps = new double[cols];
            for (int j = 0; j < cols; j++) {
                exps[j] = Math.exp(x.get(i, j) - max);
                sum += exps[j];
            }

            // Normalize
            for (int j = 0; j < cols; j++) {
                result.set(i, j, exps[j] / sum);
            }
        }

        return result;
    }

    private static double clip(double v) {
        return Math.max(-500, Math.min(500, v));
    }
}
