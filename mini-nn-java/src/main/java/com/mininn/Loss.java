package com.mininn;

/**
 * Loss functions for neural networks.
 */
public enum Loss {
    MSE,
    BINARY_CROSS_ENTROPY,
    CROSS_ENTROPY;

    private static final double EPSILON = 1e-7;

    public double compute(Tensor predictions, Tensor targets) {
        int n = predictions.getRows();
        double loss = 0;

        switch (this) {
            case MSE:
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < predictions.getCols(); j++) {
                        double diff = predictions.get(i, j) - targets.get(i, j);
                        loss += diff * diff;
                    }
                }
                return loss / (2.0 * n);

            case BINARY_CROSS_ENTROPY:
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < predictions.getCols(); j++) {
                        double p = clip(predictions.get(i, j));
                        double t = targets.get(i, j);
                        loss -= t * Math.log(p) + (1 - t) * Math.log(1 - p);
                    }
                }
                return loss / n;

            case CROSS_ENTROPY:
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < predictions.getCols(); j++) {
                        double p = clip(predictions.get(i, j));
                        double t = targets.get(i, j);
                        if (t > 0) {
                            loss -= t * Math.log(p);
                        }
                    }
                }
                return loss / n;

            default:
                return 0;
        }
    }

    public Tensor gradient(Tensor predictions, Tensor targets) {
        int rows = predictions.getRows();
        int cols = predictions.getCols();
        Tensor grad = new Tensor(rows, cols);

        switch (this) {
            case MSE:
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        grad.set(i, j, (predictions.get(i, j) - targets.get(i, j)) / rows);
                    }
                }
                break;

            case BINARY_CROSS_ENTROPY:
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        double p = clip(predictions.get(i, j));
                        double t = targets.get(i, j);
                        grad.set(i, j, (-t / p + (1 - t) / (1 - p)) / rows);
                    }
                }
                break;

            case CROSS_ENTROPY:
                // For softmax + cross-entropy, gradient simplifies to (pred - target)
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        grad.set(i, j, (predictions.get(i, j) - targets.get(i, j)) / rows);
                    }
                }
                break;
        }

        return grad;
    }

    private static double clip(double v) {
        return Math.max(EPSILON, Math.min(1 - EPSILON, v));
    }
}
