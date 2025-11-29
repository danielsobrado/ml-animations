package com.mininn;

/**
 * Stochastic Gradient Descent optimizer.
 */
public class SGDOptimizer implements Optimizer {
    private final double learningRate;

    public SGDOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void step(DenseLayer[] layers) {
        for (DenseLayer layer : layers) {
            // Update weights: W = W - lr * dW
            Tensor weights = layer.getWeights();
            Tensor weightGrad = layer.getWeightGrad();
            layer.setWeights(weights.sub(weightGrad.scale(learningRate)));

            // Update bias: b = b - lr * db
            Tensor bias = layer.getBias();
            Tensor biasGrad = layer.getBiasGrad();
            layer.setBias(bias.sub(biasGrad.scale(learningRate)));
        }
    }
}
