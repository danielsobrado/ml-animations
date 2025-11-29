package com.mininn;

import java.util.HashMap;
import java.util.Map;

/**
 * Adam optimizer implementation.
 */
public class AdamOptimizer implements Optimizer {
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private int t = 0;

    // First and second moment estimates
    private final Map<DenseLayer, Tensor> mWeights = new HashMap<>();
    private final Map<DenseLayer, Tensor> vWeights = new HashMap<>();
    private final Map<DenseLayer, Tensor> mBias = new HashMap<>();
    private final Map<DenseLayer, Tensor> vBias = new HashMap<>();

    public AdamOptimizer(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public AdamOptimizer(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public void step(DenseLayer[] layers) {
        t++;

        for (DenseLayer layer : layers) {
            // Initialize moment estimates if needed
            if (!mWeights.containsKey(layer)) {
                int[] wShape = layer.getWeights().getShape();
                int[] bShape = layer.getBias().getShape();
                mWeights.put(layer, Tensor.zeros(wShape[0], wShape[1]));
                vWeights.put(layer, Tensor.zeros(wShape[0], wShape[1]));
                mBias.put(layer, Tensor.zeros(bShape[0], bShape[1]));
                vBias.put(layer, Tensor.zeros(bShape[0], bShape[1]));
            }

            // Update weights
            updateParameter(
                layer.getWeights(),
                layer.getWeightGrad(),
                mWeights.get(layer),
                vWeights.get(layer),
                (newW) -> layer.setWeights(newW)
            );

            // Update bias
            updateParameter(
                layer.getBias(),
                layer.getBiasGrad(),
                mBias.get(layer),
                vBias.get(layer),
                (newB) -> layer.setBias(newB)
            );
        }
    }

    private void updateParameter(Tensor param, Tensor grad, Tensor m, Tensor v, 
                                 java.util.function.Consumer<Tensor> setter) {
        int rows = param.getRows();
        int cols = param.getCols();
        Tensor newParam = new Tensor(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double g = grad.get(i, j);

                // Update biased first moment estimate
                double mVal = beta1 * m.get(i, j) + (1 - beta1) * g;
                m.set(i, j, mVal);

                // Update biased second raw moment estimate
                double vVal = beta2 * v.get(i, j) + (1 - beta2) * g * g;
                v.set(i, j, vVal);

                // Compute bias-corrected first moment estimate
                double mHat = mVal / (1 - Math.pow(beta1, t));

                // Compute bias-corrected second raw moment estimate
                double vHat = vVal / (1 - Math.pow(beta2, t));

                // Update parameter
                double update = learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                newParam.set(i, j, param.get(i, j) - update);
            }
        }

        setter.accept(newParam);
    }
}
