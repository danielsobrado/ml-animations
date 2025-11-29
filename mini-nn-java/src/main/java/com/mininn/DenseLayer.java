package com.mininn;

import java.util.Random;

/**
 * Dense (fully connected) layer implementation.
 */
public class DenseLayer {
    private Tensor weights;
    private Tensor bias;
    private final Activation activation;
    
    // Cached values for backpropagation
    private Tensor input;
    private Tensor preActivation;
    private Tensor output;
    
    // Gradients
    private Tensor weightGrad;
    private Tensor biasGrad;

    public DenseLayer(int inputSize, int outputSize, Activation activation, InitType init, Random rng) {
        this.activation = activation;
        
        // Initialize weights
        switch (init) {
            case HE:
                this.weights = Tensor.he(inputSize, outputSize, rng);
                break;
            case XAVIER:
            default:
                this.weights = Tensor.xavier(inputSize, outputSize, rng);
                break;
        }
        
        // Initialize bias to zeros
        this.bias = Tensor.zeros(1, outputSize);
        
        // Initialize gradients
        this.weightGrad = Tensor.zeros(inputSize, outputSize);
        this.biasGrad = Tensor.zeros(1, outputSize);
    }

    public Tensor forward(Tensor x) {
        this.input = x;
        // z = x @ W + b
        this.preActivation = x.matmul(weights).add(bias);
        // a = activation(z)
        this.output = activation.apply(preActivation);
        return output;
    }

    public Tensor backward(Tensor gradOutput) {
        int batchSize = input.getRows();
        
        // Compute activation gradient
        Tensor activationGrad = activation.derivative(preActivation, output);
        Tensor delta = gradOutput.mul(activationGrad);
        
        // Compute weight gradient: dW = x^T @ delta
        this.weightGrad = input.transpose().matmul(delta);
        
        // Compute bias gradient: db = sum(delta, axis=0)
        this.biasGrad = delta.sumAxis(0);
        
        // Compute gradient for previous layer: dx = delta @ W^T
        return delta.matmul(weights.transpose());
    }

    public Tensor getWeights() { return weights; }
    public Tensor getBias() { return bias; }
    public Tensor getWeightGrad() { return weightGrad; }
    public Tensor getBiasGrad() { return biasGrad; }

    public void setWeights(Tensor w) { this.weights = w; }
    public void setBias(Tensor b) { this.bias = b; }

    public int getInputSize() { return weights.getRows(); }
    public int getOutputSize() { return weights.getCols(); }
    public Activation getActivation() { return activation; }

    public int getParameterCount() {
        return weights.getRows() * weights.getCols() + bias.getCols();
    }

    public enum InitType {
        XAVIER,
        HE
    }
}
