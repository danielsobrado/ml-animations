package com.mininn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Neural network implementation with a builder pattern.
 */
public class Network {
    private final List<DenseLayer> layers = new ArrayList<>();
    private final Random rng;

    public Network(Random rng) {
        this.rng = rng;
    }

    public Network addDense(int inputSize, int outputSize, Activation activation, DenseLayer.InitType init) {
        layers.add(new DenseLayer(inputSize, outputSize, activation, init, rng));
        return this;
    }

    public Tensor forward(Tensor x) {
        Tensor current = x;
        for (DenseLayer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }

    public Tensor predict(Tensor x) {
        return forward(x);
    }

    public void backward(Tensor gradOutput) {
        Tensor grad = gradOutput;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers.get(i).backward(grad);
        }
    }

    public DenseLayer[] getLayers() {
        return layers.toArray(new DenseLayer[0]);
    }

    public int getParameterCount() {
        int count = 0;
        for (DenseLayer layer : layers) {
            count += layer.getParameterCount();
        }
        return count;
    }

    public void summary() {
        System.out.println("Network Architecture:");
        for (int i = 0; i < layers.size(); i++) {
            DenseLayer layer = layers.get(i);
            System.out.printf("Layer %d: Dense(%d -> %d) + %s%n",
                i + 1, layer.getInputSize(), layer.getOutputSize(), layer.getActivation());
        }
        System.out.printf("Total parameters: %d%n", getParameterCount());
    }
}
