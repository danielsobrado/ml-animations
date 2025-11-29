package com.minidiffusion;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * A simple 4D tensor implementation for diffusion models.
 * Optimized for image data: [batch, channels, height, width]
 */
public class Tensor {
    private final double[] data;
    private final int[] shape;
    private final int[] strides;

    public Tensor(int... shape) {
        this.shape = shape.clone();
        this.strides = computeStrides(shape);
        int size = 1;
        for (int s : shape) size *= s;
        this.data = new double[size];
    }

    public Tensor(double[] data, int... shape) {
        this.shape = shape.clone();
        this.strides = computeStrides(shape);
        int size = 1;
        for (int s : shape) size *= s;
        if (data.length != size) {
            throw new IllegalArgumentException("Data length doesn't match shape");
        }
        this.data = data.clone();
    }

    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    // Factory methods
    public static Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor ones(int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.data, 1.0);
        return t;
    }

    public static Tensor randn(Random rng, int... shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = rng.nextGaussian();
        }
        return t;
    }

    public static Tensor randUniform(Random rng, double low, double high, int... shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = low + rng.nextDouble() * (high - low);
        }
        return t;
    }

    public static Tensor kaiming(Random rng, int fanIn, int fanOut) {
        double std = Math.sqrt(2.0 / fanIn);
        Tensor t = new Tensor(fanIn, fanOut);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = rng.nextGaussian() * std;
        }
        return t;
    }

    // Getters
    public int[] getShape() { return shape.clone(); }
    public int dim(int i) { return shape[i]; }
    public int size() { return data.length; }

    // Element access
    private int index(int... indices) {
        int idx = 0;
        for (int i = 0; i < indices.length; i++) {
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    public double get(int... indices) {
        return data[index(indices)];
    }

    public void set(double value, int... indices) {
        data[index(indices)] = value;
    }

    // Operations
    public Tensor clone() {
        return new Tensor(data.clone(), shape);
    }

    public Tensor add(Tensor other) {
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    public Tensor sub(Tensor other) {
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    public Tensor mul(Tensor other) {
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    public Tensor scale(double s) {
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = data[i] * s;
        }
        return result;
    }

    public Tensor addScalar(double s) {
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = data[i] + s;
        }
        return result;
    }

    public Tensor apply(DoubleUnaryOperator op) {
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = op.applyAsDouble(data[i]);
        }
        return result;
    }

    // Matrix multiply for 2D tensors
    public Tensor matmul(Tensor other) {
        if (shape.length != 2 || other.shape.length != 2) {
            throw new IllegalArgumentException("matmul requires 2D tensors");
        }
        int m = shape[0];
        int k = shape[1];
        int n = other.shape[1];
        if (k != other.shape[0]) {
            throw new IllegalArgumentException("Incompatible shapes for matmul");
        }
        
        Tensor result = new Tensor(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int p = 0; p < k; p++) {
                    sum += get(i, p) * other.get(p, j);
                }
                result.set(sum, i, j);
            }
        }
        return result;
    }

    // Activations
    public Tensor relu() {
        return apply(x -> Math.max(0, x));
    }

    public Tensor sigmoid() {
        return apply(x -> 1.0 / (1.0 + Math.exp(-Math.max(-500, Math.min(500, x)))));
    }

    public Tensor silu() {
        return apply(x -> x / (1.0 + Math.exp(-Math.max(-500, Math.min(500, x)))));
    }

    public Tensor gelu() {
        return apply(x -> 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x))));
    }

    // Statistics
    public double sum() {
        double s = 0;
        for (double v : data) s += v;
        return s;
    }

    public double mean() {
        return sum() / data.length;
    }

    public double variance() {
        double m = mean();
        double v = 0;
        for (double d : data) {
            double diff = d - m;
            v += diff * diff;
        }
        return v / data.length;
    }

    @Override
    public String toString() {
        return String.format("Tensor%s mean=%.4f", Arrays.toString(shape), mean());
    }
}
