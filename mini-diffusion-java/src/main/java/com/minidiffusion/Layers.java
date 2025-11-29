package com.minidiffusion;

import java.util.Random;

/**
 * Neural network layers for diffusion models.
 */
public class Layers {

    /**
     * Linear (fully connected) layer.
     */
    public static class Linear {
        private Tensor weight;
        private Tensor bias;
        private final int inFeatures;
        private final int outFeatures;

        public Linear(int inFeatures, int outFeatures, Random rng) {
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
            this.weight = Tensor.kaiming(rng, inFeatures, outFeatures);
            this.bias = Tensor.zeros(1, outFeatures);
        }

        public Tensor forward(Tensor x) {
            // x: [batch, inFeatures] -> [batch, outFeatures]
            Tensor out = x.matmul(weight);
            // Add bias (broadcast)
            int batch = x.dim(0);
            for (int b = 0; b < batch; b++) {
                for (int j = 0; j < outFeatures; j++) {
                    out.set(out.get(b, j) + bias.get(0, j), b, j);
                }
            }
            return out;
        }

        public int parameterCount() {
            return inFeatures * outFeatures + outFeatures;
        }
    }

    /**
     * 2D Convolution layer.
     */
    public static class Conv2d {
        private final Tensor weight;
        private final Tensor bias;
        private final int inChannels;
        private final int outChannels;
        private final int kernelSize;
        private final int stride;
        private final int padding;

        public Conv2d(int inChannels, int outChannels, int kernelSize, int stride, int padding, Random rng) {
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;

            // Kaiming initialization
            int fanIn = inChannels * kernelSize * kernelSize;
            double std = Math.sqrt(2.0 / fanIn);
            this.weight = Tensor.zeros(outChannels, inChannels, kernelSize, kernelSize);
            for (int oc = 0; oc < outChannels; oc++) {
                for (int ic = 0; ic < inChannels; ic++) {
                    for (int kh = 0; kh < kernelSize; kh++) {
                        for (int kw = 0; kw < kernelSize; kw++) {
                            weight.set(rng.nextGaussian() * std, oc, ic, kh, kw);
                        }
                    }
                }
            }
            this.bias = Tensor.zeros(outChannels);
        }

        public Tensor forward(Tensor x) {
            int batch = x.dim(0);
            int inH = x.dim(2);
            int inW = x.dim(3);

            int outH = (inH + 2 * padding - kernelSize) / stride + 1;
            int outW = (inW + 2 * padding - kernelSize) / stride + 1;

            Tensor output = Tensor.zeros(batch, outChannels, outH, outW);

            // Naive convolution
            for (int b = 0; b < batch; b++) {
                for (int oc = 0; oc < outChannels; oc++) {
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            double sum = bias.get(oc);

                            for (int ic = 0; ic < inChannels; ic++) {
                                for (int kh = 0; kh < kernelSize; kh++) {
                                    for (int kw = 0; kw < kernelSize; kw++) {
                                        int ih = oh * stride + kh - padding;
                                        int iw = ow * stride + kw - padding;

                                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                            sum += x.get(b, ic, ih, iw) * weight.get(oc, ic, kh, kw);
                                        }
                                    }
                                }
                            }
                            output.set(sum, b, oc, oh, ow);
                        }
                    }
                }
            }

            return output;
        }

        public int parameterCount() {
            return outChannels * inChannels * kernelSize * kernelSize + outChannels;
        }
    }

    /**
     * Group Normalization layer.
     */
    public static class GroupNorm {
        private final int numGroups;
        private final int numChannels;
        private final Tensor gamma;
        private final Tensor beta;
        private final double eps;

        public GroupNorm(int numGroups, int numChannels) {
            this.numGroups = Math.min(numGroups, numChannels);
            this.numChannels = numChannels;
            this.gamma = Tensor.ones(numChannels);
            this.beta = Tensor.zeros(numChannels);
            this.eps = 1e-5;
        }

        public Tensor forward(Tensor x) {
            int batch = x.dim(0);
            int channels = x.dim(1);
            int height = x.dim(2);
            int width = x.dim(3);

            Tensor result = x.clone();

            int channelsPerGroup = channels / numGroups;

            for (int b = 0; b < batch; b++) {
                for (int g = 0; g < numGroups; g++) {
                    // Compute mean and variance for this group
                    double sum = 0;
                    int count = 0;
                    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup && c < channels; c++) {
                        for (int h = 0; h < height; h++) {
                            for (int w = 0; w < width; w++) {
                                sum += x.get(b, c, h, w);
                                count++;
                            }
                        }
                    }
                    double mean = sum / count;

                    double varSum = 0;
                    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup && c < channels; c++) {
                        for (int h = 0; h < height; h++) {
                            for (int w = 0; w < width; w++) {
                                double diff = x.get(b, c, h, w) - mean;
                                varSum += diff * diff;
                            }
                        }
                    }
                    double std = Math.sqrt(varSum / count + eps);

                    // Normalize
                    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup && c < channels; c++) {
                        double gam = gamma.get(c);
                        double bet = beta.get(c);
                        for (int h = 0; h < height; h++) {
                            for (int w = 0; w < width; w++) {
                                double val = (x.get(b, c, h, w) - mean) / std;
                                result.set(gam * val + bet, b, c, h, w);
                            }
                        }
                    }
                }
            }

            return result;
        }

        public int parameterCount() {
            return 2 * numChannels;
        }
    }
}
