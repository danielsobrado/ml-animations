package com.minidiffusion;

import java.util.Random;

/**
 * U-Net architecture for diffusion models.
 */
public class UNet {

    /**
     * Residual block with time embedding.
     */
    public static class ResBlock {
        private final Layers.GroupNorm norm1;
        private final Layers.Conv2d conv1;
        private final Layers.GroupNorm norm2;
        private final Layers.Conv2d conv2;
        private final Layers.Linear timeProj;
        private final Layers.Conv2d shortcut;
        private final int inChannels;
        private final int outChannels;

        public ResBlock(int inChannels, int outChannels, int timeEmbedDim, Random rng) {
            this.inChannels = inChannels;
            this.outChannels = outChannels;

            this.norm1 = new Layers.GroupNorm(32, inChannels);
            this.conv1 = new Layers.Conv2d(inChannels, outChannels, 3, 1, 1, rng);
            this.norm2 = new Layers.GroupNorm(32, outChannels);
            this.conv2 = new Layers.Conv2d(outChannels, outChannels, 3, 1, 1, rng);
            this.timeProj = new Layers.Linear(timeEmbedDim, outChannels, rng);

            // Shortcut for channel mismatch
            if (inChannels != outChannels) {
                this.shortcut = new Layers.Conv2d(inChannels, outChannels, 1, 1, 0, rng);
            } else {
                this.shortcut = null;
            }
        }

        public Tensor forward(Tensor x, Tensor timeEmbed) {
            Tensor residual = shortcut != null ? shortcut.forward(x) : x;

            // First conv block
            Tensor h = norm1.forward(x);
            h = silu(h);
            h = conv1.forward(h);

            // Add time embedding (project and broadcast)
            Tensor timeProj2d = timeProj.forward(timeEmbed);
            h = addTimeEmbed(h, timeProj2d);

            // Second conv block
            h = norm2.forward(h);
            h = silu(h);
            h = conv2.forward(h);

            // Add residual
            return h.add(residual);
        }

        private Tensor silu(Tensor x) {
            Tensor result = x.clone();
            for (int i = 0; i < x.size(); i++) {
                double val = x.data[i];
                result.data[i] = val / (1.0 + Math.exp(-val));
            }
            return result;
        }

        private Tensor addTimeEmbed(Tensor x, Tensor timeEmbed) {
            // x: [batch, channels, height, width]
            // timeEmbed: [batch, channels]
            int batch = x.dim(0);
            int channels = x.dim(1);
            int height = x.dim(2);
            int width = x.dim(3);

            Tensor result = x.clone();
            for (int b = 0; b < batch; b++) {
                for (int c = 0; c < channels; c++) {
                    double te = timeEmbed.get(b, c);
                    for (int h = 0; h < height; h++) {
                        for (int w = 0; w < width; w++) {
                            result.set(result.get(b, c, h, w) + te, b, c, h, w);
                        }
                    }
                }
            }
            return result;
        }

        public int parameterCount() {
            int count = norm1.parameterCount() + conv1.parameterCount();
            count += norm2.parameterCount() + conv2.parameterCount();
            count += timeProj.parameterCount();
            if (shortcut != null) {
                count += shortcut.parameterCount();
            }
            return count;
        }
    }

    /**
     * Downsample block using strided convolution.
     */
    public static class Downsample {
        private final Layers.Conv2d conv;

        public Downsample(int channels, Random rng) {
            this.conv = new Layers.Conv2d(channels, channels, 3, 2, 1, rng);
        }

        public Tensor forward(Tensor x) {
            return conv.forward(x);
        }

        public int parameterCount() {
            return conv.parameterCount();
        }
    }

    /**
     * Upsample block using nearest-neighbor upsampling + convolution.
     */
    public static class Upsample {
        private final Layers.Conv2d conv;

        public Upsample(int channels, Random rng) {
            this.conv = new Layers.Conv2d(channels, channels, 3, 1, 1, rng);
        }

        public Tensor forward(Tensor x) {
            // Nearest neighbor 2x upsampling
            int batch = x.dim(0);
            int channels = x.dim(1);
            int height = x.dim(2);
            int width = x.dim(3);

            Tensor upsampled = Tensor.zeros(batch, channels, height * 2, width * 2);
            for (int b = 0; b < batch; b++) {
                for (int c = 0; c < channels; c++) {
                    for (int h = 0; h < height; h++) {
                        for (int w = 0; w < width; w++) {
                            double val = x.get(b, c, h, w);
                            upsampled.set(val, b, c, h * 2, w * 2);
                            upsampled.set(val, b, c, h * 2, w * 2 + 1);
                            upsampled.set(val, b, c, h * 2 + 1, w * 2);
                            upsampled.set(val, b, c, h * 2 + 1, w * 2 + 1);
                        }
                    }
                }
            }

            return conv.forward(upsampled);
        }

        public int parameterCount() {
            return conv.parameterCount();
        }
    }

    // U-Net components
    private final int inChannels;
    private final int outChannels;
    private final int modelChannels;
    private final int timeEmbedDim;

    private final Layers.Conv2d inputConv;
    private final Layers.Linear timeEmbed1;
    private final Layers.Linear timeEmbed2;

    // Encoder
    private final ResBlock encoderRes1;
    private final Downsample down1;
    private final ResBlock encoderRes2;
    private final Downsample down2;

    // Middle
    private final ResBlock midRes1;
    private final ResBlock midRes2;

    // Decoder
    private final Upsample up1;
    private final ResBlock decoderRes1;
    private final Upsample up2;
    private final ResBlock decoderRes2;

    private final Layers.GroupNorm outNorm;
    private final Layers.Conv2d outConv;

    public UNet(int inChannels, int outChannels, int modelChannels, Random rng) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.modelChannels = modelChannels;
        this.timeEmbedDim = modelChannels * 4;

        // Input projection
        this.inputConv = new Layers.Conv2d(inChannels, modelChannels, 3, 1, 1, rng);

        // Time embedding MLP
        this.timeEmbed1 = new Layers.Linear(modelChannels, timeEmbedDim, rng);
        this.timeEmbed2 = new Layers.Linear(timeEmbedDim, timeEmbedDim, rng);

        // Encoder
        this.encoderRes1 = new ResBlock(modelChannels, modelChannels, timeEmbedDim, rng);
        this.down1 = new Downsample(modelChannels, rng);
        this.encoderRes2 = new ResBlock(modelChannels, modelChannels * 2, timeEmbedDim, rng);
        this.down2 = new Downsample(modelChannels * 2, rng);

        // Middle
        this.midRes1 = new ResBlock(modelChannels * 2, modelChannels * 2, timeEmbedDim, rng);
        this.midRes2 = new ResBlock(modelChannels * 2, modelChannels * 2, timeEmbedDim, rng);

        // Decoder (with skip connections, so double channels)
        this.up1 = new Upsample(modelChannels * 2, rng);
        this.decoderRes1 = new ResBlock(modelChannels * 4, modelChannels, timeEmbedDim, rng);
        this.up2 = new Upsample(modelChannels, rng);
        this.decoderRes2 = new ResBlock(modelChannels * 2, modelChannels, timeEmbedDim, rng);

        // Output
        this.outNorm = new Layers.GroupNorm(32, modelChannels);
        this.outConv = new Layers.Conv2d(modelChannels, outChannels, 3, 1, 1, rng);
    }

    /**
     * Forward pass through U-Net.
     */
    public Tensor forward(Tensor x, int timestep) {
        int batch = x.dim(0);

        // Time embedding
        Tensor t = timestepEmbedding(timestep, modelChannels, batch);
        t = timeEmbed1.forward(t);
        t = silu(t);
        t = timeEmbed2.forward(t);

        // Input projection
        Tensor h = inputConv.forward(x);

        // Encoder with skip connections
        Tensor h1 = encoderRes1.forward(h, t); // [batch, modelChannels, H, W]
        Tensor h2 = down1.forward(h1); // [batch, modelChannels, H/2, W/2]
        Tensor h3 = encoderRes2.forward(h2, t); // [batch, modelChannels*2, H/2, W/2]
        Tensor h4 = down2.forward(h3); // [batch, modelChannels*2, H/4, W/4]

        // Middle
        Tensor mid = midRes1.forward(h4, t);
        mid = midRes2.forward(mid, t);

        // Decoder with skip connections
        Tensor up = up1.forward(mid); // [batch, modelChannels*2, H/2, W/2]
        up = concat(up, h3); // [batch, modelChannels*4, H/2, W/2]
        up = decoderRes1.forward(up, t); // [batch, modelChannels, H/2, W/2]

        up = up2.forward(up); // [batch, modelChannels, H, W]
        up = concat(up, h1); // [batch, modelChannels*2, H, W]
        up = decoderRes2.forward(up, t); // [batch, modelChannels, H, W]

        // Output
        Tensor out = outNorm.forward(up);
        out = silu(out);
        out = outConv.forward(out);

        return out;
    }

    /**
     * Sinusoidal timestep embedding.
     */
    private Tensor timestepEmbedding(int timestep, int dim, int batch) {
        int halfDim = dim / 2;
        double logMax = Math.log(10000.0);

        Tensor emb = Tensor.zeros(batch, dim);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < halfDim; i++) {
                double freq = Math.exp(-logMax * i / halfDim);
                double angle = timestep * freq;
                emb.set(Math.sin(angle), b, i);
                emb.set(Math.cos(angle), b, i + halfDim);
            }
        }
        return emb;
    }

    private Tensor silu(Tensor x) {
        Tensor result = x.clone();
        for (int i = 0; i < x.size(); i++) {
            double val = x.data[i];
            result.data[i] = val / (1.0 + Math.exp(-val));
        }
        return result;
    }

    /**
     * Concatenate two tensors along the channel dimension.
     */
    private Tensor concat(Tensor a, Tensor b) {
        int batch = a.dim(0);
        int ca = a.dim(1);
        int cb = b.dim(1);
        int height = a.dim(2);
        int width = a.dim(3);

        Tensor result = Tensor.zeros(batch, ca + cb, height, width);

        // Copy a
        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < ca; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        result.set(a.get(n, c, h, w), n, c, h, w);
                    }
                }
            }
        }

        // Copy b
        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < cb; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        result.set(b.get(n, c, h, w), n, ca + c, h, w);
                    }
                }
            }
        }

        return result;
    }

    public int parameterCount() {
        int count = inputConv.parameterCount();
        count += timeEmbed1.parameterCount() + timeEmbed2.parameterCount();
        count += encoderRes1.parameterCount() + down1.parameterCount();
        count += encoderRes2.parameterCount() + down2.parameterCount();
        count += midRes1.parameterCount() + midRes2.parameterCount();
        count += up1.parameterCount() + decoderRes1.parameterCount();
        count += up2.parameterCount() + decoderRes2.parameterCount();
        count += outNorm.parameterCount() + outConv.parameterCount();
        return count;
    }
}
