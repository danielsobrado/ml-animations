package com.minidiffusion;

import java.util.Random;

/**
 * Mini-Diffusion Demo
 * 
 * Demonstrates the diffusion model components:
 * - Tensor operations
 * - Noise scheduler
 * - U-Net architecture
 * - Sampling
 */
public class Demo {

    public static void main(String[] args) {
        System.out.println("=== Mini-Diffusion Java Demo ===\n");

        Random rng = new Random(42);

        // Demo 1: Tensor operations
        demoTensorOps(rng);

        // Demo 2: Noise scheduler
        demoNoiseScheduler(rng);

        // Demo 3: U-Net architecture
        demoUNet(rng);

        // Demo 4: Sampling
        demoSampling(rng);

        System.out.println("\n=== Demo Complete ===");
    }

    private static void demoTensorOps(Random rng) {
        System.out.println("--- Tensor Operations ---");

        // Create tensors
        Tensor zeros = Tensor.zeros(2, 3, 4, 4);
        System.out.printf("zeros shape: [%d, %d, %d, %d]%n", 
            zeros.dim(0), zeros.dim(1), zeros.dim(2), zeros.dim(3));

        Tensor ones = Tensor.ones(2, 3, 4, 4);
        System.out.printf("ones shape: [%d, %d, %d, %d]%n",
            ones.dim(0), ones.dim(1), ones.dim(2), ones.dim(3));

        Tensor randn = Tensor.randn(rng, new int[]{2, 3, 4, 4});
        System.out.printf("randn mean: %.4f, std: %.4f%n", randn.mean(), randn.std());

        // Arithmetic
        Tensor sum = zeros.add(ones);
        System.out.printf("zeros + ones mean: %.4f%n", sum.mean());

        Tensor scaled = randn.mul(2.0);
        System.out.printf("randn * 2 mean: %.4f, std: %.4f%n", scaled.mean(), scaled.std());

        // Activations
        Tensor x = Tensor.randn(rng, new int[]{1, 1, 2, 2});
        System.out.printf("Original values: [%.4f, %.4f, %.4f, %.4f]%n",
            x.get(0, 0, 0, 0), x.get(0, 0, 0, 1), x.get(0, 0, 1, 0), x.get(0, 0, 1, 1));

        Tensor relu = x.relu();
        System.out.printf("After ReLU: [%.4f, %.4f, %.4f, %.4f]%n",
            relu.get(0, 0, 0, 0), relu.get(0, 0, 0, 1), relu.get(0, 0, 1, 0), relu.get(0, 0, 1, 1));

        System.out.println();
    }

    private static void demoNoiseScheduler(Random rng) {
        System.out.println("--- Noise Scheduler ---");

        // Create schedulers with different schedules
        NoiseScheduler linear = NoiseScheduler.linear(1000);
        NoiseScheduler cosine = NoiseScheduler.cosine(1000);
        NoiseScheduler quadratic = NoiseScheduler.quadratic(1000);

        System.out.println("Linear schedule:");
        System.out.printf("  beta[0]=%.6f, beta[500]=%.6f, beta[999]=%.6f%n",
            linear.getBeta(0), linear.getBeta(500), linear.getBeta(999));
        System.out.printf("  alpha_cumprod[0]=%.6f, alpha_cumprod[500]=%.6f, alpha_cumprod[999]=%.6f%n",
            linear.getAlphaCumprod(0), linear.getAlphaCumprod(500), linear.getAlphaCumprod(999));

        System.out.println("Cosine schedule:");
        System.out.printf("  alpha_cumprod[0]=%.6f, alpha_cumprod[500]=%.6f, alpha_cumprod[999]=%.6f%n",
            cosine.getAlphaCumprod(0), cosine.getAlphaCumprod(500), cosine.getAlphaCumprod(999));

        System.out.println("Quadratic schedule:");
        System.out.printf("  alpha_cumprod[0]=%.6f, alpha_cumprod[500]=%.6f, alpha_cumprod[999]=%.6f%n",
            quadratic.getAlphaCumprod(0), quadratic.getAlphaCumprod(500), quadratic.getAlphaCumprod(999));

        // Demo add_noise
        Tensor sample = Tensor.ones(1, 3, 8, 8);
        Tensor noise = Tensor.randn(rng, new int[]{1, 3, 8, 8});

        System.out.println("\nAdding noise at different timesteps:");
        for (int t : new int[]{0, 250, 500, 750, 999}) {
            Tensor noisy = linear.addNoise(sample, noise, t);
            System.out.printf("  t=%d: mean=%.4f, std=%.4f%n", t, noisy.mean(), noisy.std());
        }

        System.out.println();
    }

    private static void demoUNet(Random rng) {
        System.out.println("--- U-Net Architecture ---");

        // Create a small U-Net
        int inChannels = 3;
        int outChannels = 3;
        int modelChannels = 32;

        UNet unet = new UNet(inChannels, outChannels, modelChannels, rng);
        System.out.printf("U-Net created: in=%d, out=%d, model=%d%n", 
            inChannels, outChannels, modelChannels);
        System.out.printf("Total parameters: %,d%n", unet.parameterCount());

        // Forward pass
        Tensor x = Tensor.randn(rng, new int[]{1, inChannels, 32, 32});
        System.out.printf("Input shape: [%d, %d, %d, %d]%n",
            x.dim(0), x.dim(1), x.dim(2), x.dim(3));

        long startTime = System.currentTimeMillis();
        Tensor out = unet.forward(x, 500);
        long endTime = System.currentTimeMillis();

        System.out.printf("Output shape: [%d, %d, %d, %d]%n",
            out.dim(0), out.dim(1), out.dim(2), out.dim(3));
        System.out.printf("Forward pass time: %d ms%n", endTime - startTime);
        System.out.printf("Output mean: %.4f, std: %.4f%n", out.mean(), out.std());

        System.out.println();
    }

    private static void demoSampling(Random rng) {
        System.out.println("--- Sampling ---");

        // Create scheduler and model
        NoiseScheduler scheduler = NoiseScheduler.linear(1000);
        UNet model = new UNet(3, 3, 32, rng);

        // Create sampler
        int numSteps = 10;  // Use few steps for demo
        Sampler ddpmSampler = new Sampler(scheduler, numSteps, Sampler.SamplerType.DDPM);
        Sampler ddimSampler = new Sampler(scheduler, numSteps, Sampler.SamplerType.DDIM);

        System.out.printf("Sampler with %d inference steps%n", numSteps);
        System.out.println("Timesteps: " + formatArray(ddpmSampler.getTimesteps()));

        // Sample with DDPM
        System.out.println("\nDDPM Sampling:");
        int[] shape = {1, 3, 16, 16};  // Small for demo
        long startTime = System.currentTimeMillis();
        Tensor ddpmSample = ddpmSampler.sample(model, shape, rng, (step, total, t, current) -> {
            System.out.printf("  Step %d/%d (t=%d): mean=%.4f%n", step, total, t, current.mean());
        });
        long endTime = System.currentTimeMillis();
        System.out.printf("DDPM sampling time: %d ms%n", endTime - startTime);
        System.out.printf("Final sample mean: %.4f, std: %.4f%n", ddpmSample.mean(), ddpmSample.std());

        // Sample with DDIM
        System.out.println("\nDDIM Sampling:");
        startTime = System.currentTimeMillis();
        Tensor ddimSample = ddimSampler.sample(model, shape, rng, (step, total, t, current) -> {
            System.out.printf("  Step %d/%d (t=%d): mean=%.4f%n", step, total, t, current.mean());
        });
        endTime = System.currentTimeMillis();
        System.out.printf("DDIM sampling time: %d ms%n", endTime - startTime);
        System.out.printf("Final sample mean: %.4f, std: %.4f%n", ddimSample.mean(), ddimSample.std());
    }

    private static String formatArray(int[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(arr[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}
