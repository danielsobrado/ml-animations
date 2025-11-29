package com.minidiffusion;

import java.util.Random;

/**
 * Sampler for diffusion models.
 * 
 * Implements DDPM and DDIM sampling.
 */
public class Sampler {

    public enum SamplerType {
        DDPM,
        DDIM
    }

    private final NoiseScheduler scheduler;
    private final int numInferenceSteps;
    private final SamplerType samplerType;
    private final int[] timesteps;

    public Sampler(NoiseScheduler scheduler, int numInferenceSteps, SamplerType samplerType) {
        this.scheduler = scheduler;
        this.numInferenceSteps = numInferenceSteps;
        this.samplerType = samplerType;

        // Create timestep schedule (evenly spaced)
        this.timesteps = new int[numInferenceSteps];
        int totalSteps = scheduler.getNumTimesteps();
        for (int i = 0; i < numInferenceSteps; i++) {
            timesteps[i] = totalSteps - 1 - (i * totalSteps / numInferenceSteps);
        }
    }

    /**
     * Sample from pure noise using the model.
     */
    public Tensor sample(UNet model, int[] shape, Random rng) {
        // Start with pure noise
        Tensor x = Tensor.randn(rng, shape);

        // Iterate through timesteps
        for (int i = 0; i < timesteps.length; i++) {
            int t = timesteps[i];
            int prevT = i < timesteps.length - 1 ? timesteps[i + 1] : -1;

            // Predict noise
            Tensor noisePred = model.forward(x, t);

            // Step
            if (samplerType == SamplerType.DDPM) {
                x = scheduler.step(x, noisePred, t, rng);
            } else {
                x = scheduler.stepDdim(x, noisePred, t, prevT);
            }
        }

        return x;
    }

    /**
     * Sample with progress callback.
     */
    public Tensor sample(UNet model, int[] shape, Random rng, ProgressCallback callback) {
        // Start with pure noise
        Tensor x = Tensor.randn(rng, shape);

        // Iterate through timesteps
        for (int i = 0; i < timesteps.length; i++) {
            int t = timesteps[i];
            int prevT = i < timesteps.length - 1 ? timesteps[i + 1] : -1;

            // Predict noise
            Tensor noisePred = model.forward(x, t);

            // Step
            if (samplerType == SamplerType.DDPM) {
                x = scheduler.step(x, noisePred, t, rng);
            } else {
                x = scheduler.stepDdim(x, noisePred, t, prevT);
            }

            if (callback != null) {
                callback.onProgress(i + 1, timesteps.length, t, x);
            }
        }

        return x;
    }

    /**
     * Progress callback interface.
     */
    public interface ProgressCallback {
        void onProgress(int step, int totalSteps, int timestep, Tensor current);
    }

    public int[] getTimesteps() {
        return timesteps.clone();
    }

    public int getNumInferenceSteps() {
        return numInferenceSteps;
    }

    public SamplerType getSamplerType() {
        return samplerType;
    }
}
