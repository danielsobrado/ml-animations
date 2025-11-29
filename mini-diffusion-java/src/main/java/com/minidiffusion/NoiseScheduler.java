package com.minidiffusion;

import java.util.Random;

/**
 * Noise scheduler for diffusion models.
 * 
 * Implements linear, cosine, and quadratic beta schedules.
 */
public class NoiseScheduler {

    public enum ScheduleType {
        LINEAR,
        COSINE,
        QUADRATIC
    }

    private final int numTimesteps;
    private final double[] betas;
    private final double[] alphas;
    private final double[] alphasCumprod;
    private final double[] sqrtAlphasCumprod;
    private final double[] sqrtOneMinusAlphasCumprod;
    private final double[] posteriorVariance;
    private final ScheduleType scheduleType;

    public NoiseScheduler(int numTimesteps, ScheduleType scheduleType, double betaStart, double betaEnd) {
        this.numTimesteps = numTimesteps;
        this.scheduleType = scheduleType;

        this.betas = new double[numTimesteps];
        this.alphas = new double[numTimesteps];
        this.alphasCumprod = new double[numTimesteps];
        this.sqrtAlphasCumprod = new double[numTimesteps];
        this.sqrtOneMinusAlphasCumprod = new double[numTimesteps];
        this.posteriorVariance = new double[numTimesteps];

        // Generate beta schedule
        switch (scheduleType) {
            case LINEAR:
                for (int t = 0; t < numTimesteps; t++) {
                    betas[t] = betaStart + (betaEnd - betaStart) * t / (numTimesteps - 1);
                }
                break;

            case COSINE:
                double s = 0.008;
                for (int t = 0; t < numTimesteps; t++) {
                    double t1 = (double) t / numTimesteps;
                    double t2 = (double) (t + 1) / numTimesteps;
                    double alpha1 = Math.pow(Math.cos((t1 + s) / (1 + s) * Math.PI / 2), 2);
                    double alpha2 = Math.pow(Math.cos((t2 + s) / (1 + s) * Math.PI / 2), 2);
                    betas[t] = Math.min(1 - alpha2 / alpha1, 0.999);
                }
                break;

            case QUADRATIC:
                double sqrtBetaStart = Math.sqrt(betaStart);
                double sqrtBetaEnd = Math.sqrt(betaEnd);
                for (int t = 0; t < numTimesteps; t++) {
                    double sqrtBeta = sqrtBetaStart + (sqrtBetaEnd - sqrtBetaStart) * t / (numTimesteps - 1);
                    betas[t] = sqrtBeta * sqrtBeta;
                }
                break;
        }

        // Compute alphas and cumulative products
        double cumProd = 1.0;
        for (int t = 0; t < numTimesteps; t++) {
            alphas[t] = 1.0 - betas[t];
            cumProd *= alphas[t];
            alphasCumprod[t] = cumProd;
            sqrtAlphasCumprod[t] = Math.sqrt(cumProd);
            sqrtOneMinusAlphasCumprod[t] = Math.sqrt(1.0 - cumProd);
        }

        // Compute posterior variance
        for (int t = 0; t < numTimesteps; t++) {
            if (t == 0) {
                posteriorVariance[t] = betas[t];
            } else {
                posteriorVariance[t] = betas[t] * (1.0 - alphasCumprod[t - 1]) / (1.0 - alphasCumprod[t]);
            }
        }
    }

    /**
     * Create a linear schedule scheduler.
     */
    public static NoiseScheduler linear(int numTimesteps) {
        return new NoiseScheduler(numTimesteps, ScheduleType.LINEAR, 0.0001, 0.02);
    }

    /**
     * Create a cosine schedule scheduler.
     */
    public static NoiseScheduler cosine(int numTimesteps) {
        return new NoiseScheduler(numTimesteps, ScheduleType.COSINE, 0.0001, 0.02);
    }

    /**
     * Create a quadratic schedule scheduler.
     */
    public static NoiseScheduler quadratic(int numTimesteps) {
        return new NoiseScheduler(numTimesteps, ScheduleType.QUADRATIC, 0.0001, 0.02);
    }

    /**
     * Add noise to a sample at a given timestep.
     * q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
     */
    public Tensor addNoise(Tensor sample, Tensor noise, int timestep) {
        double sqrtAlpha = sqrtAlphasCumprod[timestep];
        double sqrtOneMinusAlpha = sqrtOneMinusAlphasCumprod[timestep];

        Tensor result = sample.mul(sqrtAlpha).add(noise.mul(sqrtOneMinusAlpha));
        return result;
    }

    /**
     * DDPM step: predict x_{t-1} from x_t and noise prediction.
     */
    public Tensor step(Tensor sample, Tensor noisePredict, int timestep, Random rng) {
        double beta = betas[timestep];
        double alpha = alphas[timestep];
        double alphaCumprod = alphasCumprod[timestep];
        double sqrtAlpha = Math.sqrt(alpha);
        double sqrtOneMinusAlphaCumprod = sqrtOneMinusAlphasCumprod[timestep];

        // Predict x_0
        // x_0 = (x_t - sqrt(1-alpha_cumprod) * noise) / sqrt(alpha_cumprod)
        Tensor x0Pred = sample.sub(noisePredict.mul(sqrtOneMinusAlphaCumprod)).mul(1.0 / sqrtAlphasCumprod[timestep]);

        // Compute mean for x_{t-1}
        double coeff1 = beta * Math.sqrt(alphaCumprod > 0 ? alphasCumprod[Math.max(0, timestep - 1)] : 1.0) / (1.0 - alphaCumprod);
        double coeff2 = (1.0 - (timestep > 0 ? alphasCumprod[timestep - 1] : 1.0)) * sqrtAlpha / (1.0 - alphaCumprod);

        Tensor mean = x0Pred.mul(coeff1).add(sample.mul(coeff2));

        // Add noise if not the last step
        if (timestep > 0) {
            double stdDev = Math.sqrt(posteriorVariance[timestep]);
            Tensor noise = Tensor.randn(rng, sample.shape());
            mean = mean.add(noise.mul(stdDev));
        }

        return mean;
    }

    /**
     * DDIM step: deterministic sampling.
     */
    public Tensor stepDdim(Tensor sample, Tensor noisePredict, int timestep, int prevTimestep) {
        double alphaCumprod = alphasCumprod[timestep];
        double alphaCumprodPrev = prevTimestep >= 0 ? alphasCumprod[prevTimestep] : 1.0;

        double sqrtAlphaCumprod = sqrtAlphasCumprod[timestep];
        double sqrtOneMinusAlphaCumprod = sqrtOneMinusAlphasCumprod[timestep];

        // Predict x_0
        Tensor x0Pred = sample.sub(noisePredict.mul(sqrtOneMinusAlphaCumprod)).mul(1.0 / sqrtAlphaCumprod);

        // DDIM formula
        double sqrtAlphaCumprodPrev = Math.sqrt(alphaCumprodPrev);
        double sqrtOneMinusAlphaCumprodPrev = Math.sqrt(1.0 - alphaCumprodPrev);

        Tensor result = x0Pred.mul(sqrtAlphaCumprodPrev).add(noisePredict.mul(sqrtOneMinusAlphaCumprodPrev));
        return result;
    }

    public int getNumTimesteps() {
        return numTimesteps;
    }

    public double getBeta(int t) {
        return betas[t];
    }

    public double getAlpha(int t) {
        return alphas[t];
    }

    public double getAlphaCumprod(int t) {
        return alphasCumprod[t];
    }

    public double getSqrtAlphaCumprod(int t) {
        return sqrtAlphasCumprod[t];
    }

    public double getSqrtOneMinusAlphaCumprod(int t) {
        return sqrtOneMinusAlphasCumprod[t];
    }
}
