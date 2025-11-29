package diffusion

import (
	"fmt"
	"math/rand"
)

// SamplerConfig holds sampler configuration.
type SamplerConfig struct {
	NumSteps      int
	GuidanceScale float64
	UseDDIM       bool
	Eta           float64 // For DDIM: 0 = deterministic, 1 = DDPM-like
}

// DefaultSamplerConfig returns default sampler config.
func DefaultSamplerConfig() SamplerConfig {
	return SamplerConfig{
		NumSteps:      50,
		GuidanceScale: 1.0,
		UseDDIM:       true,
		Eta:           0.0,
	}
}

// Sampler generates images from noise.
type Sampler struct {
	Config    SamplerConfig
	Scheduler *NoiseScheduler
	rng       *rand.Rand
}

// NewSampler creates a new sampler.
func NewSampler(config SamplerConfig, scheduler *NoiseScheduler, rng *rand.Rand) *Sampler {
	return &Sampler{
		Config:    config,
		Scheduler: scheduler,
		rng:       rng,
	}
}

// Sample generates images from random noise.
func (s *Sampler) Sample(model *UNet, shape []int, verbose bool) *Tensor {
	if s.Config.UseDDIM {
		return s.SampleDDIM(model, shape, verbose)
	}
	return s.SampleDDPM(model, shape, verbose)
}

// SampleDDPM performs stochastic DDPM sampling.
func (s *Sampler) SampleDDPM(model *UNet, shape []int, verbose bool) *Tensor {
	batchSize := shape[0]
	totalSteps := s.Scheduler.Config.NumTimesteps

	// Start from pure noise
	x := Randn(shape, s.rng)

	// Iterate from T-1 to 0
	for t := totalSteps - 1; t >= 0; t-- {
		if verbose && t%100 == 0 {
			fmt.Printf("  DDPM Step %d/%d\n", totalSteps-t, totalSteps)
		}

		timesteps := make([]int, batchSize)
		for i := range timesteps {
			timesteps[i] = t
		}

		// Predict noise
		predictedNoise := model.Forward(x, timesteps)

		// Denoising step
		x = s.Scheduler.Step(x, predictedNoise, t)
	}

	return x
}

// SampleDDIM performs deterministic DDIM sampling.
func (s *Sampler) SampleDDIM(model *UNet, shape []int, verbose bool) *Tensor {
	batchSize := shape[0]
	totalSteps := s.Scheduler.Config.NumTimesteps
	numSteps := s.Config.NumSteps

	// Create time schedule (evenly spaced)
	timeSteps := make([]int, numSteps)
	for i := 0; i < numSteps; i++ {
		timeSteps[i] = totalSteps - 1 - i*(totalSteps/numSteps)
	}

	// Start from pure noise
	x := Randn(shape, s.rng)

	for step, t := range timeSteps {
		if verbose && step%10 == 0 {
			fmt.Printf("  DDIM Step %d/%d (t=%d)\n", step+1, numSteps, t)
		}

		timestepsBatch := make([]int, batchSize)
		for i := range timestepsBatch {
			timestepsBatch[i] = t
		}

		// Predict noise
		predictedNoise := model.Forward(x, timestepsBatch)

		// DDIM update
		alphaT := s.Scheduler.AlphasCumprod[t]

		// Get alpha for previous timestep
		var alphaPrev float64
		if step < len(timeSteps)-1 {
			alphaPrev = s.Scheduler.AlphasCumprod[timeSteps[step+1]]
		} else {
			alphaPrev = 1.0
		}

		// Predict x0
		sqrtAlphaT := s.Scheduler.SqrtAlphasCumprod[t]
		sqrtOneMinusAlphaT := s.Scheduler.SqrtOneMinusAlphasCumprod[t]

		// x0_pred = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
		x0Pred := x.Sub(predictedNoise.Scale(sqrtOneMinusAlphaT)).Scale(1.0 / sqrtAlphaT)

		// Direction pointing to x_t
		dirXt := predictedNoise.Scale(1.0 - alphaPrev)

		// Compute x_{t-1}
		sqrtAlphaPrev := alphaPrev
		if sqrtAlphaPrev > 0 {
			sqrtAlphaPrev = sqrtFloat64(sqrtAlphaPrev)
		}

		x = x0Pred.Scale(sqrtAlphaPrev).Add(dirXt.Scale(sqrtFloat64(1.0 - alphaPrev)))

		// Add noise if eta > 0
		if s.Config.Eta > 0 && step < len(timeSteps)-1 {
			sigma := s.Config.Eta * sqrtFloat64((1.0-alphaPrev)/(1.0-alphaT)) * sqrtFloat64(1.0-alphaT/alphaPrev)
			noise := Randn(shape, s.rng)
			x = x.Add(noise.Scale(sigma))
		}
	}

	return x
}

func sqrtFloat64(x float64) float64 {
	if x < 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 50; i++ {
		z = (z + x/z) / 2
	}
	return z
}
