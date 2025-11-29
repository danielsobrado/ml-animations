package diffusion

import (
	"math"
	"math/rand"
)

// ScheduleType represents the noise schedule type.
type ScheduleType string

const (
	Linear    ScheduleType = "linear"
	Cosine    ScheduleType = "cosine"
	Quadratic ScheduleType = "quadratic"
)

// DiffusionConfig holds configuration for the diffusion process.
type DiffusionConfig struct {
	NumTimesteps int
	BetaStart    float64
	BetaEnd      float64
	Schedule     ScheduleType
}

// DefaultDiffusionConfig returns default configuration.
func DefaultDiffusionConfig() DiffusionConfig {
	return DiffusionConfig{
		NumTimesteps: 1000,
		BetaStart:    1e-4,
		BetaEnd:      0.02,
		Schedule:     Linear,
	}
}

// NoiseScheduler handles adding and removing noise at different timesteps.
type NoiseScheduler struct {
	Config                    DiffusionConfig
	Betas                     []float64
	Alphas                    []float64
	AlphasCumprod             []float64
	SqrtAlphasCumprod         []float64
	SqrtOneMinusAlphasCumprod []float64
	rng                       *rand.Rand
}

// NewNoiseScheduler creates a new noise scheduler.
func NewNoiseScheduler(config DiffusionConfig, rng *rand.Rand) *NoiseScheduler {
	betas := getBetas(config)

	alphas := make([]float64, len(betas))
	for i, b := range betas {
		alphas[i] = 1.0 - b
	}

	alphasCumprod := make([]float64, len(alphas))
	cumprod := 1.0
	for i, a := range alphas {
		cumprod *= a
		alphasCumprod[i] = cumprod
	}

	sqrtAlphasCumprod := make([]float64, len(alphasCumprod))
	sqrtOneMinusAlphasCumprod := make([]float64, len(alphasCumprod))
	for i, a := range alphasCumprod {
		sqrtAlphasCumprod[i] = math.Sqrt(a)
		sqrtOneMinusAlphasCumprod[i] = math.Sqrt(1.0 - a)
	}

	return &NoiseScheduler{
		Config:                    config,
		Betas:                     betas,
		Alphas:                    alphas,
		AlphasCumprod:             alphasCumprod,
		SqrtAlphasCumprod:         sqrtAlphasCumprod,
		SqrtOneMinusAlphasCumprod: sqrtOneMinusAlphasCumprod,
		rng:                       rng,
	}
}

// getBetas generates the beta schedule.
func getBetas(config DiffusionConfig) []float64 {
	t := config.NumTimesteps
	betas := make([]float64, t)

	switch config.Schedule {
	case Linear:
		for i := 0; i < t; i++ {
			betas[i] = config.BetaStart + (config.BetaEnd-config.BetaStart)*float64(i)/float64(t-1)
		}
	case Cosine:
		s := 0.008
		maxBeta := 0.999
		for i := 0; i < t; i++ {
			t1 := float64(i) / float64(t)
			t2 := float64(i+1) / float64(t)
			alphaBarT1 := math.Pow(math.Cos((t1+s)/(1.0+s)*math.Pi/2), 2)
			alphaBarT2 := math.Pow(math.Cos((t2+s)/(1.0+s)*math.Pi/2), 2)
			beta := 1.0 - alphaBarT2/alphaBarT1
			if beta > maxBeta {
				beta = maxBeta
			}
			betas[i] = beta
		}
	case Quadratic:
		for i := 0; i < t; i++ {
			frac := float64(i) / float64(t-1)
			betas[i] = config.BetaStart + (config.BetaEnd-config.BetaStart)*frac*frac
		}
	}

	return betas
}

// AddNoise adds noise to data at timestep t (forward diffusion).
// q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
// Returns: (noisy_data, noise)
func (s *NoiseScheduler) AddNoise(x0 *Tensor, t int) (*Tensor, *Tensor) {
	noise := Randn(x0.Shape, s.rng)
	noisy := s.AddNoiseWith(x0, noise, t)
	return noisy, noise
}

// AddNoiseWith adds specific noise to data at timestep t.
func (s *NoiseScheduler) AddNoiseWith(x0, noise *Tensor, t int) *Tensor {
	sqrtAlpha := s.SqrtAlphasCumprod[t]
	sqrtOneMinusAlpha := s.SqrtOneMinusAlphasCumprod[t]

	// x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
	return x0.Scale(sqrtAlpha).Add(noise.Scale(sqrtOneMinusAlpha))
}

// SampleTimesteps samples random timesteps for a training batch.
func (s *NoiseScheduler) SampleTimesteps(batchSize int) []int {
	timesteps := make([]int, batchSize)
	for i := range timesteps {
		timesteps[i] = s.rng.Intn(s.Config.NumTimesteps)
	}
	return timesteps
}

// Step computes a single denoising step.
// Given x_t and predicted noise, compute x_{t-1}
func (s *NoiseScheduler) Step(xT, predictedNoise *Tensor, t int) *Tensor {
	alphaT := s.Alphas[t]
	alphaBarT := s.AlphasCumprod[t]
	betaT := s.Betas[t]

	// Compute coefficient
	coef := betaT / math.Sqrt(1.0-alphaBarT)

	// Mean: (x_t - coef * predicted_noise) / sqrt(alpha_t)
	mean := xT.Sub(predictedNoise.Scale(coef)).Scale(1.0 / math.Sqrt(alphaT))

	// Add noise (except at final step)
	if t > 0 {
		sigma := math.Sqrt(betaT)
		noise := Randn(xT.Shape, s.rng)
		return mean.Add(noise.Scale(sigma))
	}

	return mean
}

// GetTimestepEmbedding creates sinusoidal timestep embeddings.
func GetTimestepEmbedding(timesteps []int, embDim int) *Tensor {
	batchSize := len(timesteps)
	halfDim := embDim / 2

	emb := Zeros([]int{batchSize, embDim})

	for b, t := range timesteps {
		for i := 0; i < halfDim; i++ {
			freq := math.Exp(-math.Log(10000.0) * float64(i) / float64(halfDim))
			arg := float64(t) * freq
			emb.Data.Set(b, i, math.Sin(arg))
			emb.Data.Set(b, i+halfDim, math.Cos(arg))
		}
	}

	return emb
}
