package main

import (
	"fmt"
	"math/rand"
	"time"

	"mini-diffusion-go/diffusion"
)

func main() {
	fmt.Println("=== Mini-Diffusion Go: Demo ===")
	fmt.Println()

	// Set random seed
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Configuration
	imageSize := 16 // Small for demo
	channels := 3
	batchSize := 2

	fmt.Println("Configuration:")
	fmt.Printf("  Image size: %dx%d\n", imageSize, imageSize)
	fmt.Printf("  Channels: %d\n", channels)
	fmt.Printf("  Batch size: %d\n", batchSize)
	fmt.Println()

	// Create noise scheduler
	diffConfig := diffusion.DefaultDiffusionConfig()
	diffConfig.NumTimesteps = 100 // Reduced for demo
	scheduler := diffusion.NewNoiseScheduler(diffConfig, rng)

	fmt.Println("Noise Scheduler:")
	fmt.Printf("  Timesteps: %d\n", diffConfig.NumTimesteps)
	fmt.Printf("  Schedule: %s\n", diffConfig.Schedule)
	fmt.Printf("  Beta range: [%.6f, %.6f]\n", diffConfig.BetaStart, diffConfig.BetaEnd)
	fmt.Println()

	// Demo: Forward diffusion process
	fmt.Println("=== Forward Diffusion Demo ===")

	// Create a "clean" image (simulated)
	x0 := diffusion.RandUniform([]int{1, channels, imageSize, imageSize}, 0, 1, rng)
	fmt.Printf("Original image - mean: %.4f\n", x0.Mean())

	// Add noise at different timesteps
	timestepsToShow := []int{0, 25, 50, 75, 99}
	for _, t := range timestepsToShow {
		noisy, _ := scheduler.AddNoise(x0, t)
		fmt.Printf("  t=%3d: mean=%.4f, signal_ratio=%.4f\n",
			t, noisy.Mean(), scheduler.SqrtAlphasCumprod[t])
	}
	fmt.Println()

	// Create U-Net
	fmt.Println("=== U-Net Model ===")
	modelChannels := 32
	channelMult := []int{1, 2}
	numResBlocks := 1

	unet := diffusion.NewUNet(
		channels,
		modelChannels,
		channelMult,
		numResBlocks,
		rng,
	)

	fmt.Printf("Model channels: %d\n", modelChannels)
	fmt.Printf("Channel multipliers: %v\n", channelMult)
	fmt.Printf("Res blocks per level: %d\n", numResBlocks)
	fmt.Printf("Total parameters: %d\n", unet.ParameterCount())
	fmt.Println()

	// Demo: Single forward pass
	fmt.Println("=== Forward Pass Demo ===")
	testInput := diffusion.Randn([]int{batchSize, channels, imageSize, imageSize}, rng)
	testTimesteps := []int{50, 50}

	fmt.Printf("Input shape: %v\n", testInput.Shape)
	output := unet.Forward(testInput, testTimesteps)
	fmt.Printf("Output shape: %v\n", output.Shape)
	fmt.Printf("Output mean: %.4f\n", output.Mean())
	fmt.Println()

	// Demo: Sampling (abbreviated)
	fmt.Println("=== Sampling Demo ===")
	fmt.Println("Note: With random weights, output will be noise-like")
	fmt.Println()

	samplerConfig := diffusion.SamplerConfig{
		NumSteps: 10, // Very few steps for demo
		UseDDIM:  true,
		Eta:      0.0,
	}
	sampler := diffusion.NewSampler(samplerConfig, scheduler, rng)

	fmt.Printf("Sampler: DDIM with %d steps\n", samplerConfig.NumSteps)
	fmt.Println("Generating...")

	generated := sampler.Sample(unet, []int{1, channels, imageSize, imageSize}, true)
	fmt.Printf("\nGenerated image shape: %v\n", generated.Shape)
	fmt.Printf("Generated mean: %.4f\n", generated.Mean())
	fmt.Println()

	// Timestep embedding demo
	fmt.Println("=== Timestep Embedding Demo ===")
	embDim := 64
	timesteps := []int{0, 100, 500, 999}
	emb := diffusion.GetTimestepEmbedding(timesteps, embDim)
	fmt.Printf("Timesteps: %v\n", timesteps)
	fmt.Printf("Embedding shape: %v\n", emb.Shape)
	fmt.Printf("Embedding dim: %d\n", embDim)

	// Show first few values for t=0
	fmt.Print("t=0 embedding (first 8): [")
	for i := 0; i < 8; i++ {
		fmt.Printf("%.3f", emb.At(0, i))
		if i < 7 {
			fmt.Print(", ")
		}
	}
	fmt.Println("]")

	fmt.Println()
	fmt.Println("✅ Mini-Diffusion Go demo completed!")
}
