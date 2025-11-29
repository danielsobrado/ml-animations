package main

import (
	"fmt"
	"math/rand"
	"time"

	"mini-nn-go/nn"
)

func main() {
	fmt.Println("=== Mini-NN Go: XOR Demo ===")
	fmt.Println()

	// Set random seed for reproducibility
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Generate expanded XOR data for training
	trainSize := 1000
	xTrain, yTrain := nn.GenerateExpandedXORData(trainSize, rng)

	// Build network: 2 -> 8 -> 8 -> 1
	network := nn.NewNetworkBuilder().
		AddDense(2, 8, nn.ReLU, nn.HeInit).
		AddDense(8, 8, nn.ReLU, nn.HeInit).
		AddDense(8, 1, nn.Sigmoid, nn.XavierInit).
		Build()

	fmt.Println("Network Architecture:")
	network.Summary()
	fmt.Println()

	// Create optimizer
	optimizer := nn.NewAdam(0.01, 0.9, 0.999, 1e-8)

	// Create loss function
	loss := nn.NewBinaryCrossEntropyLoss()

	// Create trainer
	config := nn.TrainingConfig{
		Epochs:            100,
		BatchSize:         32,
		ValidationSplit:   0.2,
		Shuffle:           true,
		Verbose:           true,
		EarlyStopPatience: 15,
	}
	trainer := nn.NewTrainer(config, rng)

	// Train
	fmt.Println("Training...")
	history := trainer.Fit(network, xTrain, yTrain, loss, optimizer)

	// Evaluate on original XOR patterns
	fmt.Println()
	fmt.Println("=== Final Evaluation on Pure XOR ===")

	xTest, yTest := nn.GenerateXORData()

	correct := 0
	for i := 0; i < 4; i++ {
		input := xTest.SliceRows(i, i+1)
		target := yTest.At(i, 0)
		pred := network.Predict(input)
		predVal := pred.At(0, 0)
		predClass := 0
		if predVal >= 0.5 {
			predClass = 1
		}
		targetClass := int(target)

		status := "✗"
		if predClass == targetClass {
			correct++
			status = "✓"
		}

		fmt.Printf("Input: [%.0f, %.0f] -> Expected: %d, Predicted: %.4f (%d) %s\n",
			input.At(0, 0), input.At(0, 1), targetClass, predVal, predClass, status)
	}

	accuracy := float64(correct) / 4.0 * 100
	fmt.Printf("\nFinal Accuracy: %.1f%% (%d/4)\n", accuracy, correct)

	// Print final training stats
	if len(history.ValAccuracy) > 0 {
		finalValAcc := history.ValAccuracy[len(history.ValAccuracy)-1] * 100
		fmt.Printf("Final Validation Accuracy: %.1f%%\n", finalValAcc)
	}
}
