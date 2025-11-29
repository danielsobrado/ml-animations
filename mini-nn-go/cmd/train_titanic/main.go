package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"mini-nn-go/nn"
)

func main() {
	fmt.Println("=== Mini-NN Go: Titanic Survival Prediction ===")
	fmt.Println()

	// Set random seed
	rng := rand.New(rand.NewSource(42))

	// Find data files
	dataDir := findDataDir()
	trainPath := filepath.Join(dataDir, "train.csv")
	testPath := filepath.Join(dataDir, "test_with_labels.csv")

	// Check if files exist
	if _, err := os.Stat(trainPath); os.IsNotExist(err) {
		fmt.Printf("Error: Training data not found at %s\n", trainPath)
		fmt.Println("Please ensure the Titanic dataset is available.")
		os.Exit(1)
	}

	fmt.Printf("Loading data from: %s\n", dataDir)

	// Load data
	data, err := nn.LoadTitanicData(trainPath, testPath)
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		os.Exit(1)
	}

	trainRows, trainCols := data.XTrain.Shape()
	testRows, _ := data.XTest.Shape()
	fmt.Printf("Training samples: %d, Features: %d\n", trainRows, trainCols)
	fmt.Printf("Test samples: %d\n", testRows)
	fmt.Println()

	// Build network: 7 -> 32 -> 16 -> 1
	network := nn.NewNetworkBuilder().
		AddDense(trainCols, 32, nn.ReLU, nn.HeInit).
		AddDense(32, 16, nn.ReLU, nn.HeInit).
		AddDense(16, 1, nn.Sigmoid, nn.XavierInit).
		Build()

	fmt.Println("Network Architecture:")
	network.Summary()
	fmt.Println()

	// Create optimizer
	optimizer := nn.NewAdam(0.001, 0.9, 0.999, 1e-8)

	// Create loss function
	loss := nn.NewBinaryCrossEntropyLoss()

	// Create trainer
	config := nn.TrainingConfig{
		Epochs:            200,
		BatchSize:         32,
		ValidationSplit:   0.2,
		Shuffle:           true,
		Verbose:           true,
		EarlyStopPatience: 20,
	}
	trainer := nn.NewTrainer(config, rng)

	// Train
	startTime := time.Now()
	fmt.Println("Training...")
	history := trainer.Fit(network, data.XTrain, data.YTrain, loss, optimizer)
	elapsed := time.Since(startTime)

	fmt.Printf("\nTraining completed in %.2fs\n", elapsed.Seconds())
	fmt.Println()

	// Evaluate on test set
	fmt.Println("=== Test Set Evaluation ===")
	testPred := network.Predict(data.XTest)

	correct := 0
	for i := 0; i < testRows; i++ {
		predVal := testPred.At(i, 0)
		targetVal := data.YTest.At(i, 0)

		predClass := 0
		if predVal >= 0.5 {
			predClass = 1
		}
		targetClass := int(targetVal)

		if predClass == targetClass {
			correct++
		}
	}

	testAcc := float64(correct) / float64(testRows) * 100
	fmt.Printf("Test Accuracy: %.1f%% (%d/%d)\n", testAcc, correct, testRows)

	// Print summary
	fmt.Println()
	fmt.Println("=== Training Summary ===")
	if len(history.TrainLoss) > 0 {
		fmt.Printf("Final Train Loss: %.4f\n", history.TrainLoss[len(history.TrainLoss)-1])
	}
	if len(history.ValLoss) > 0 {
		fmt.Printf("Final Val Loss: %.4f\n", history.ValLoss[len(history.ValLoss)-1])
	}
	if len(history.TrainAccuracy) > 0 {
		fmt.Printf("Final Train Accuracy: %.1f%%\n", history.TrainAccuracy[len(history.TrainAccuracy)-1]*100)
	}
	if len(history.ValAccuracy) > 0 {
		fmt.Printf("Final Val Accuracy: %.1f%%\n", history.ValAccuracy[len(history.ValAccuracy)-1]*100)
	}
	fmt.Printf("Test Accuracy: %.1f%%\n", testAcc)
	fmt.Printf("Epochs trained: %d\n", len(history.TrainLoss))
}

// findDataDir looks for the Titanic data directory.
func findDataDir() string {
	// Try common locations
	paths := []string{
		"data",
		"../data",
		"../../data",
		"../mini-nn/data",
		"../../mini-nn/data",
	}

	for _, p := range paths {
		if _, err := os.Stat(filepath.Join(p, "train.csv")); err == nil {
			return p
		}
	}

	// Default to data directory
	return "data"
}
