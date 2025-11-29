package nn

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// TrainingConfig holds training configuration.
type TrainingConfig struct {
	Epochs            int
	BatchSize         int
	ValidationSplit   float64
	Shuffle           bool
	Verbose           bool
	EarlyStopPatience int
}

// DefaultConfig returns default training configuration.
func DefaultConfig() TrainingConfig {
	return TrainingConfig{
		Epochs:            100,
		BatchSize:         32,
		ValidationSplit:   0.2,
		Shuffle:           true,
		Verbose:           true,
		EarlyStopPatience: 10,
	}
}

// History stores training history.
type History struct {
	TrainLoss     []float64
	TrainAccuracy []float64
	ValLoss       []float64
	ValAccuracy   []float64
}

// BestValLoss returns the best validation loss.
func (h *History) BestValLoss() float64 {
	if len(h.ValLoss) == 0 {
		return 0
	}
	best := h.ValLoss[0]
	for _, v := range h.ValLoss {
		if v < best {
			best = v
		}
	}
	return best
}

// BestValAccuracy returns the best validation accuracy.
func (h *History) BestValAccuracy() float64 {
	if len(h.ValAccuracy) == 0 {
		return 0
	}
	best := h.ValAccuracy[0]
	for _, v := range h.ValAccuracy {
		if v > best {
			best = v
		}
	}
	return best
}

// Trainer handles training loops.
type Trainer struct {
	Config TrainingConfig
	rng    *rand.Rand
}

// NewTrainer creates a new trainer.
func NewTrainer(config TrainingConfig, seed int64) *Trainer {
	return &Trainer{
		Config: config,
		rng:    rand.New(rand.NewSource(seed)),
	}
}

// Fit trains the network on data.
func (t *Trainer) Fit(
	network *Network,
	x, y *Tensor,
	loss Loss,
	optimizer Optimizer,
) *History {
	history := &History{}

	// Split data into train/validation
	nSamples := x.Rows()
	nVal := int(float64(nSamples) * t.Config.ValidationSplit)
	nTrain := nSamples - nVal

	// Create indices for shuffling
	indices := make([]int, nSamples)
	for i := range indices {
		indices[i] = i
	}
	t.rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// Split indices
	trainIdx := indices[:nTrain]
	valIdx := indices[nTrain:]

	// Create train/val tensors
	xTrain := selectRows(x, trainIdx)
	yTrain := selectRows(y, trainIdx)
	xVal := selectRows(x, valIdx)
	yVal := selectRows(y, valIdx)

	bestValLoss := math.Inf(1)
	patienceCounter := 0

	for epoch := 0; epoch < t.Config.Epochs; epoch++ {
		// Shuffle training data
		if t.Config.Shuffle {
			shuffleData(xTrain, yTrain, t.rng)
		}

		// Training
		trainLoss, trainAcc := t.trainEpoch(network, xTrain, yTrain, loss, optimizer)

		// Validation
		valLoss, valAcc := t.evaluate(network, xVal, yVal, loss)

		history.TrainLoss = append(history.TrainLoss, trainLoss)
		history.TrainAccuracy = append(history.TrainAccuracy, trainAcc)
		history.ValLoss = append(history.ValLoss, valLoss)
		history.ValAccuracy = append(history.ValAccuracy, valAcc)

		if t.Config.Verbose && (epoch+1)%10 == 0 {
			fmt.Printf("Epoch %3d: train_loss=%.4f, train_acc=%.2f%%, val_loss=%.4f, val_acc=%.2f%%\n",
				epoch+1, trainLoss, trainAcc*100, valLoss, valAcc*100)
		}

		// Early stopping
		if valLoss < bestValLoss {
			bestValLoss = valLoss
			patienceCounter = 0
		} else {
			patienceCounter++
			if patienceCounter >= t.Config.EarlyStopPatience {
				if t.Config.Verbose {
					fmt.Printf("Early stopping at epoch %d\n", epoch+1)
				}
				break
			}
		}
	}

	return history
}

// trainEpoch runs one training epoch.
func (t *Trainer) trainEpoch(
	network *Network,
	x, y *Tensor,
	lossFunc Loss,
	optimizer Optimizer,
) (float64, float64) {
	nSamples := x.Rows()
	batchSize := t.Config.BatchSize
	nBatches := (nSamples + batchSize - 1) / batchSize

	totalLoss := 0.0
	correct := 0
	total := 0

	for b := 0; b < nBatches; b++ {
		start := b * batchSize
		end := start + batchSize
		if end > nSamples {
			end = nSamples
		}

		xBatch := x.SliceRows(start, end)
		yBatch := y.SliceRows(start, end)

		// Forward pass
		pred := network.Forward(xBatch)

		// Compute loss
		batchLoss := lossFunc.Compute(pred, yBatch)
		totalLoss += batchLoss * float64(end-start)

		// Compute accuracy
		batchCorrect := computeAccuracy(pred, yBatch)
		correct += batchCorrect
		total += end - start

		// Backward pass
		grad := lossFunc.Gradient(pred, yBatch)
		network.Backward(grad)

		// Update weights
		optimizer.Step(network.DenseLayers())
	}

	return totalLoss / float64(nSamples), float64(correct) / float64(total)
}

// evaluate evaluates the network on data.
func (t *Trainer) evaluate(
	network *Network,
	x, y *Tensor,
	lossFunc Loss,
) (float64, float64) {
	pred := network.Forward(x)
	loss := lossFunc.Compute(pred, y)
	correct := computeAccuracy(pred, y)
	return loss, float64(correct) / float64(x.Rows())
}

// computeAccuracy computes classification accuracy.
func computeAccuracy(pred, target *Tensor) int {
	r, c := pred.Shape()
	correct := 0

	for i := 0; i < r; i++ {
		if c == 1 {
			// Binary classification
			p := pred.At(i, 0)
			t := target.At(i, 0)
			predClass := 0.0
			if p >= 0.5 {
				predClass = 1.0
			}
			if math.Abs(predClass-t) < 0.5 {
				correct++
			}
		} else {
			// Multi-class classification
			predArgmax := argmax(pred.GetRow(i))
			targetArgmax := argmax(target.GetRow(i))
			if predArgmax == targetArgmax {
				correct++
			}
		}
	}

	return correct
}

// argmax returns the index of the maximum value.
func argmax(arr []float64) int {
	maxIdx := 0
	maxVal := arr[0]
	for i, v := range arr {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// selectRows creates a new tensor with selected rows.
func selectRows(t *Tensor, indices []int) *Tensor {
	_, c := t.Shape()
	data := make([]float64, len(indices)*c)
	for i, idx := range indices {
		for j := 0; j < c; j++ {
			data[i*c+j] = t.At(idx, j)
		}
	}
	return &Tensor{Data: mat.NewDense(len(indices), c, data)}
}

// shuffleData shuffles x and y in sync.
func shuffleData(x, y *Tensor, rng *rand.Rand) {
	n := x.Rows()
	_, cx := x.Shape()
	_, cy := y.Shape()

	for i := n - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		// Swap rows i and j in both x and y
		for k := 0; k < cx; k++ {
			xi := x.At(i, k)
			xj := x.At(j, k)
			x.Set(i, k, xj)
			x.Set(j, k, xi)
		}
		for k := 0; k < cy; k++ {
			yi := y.At(i, k)
			yj := y.At(j, k)
			y.Set(i, k, yj)
			y.Set(j, k, yi)
		}
	}
}
