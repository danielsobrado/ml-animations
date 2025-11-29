package nn

import "math"

// LossType represents the type of loss function.
type LossType int

const (
	MSE LossType = iota
	BinaryCrossEntropy
	CrossEntropy
)

// Loss represents a loss function.
type Loss struct {
	Type LossType
}

// NewLoss creates a new loss function.
func NewLoss(t LossType) Loss {
	return Loss{Type: t}
}

// Compute computes the loss between predictions and targets.
// predictions: (batch_size, output_dim)
// targets: (batch_size, output_dim)
func (l Loss) Compute(predictions, targets *Tensor) float64 {
	switch l.Type {
	case MSE:
		return computeMSE(predictions, targets)
	case BinaryCrossEntropy:
		return computeBCE(predictions, targets)
	case CrossEntropy:
		return computeCE(predictions, targets)
	default:
		return computeMSE(predictions, targets)
	}
}

// Gradient computes the gradient of the loss w.r.t. predictions.
func (l Loss) Gradient(predictions, targets *Tensor) *Tensor {
	switch l.Type {
	case MSE:
		return gradientMSE(predictions, targets)
	case BinaryCrossEntropy:
		return gradientBCE(predictions, targets)
	case CrossEntropy:
		return gradientCE(predictions, targets)
	default:
		return gradientMSE(predictions, targets)
	}
}

// computeMSE: Mean Squared Error = mean((y - ŷ)²)
func computeMSE(pred, target *Tensor) float64 {
	diff := pred.Sub(target)
	squared := diff.Mul(diff)
	return squared.Mean()
}

// gradientMSE: d(MSE)/dŷ = 2(ŷ - y) / n
func gradientMSE(pred, target *Tensor) *Tensor {
	r, c := pred.Shape()
	n := float64(r * c)
	diff := pred.Sub(target)
	return diff.Scale(2.0 / n)
}

// computeBCE: Binary Cross-Entropy = -mean(y*log(ŷ) + (1-y)*log(1-ŷ))
func computeBCE(pred, target *Tensor) float64 {
	eps := 1e-7
	r, c := pred.Shape()
	sum := 0.0

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			p := clamp(pred.At(i, j), eps, 1-eps)
			t := target.At(i, j)
			sum += t*math.Log(p) + (1-t)*math.Log(1-p)
		}
	}

	return -sum / float64(r*c)
}

// gradientBCE: d(BCE)/dŷ = (ŷ - y) / (ŷ * (1 - ŷ))
// Simplified when combined with sigmoid: ŷ - y
func gradientBCE(pred, target *Tensor) *Tensor {
	eps := 1e-7
	r, c := pred.Shape()
	result := Zeros(r, c)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			p := clamp(pred.At(i, j), eps, 1-eps)
			t := target.At(i, j)
			// Gradient: (p - t) / (p * (1 - p))
			// For numerical stability with sigmoid output, we use (p - t) / batch_size
			result.Set(i, j, (p-t)/float64(r))
		}
	}

	return result
}

// computeCE: Cross-Entropy = -mean(sum(y * log(ŷ)))
func computeCE(pred, target *Tensor) float64 {
	eps := 1e-7
	r, c := pred.Shape()
	sum := 0.0

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			p := clamp(pred.At(i, j), eps, 1-eps)
			t := target.At(i, j)
			sum += t * math.Log(p)
		}
	}

	return -sum / float64(r)
}

// gradientCE: For softmax + cross-entropy: ŷ - y (per sample)
func gradientCE(pred, target *Tensor) *Tensor {
	r, _ := pred.Shape()
	diff := pred.Sub(target)
	return diff.Scale(1.0 / float64(r))
}

// clamp restricts a value to [min, max].
func clamp(v, minV, maxV float64) float64 {
	if v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
}

// String returns the name of the loss function.
func (l Loss) String() string {
	switch l.Type {
	case MSE:
		return "MSE"
	case BinaryCrossEntropy:
		return "BinaryCrossEntropy"
	case CrossEntropy:
		return "CrossEntropy"
	default:
		return "Unknown"
	}
}
