package nn

import "math"

// ActivationType represents the type of activation function.
type ActivationType int

const (
	Linear ActivationType = iota
	ReLU
	LeakyReLU
	Sigmoid
	Tanh
	Softmax
)

// Activation holds activation function and its parameter.
type Activation struct {
	Type  ActivationType
	Alpha float64 // For LeakyReLU
}

// NewActivation creates a new activation.
func NewActivation(t ActivationType) Activation {
	return Activation{Type: t, Alpha: 0.01}
}

// NewLeakyReLU creates LeakyReLU with custom alpha.
func NewLeakyReLU(alpha float64) Activation {
	return Activation{Type: LeakyReLU, Alpha: alpha}
}

// Apply applies the activation function to a tensor.
func (a Activation) Apply(x *Tensor) *Tensor {
	switch a.Type {
	case Linear:
		return x.Clone()
	case ReLU:
		return x.Apply(func(v float64) float64 {
			return math.Max(0, v)
		})
	case LeakyReLU:
		alpha := a.Alpha
		return x.Apply(func(v float64) float64 {
			if v > 0 {
				return v
			}
			return alpha * v
		})
	case Sigmoid:
		return x.Apply(sigmoid)
	case Tanh:
		return x.Apply(math.Tanh)
	case Softmax:
		return applySoftmax(x)
	default:
		return x.Clone()
	}
}

// Derivative computes the derivative of the activation.
// For backpropagation, we need the derivative w.r.t. the pre-activation input.
func (a Activation) Derivative(input, output *Tensor) *Tensor {
	switch a.Type {
	case Linear:
		return Ones(input.Rows(), input.Cols())
	case ReLU:
		return input.Apply(func(v float64) float64 {
			if v > 0 {
				return 1.0
			}
			return 0.0
		})
	case LeakyReLU:
		alpha := a.Alpha
		return input.Apply(func(v float64) float64 {
			if v > 0 {
				return 1.0
			}
			return alpha
		})
	case Sigmoid:
		// Sigmoid derivative: σ(x) * (1 - σ(x))
		// We use the cached output
		return output.Apply(func(v float64) float64 {
			return v * (1.0 - v)
		})
	case Tanh:
		// Tanh derivative: 1 - tanh²(x)
		return output.Apply(func(v float64) float64 {
			return 1.0 - v*v
		})
	case Softmax:
		// For softmax with cross-entropy, the gradient is handled specially
		// This returns 1 as a placeholder (actual gradient computed in loss)
		return Ones(input.Rows(), input.Cols())
	default:
		return Ones(input.Rows(), input.Cols())
	}
}

// sigmoid computes 1 / (1 + exp(-x))
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	// For numerical stability with negative x
	expX := math.Exp(x)
	return expX / (1.0 + expX)
}

// applySoftmax applies softmax row-wise (each sample independently).
func applySoftmax(x *Tensor) *Tensor {
	r, c := x.Shape()
	result := Zeros(r, c)

	for i := 0; i < r; i++ {
		// Find max for numerical stability
		maxVal := x.At(i, 0)
		for j := 1; j < c; j++ {
			if x.At(i, j) > maxVal {
				maxVal = x.At(i, j)
			}
		}

		// Compute exp(x - max) and sum
		sum := 0.0
		expVals := make([]float64, c)
		for j := 0; j < c; j++ {
			expVals[j] = math.Exp(x.At(i, j) - maxVal)
			sum += expVals[j]
		}

		// Normalize
		for j := 0; j < c; j++ {
			result.Set(i, j, expVals[j]/sum)
		}
	}

	return result
}

// String returns the name of the activation.
func (a Activation) String() string {
	switch a.Type {
	case Linear:
		return "Linear"
	case ReLU:
		return "ReLU"
	case LeakyReLU:
		return "LeakyReLU"
	case Sigmoid:
		return "Sigmoid"
	case Tanh:
		return "Tanh"
	case Softmax:
		return "Softmax"
	default:
		return "Unknown"
	}
}
