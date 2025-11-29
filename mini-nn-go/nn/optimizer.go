package nn

import "math"

// OptimizerType represents the type of optimizer.
type OptimizerType int

const (
	SGDType OptimizerType = iota
	SGDMomentumType
	AdamType
)

// Optimizer interface for all optimizers.
type Optimizer interface {
	Step(layers []*Dense)
	LearningRate() float64
}

// SGD implements stochastic gradient descent.
type SGD struct {
	LR float64
}

// NewSGD creates an SGD optimizer.
func NewSGD(lr float64) *SGD {
	return &SGD{LR: lr}
}

// Step updates weights using gradients.
func (o *SGD) Step(layers []*Dense) {
	for _, layer := range layers {
		// Update weights: W = W - lr * grad_W
		r, c := layer.Weights.Shape()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				w := layer.Weights.At(i, j)
				g := layer.GradWeights.At(i, j)
				layer.Weights.Set(i, j, w-o.LR*g)
			}
		}

		// Update biases: b = b - lr * grad_b
		for i := range layer.Bias {
			layer.Bias[i] -= o.LR * layer.GradBias[i]
		}
	}
}

// LearningRate returns the learning rate.
func (o *SGD) LearningRate() float64 {
	return o.LR
}

// SGDMomentum implements SGD with momentum.
type SGDMomentum struct {
	LR       float64
	Momentum float64
	vWeights []*Tensor
	vBias    [][]float64
	init     bool
}

// NewSGDMomentum creates an SGD optimizer with momentum.
func NewSGDMomentum(lr, momentum float64) *SGDMomentum {
	return &SGDMomentum{LR: lr, Momentum: momentum}
}

// Step updates weights using gradients with momentum.
func (o *SGDMomentum) Step(layers []*Dense) {
	if !o.init {
		o.vWeights = make([]*Tensor, len(layers))
		o.vBias = make([][]float64, len(layers))
		for i, layer := range layers {
			o.vWeights[i] = Zeros(layer.InFeatures, layer.OutFeatures)
			o.vBias[i] = make([]float64, layer.OutFeatures)
		}
		o.init = true
	}

	for i, layer := range layers {
		r, c := layer.Weights.Shape()

		// Update velocity and weights
		for ri := 0; ri < r; ri++ {
			for ci := 0; ci < c; ci++ {
				v := o.Momentum*o.vWeights[i].At(ri, ci) - o.LR*layer.GradWeights.At(ri, ci)
				o.vWeights[i].Set(ri, ci, v)
				layer.Weights.Set(ri, ci, layer.Weights.At(ri, ci)+v)
			}
		}

		// Update bias velocity and biases
		for bi := range layer.Bias {
			o.vBias[i][bi] = o.Momentum*o.vBias[i][bi] - o.LR*layer.GradBias[bi]
			layer.Bias[bi] += o.vBias[i][bi]
		}
	}
}

// LearningRate returns the learning rate.
func (o *SGDMomentum) LearningRate() float64 {
	return o.LR
}

// Adam implements the Adam optimizer.
type Adam struct {
	LR      float64
	Beta1   float64
	Beta2   float64
	Epsilon float64
	mW      []*Tensor // First moment for weights
	vW      []*Tensor // Second moment for weights
	mB      [][]float64
	vB      [][]float64
	t       int // Time step
	init    bool
}

// NewAdam creates an Adam optimizer with default hyperparameters.
func NewAdam(lr float64) *Adam {
	return &Adam{
		LR:      lr,
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
	}
}

// NewAdamFull creates an Adam optimizer with custom hyperparameters.
func NewAdamFull(lr, beta1, beta2, epsilon float64) *Adam {
	return &Adam{
		LR:      lr,
		Beta1:   beta1,
		Beta2:   beta2,
		Epsilon: epsilon,
	}
}

// Step updates weights using Adam algorithm.
func (o *Adam) Step(layers []*Dense) {
	if !o.init {
		o.mW = make([]*Tensor, len(layers))
		o.vW = make([]*Tensor, len(layers))
		o.mB = make([][]float64, len(layers))
		o.vB = make([][]float64, len(layers))
		for i, layer := range layers {
			o.mW[i] = Zeros(layer.InFeatures, layer.OutFeatures)
			o.vW[i] = Zeros(layer.InFeatures, layer.OutFeatures)
			o.mB[i] = make([]float64, layer.OutFeatures)
			o.vB[i] = make([]float64, layer.OutFeatures)
		}
		o.init = true
	}

	o.t++
	biasCorr1 := 1.0 - math.Pow(o.Beta1, float64(o.t))
	biasCorr2 := 1.0 - math.Pow(o.Beta2, float64(o.t))

	for i, layer := range layers {
		r, c := layer.Weights.Shape()

		// Update weights
		for ri := 0; ri < r; ri++ {
			for ci := 0; ci < c; ci++ {
				g := layer.GradWeights.At(ri, ci)

				// Update biased first moment estimate
				m := o.Beta1*o.mW[i].At(ri, ci) + (1-o.Beta1)*g
				o.mW[i].Set(ri, ci, m)

				// Update biased second raw moment estimate
				v := o.Beta2*o.vW[i].At(ri, ci) + (1-o.Beta2)*g*g
				o.vW[i].Set(ri, ci, v)

				// Bias-corrected estimates
				mHat := m / biasCorr1
				vHat := v / biasCorr2

				// Update weights
				w := layer.Weights.At(ri, ci)
				layer.Weights.Set(ri, ci, w-o.LR*mHat/(math.Sqrt(vHat)+o.Epsilon))
			}
		}

		// Update biases
		for bi := range layer.Bias {
			g := layer.GradBias[bi]

			o.mB[i][bi] = o.Beta1*o.mB[i][bi] + (1-o.Beta1)*g
			o.vB[i][bi] = o.Beta2*o.vB[i][bi] + (1-o.Beta2)*g*g

			mHat := o.mB[i][bi] / biasCorr1
			vHat := o.vB[i][bi] / biasCorr2

			layer.Bias[bi] -= o.LR * mHat / (math.Sqrt(vHat) + o.Epsilon)
		}
	}
}

// LearningRate returns the learning rate.
func (o *Adam) LearningRate() float64 {
	return o.LR
}
