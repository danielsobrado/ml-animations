package nn

import "math/rand"

// Layer interface for all layer types.
type Layer interface {
	Forward(input *Tensor) *Tensor
	Backward(gradOutput *Tensor) *Tensor
	Parameters() []*Tensor
	Gradients() []*Tensor
	NumParameters() int
	Name() string
}

// Dense is a fully connected layer.
// Computes: output = input @ weights + bias
type Dense struct {
	InFeatures  int
	OutFeatures int
	Weights     *Tensor
	Bias        []float64
	GradWeights *Tensor
	GradBias    []float64
	input       *Tensor // Cached for backprop
}

// NewDense creates a new dense layer with He initialization.
func NewDense(inFeatures, outFeatures int, rng *rand.Rand) *Dense {
	return &Dense{
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
		Weights:     He(inFeatures, outFeatures, rng),
		Bias:        make([]float64, outFeatures),
		GradWeights: Zeros(inFeatures, outFeatures),
		GradBias:    make([]float64, outFeatures),
	}
}

// NewDenseXavier creates a dense layer with Xavier initialization.
func NewDenseXavier(inFeatures, outFeatures int, rng *rand.Rand) *Dense {
	return &Dense{
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
		Weights:     Xavier(inFeatures, outFeatures, rng),
		Bias:        make([]float64, outFeatures),
		GradWeights: Zeros(inFeatures, outFeatures),
		GradBias:    make([]float64, outFeatures),
	}
}

// Forward computes input @ weights + bias.
func (d *Dense) Forward(input *Tensor) *Tensor {
	d.input = input
	output := input.MatMul(d.Weights)
	return output.AddBias(d.Bias)
}

// Backward computes gradients and returns gradient w.r.t. input.
func (d *Dense) Backward(gradOutput *Tensor) *Tensor {
	batchSize := d.input.Rows()

	// Gradient w.r.t. weights: input.T @ gradOutput
	d.GradWeights = d.input.Transpose().MatMul(gradOutput)

	// Gradient w.r.t. bias: sum of gradOutput over batch
	d.GradBias = gradOutput.SumAxis(0)
	for i := range d.GradBias {
		d.GradBias[i] /= float64(batchSize)
	}

	// Scale weight gradients
	d.GradWeights = d.GradWeights.Scale(1.0 / float64(batchSize))

	// Gradient w.r.t. input: gradOutput @ weights.T
	return gradOutput.MatMul(d.Weights.Transpose())
}

// Parameters returns the weight tensor.
func (d *Dense) Parameters() []*Tensor {
	return []*Tensor{d.Weights}
}

// Gradients returns the weight gradient tensor.
func (d *Dense) Gradients() []*Tensor {
	return []*Tensor{d.GradWeights}
}

// NumParameters returns total number of trainable parameters.
func (d *Dense) NumParameters() int {
	return d.InFeatures*d.OutFeatures + d.OutFeatures
}

// Name returns the layer name.
func (d *Dense) Name() string {
	return "Dense"
}

// ActivationLayer wraps an activation function as a layer.
type ActivationLayer struct {
	Activation Activation
	input      *Tensor
	output     *Tensor
}

// NewActivationLayer creates an activation layer.
func NewActivationLayer(activation Activation) *ActivationLayer {
	return &ActivationLayer{Activation: activation}
}

// Forward applies the activation.
func (a *ActivationLayer) Forward(input *Tensor) *Tensor {
	a.input = input
	a.output = a.Activation.Apply(input)
	return a.output
}

// Backward computes gradient w.r.t. input.
func (a *ActivationLayer) Backward(gradOutput *Tensor) *Tensor {
	grad := a.Activation.Derivative(a.input, a.output)
	return gradOutput.Mul(grad)
}

// Parameters returns empty (no learnable parameters).
func (a *ActivationLayer) Parameters() []*Tensor {
	return nil
}

// Gradients returns empty.
func (a *ActivationLayer) Gradients() []*Tensor {
	return nil
}

// NumParameters returns 0.
func (a *ActivationLayer) NumParameters() int {
	return 0
}

// Name returns the layer name.
func (a *ActivationLayer) Name() string {
	return "Activation(" + a.Activation.String() + ")"
}
