package diffusion

import (
	"math"
	"math/rand"
)

// Linear layer: y = xW + b
type Linear struct {
	Weight      *Tensor
	Bias        *Tensor
	InFeatures  int
	OutFeatures int
}

// NewLinear creates a new linear layer with Kaiming initialization.
func NewLinear(inFeatures, outFeatures int, rng *rand.Rand) *Linear {
	return &Linear{
		Weight:      KaimingInit(inFeatures, outFeatures, rng),
		Bias:        Zeros([]int{1, outFeatures}),
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
	}
}

// Forward computes y = xW + b.
func (l *Linear) Forward(x *Tensor) *Tensor {
	out := x.MatMul(l.Weight)
	// Broadcast bias
	r, _ := out.Data.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < l.OutFeatures; j++ {
			out.Data.Set(i, j, out.Data.At(i, j)+l.Bias.Data.At(0, j))
		}
	}
	return out
}

// ParameterCount returns total parameters.
func (l *Linear) ParameterCount() int {
	return l.InFeatures*l.OutFeatures + l.OutFeatures
}

// Conv2d implements 2D convolution.
type Conv2d struct {
	Weight      *Tensor // [outChannels, inChannels, kernelSize, kernelSize]
	Bias        *Tensor // [outChannels]
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	rng         *rand.Rand
}

// NewConv2d creates a new Conv2d layer.
func NewConv2d(inChannels, outChannels, kernelSize, stride, padding int, rng *rand.Rand) *Conv2d {
	fanIn := inChannels * kernelSize * kernelSize
	std := math.Sqrt(2.0 / float64(fanIn))

	weightSize := outChannels * inChannels * kernelSize * kernelSize
	weight := NewTensor([]int{outChannels, inChannels, kernelSize, kernelSize})
	r, c := weight.Data.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			weight.Data.Set(i, j, rng.NormFloat64()*std)
		}
	}
	_ = weightSize

	return &Conv2d{
		Weight:      weight,
		Bias:        Zeros([]int{outChannels, 1}),
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		rng:         rng,
	}
}

// Forward performs convolution.
// Input shape: [batch, channels, height, width]
// Output shape: [batch, outChannels, outHeight, outWidth]
func (c *Conv2d) Forward(x *Tensor) *Tensor {
	batch := x.Shape[0]
	inH := x.Shape[2]
	inW := x.Shape[3]

	outH := (inH+2*c.Padding-c.KernelSize)/c.Stride + 1
	outW := (inW+2*c.Padding-c.KernelSize)/c.Stride + 1

	output := Zeros([]int{batch, c.OutChannels, outH, outW})

	// Naive convolution for clarity
	for b := 0; b < batch; b++ {
		for oc := 0; oc < c.OutChannels; oc++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					sum := c.Bias.At(oc, 0)

					for ic := 0; ic < c.InChannels; ic++ {
						for kh := 0; kh < c.KernelSize; kh++ {
							for kw := 0; kw < c.KernelSize; kw++ {
								ih := oh*c.Stride + kh - c.Padding
								iw := ow*c.Stride + kw - c.Padding

								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									xVal := getVal4D(x, b, ic, ih, iw)
									wVal := getVal4D(c.Weight, oc, ic, kh, kw)
									sum += xVal * wVal
								}
							}
						}
					}
					setVal4D(output, sum, b, oc, oh, ow)
				}
			}
		}
	}

	return output
}

// ParameterCount returns total parameters.
func (c *Conv2d) ParameterCount() int {
	return c.OutChannels*c.InChannels*c.KernelSize*c.KernelSize + c.OutChannels
}

// getVal4D gets value from 4D tensor.
func getVal4D(t *Tensor, b, c, h, w int) float64 {
	idx := b*t.Shape[1]*t.Shape[2]*t.Shape[3] + c*t.Shape[2]*t.Shape[3] + h*t.Shape[3] + w
	r, cols := t.Data.Dims()
	_ = r
	return t.Data.At(idx/cols, idx%cols)
}

// setVal4D sets value in 4D tensor.
func setVal4D(t *Tensor, val float64, b, c, h, w int) {
	idx := b*t.Shape[1]*t.Shape[2]*t.Shape[3] + c*t.Shape[2]*t.Shape[3] + h*t.Shape[3] + w
	_, cols := t.Data.Dims()
	t.Data.Set(idx/cols, idx%cols, val)
}

// GroupNorm implements group normalization.
type GroupNorm struct {
	NumGroups   int
	NumChannels int
	Gamma       *Tensor
	Beta        *Tensor
	Eps         float64
}

// NewGroupNorm creates a GroupNorm layer.
func NewGroupNorm(numGroups, numChannels int) *GroupNorm {
	return &GroupNorm{
		NumGroups:   numGroups,
		NumChannels: numChannels,
		Gamma:       Ones([]int{1, numChannels}),
		Beta:        Zeros([]int{1, numChannels}),
		Eps:         1e-5,
	}
}

// Forward applies group normalization.
func (g *GroupNorm) Forward(x *Tensor) *Tensor {
	// Simplified: just normalize each channel independently
	result := x.Clone()
	batch := x.Shape[0]
	channels := x.Shape[1]
	spatial := 1
	if len(x.Shape) > 2 {
		for i := 2; i < len(x.Shape); i++ {
			spatial *= x.Shape[i]
		}
	}

	for b := 0; b < batch; b++ {
		for c := 0; c < channels; c++ {
			// Compute mean and variance for this channel
			sum := 0.0
			for s := 0; s < spatial; s++ {
				idx := b*channels*spatial + c*spatial + s
				_, cols := result.Data.Dims()
				sum += result.Data.At(idx/cols, idx%cols)
			}
			mean := sum / float64(spatial)

			varSum := 0.0
			for s := 0; s < spatial; s++ {
				idx := b*channels*spatial + c*spatial + s
				_, cols := result.Data.Dims()
				diff := result.Data.At(idx/cols, idx%cols) - mean
				varSum += diff * diff
			}
			variance := varSum / float64(spatial)
			std := math.Sqrt(variance + g.Eps)

			// Normalize
			gamma := g.Gamma.At(0, c)
			beta := g.Beta.At(0, c)
			for s := 0; s < spatial; s++ {
				idx := b*channels*spatial + c*spatial + s
				_, cols := result.Data.Dims()
				val := result.Data.At(idx/cols, idx%cols)
				normalized := (val - mean) / std
				result.Data.Set(idx/cols, idx%cols, gamma*normalized+beta)
			}
		}
	}

	return result
}

// ParameterCount returns total parameters.
func (g *GroupNorm) ParameterCount() int {
	return 2 * g.NumChannels
}
