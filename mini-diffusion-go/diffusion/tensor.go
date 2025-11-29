package diffusion

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Tensor wraps gonum Dense matrix with additional operations.
type Tensor struct {
	Data  *mat.Dense
	Shape []int
}

// NewTensor creates a tensor with the given shape.
func NewTensor(shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	// For 2D operations, we use rows x cols
	rows, cols := flattenShape(shape)
	return &Tensor{
		Data:  mat.NewDense(rows, cols, make([]float64, rows*cols)),
		Shape: shape,
	}
}

// flattenShape converts multi-dim shape to 2D for gonum
func flattenShape(shape []int) (int, int) {
	if len(shape) == 0 {
		return 1, 1
	}
	if len(shape) == 1 {
		return shape[0], 1
	}
	if len(shape) == 2 {
		return shape[0], shape[1]
	}
	// For higher dims, flatten all but last into rows
	rows := 1
	for i := 0; i < len(shape)-1; i++ {
		rows *= shape[i]
	}
	return rows, shape[len(shape)-1]
}

// Zeros creates a tensor filled with zeros.
func Zeros(shape []int) *Tensor {
	return NewTensor(shape)
}

// Ones creates a tensor filled with ones.
func Ones(shape []int) *Tensor {
	t := NewTensor(shape)
	r, c := t.Data.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			t.Data.Set(i, j, 1.0)
		}
	}
	return t
}

// Randn creates a tensor with random normal values.
func Randn(shape []int, rng *rand.Rand) *Tensor {
	t := NewTensor(shape)
	r, c := t.Data.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			t.Data.Set(i, j, rng.NormFloat64())
		}
	}
	return t
}

// RandUniform creates a tensor with uniform random values.
func RandUniform(shape []int, low, high float64, rng *rand.Rand) *Tensor {
	t := NewTensor(shape)
	r, c := t.Data.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			t.Data.Set(i, j, low+rng.Float64()*(high-low))
		}
	}
	return t
}

// XavierInit creates Xavier/Glorot initialized weights.
func XavierInit(fanIn, fanOut int, rng *rand.Rand) *Tensor {
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
	return RandUniform([]int{fanIn, fanOut}, -limit, limit, rng)
}

// KaimingInit creates He initialization for ReLU networks.
func KaimingInit(fanIn, fanOut int, rng *rand.Rand) *Tensor {
	std := math.Sqrt(2.0 / float64(fanIn))
	t := NewTensor([]int{fanIn, fanOut})
	r, c := t.Data.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			t.Data.Set(i, j, rng.NormFloat64()*std)
		}
	}
	return t
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	r, c := t.Data.Dims()
	newData := mat.NewDense(r, c, nil)
	newData.Copy(t.Data)
	shapeCopy := make([]int, len(t.Shape))
	copy(shapeCopy, t.Shape)
	return &Tensor{Data: newData, Shape: shapeCopy}
}

// Add performs element-wise addition.
func (t *Tensor) Add(other *Tensor) *Tensor {
	result := t.Clone()
	result.Data.Add(t.Data, other.Data)
	return result
}

// Sub performs element-wise subtraction.
func (t *Tensor) Sub(other *Tensor) *Tensor {
	result := t.Clone()
	result.Data.Sub(t.Data, other.Data)
	return result
}

// Mul performs element-wise multiplication.
func (t *Tensor) Mul(other *Tensor) *Tensor {
	result := t.Clone()
	result.Data.MulElem(t.Data, other.Data)
	return result
}

// Scale multiplies all elements by a scalar.
func (t *Tensor) Scale(s float64) *Tensor {
	result := t.Clone()
	result.Data.Scale(s, t.Data)
	return result
}

// AddScalar adds a scalar to all elements.
func (t *Tensor) AddScalar(s float64) *Tensor {
	result := t.Clone()
	r, c := result.Data.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Data.Set(i, j, result.Data.At(i, j)+s)
		}
	}
	return result
}

// MatMul performs matrix multiplication.
func (t *Tensor) MatMul(other *Tensor) *Tensor {
	r1, _ := t.Data.Dims()
	_, c2 := other.Data.Dims()
	result := mat.NewDense(r1, c2, nil)
	result.Mul(t.Data, other.Data)
	return &Tensor{Data: result, Shape: []int{r1, c2}}
}

// Apply applies a function to each element.
func (t *Tensor) Apply(f func(float64) float64) *Tensor {
	result := t.Clone()
	r, c := result.Data.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Data.Set(i, j, f(result.Data.At(i, j)))
		}
	}
	return result
}

// ReLU activation.
func (t *Tensor) ReLU() *Tensor {
	return t.Apply(func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0
	})
}

// Sigmoid activation.
func (t *Tensor) Sigmoid() *Tensor {
	return t.Apply(func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-clip(x, -500, 500)))
	})
}

// SiLU (Swish) activation: x * sigmoid(x)
func (t *Tensor) SiLU() *Tensor {
	return t.Apply(func(x float64) float64 {
		return x / (1.0 + math.Exp(-clip(x, -500, 500)))
	})
}

// GELU activation (approximation).
func (t *Tensor) GELU() *Tensor {
	return t.Apply(func(x float64) float64 {
		return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x+0.044715*x*x*x)))
	})
}

// Sqrt element-wise square root.
func (t *Tensor) Sqrt() *Tensor {
	return t.Apply(math.Sqrt)
}

// Sum returns the sum of all elements.
func (t *Tensor) Sum() float64 {
	sum := 0.0
	r, c := t.Data.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += t.Data.At(i, j)
		}
	}
	return sum
}

// Mean returns the mean of all elements.
func (t *Tensor) Mean() float64 {
	r, c := t.Data.Dims()
	return t.Sum() / float64(r*c)
}

// At returns the element at the given indices.
func (t *Tensor) At(indices ...int) float64 {
	if len(indices) == 2 {
		return t.Data.At(indices[0], indices[1])
	}
	// For higher dims, compute flat index
	r, c := t.Data.Dims()
	idx := 0
	stride := r * c
	for i, s := range indices[:len(indices)-1] {
		stride /= t.Shape[i]
		idx += s * stride
	}
	return t.Data.At(idx/c, idx%c)
}

// Set sets the element at the given indices.
func (t *Tensor) Set(value float64, indices ...int) {
	if len(indices) == 2 {
		t.Data.Set(indices[0], indices[1], value)
		return
	}
	r, c := t.Data.Dims()
	idx := 0
	stride := r * c
	for i, s := range indices[:len(indices)-1] {
		stride /= t.Shape[i]
		idx += s * stride
	}
	t.Data.Set(idx/c, idx%c, value)
}

// Numel returns total number of elements.
func (t *Tensor) Numel() int {
	size := 1
	for _, s := range t.Shape {
		size *= s
	}
	return size
}

func clip(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}
