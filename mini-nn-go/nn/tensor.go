// Package nn provides neural network primitives from scratch in Go.
package nn

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Tensor wraps a gonum matrix for neural network operations.
type Tensor struct {
	Data *mat.Dense
}

// NewTensor creates a tensor from a gonum Dense matrix.
func NewTensor(data *mat.Dense) *Tensor {
	return &Tensor{Data: data}
}

// Zeros creates a tensor of zeros with given shape.
func Zeros(rows, cols int) *Tensor {
	return &Tensor{Data: mat.NewDense(rows, cols, nil)}
}

// Ones creates a tensor of ones with given shape.
func Ones(rows, cols int) *Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 1.0
	}
	return &Tensor{Data: mat.NewDense(rows, cols, data)}
}

// RandomUniform creates a tensor with random values from uniform distribution.
func RandomUniform(rows, cols int, low, high float64, rng *rand.Rand) *Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = low + rng.Float64()*(high-low)
	}
	return &Tensor{Data: mat.NewDense(rows, cols, data)}
}

// RandomNormal creates a tensor with random values from normal distribution.
func RandomNormal(rows, cols int, mean, std float64, rng *rand.Rand) *Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rng.NormFloat64()*std + mean
	}
	return &Tensor{Data: mat.NewDense(rows, cols, data)}
}

// Xavier initialization (good for sigmoid/tanh).
func Xavier(fanIn, fanOut int, rng *rand.Rand) *Tensor {
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
	return RandomUniform(fanIn, fanOut, -limit, limit, rng)
}

// He initialization (good for ReLU).
func He(fanIn, fanOut int, rng *rand.Rand) *Tensor {
	std := math.Sqrt(2.0 / float64(fanIn))
	return RandomNormal(fanIn, fanOut, 0, std, rng)
}

// Shape returns (rows, cols).
func (t *Tensor) Shape() (int, int) {
	return t.Data.Dims()
}

// Rows returns number of rows.
func (t *Tensor) Rows() int {
	r, _ := t.Data.Dims()
	return r
}

// Cols returns number of columns.
func (t *Tensor) Cols() int {
	_, c := t.Data.Dims()
	return c
}

// Clone creates a deep copy.
func (t *Tensor) Clone() *Tensor {
	r, c := t.Shape()
	result := mat.NewDense(r, c, nil)
	result.Copy(t.Data)
	return &Tensor{Data: result}
}

// MatMul performs matrix multiplication: self @ other.
func (t *Tensor) MatMul(other *Tensor) *Tensor {
	r, _ := t.Shape()
	_, c := other.Shape()
	result := mat.NewDense(r, c, nil)
	result.Mul(t.Data, other.Data)
	return &Tensor{Data: result}
}

// Transpose returns the transpose of the tensor.
func (t *Tensor) Transpose() *Tensor {
	r, c := t.Shape()
	result := mat.NewDense(c, r, nil)
	result.Copy(t.Data.T())
	return &Tensor{Data: result}
}

// Add performs element-wise addition.
func (t *Tensor) Add(other *Tensor) *Tensor {
	r, c := t.Shape()
	result := mat.NewDense(r, c, nil)
	result.Add(t.Data, other.Data)
	return &Tensor{Data: result}
}

// Sub performs element-wise subtraction.
func (t *Tensor) Sub(other *Tensor) *Tensor {
	r, c := t.Shape()
	result := mat.NewDense(r, c, nil)
	result.Sub(t.Data, other.Data)
	return &Tensor{Data: result}
}

// Mul performs element-wise multiplication.
func (t *Tensor) Mul(other *Tensor) *Tensor {
	r, c := t.Shape()
	result := mat.NewDense(r, c, nil)
	result.MulElem(t.Data, other.Data)
	return &Tensor{Data: result}
}

// Scale multiplies all elements by a scalar.
func (t *Tensor) Scale(s float64) *Tensor {
	r, c := t.Shape()
	result := mat.NewDense(r, c, nil)
	result.Scale(s, t.Data)
	return &Tensor{Data: result}
}

// AddScalar adds a scalar to all elements.
func (t *Tensor) AddScalar(s float64) *Tensor {
	r, c := t.Shape()
	data := make([]float64, r*c)
	raw := t.Data.RawMatrix().Data
	for i, v := range raw {
		data[i] = v + s
	}
	return &Tensor{Data: mat.NewDense(r, c, data)}
}

// Apply applies a function element-wise.
func (t *Tensor) Apply(f func(float64) float64) *Tensor {
	r, c := t.Shape()
	data := make([]float64, r*c)
	raw := t.Data.RawMatrix().Data
	for i, v := range raw {
		data[i] = f(v)
	}
	return &Tensor{Data: mat.NewDense(r, c, data)}
}

// AddBias adds a bias vector to each row.
func (t *Tensor) AddBias(bias []float64) *Tensor {
	r, c := t.Shape()
	result := mat.NewDense(r, c, nil)
	result.Copy(t.Data)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, result.At(i, j)+bias[j])
		}
	}
	return &Tensor{Data: result}
}

// Sum returns the sum of all elements.
func (t *Tensor) Sum() float64 {
	sum := 0.0
	raw := t.Data.RawMatrix().Data
	for _, v := range raw {
		sum += v
	}
	return sum
}

// Mean returns the mean of all elements.
func (t *Tensor) Mean() float64 {
	r, c := t.Shape()
	return t.Sum() / float64(r*c)
}

// SumAxis sums along an axis (0=rows, 1=cols).
func (t *Tensor) SumAxis(axis int) []float64 {
	r, c := t.Shape()
	if axis == 0 {
		// Sum each column
		result := make([]float64, c)
		for j := 0; j < c; j++ {
			for i := 0; i < r; i++ {
				result[j] += t.Data.At(i, j)
			}
		}
		return result
	}
	// Sum each row
	result := make([]float64, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result[i] += t.Data.At(i, j)
		}
	}
	return result
}

// GetRow returns a copy of the i-th row.
func (t *Tensor) GetRow(i int) []float64 {
	_, c := t.Shape()
	row := make([]float64, c)
	for j := 0; j < c; j++ {
		row[j] = t.Data.At(i, j)
	}
	return row
}

// SliceRows returns rows from start to end (exclusive).
func (t *Tensor) SliceRows(start, end int) *Tensor {
	_, c := t.Shape()
	rows := end - start
	data := make([]float64, rows*c)
	for i := 0; i < rows; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = t.Data.At(start+i, j)
		}
	}
	return &Tensor{Data: mat.NewDense(rows, c, data)}
}

// FromSlice creates a tensor from a 2D slice.
func FromSlice(data [][]float64) *Tensor {
	if len(data) == 0 {
		return &Tensor{Data: mat.NewDense(0, 0, nil)}
	}
	rows := len(data)
	cols := len(data[0])
	flat := make([]float64, rows*cols)
	for i, row := range data {
		for j, v := range row {
			flat[i*cols+j] = v
		}
	}
	return &Tensor{Data: mat.NewDense(rows, cols, flat)}
}

// At returns the element at (i, j).
func (t *Tensor) At(i, j int) float64 {
	return t.Data.At(i, j)
}

// Set sets the element at (i, j).
func (t *Tensor) Set(i, j int, v float64) {
	t.Data.Set(i, j, v)
}

// Print displays the tensor.
func (t *Tensor) Print() {
	r, c := t.Shape()
	fmt.Printf("Tensor(%d x %d)\n", r, c)
	fa := mat.Formatted(t.Data, mat.Prefix("  "), mat.Squeeze())
	fmt.Printf("  %v\n", fa)
}
