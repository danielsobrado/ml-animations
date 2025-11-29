package nn

import (
	"fmt"
	"math/rand"
)

// Network represents a sequential neural network.
type Network struct {
	layers     []Layer
	denseLayer []*Dense // Keep track of dense layers for optimizer
	rng        *rand.Rand
}

// NewNetwork creates a new empty network.
func NewNetwork(seed int64) *Network {
	return &Network{
		layers:     make([]Layer, 0),
		denseLayer: make([]*Dense, 0),
		rng:        rand.New(rand.NewSource(seed)),
	}
}

// AddDense adds a dense layer.
func (n *Network) AddDense(inFeatures, outFeatures int) *Network {
	dense := NewDense(inFeatures, outFeatures, n.rng)
	n.layers = append(n.layers, dense)
	n.denseLayer = append(n.denseLayer, dense)
	return n
}

// AddDenseXavier adds a dense layer with Xavier initialization.
func (n *Network) AddDenseXavier(inFeatures, outFeatures int) *Network {
	dense := NewDenseXavier(inFeatures, outFeatures, n.rng)
	n.layers = append(n.layers, dense)
	n.denseLayer = append(n.denseLayer, dense)
	return n
}

// AddActivation adds an activation layer.
func (n *Network) AddActivation(activation ActivationType) *Network {
	n.layers = append(n.layers, NewActivationLayer(NewActivation(activation)))
	return n
}

// AddLeakyReLU adds a LeakyReLU activation layer.
func (n *Network) AddLeakyReLU(alpha float64) *Network {
	n.layers = append(n.layers, NewActivationLayer(NewLeakyReLU(alpha)))
	return n
}

// Forward performs a forward pass through all layers.
func (n *Network) Forward(input *Tensor) *Tensor {
	x := input
	for _, layer := range n.layers {
		x = layer.Forward(x)
	}
	return x
}

// Backward performs a backward pass through all layers.
func (n *Network) Backward(gradOutput *Tensor) {
	grad := gradOutput
	for i := len(n.layers) - 1; i >= 0; i-- {
		grad = n.layers[i].Backward(grad)
	}
}

// Predict makes a prediction (forward pass without training).
func (n *Network) Predict(input *Tensor) *Tensor {
	return n.Forward(input)
}

// Summary prints a summary of the network architecture.
func (n *Network) Summary() {
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║                    Network Summary                          ║")
	fmt.Println("╠════════════════════════════════════════════════════════════╣")

	totalParams := 0
	for i, layer := range n.layers {
		params := layer.NumParameters()
		totalParams += params
		fmt.Printf("║ Layer %d: %-20s  Params: %-10d      ║\n", i, layer.Name(), params)
	}

	fmt.Println("╠════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Total Parameters: %-40d ║\n", totalParams)
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
}

// NumParameters returns total number of trainable parameters.
func (n *Network) NumParameters() int {
	total := 0
	for _, layer := range n.layers {
		total += layer.NumParameters()
	}
	return total
}

// DenseLayers returns the dense layers (for optimizer).
func (n *Network) DenseLayers() []*Dense {
	return n.denseLayer
}
