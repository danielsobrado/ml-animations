// Package markov provides a generic Markov chain implementation.
package markov

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// Chain represents a generic Markov chain.
type Chain struct {
	Order        int
	Transitions  map[string]map[string]int
	StartStates  map[string]int
	NumSequences int
}

// NewChain creates a new Markov chain with the specified order.
func NewChain(order int) *Chain {
	if order < 1 {
		panic("Order must be at least 1")
	}
	return &Chain{
		Order:       order,
		Transitions: make(map[string]map[string]int),
		StartStates: make(map[string]int),
	}
}

// stateKey converts a slice of states to a string key.
func stateKey(states []string) string {
	return strings.Join(states, "\x00")
}

// Train trains the chain on a sequence of states.
func (c *Chain) Train(sequence []string) {
	if len(sequence) <= c.Order {
		return
	}

	c.NumSequences++

	// Record start state
	start := stateKey(sequence[:c.Order])
	c.StartStates[start]++

	// Record transitions
	for i := 0; i <= len(sequence)-c.Order-1; i++ {
		current := stateKey(sequence[i : i+c.Order])
		next := sequence[i+c.Order]

		if c.Transitions[current] == nil {
			c.Transitions[current] = make(map[string]int)
		}
		c.Transitions[current][next]++
	}
}

// TrainMany trains on multiple sequences.
func (c *Chain) TrainMany(sequences [][]string) {
	for _, seq := range sequences {
		c.Train(seq)
	}
}

// GetProbabilities returns the probability distribution for next states.
func (c *Chain) GetProbabilities(current []string) map[string]float64 {
	if len(current) != c.Order {
		return nil
	}

	key := stateKey(current)
	counts, exists := c.Transitions[key]
	if !exists {
		return nil
	}

	total := 0
	for _, count := range counts {
		total += count
	}

	probs := make(map[string]float64)
	for state, count := range counts {
		probs[state] = float64(count) / float64(total)
	}
	return probs
}

// SampleNext samples the next state given the current state.
func (c *Chain) SampleNext(current []string, rng *rand.Rand) (string, bool) {
	if len(current) != c.Order {
		return "", false
	}

	key := stateKey(current)
	counts, exists := c.Transitions[key]
	if !exists {
		return "", false
	}

	total := 0
	for _, count := range counts {
		total += count
	}

	threshold := rng.Intn(total)
	for state, count := range counts {
		threshold -= count
		if threshold < 0 {
			return state, true
		}
	}

	// Should never reach here
	for state := range counts {
		return state, true
	}
	return "", false
}

// SampleStart samples a random starting state.
func (c *Chain) SampleStart(rng *rand.Rand) ([]string, bool) {
	if len(c.StartStates) == 0 {
		return nil, false
	}

	total := 0
	for _, count := range c.StartStates {
		total += count
	}

	threshold := rng.Intn(total)
	for key, count := range c.StartStates {
		threshold -= count
		if threshold < 0 {
			return strings.Split(key, "\x00"), true
		}
	}

	for key := range c.StartStates {
		return strings.Split(key, "\x00"), true
	}
	return nil, false
}

// Generate generates a sequence of states.
func (c *Chain) Generate(length int, rng *rand.Rand) []string {
	start, ok := c.SampleStart(rng)
	if !ok {
		return nil
	}

	result := make([]string, len(start))
	copy(result, start)

	for len(result) < length {
		current := result[len(result)-c.Order:]
		next, ok := c.SampleNext(current, rng)
		if !ok {
			break
		}
		result = append(result, next)
	}

	return result
}

// NumStates returns the number of unique state sequences.
func (c *Chain) NumStates() int {
	return len(c.Transitions)
}

// NumTransitions returns the total number of transitions recorded.
func (c *Chain) NumTransitions() int {
	total := 0
	for _, counts := range c.Transitions {
		for _, count := range counts {
			total += count
		}
	}
	return total
}

// Entropy calculates the entropy of the chain.
func (c *Chain) Entropy() float64 {
	totalEntropy := 0.0
	totalWeight := 0.0

	for _, counts := range c.Transitions {
		total := 0
		for _, count := range counts {
			total += count
		}
		if total == 0 {
			continue
		}

		stateEntropy := 0.0
		for _, count := range counts {
			if count > 0 {
				p := float64(count) / float64(total)
				stateEntropy -= p * math.Log2(p)
			}
		}

		totalEntropy += stateEntropy * float64(total)
		totalWeight += float64(total)
	}

	if totalWeight > 0 {
		return totalEntropy / totalWeight
	}
	return 0.0
}

// String returns a string representation of the chain.
func (c *Chain) String() string {
	return fmt.Sprintf("MarkovChain(order=%d, states=%d, transitions=%d)",
		c.Order, c.NumStates(), c.NumTransitions())
}
