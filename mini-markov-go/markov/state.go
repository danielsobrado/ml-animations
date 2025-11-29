package markov

import (
	"math/rand"
)

// StateChain models discrete state systems using Markov chains.
type StateChain struct {
	Chain  *Chain
	States []string
}

// NewStateChain creates a new first-order state chain.
func NewStateChain() *StateChain {
	return &StateChain{
		Chain: NewChain(1),
	}
}

// NewStateChainWithOrder creates a state chain with specified order.
func NewStateChainWithOrder(order int) *StateChain {
	return &StateChain{
		Chain: NewChain(order),
	}
}

// WithStates sets the possible states for validation.
func (s *StateChain) WithStates(states []string) *StateChain {
	s.States = make([]string, len(states))
	copy(s.States, states)
	return s
}

// AddTransition adds a single transition observation.
func (s *StateChain) AddTransition(from, to string) {
	s.Chain.Train([]string{from, to})
}

// AddTransitionCount adds a transition with a specific count.
func (s *StateChain) AddTransitionCount(from, to string, count int) {
	for i := 0; i < count; i++ {
		s.AddTransition(from, to)
	}
}

// Train trains on a sequence of states.
func (s *StateChain) Train(sequence []string) {
	s.Chain.Train(sequence)
}

// TrainMany trains on multiple sequences.
func (s *StateChain) TrainMany(sequences [][]string) {
	for _, seq := range sequences {
		s.Train(seq)
	}
}

// Probability gets the probability of transitioning from one state to another.
func (s *StateChain) Probability(from, to string) float64 {
	probs := s.Chain.GetProbabilities([]string{from})
	if probs == nil {
		return 0.0
	}
	return probs[to]
}

// ProbabilitiesFrom gets all transition probabilities from a state.
func (s *StateChain) ProbabilitiesFrom(state string) map[string]float64 {
	probs := s.Chain.GetProbabilities([]string{state})
	if probs == nil {
		return make(map[string]float64)
	}
	return probs
}

// NextState samples the next state.
func (s *StateChain) NextState(current string, rng *rand.Rand) (string, bool) {
	return s.Chain.SampleNext([]string{current}, rng)
}

// Simulate simulates the chain for a number of steps.
func (s *StateChain) Simulate(start string, steps int, rng *rand.Rand) []string {
	result := []string{start}

	for i := 0; i < steps; i++ {
		current := result[len(result)-1]
		next, ok := s.NextState(current, rng)
		if !ok {
			break
		}
		result = append(result, next)
	}

	return result
}

// StationaryDistribution calculates the long-term probability distribution.
func (s *StateChain) StationaryDistribution(iterations int) map[string]float64 {
	// Get all unique states
	stateSet := make(map[string]bool)
	for key := range s.Chain.Transitions {
		stateSet[key] = true
	}
	for _, counts := range s.Chain.Transitions {
		for state := range counts {
			stateSet[state] = true
		}
	}

	if len(stateSet) == 0 {
		return make(map[string]float64)
	}

	// Convert to slice for indexing
	states := make([]string, 0, len(stateSet))
	for state := range stateSet {
		states = append(states, state)
	}
	n := len(states)

	stateToIdx := make(map[string]int)
	for i, state := range states {
		stateToIdx[state] = i
	}

	// Build transition matrix
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
	}

	for fromState, counts := range s.Chain.Transitions {
		fromIdx, ok := stateToIdx[fromState]
		if !ok {
			continue
		}

		total := 0
		for _, count := range counts {
			total += count
		}

		for toState, count := range counts {
			toIdx, ok := stateToIdx[toState]
			if ok {
				matrix[fromIdx][toIdx] = float64(count) / float64(total)
			}
		}
	}

	// Power iteration
	dist := make([]float64, n)
	for i := range dist {
		dist[i] = 1.0 / float64(n)
	}

	for iter := 0; iter < iterations; iter++ {
		newDist := make([]float64, n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				newDist[j] += dist[i] * matrix[i][j]
			}
		}
		dist = newDist
	}

	// Normalize
	sum := 0.0
	for _, d := range dist {
		sum += d
	}
	if sum > 0 {
		for i := range dist {
			dist[i] /= sum
		}
	}

	// Convert to map
	result := make(map[string]float64)
	for i, state := range states {
		result[state] = dist[i]
	}
	return result
}

// ExpectedStepsTo estimates the expected number of steps to reach a target state.
func (s *StateChain) ExpectedStepsTo(from, to string, maxSteps, simulations int, rng *rand.Rand) float64 {
	totalSteps := 0
	successCount := 0

	for sim := 0; sim < simulations; sim++ {
		current := from
		for step := 1; step <= maxSteps; step++ {
			next, ok := s.NextState(current, rng)
			if !ok {
				break
			}
			if next == to {
				totalSteps += step
				successCount++
				break
			}
			current = next
		}
	}

	if successCount == 0 {
		return -1 // Unreachable
	}
	return float64(totalSteps) / float64(successCount)
}
