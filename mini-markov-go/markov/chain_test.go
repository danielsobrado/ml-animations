package markov

import (
	"math"
	"math/rand"
	"testing"
)

func TestBasicChain(t *testing.T) {
	chain := NewChain(1)
	chain.Train([]string{"a", "b", "a", "b", "a", "b"})

	if chain.Order != 1 {
		t.Errorf("Expected order 1, got %d", chain.Order)
	}

	if chain.NumStates() == 0 {
		t.Error("Expected non-zero states")
	}

	// After 'a', should always go to 'b' (probability 1.0)
	probs := chain.GetProbabilities([]string{"a"})
	if probs == nil {
		t.Fatal("Expected probabilities for 'a'")
	}
	if probs["b"] != 1.0 {
		t.Errorf("Expected P(b|a) = 1.0, got %f", probs["b"])
	}
}

func TestSecondOrder(t *testing.T) {
	chain := NewChain(2)
	chain.Train([]string{"a", "b", "c", "a", "b", "d"})

	// After 'a','b', we've seen 'c' once and 'd' once
	probs := chain.GetProbabilities([]string{"a", "b"})
	if probs == nil {
		t.Fatal("Expected probabilities")
	}
	if probs["c"] != 0.5 {
		t.Errorf("Expected P(c|ab) = 0.5, got %f", probs["c"])
	}
	if probs["d"] != 0.5 {
		t.Errorf("Expected P(d|ab) = 0.5, got %f", probs["d"])
	}
}

func TestGeneration(t *testing.T) {
	chain := NewChain(1)
	chain.Train([]string{"1", "2", "3", "1", "2", "3", "1", "2", "3"})

	rng := rand.New(rand.NewSource(42))
	generated := chain.Generate(10, rng)

	if len(generated) == 0 {
		t.Error("Expected non-empty generated sequence")
	}
	if len(generated) > 10 {
		t.Errorf("Expected max 10 elements, got %d", len(generated))
	}
}

func TestEntropy(t *testing.T) {
	// Deterministic chain (entropy = 0)
	chain1 := NewChain(1)
	chain1.Train([]string{"a", "b", "a", "b", "a", "b"})
	if chain1.Entropy() > 0.01 {
		t.Errorf("Expected near-zero entropy for deterministic chain, got %f", chain1.Entropy())
	}

	// Random chain (higher entropy)
	chain2 := NewChain(1)
	chain2.Train([]string{"a", "b", "a", "c", "a", "d", "a", "e"})
	if chain2.Entropy() <= 1.0 {
		t.Errorf("Expected entropy > 1.0 for random chain, got %f", chain2.Entropy())
	}
}

func TestStateChain(t *testing.T) {
	weather := NewStateChain()
	weather.AddTransitionCount("sunny", "sunny", 70)
	weather.AddTransitionCount("sunny", "cloudy", 20)
	weather.AddTransitionCount("sunny", "rainy", 10)

	prob := weather.Probability("sunny", "sunny")
	if math.Abs(prob-0.70) > 0.01 {
		t.Errorf("Expected P(sunny|sunny) = 0.70, got %f", prob)
	}
}

func TestStationaryDistribution(t *testing.T) {
	weather := NewStateChain()
	weather.AddTransitionCount("sunny", "sunny", 70)
	weather.AddTransitionCount("sunny", "cloudy", 20)
	weather.AddTransitionCount("sunny", "rainy", 10)
	weather.AddTransitionCount("cloudy", "sunny", 30)
	weather.AddTransitionCount("cloudy", "cloudy", 40)
	weather.AddTransitionCount("cloudy", "rainy", 30)
	weather.AddTransitionCount("rainy", "sunny", 20)
	weather.AddTransitionCount("rainy", "cloudy", 40)
	weather.AddTransitionCount("rainy", "rainy", 40)

	stationary := weather.StationaryDistribution(1000)

	// Check that probabilities sum to ~1
	sum := 0.0
	for _, p := range stationary {
		sum += p
	}
	if math.Abs(sum-1.0) > 0.01 {
		t.Errorf("Stationary distribution should sum to 1.0, got %f", sum)
	}

	// Sunny should be most likely (~46%)
	if stationary["sunny"] < 0.40 || stationary["sunny"] > 0.52 {
		t.Errorf("Expected sunny ~46%%, got %f%%", stationary["sunny"]*100)
	}
}

func TestTextGenerator(t *testing.T) {
	gen := NewTextGenerator(2)
	gen.Train("the quick brown fox jumps over the lazy dog")

	rng := rand.New(rand.NewSource(42))
	text := gen.Generate(20, rng)

	if text == "" {
		t.Error("Expected non-empty generated text")
	}
}
