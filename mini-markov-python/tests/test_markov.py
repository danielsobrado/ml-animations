"""Tests for mini_markov package."""

import random
import pytest
from mini_markov import MarkovChain, TextGenerator, StateChain


class TestMarkovChain:
    def test_basic_chain(self):
        chain = MarkovChain[str](1)
        chain.train(list("ababab"))
        
        assert chain.order == 1
        assert chain.num_states > 0
        
        # After 'a', should always go to 'b' (probability 1.0)
        probs = chain.get_probabilities(["a"])
        assert probs is not None
        assert abs(probs["b"] - 1.0) < 0.001
    
    def test_second_order(self):
        chain = MarkovChain[str](2)
        chain.train(list("abcabd"))
        
        # After 'a','b', we've seen 'c' once and 'd' once
        probs = chain.get_probabilities(["a", "b"])
        assert probs is not None
        assert abs(probs["c"] - 0.5) < 0.001
        assert abs(probs["d"] - 0.5) < 0.001
    
    def test_generation(self):
        chain = MarkovChain[int](1)
        chain.train([1, 2, 3, 1, 2, 3, 1, 2, 3])
        
        rng = random.Random(42)
        generated = chain.generate(10, rng)
        
        assert len(generated) > 0
        assert len(generated) <= 10
    
    def test_entropy(self):
        # Deterministic chain (entropy ~ 0)
        chain1 = MarkovChain[str](1)
        chain1.train(list("ababab"))
        assert chain1.entropy() < 0.01
        
        # Random chain (higher entropy)
        chain2 = MarkovChain[str](1)
        chain2.train(list("abacadae"))
        assert chain2.entropy() > 1.0


class TestStateChain:
    def test_transitions(self):
        weather = StateChain()
        weather.add_transition_count("sunny", "sunny", 70)
        weather.add_transition_count("sunny", "cloudy", 20)
        weather.add_transition_count("sunny", "rainy", 10)
        
        prob = weather.probability("sunny", "sunny")
        assert abs(prob - 0.70) < 0.01
    
    def test_stationary_distribution(self):
        weather = StateChain()
        weather.add_transition_count("sunny", "sunny", 70)
        weather.add_transition_count("sunny", "cloudy", 20)
        weather.add_transition_count("sunny", "rainy", 10)
        weather.add_transition_count("cloudy", "sunny", 30)
        weather.add_transition_count("cloudy", "cloudy", 40)
        weather.add_transition_count("cloudy", "rainy", 30)
        weather.add_transition_count("rainy", "sunny", 20)
        weather.add_transition_count("rainy", "cloudy", 40)
        weather.add_transition_count("rainy", "rainy", 40)
        
        stationary = weather.stationary_distribution(1000)
        
        # Check that probabilities sum to ~1
        total = sum(stationary.values())
        assert abs(total - 1.0) < 0.01
        
        # Sunny should be most likely (~46%)
        assert 0.40 < stationary["sunny"] < 0.52


class TestTextGenerator:
    def test_basic_generation(self):
        gen = TextGenerator(2)
        gen.train("the quick brown fox jumps over the lazy dog")
        
        rng = random.Random(42)
        text = gen.generate(20, rng)
        
        assert len(text) > 0
    
    def test_stats(self):
        gen = TextGenerator(1)
        gen.train("a b c a b c a b c")
        
        stats = gen.stats
        assert stats["order"] == 1
        assert stats["num_states"] > 0
        assert stats["entropy"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
