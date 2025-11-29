"""State chain for modeling discrete state systems."""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import random

from .chain import MarkovChain


class StateChain:
    """
    State chain for modeling discrete state systems like weather,
    game states, or any system with finite states and probabilistic transitions.
    """
    
    def __init__(self, order: int = 1):
        """
        Creates a new state chain.
        
        Args:
            order: The order of the Markov chain (default: 1, first-order).
        """
        self.chain = MarkovChain[str](order)
        self.states: Set[str] = set()
    
    def with_states(self, *states: str) -> "StateChain":
        """Sets the possible states for validation."""
        self.states = set(states)
        return self
    
    def add_transition(self, from_state: str, to_state: str) -> None:
        """Adds a single transition observation."""
        self.chain.train([from_state, to_state])
    
    def add_transition_count(self, from_state: str, to_state: str, count: int) -> None:
        """Adds a transition with a specific count."""
        for _ in range(count):
            self.add_transition(from_state, to_state)
    
    def train(self, sequence: List[str]) -> None:
        """Trains on a sequence of states."""
        self.chain.train(sequence)
    
    def train_many(self, sequences: List[List[str]]) -> None:
        """Trains on multiple sequences."""
        for seq in sequences:
            self.train(seq)
    
    def probability(self, from_state: str, to_state: str) -> float:
        """Gets the probability of transitioning from one state to another."""
        probs = self.chain.get_probabilities([from_state])
        return probs.get(to_state, 0.0) if probs else 0.0
    
    def probabilities_from(self, state: str) -> Dict[str, float]:
        """Gets all transition probabilities from a state."""
        probs = self.chain.get_probabilities([state])
        return probs if probs else {}
    
    def next_state(self, current: str, rng: random.Random = None) -> Optional[str]:
        """Samples the next state."""
        return self.chain.sample_next([current], rng)
    
    def simulate(self, start: str, steps: int, rng: random.Random = None) -> List[str]:
        """
        Simulates the chain for a number of steps.
        
        Args:
            start: Starting state.
            steps: Number of steps to simulate.
            rng: Random number generator.
            
        Returns:
            List of states including the start state.
        """
        result = [start]
        
        for _ in range(steps):
            current = result[-1]
            next_s = self.next_state(current, rng)
            if next_s is None:
                break
            result.append(next_s)
        
        return result
    
    def stationary_distribution(self, iterations: int = 1000) -> Dict[str, float]:
        """
        Calculates the stationary distribution using power iteration.
        
        Args:
            iterations: Number of power iteration steps.
            
        Returns:
            Dictionary mapping states to their stationary probabilities.
        """
        # Get all unique states
        state_set: Set[str] = set()
        for key in self.chain._transitions:
            state_set.update(key)
        for counts in self.chain._transitions.values():
            state_set.update(counts.keys())
        
        if not state_set:
            return {}
        
        state_list = list(state_set)
        n = len(state_list)
        state_to_idx = {s: i for i, s in enumerate(state_list)}
        
        # Build transition matrix
        matrix = [[0.0] * n for _ in range(n)]
        
        for key, counts in self.chain._transitions.items():
            if len(key) != 1:
                continue
            
            from_state = key[0]
            from_idx = state_to_idx.get(from_state)
            if from_idx is None:
                continue
            
            total = sum(counts.values())
            
            for to_state, count in counts.items():
                to_idx = state_to_idx.get(to_state)
                if to_idx is not None:
                    matrix[from_idx][to_idx] = count / total
        
        # Power iteration
        dist = [1.0 / n] * n
        
        for _ in range(iterations):
            new_dist = [0.0] * n
            for i in range(n):
                for j in range(n):
                    new_dist[j] += dist[i] * matrix[i][j]
            dist = new_dist
        
        # Normalize
        total = sum(dist)
        if total > 0:
            dist = [d / total for d in dist]
        
        return {state_list[i]: dist[i] for i in range(n)}
    
    def expected_steps_to(
        self,
        from_state: str,
        to_state: str,
        max_steps: int = 1000,
        simulations: int = 1000,
        rng: random.Random = None
    ) -> float:
        """
        Estimates the expected number of steps to reach a target state.
        
        Args:
            from_state: Starting state.
            to_state: Target state.
            max_steps: Maximum steps per simulation.
            simulations: Number of simulations to run.
            rng: Random number generator.
            
        Returns:
            Expected number of steps, or -1 if unreachable.
        """
        total_steps = 0
        success_count = 0
        
        for _ in range(simulations):
            current = from_state
            for step in range(1, max_steps + 1):
                next_s = self.next_state(current, rng)
                if next_s is None:
                    break
                if next_s == to_state:
                    total_steps += step
                    success_count += 1
                    break
                current = next_s
        
        return total_steps / success_count if success_count > 0 else -1.0
    
    def __repr__(self) -> str:
        return f"StateChain(order={self.chain.order}, states={self.chain.num_states})"
