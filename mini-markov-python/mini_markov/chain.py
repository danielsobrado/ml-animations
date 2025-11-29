"""Generic Markov Chain implementation."""

from typing import TypeVar, Generic, Dict, List, Optional, Tuple
from collections import defaultdict
import random
import math

T = TypeVar("T")


class MarkovChain(Generic[T]):
    """
    A generic Markov chain implementation supporting any hashable type.
    
    Attributes:
        order: The number of previous states to consider (n-gram size).
    """
    
    def __init__(self, order: int = 1):
        """
        Creates a new Markov chain.
        
        Args:
            order: Number of previous states to consider (must be >= 1).
        """
        if order < 1:
            raise ValueError("Order must be at least 1")
        
        self.order = order
        self._transitions: Dict[Tuple[T, ...], Dict[T, int]] = defaultdict(lambda: defaultdict(int))
        self._start_states: Dict[Tuple[T, ...], int] = defaultdict(int)
        self._num_sequences = 0
    
    def train(self, sequence: List[T]) -> None:
        """
        Trains the chain on a sequence of states.
        
        Args:
            sequence: The sequence to train on.
        """
        if len(sequence) <= self.order:
            return
        
        self._num_sequences += 1
        
        # Record start state
        start = tuple(sequence[:self.order])
        self._start_states[start] += 1
        
        # Record transitions
        for i in range(len(sequence) - self.order):
            current = tuple(sequence[i:i + self.order])
            next_state = sequence[i + self.order]
            self._transitions[current][next_state] += 1
    
    def train_many(self, sequences: List[List[T]]) -> None:
        """Trains the chain on multiple sequences."""
        for seq in sequences:
            self.train(seq)
    
    def get_probabilities(self, current: List[T]) -> Optional[Dict[T, float]]:
        """
        Gets the probability distribution for next states.
        
        Args:
            current: The current state sequence (must have length == order).
            
        Returns:
            Dictionary mapping states to probabilities, or None if unknown state.
        """
        if len(current) != self.order:
            return None
        
        key = tuple(current)
        if key not in self._transitions:
            return None
        
        counts = self._transitions[key]
        total = sum(counts.values())
        
        return {state: count / total for state, count in counts.items()}
    
    def sample_next(self, current: List[T], rng: random.Random = None) -> Optional[T]:
        """
        Samples the next state given the current state.
        
        Args:
            current: The current state sequence.
            rng: Random number generator (uses default if None).
            
        Returns:
            The sampled next state, or None if unknown state.
        """
        if rng is None:
            rng = random.Random()
            
        if len(current) != self.order:
            return None
        
        key = tuple(current)
        if key not in self._transitions:
            return None
        
        counts = self._transitions[key]
        total = sum(counts.values())
        
        threshold = rng.randint(0, total - 1)
        
        for state, count in counts.items():
            threshold -= count
            if threshold < 0:
                return state
        
        return next(iter(counts.keys()))
    
    def sample_start(self, rng: random.Random = None) -> Optional[List[T]]:
        """Samples a random starting state."""
        if rng is None:
            rng = random.Random()
            
        if not self._start_states:
            return None
        
        total = sum(self._start_states.values())
        threshold = rng.randint(0, total - 1)
        
        for start, count in self._start_states.items():
            threshold -= count
            if threshold < 0:
                return list(start)
        
        return list(next(iter(self._start_states.keys())))
    
    def generate(self, length: int, rng: random.Random = None) -> List[T]:
        """
        Generates a sequence of states.
        
        Args:
            length: Maximum length of the generated sequence.
            rng: Random number generator.
            
        Returns:
            Generated sequence of states.
        """
        if rng is None:
            rng = random.Random()
            
        start = self.sample_start(rng)
        if start is None:
            return []
        
        result = list(start)
        
        while len(result) < length:
            current = result[-self.order:]
            next_state = self.sample_next(current, rng)
            if next_state is None:
                break
            result.append(next_state)
        
        return result
    
    @property
    def num_states(self) -> int:
        """Returns the number of unique state sequences."""
        return len(self._transitions)
    
    @property
    def num_transitions(self) -> int:
        """Returns the total number of transitions recorded."""
        return sum(sum(counts.values()) for counts in self._transitions.values())
    
    @property
    def num_sequences(self) -> int:
        """Returns the number of training sequences."""
        return self._num_sequences
    
    def entropy(self) -> float:
        """Calculates the entropy of the chain in bits."""
        total_entropy = 0.0
        total_weight = 0.0
        
        for counts in self._transitions.values():
            total = sum(counts.values())
            if total == 0:
                continue
            
            state_entropy = 0.0
            for count in counts.values():
                if count > 0:
                    p = count / total
                    state_entropy -= p * math.log2(p)
            
            total_entropy += state_entropy * total
            total_weight += total
        
        return total_entropy / total_weight if total_weight > 0 else 0.0
    
    def clear(self) -> None:
        """Clears all learned transitions."""
        self._transitions.clear()
        self._start_states.clear()
        self._num_sequences = 0
    
    def __repr__(self) -> str:
        return f"MarkovChain(order={self.order}, states={self.num_states}, transitions={self.num_transitions})"
