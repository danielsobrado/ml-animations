"""Text generation using Markov chains."""

from typing import List, Set, Optional
import re
import random

from .chain import MarkovChain


class TextGenerator:
    """
    Text generator using word-level Markov chains.
    """
    
    def __init__(self, n: int = 2, preserve_case: bool = False):
        """
        Creates a new text generator.
        
        Args:
            n: The n-gram size (1 = unigram, 2 = bigram, etc.).
            preserve_case: Whether to preserve letter case.
        """
        self.chain = MarkovChain[str](n)
        self.preserve_case = preserve_case
        self.end_tokens: Set[str] = {".", "!", "?"}
    
    def add_end_token(self, token: str) -> "TextGenerator":
        """Adds a custom end token."""
        self.end_tokens.add(token)
        return self
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text into words and punctuation."""
        if not self.preserve_case:
            text = text.lower()
        
        # Split on whitespace first
        tokens = []
        for word in text.split():
            tokens.extend(self._split_punctuation(word))
        
        return [t for t in tokens if t]
    
    def _split_punctuation(self, word: str) -> List[str]:
        """Splits punctuation from words."""
        result = []
        current = []
        
        for char in word:
            if char.isalnum():
                current.append(char)
            else:
                if current:
                    result.append("".join(current))
                    current = []
                result.append(char)
        
        if current:
            result.append("".join(current))
        
        return result
    
    def train(self, text: str) -> None:
        """Trains the generator on a text corpus."""
        tokens = self._tokenize(text)
        if tokens:
            self.chain.train(tokens)
    
    def train_many(self, *texts: str) -> None:
        """Trains on multiple texts."""
        for text in texts:
            self.train(text)
    
    def generate(self, max_words: int, rng: random.Random = None) -> str:
        """Generates text with a maximum number of words."""
        tokens = self.chain.generate(max_words, rng)
        return self._join_tokens(tokens)
    
    def generate_from(self, prompt: str, max_words: int, rng: random.Random = None) -> str:
        """Generates text starting with a specific prompt."""
        prompt_tokens = self._tokenize(prompt)
        
        if len(prompt_tokens) < self.chain.order:
            return self.generate(max_words, rng)
        
        result = list(prompt_tokens)
        
        while len(result) < len(prompt_tokens) + max_words:
            current = result[-self.chain.order:]
            next_token = self.chain.sample_next(current, rng)
            if next_token is None:
                break
            result.append(next_token)
            if next_token in self.end_tokens:
                break
        
        return self._join_tokens(result)
    
    def generate_sentence(self, max_words: int, rng: random.Random = None) -> str:
        """Generates a complete sentence."""
        start = self.chain.sample_start(rng)
        if start is None:
            return ""
        
        result = list(start)
        
        while len(result) < max_words:
            current = result[-self.chain.order:]
            next_token = self.chain.sample_next(current, rng)
            if next_token is None:
                break
            result.append(next_token)
            if next_token in self.end_tokens:
                break
        
        return self._join_tokens(result)
    
    def _join_tokens(self, tokens: List[str]) -> str:
        """Joins tokens back into text with proper spacing."""
        if not tokens:
            return ""
        
        punctuation = re.compile(r'^[.!?,;:\'")\]]$')
        result = []
        
        for i, token in enumerate(tokens):
            if i > 0 and not punctuation.match(token):
                result.append(" ")
            result.append(token)
        
        return "".join(result)
    
    @property
    def stats(self) -> dict:
        """Returns statistics about the generator."""
        return {
            "order": self.chain.order,
            "num_states": self.chain.num_states,
            "num_transitions": self.chain.num_transitions,
            "num_sequences": self.chain.num_sequences,
            "entropy": self.chain.entropy(),
        }
    
    def __repr__(self) -> str:
        return f"TextGenerator(order={self.chain.order}, states={self.chain.num_states})"


class CharGenerator:
    """
    Character-level text generator using Markov chains.
    """
    
    def __init__(self, n: int = 3):
        """
        Creates a new character generator.
        
        Args:
            n: The n-gram size (number of characters to consider).
        """
        self.chain = MarkovChain[str](n)
    
    def train(self, text: str) -> None:
        """Trains the generator on text."""
        chars = list(text)
        if chars:
            self.chain.train(chars)
    
    def generate(self, max_chars: int, rng: random.Random = None) -> str:
        """Generates text with a maximum number of characters."""
        chars = self.chain.generate(max_chars, rng)
        return "".join(chars)
    
    @property
    def stats(self) -> dict:
        """Returns statistics about the generator."""
        return {
            "order": self.chain.order,
            "num_states": self.chain.num_states,
            "num_transitions": self.chain.num_transitions,
            "entropy": self.chain.entropy(),
        }
    
    def __repr__(self) -> str:
        return f"CharGenerator(order={self.chain.order}, states={self.chain.num_states})"
