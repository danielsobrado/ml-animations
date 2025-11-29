# Mini Markov Python

A from-scratch implementation of Markov Chains in Python for educational purposes.

## Overview

This is a Python port of the mini-markov Rust implementation, providing the same features and similar results.

## Features

- ✅ Generic Markov chain implementation
- ✅ N-gram support (configurable order)
- ✅ Text generation (word-level and character-level)
- ✅ State machine modeling
- ✅ Stationary distribution calculation
- ✅ Entropy measurement

## Project Structure

```
mini-markov-python/
├── pyproject.toml
├── README.md
├── mini_markov/
│   ├── __init__.py
│   ├── chain.py          # Core implementation
│   ├── text.py           # Text generation
│   ├── state.py          # State modeling
│   ├── demo_text.py      # Text demo
│   ├── demo_weather.py   # Weather demo
│   └── demo_music.py     # Music demo
└── tests/
    └── test_markov.py
```

## Requirements

- Python 3.9+
- pytest (for testing)

## Installation

```bash
# From the mini-markov-python directory
pip install -e .

# Or for development with test dependencies
pip install -e ".[dev]"
```

## Usage

### Running Demos

```bash
# Text generation demo
python -m mini_markov.demo_text

# Weather simulation demo
python -m mini_markov.demo_weather

# Music chord progression demo
python -m mini_markov.demo_music
```

### Running Tests

```bash
pytest tests/ -v
```

## Example

```python
from mini_markov import MarkovChain, TextGenerator, StateChain
import random

# Basic chain example
chain = MarkovChain[str](1)
chain.train(list("ababac"))

probs = chain.get_probabilities(["a"])
print(probs)  # {'b': 0.666..., 'c': 0.333...}

# Text generation
gen = TextGenerator(2)
gen.train("the quick brown fox jumps over the lazy dog")

rng = random.Random(42)
print(gen.generate(20, rng))

# State machine
weather = StateChain()
weather.add_transition_count("sunny", "sunny", 70)
weather.add_transition_count("sunny", "cloudy", 30)

print(weather.probability("sunny", "sunny"))  # 0.7
```

## Comparison with Other Implementations

This Python implementation gives results similar to:
- mini-markov (Rust)
- mini-markov-go (Go)
- mini-markov-java (Java)

All implementations use the same:
- Transition probability calculations
- Stationary distribution algorithm (power iteration)
- Entropy formula

## License

MIT License
