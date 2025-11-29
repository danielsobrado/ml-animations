# Mini Markov Go

A from-scratch implementation of Markov Chains in Go for educational purposes.

## Overview

This is a Go port of the mini-markov Rust implementation, providing the same features and similar results.

## Features

- ✅ Generic Markov chain implementation
- ✅ N-gram support (configurable order)
- ✅ Text generation (word-level)
- ✅ State machine modeling
- ✅ Stationary distribution calculation
- ✅ Entropy measurement

## Project Structure

```
mini-markov-go/
├── go.mod
├── README.md
└── markov/
│   ├── chain.go        # Core MarkovChain implementation
│   ├── chain_test.go   # Unit tests
│   ├── text.go         # TextGenerator
│   └── state.go        # StateChain
└── cmd/
    ├── demo_text/      # Text generation demo
    ├── demo_weather/   # Weather simulation demo
    └── demo_music/     # Chord progression demo
```

## Usage

### Building

```bash
go build ./...
```

### Running Tests

```bash
go test ./markov/...
```

### Running Demos

```bash
# Text generation demo
go run ./cmd/demo_text

# Weather simulation demo
go run ./cmd/demo_weather

# Music chord progression demo
go run ./cmd/demo_music
```

## Example

```go
package main

import (
    "fmt"
    "math/rand"
    "mini-markov-go/markov"
)

func main() {
    // Create a first-order chain
    chain := markov.NewChain(1)
    
    // Train on a sequence
    chain.Train([]string{"a", "b", "a", "b", "a", "c"})
    
    // Get probabilities
    probs := chain.GetProbabilities([]string{"a"})
    fmt.Println(probs) // map[b:0.666... c:0.333...]
    
    // Generate sequence
    rng := rand.New(rand.NewSource(42))
    generated := chain.Generate(10, rng)
    fmt.Println(generated)
}
```

## License

MIT License
