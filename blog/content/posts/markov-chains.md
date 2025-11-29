---
title: "Markov Chains - states and transitions"
date: 2024-10-26
draft: false
tags: ["markov-chains", "probability", "stochastic-process"]
categories: ["Reinforcement Learning"]
---

A system that jumps between states with fixed probabilities. Where you go next depends only on where you are now, not how you got there. This "memoryless" property is surprisingly powerful.

## Definition

Sequence of random variables X₀, X₁, X₂, ... where:

$$P(X_{n+1} = j | X_n = i, X_{n-1}, ..., X_0) = P(X_{n+1} = j | X_n = i)$$

Only current state matters. History irrelevant.

![Markov Chains](https://danielsobrado.github.io/ml-animations/animation/markov-chains)

See states evolve: [Markov Chains Animation](https://danielsobrado.github.io/ml-animations/animation/markov-chains)

## Transition matrix

Probabilities stored in matrix P:
$$P_{ij} = P(X_{n+1} = j | X_n = i)$$

Each row sums to 1 (must go somewhere).

Example - weather:
```
         Sunny  Rainy
Sunny  [  0.8    0.2  ]
Rainy  [  0.4    0.6  ]
```

If sunny today: 80% sunny tomorrow, 20% rainy.

## Computing future states

If π_n is probability distribution over states at time n:

$$\pi_{n+1} = \pi_n P$$

After k steps:
$$\pi_{n+k} = \pi_n P^k$$

```python
import numpy as np

P = np.array([[0.8, 0.2],
              [0.4, 0.6]])

# Start sunny (100% probability)
pi = np.array([1.0, 0.0])

# After 10 days
pi_10 = pi @ np.linalg.matrix_power(P, 10)
```

## Stationary distribution

Eventually, distribution stabilizes:
$$\pi^* = \pi^* P$$

Regardless of starting state!

Find it by solving:
$$\pi^* (P - I) = 0, \quad \sum_i \pi^*_i = 1$$

Or: dominant eigenvector of P^T.

```python
# Find stationary distribution
eigenvalues, eigenvectors = np.linalg.eig(P.T)
# Eigenvalue 1 corresponds to stationary
idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
stationary = eigenvectors[:, idx].real
stationary = stationary / stationary.sum()
```

## Properties

**Irreducible:** Can reach any state from any state.

**Aperiodic:** No fixed cycle length.

**Ergodic:** Irreducible + aperiodic → unique stationary distribution exists.

## PageRank

Web pages as states. Links as transitions.

$$P(i \to j) = \frac{1}{\text{outlinks from } i}$$

Stationary distribution = page importance!

With damping (random jumps):
$$\tilde{P} = \alpha P + (1-\alpha)\frac{1}{n}\mathbf{1}$$

More details: [PageRank post](/posts/pagerank/)

## Applications in ML

**MCMC (Markov Chain Monte Carlo):**
Sample from complex distributions by constructing chain with target as stationary distribution.

**Language models (simple):**
Next word depends only on previous word(s).

**Hidden Markov Models:**
States are hidden. Emissions are observed.

**RL:**
MDPs are Markov chains with actions.

## Absorbing states

State you can't leave. Once there, stay forever.

P(i,i) = 1 for absorbing state i.

Example: gambler's ruin. Win everything or lose everything = absorbing.

## Expected hitting time

How many steps to reach state j from state i?

$$h_i = 1 + \sum_{k \neq j} P_{ik} h_k$$

Solve system of equations.

```python
# Expected time to reach state 0 from each state
# h = 1 + P * h (excluding target state)
# Solve: (I - P_subset) h = 1
```

## Simulation

```python
def simulate_chain(P, initial_state, steps):
    n_states = P.shape[0]
    states = [initial_state]
    state = initial_state
    
    for _ in range(steps):
        state = np.random.choice(n_states, p=P[state])
        states.append(state)
    
    return states

# Run simulation
trajectory = simulate_chain(P, initial_state=0, steps=100)
```

## Connection to eigenvalues

Eigenvalues of P reveal chain properties:
- Largest eigenvalue always 1 (for stochastic matrix)
- Second largest eigenvalue: mixing rate

$$||\pi_n - \pi^*|| \leq C \cdot \lambda_2^n$$

Larger λ₂ = slower mixing.

## Time reversal

Detailed balance condition:
$$\pi_i P_{ij} = \pi_j P_{ji}$$

Chain looks same forward and backward. Important for MCMC algorithms.

The animation shows how probability flows through states: [Markov Chains Animation](https://danielsobrado.github.io/ml-animations/animation/markov-chains)

---

Related:
- [PageRank uses Markov chains](/posts/pagerank/)
- [RL extends to MDPs](/posts/rl-foundations/)
- [Conditional Probability](/posts/conditional-probability/)
