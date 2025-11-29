---
title: "Exploration in RL - how agents discover"
date: 2024-10-27
draft: false
tags: ["exploration", "exploitation", "epsilon-greedy", "ucb"]
categories: ["Reinforcement Learning"]
---

Explore or exploit? The fundamental dilemma. Stick with what works or try something new? Get it wrong and agent either misses good options or wastes time on bad ones.

## The problem

Agent knows action A gives reward 5. Should it try action B?

If B gives 10: should have explored earlier!
If B gives 1: wasted time exploring.

Can't know without trying. But trying has cost.

![Exploration Strategies](https://danielsobrado.github.io/ml-animations/animation/rl-exploration)

Compare strategies: [Exploration Animation](https://danielsobrado.github.io/ml-animations/animation/rl-exploration)

## ε-greedy

Simplest approach:
- With probability ε: random action
- With probability 1-ε: best known action

```python
def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    return np.argmax(Q[state])
```

## Decaying ε

Start exploring more, reduce over time:

```python
epsilon = max(epsilon_min, epsilon * decay_rate)

# Or schedule-based
epsilon = epsilon_start * (epsilon_end / epsilon_start) ** (episode / total_episodes)
```

Early: lots of random exploration
Later: mostly exploitation with occasional exploration

## Problems with ε-greedy

1. **Undirected:** Explores randomly, not intelligently
2. **Doesn't consider uncertainty:** Equal chance to try known-bad vs unknown
3. **Never stops:** Still random after millions of steps

## Boltzmann (Softmax) exploration

Actions chosen proportionally to their values:

$$\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'}\exp(Q(s,a')/\tau)}$$

Temperature τ:
- High τ: nearly uniform (explore)
- Low τ: nearly greedy (exploit)
- τ → 0: argmax

```python
def boltzmann_action(Q, state, temperature):
    q_values = Q[state]
    exp_q = np.exp(q_values / temperature)
    probs = exp_q / exp_q.sum()
    return np.random.choice(len(q_values), p=probs)
```

Better than ε-greedy: prefers good actions while still exploring.

## UCB - Upper Confidence Bound

Explore actions with high uncertainty:

$$a = \arg\max_a \left[ Q(s, a) + c\sqrt{\frac{\ln t}{N(s, a)}} \right]$$

- Q(s, a): value estimate
- N(s, a): times action tried
- t: total timesteps
- c: exploration parameter

Untried actions have infinite UCB → tried first.
As N increases, bonus decreases → exploit.

```python
def ucb_action(Q, state, counts, total_steps, c=2):
    if 0 in counts[state]:  # unexplored action exists
        return np.argmin(counts[state])
    
    ucb_values = Q[state] + c * np.sqrt(np.log(total_steps) / counts[state])
    return np.argmax(ucb_values)
```

## Optimism in the face of uncertainty

Initialize Q-values high. Agent naturally explores to verify.

```python
Q = defaultdict(lambda: np.ones(n_actions) * 10)  # optimistic init
```

Simple but effective for finite state spaces.

## Thompson Sampling

Bayesian approach. Maintain probability distribution over Q-values.
Sample from posterior, act greedily on sample.

```python
# For bandits with beta distribution
def thompson_sampling(successes, failures):
    samples = [np.random.beta(s + 1, f + 1) for s, f in zip(successes, failures)]
    return np.argmax(samples)
```

Naturally balances exploration and exploitation.

## Intrinsic motivation

Add exploration bonus to reward:

$$r' = r + \beta \cdot \text{bonus}(s, a)$$

**Count-based:**
$$\text{bonus} = \frac{1}{\sqrt{N(s)}}$$

Less visited states get higher bonus.

**Curiosity-driven:**
$$\text{bonus} = ||\hat{s}' - s'||^2$$

Predict next state. Bonus from prediction error. Novel states hard to predict.

## Noisy Networks

Add learned noise to network weights:

$$w = \mu + \sigma \odot \epsilon$$

Network learns how much to explore per weight. No need for ε-schedule.

```python
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma = nn.Parameter(torch.ones(out_features, in_features) * std_init)
    
    def forward(self, x):
        if self.training:
            epsilon = torch.randn_like(self.sigma)
            weight = self.mu + self.sigma * epsilon
        else:
            weight = self.mu
        return F.linear(x, weight)
```

## Comparison

| Method | Intelligence | Computation | Works well when |
|--------|--------------|-------------|-----------------|
| ε-greedy | Low | Low | Simple problems |
| Boltzmann | Medium | Low | Clear value differences |
| UCB | High | Medium | Known uncertainty |
| Thompson | High | Medium | Bayesian setting |
| Curiosity | High | High | Large state spaces |

## Practical advice

1. Start with ε-greedy, decay from 1.0 to 0.01
2. If that fails, try UCB or Boltzmann
3. For deep RL, noisy networks often better
4. For sparse rewards, intrinsic motivation helps

The animation compares these strategies head-to-head: [Exploration Animation](https://danielsobrado.github.io/ml-animations/animation/rl-exploration)

---

Related:
- [RL Foundations](/posts/rl-foundations/)
- [Q-Learning](/posts/q-learning/)
