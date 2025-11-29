---
title: "Reinforcement Learning Foundations"
date: 2024-10-29
draft: false
tags: ["reinforcement-learning", "mdp", "reward", "policy"]
categories: ["Reinforcement Learning"]
---

Agent interacts with environment. Gets rewards. Learns to maximize rewards. That's RL in a nutshell. But the details matter.

## The setup

- **Agent:** The learner/decision maker
- **Environment:** What agent interacts with
- **State (s):** Current situation
- **Action (a):** What agent can do
- **Reward (r):** Feedback signal
- **Policy (π):** Strategy for choosing actions

![RL Foundations](https://danielsobrado.github.io/ml-animations/animation/rl-foundations)

Interactive diagram: [RL Foundations Animation](https://danielsobrado.github.io/ml-animations/animation/rl-foundations)

## MDP framework

Markov Decision Process formalizes the problem.

Components:
- S: set of states
- A: set of actions
- P(s'|s,a): transition probabilities
- R(s,a,s'): reward function
- γ: discount factor (0 < γ ≤ 1)

Markov property: next state depends only on current state and action, not history.

## Goal

Maximize expected cumulative reward:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

Discount factor γ makes future rewards worth less. Why?
- Future is uncertain
- Prefer reward now
- Makes math converge

## Value functions

**State value V(s):** Expected return starting from state s
$$V^\pi(s) = E_\pi[G_t | S_t = s]$$

**Action value Q(s,a):** Expected return starting from s, taking action a
$$Q^\pi(s,a) = E_\pi[G_t | S_t = s, A_t = a]$$

## Bellman equations

Value functions satisfy recursive relationships:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

These are key to solving MDPs.

## Policy

**Deterministic:** a = π(s)
**Stochastic:** π(a|s) = P(A=a|S=s)

**Optimal policy π*:** Achieves maximum value for all states
$$\pi^* = \arg\max_\pi V^\pi(s) \quad \forall s$$

## Types of RL algorithms

**Model-based:** Learn environment dynamics, plan
- Know P(s'|s,a)
- Can simulate ahead

**Model-free:** Learn directly from experience
- Don't need environment model
- More flexible but sample inefficient

**Value-based:** Learn value function, derive policy
- Q-learning, DQN
- π(s) = argmax_a Q(s,a)

**Policy-based:** Learn policy directly
- REINFORCE, PPO
- Parameterize π_θ(a|s)

**Actor-critic:** Both value and policy
- Actor: policy
- Critic: value function

## Simple example

Grid world:
- States: grid positions
- Actions: up, down, left, right
- Reward: +1 at goal, -1 at trap, -0.01 per step
- Goal: reach goal without traps

```python
# Simple environment
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)
        self.goal = (size-1, size-1)
    
    def step(self, action):
        # action: 0=up, 1=down, 2=left, 3=right
        x, y = self.state
        if action == 0: y = min(y+1, self.size-1)
        elif action == 1: y = max(y-1, 0)
        elif action == 2: x = max(x-1, 0)
        elif action == 3: x = min(x+1, self.size-1)
        
        self.state = (x, y)
        
        if self.state == self.goal:
            return self.state, 1.0, True
        return self.state, -0.01, False
```

## Exploration vs exploitation

**Exploit:** Choose best known action
**Explore:** Try something different to learn more

Too much exploitation: miss better options
Too much exploration: waste time on bad actions

Balance is crucial. ε-greedy is simplest:
- With prob ε: random action
- With prob 1-ε: best action

More in: [Exploration post](/posts/rl-exploration/)

## Key challenges

1. **Credit assignment:** Which action caused the reward?
2. **Sample efficiency:** Learning from limited experience
3. **Generalization:** Transfer to new states
4. **Stability:** Training can be unstable

The animation shows agent-environment interaction: [RL Foundations Animation](https://danielsobrado.github.io/ml-animations/animation/rl-foundations)

---

Related:
- [Q-Learning algorithm](/posts/q-learning/)
- [Exploration strategies](/posts/rl-exploration/)
- [Markov Chains](/posts/markov-chains/)
