---
title: "Q-Learning - learning without a model"
date: 2024-10-28
draft: false
tags: ["q-learning", "reinforcement-learning", "temporal-difference"]
categories: ["Reinforcement Learning"]
---

Q-learning learns optimal action values without knowing the environment. Just from experience. One of the most important RL algorithms.

## The Q-function

Q(s, a) = expected return from state s, taking action a, then acting optimally.

$$Q^*(s, a) = E[r + \gamma \max_{a'} Q^*(s', a')]$$

If we know Q*, optimal policy is simple: always pick argmax_a Q*(s, a).

![Q-Learning](https://danielsobrado.github.io/ml-animations/animation/q-learning)

Watch it learn: [Q-Learning Animation](https://danielsobrado.github.io/ml-animations/animation/q-learning)

## The update rule

After taking action a in state s, getting reward r, reaching state s':

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

Where:
- α: learning rate
- γ: discount factor
- r + γ max Q(s', a'): target (what Q should be)
- Q(s, a): current estimate
- Difference: temporal difference error

## Algorithm

```python
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning update
            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state = next_state
    
    return Q
```

## Off-policy learning

Q-learning is off-policy: learns about optimal policy while following different policy (ε-greedy).

The max in update rule assumes optimal action. Doesn't matter what action was actually taken.

Contrast with SARSA (on-policy):
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s', a') - Q(s,a)]$$

Uses actual next action a', not max.

## Convergence

Q-learning converges to optimal Q* if:
1. All state-action pairs visited infinitely often
2. Learning rate decreases appropriately
3. Bounded rewards

In practice, these conditions approximately met.

## The exploration problem

Need to explore to find good strategies. But also need to exploit to get rewards.

**ε-greedy:** Random action with probability ε

**Decaying ε:** Start high, decrease over time
```python
epsilon = max(0.01, epsilon * 0.995)
```

**Softmax/Boltzmann:**
$$\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}$$

Temperature τ controls randomness.

## Q-table limitations

Q-table stores one value per (state, action) pair.

Problems:
- Continuous states → infinite table
- Large state spaces → huge table
- No generalization between similar states

Solution: function approximation (DQN)

## Deep Q-Networks (DQN)

Replace Q-table with neural network:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

Key tricks:
- **Experience replay:** Store transitions, sample randomly
- **Target network:** Separate network for stable targets

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)
```

## Experience replay

Store (s, a, r, s', done) transitions in buffer. Sample random batches for training.

Why it helps:
- Breaks correlation between consecutive samples
- Reuses experience multiple times
- More efficient learning

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

## Target network

Using same network for prediction and target causes instability.

Solution: separate target network, update slowly:
```python
# Soft update
for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

Or hard update every N steps.

## Improvements

**Double DQN:** Reduces overestimation
$$y = r + \gamma Q_{target}(s', \arg\max_{a'} Q_{policy}(s', a'))$$

**Dueling DQN:** Separate value and advantage streams

**Prioritized replay:** Sample important transitions more often

The animation shows Q-values evolving: [Q-Learning Animation](https://danielsobrado.github.io/ml-animations/animation/q-learning)

---

Related:
- [RL Foundations](/posts/rl-foundations/)
- [Exploration strategies](/posts/rl-exploration/)
- [Markov Chains](/posts/markov-chains/)
