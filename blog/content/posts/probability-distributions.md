---
title: "Probability Distributions - the shapes of randomness"
date: 2024-11-05
draft: false
tags: ["probability", "distributions", "statistics"]
categories: ["Probability & Statistics"]
---

ML is full of probability distributions. Knowing when to use which one matters.

## Why distributions?

Data has patterns. Heights cluster around average. Rare events are rare. Distributions capture these patterns mathematically.

![Probability Distributions](https://danielsobrado.github.io/ml-animations/animation/probability-distributions)

Interactive examples: [Probability Distributions Animation](https://danielsobrado.github.io/ml-animations/animation/probability-distributions)

## Normal (Gaussian)

The most important distribution. Bell curve.

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Parameters: μ (mean), σ (standard deviation)

```python
import numpy as np
from scipy import stats

# Generate samples
samples = np.random.normal(mu=0, sigma=1, size=1000)

# PDF at point
prob = stats.norm.pdf(x, loc=mu, scale=sigma)
```

Used when:
- Central limit theorem applies
- Sum of many small effects
- Default assumption for continuous data

## Bernoulli and Binomial

**Bernoulli:** Single yes/no trial
- P(X=1) = p
- P(X=0) = 1-p

**Binomial:** Sum of n Bernoulli trials
$$P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$$

```python
# Bernoulli
samples = np.random.binomial(n=1, p=0.3, size=1000)

# Binomial
samples = np.random.binomial(n=10, p=0.3, size=1000)
```

Used for: coin flips, success/failure counts, binary classification probabilities.

## Poisson

Counts of rare events in fixed interval.

$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Parameter: λ (average rate)

```python
samples = np.random.poisson(lam=5, size=1000)
```

Used for: website visits per minute, defects per unit, arrivals per hour.

## Exponential

Time between Poisson events.

$$f(x) = \lambda e^{-\lambda x}$$

```python
samples = np.random.exponential(scale=1/lam, size=1000)
```

Memoryless property: P(X > s+t | X > s) = P(X > t)

## Uniform

All values equally likely in range.

```python
samples = np.random.uniform(low=0, high=1, size=1000)
```

Used for: initialization, random sampling, random number generation base.

## Categorical and Multinomial

**Categorical:** One trial with k possible outcomes
**Multinomial:** n trials with k possible outcomes

```python
# Categorical (one-hot encoded)
probs = [0.2, 0.3, 0.5]
sample = np.random.multinomial(n=1, pvals=probs)

# Multinomial
samples = np.random.multinomial(n=10, pvals=probs)
```

Softmax outputs follow categorical distribution.

## Beta

Distribution over probabilities (values in [0,1]).

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$$

```python
samples = np.random.beta(a=2, b=5, size=1000)
```

Used for: Bayesian inference on proportions, uncertainty over probabilities.

## Dirichlet

Multivariate Beta. Distribution over probability vectors.

```python
samples = np.random.dirichlet(alpha=[1, 1, 1], size=1000)
# Each sample sums to 1
```

Used for: topic models, mixture models, Bayesian categorical.

## Summary table

| Distribution | Domain | Parameters | Use case |
|--------------|--------|------------|----------|
| Normal | (-∞, ∞) | μ, σ | Continuous data |
| Bernoulli | {0, 1} | p | Single binary trial |
| Binomial | {0,...,n} | n, p | Count of successes |
| Poisson | {0,1,2,...} | λ | Rare event counts |
| Exponential | [0, ∞) | λ | Time between events |
| Uniform | [a, b] | a, b | Equal likelihood |
| Categorical | {1,...,k} | p₁...pₖ | k-way classification |
| Beta | [0, 1] | α, β | Probability values |

## Fitting distributions

```python
from scipy import stats

# Fit normal to data
mu, sigma = stats.norm.fit(data)

# Fit any distribution
params = stats.expon.fit(data)
```

## Checking fit

```python
# QQ plot
import matplotlib.pyplot as plt
stats.probplot(data, dist="norm", plot=plt)

# Kolmogorov-Smirnov test
statistic, pvalue = stats.kstest(data, 'norm', args=(mu, sigma))
```

Explore different distributions interactively: [Probability Distributions Animation](https://danielsobrado.github.io/ml-animations/animation/probability-distributions)

---

Related:
- [Conditional Probability](/posts/conditional-probability/)
- [Expected Value and Variance](/posts/expected-value-variance/)
- [Cross-Entropy loss](/posts/cross-entropy/)
