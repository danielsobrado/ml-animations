---
title: "Expected Value and Variance - summarizing distributions"
date: 2024-11-03
draft: false
tags: ["expected-value", "variance", "statistics"]
categories: ["Probability & Statistics"]
---

Two numbers that capture essence of a distribution: expected value (center) and variance (spread).

## Expected value

The "average" value. Weighted sum of outcomes by probability.

Discrete:
$$E[X] = \sum_x x \cdot P(X=x)$$

Continuous:
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$$

![Expected Value and Variance](https://danielsobrado.github.io/ml-animations/animation/expected-value-variance)

Visual explanation: [Expected Value Animation](https://danielsobrado.github.io/ml-animations/animation/expected-value-variance)

## Example

Fair die:
$$E[X] = 1(\frac{1}{6}) + 2(\frac{1}{6}) + ... + 6(\frac{1}{6}) = \frac{21}{6} = 3.5$$

You never actually roll 3.5, but it's the long-run average.

## Properties

**Linearity:**
$$E[aX + b] = aE[X] + b$$
$$E[X + Y] = E[X] + E[Y]$$

Always true, even if X and Y dependent.

**For independent X, Y:**
$$E[XY] = E[X] \cdot E[Y]$$

Only for independent variables!

## Variance

How spread out the distribution is.

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

Standard deviation: σ = √Var(X)

```python
import numpy as np

data = [1, 2, 3, 4, 5]
mean = np.mean(data)
var = np.var(data)  # or np.var(data, ddof=1) for sample variance
std = np.std(data)
```

## Variance properties

**Scaling:**
$$\text{Var}(aX) = a^2 \text{Var}(X)$$

**Shift:**
$$\text{Var}(X + b) = \text{Var}(X)$$

**Sum (independent):**
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

If not independent, need covariance:
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$$

## Covariance

Measure of joint variability:

$$\text{Cov}(X,Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$$

Positive: X and Y move together
Negative: X up means Y down
Zero: no linear relationship

## Common distributions

| Distribution | E[X] | Var(X) |
|-------------|------|--------|
| Bernoulli(p) | p | p(1-p) |
| Binomial(n,p) | np | np(1-p) |
| Poisson(λ) | λ | λ |
| Normal(μ,σ²) | μ | σ² |
| Uniform(a,b) | (a+b)/2 | (b-a)²/12 |
| Exponential(λ) | 1/λ | 1/λ² |

## In machine learning

**Loss functions:**
Minimize expected loss:
$$\min_\theta E_{(x,y)}[\text{Loss}(f_\theta(x), y)]$$

**Bias-variance tradeoff:**
$$E[\text{error}] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

**Monte Carlo estimation:**
$$E[f(X)] \approx \frac{1}{n}\sum_{i=1}^n f(x_i)$$

Sample mean converges to expected value.

## Law of Large Numbers

Sample mean → expected value as n → ∞

$$\bar{X}_n = \frac{1}{n}\sum X_i \to E[X]$$

Why training on more data helps: better estimate of true expected loss.

## Central Limit Theorem

Sum of many independent random variables → Normal distribution

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \to \mathcal{N}(0,1)$$

Why Normal shows up everywhere.

## Computing from data

```python
# Sample estimates
mean = np.mean(X)
var = np.var(X, ddof=1)  # ddof=1 for unbiased sample variance

# Covariance matrix
cov_matrix = np.cov(X, Y)

# Correlation (normalized covariance)
corr = np.corrcoef(X, Y)
```

## Moment generating functions

More advanced: MGF encodes all moments.

$$M_X(t) = E[e^{tX}]$$

Derivatives at t=0 give moments:
- M'(0) = E[X]
- M''(0) = E[X²]

Useful for proofs and deriving distributions.

The animation shows how mean and variance describe distributions: [Expected Value Animation](https://danielsobrado.github.io/ml-animations/animation/expected-value-variance)

---

Related:
- [Probability Distributions](/posts/probability-distributions/)
- [Entropy - another way to measure spread](/posts/entropy/)
