---
title: "Spearman Correlation - ranking relationships"
date: 2024-10-30
draft: false
tags: ["spearman", "correlation", "statistics", "rank"]
categories: ["Probability & Statistics"]
---

Pearson correlation measures linear relationships. Spearman measures monotonic ones. If X goes up, does Y tend to go up? Works even for non-linear relationships.

## Definition

Spearman correlation = Pearson correlation on ranks.

1. Convert values to ranks
2. Compute Pearson correlation on ranks

$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

Where $d_i$ = difference between ranks.

![Spearman Correlation](https://danielsobrado.github.io/ml-animations/animation/spearman-correlation)

See ranking in action: [Spearman Animation](https://danielsobrado.github.io/ml-animations/animation/spearman-correlation)

## Example

| X | Y | Rank(X) | Rank(Y) | d | d² |
|---|---|---------|---------|---|-----|
| 10 | 100 | 1 | 1 | 0 | 0 |
| 20 | 80 | 2 | 3 | -1 | 1 |
| 30 | 90 | 3 | 2 | 1 | 1 |
| 40 | 50 | 4 | 4 | 0 | 0 |

$$\rho = 1 - \frac{6 \times 2}{4 \times 15} = 1 - 0.2 = 0.8$$

Strong positive rank correlation.

## Computing in Python

```python
from scipy.stats import spearmanr

# Basic usage
correlation, p_value = spearmanr(X, Y)

# With pandas
df['A'].corr(df['B'], method='spearman')

# Multiple variables at once
correlation_matrix = df.corr(method='spearman')
```

## Spearman vs Pearson

**Pearson:** Linear relationship only
- Measures: how close to a line?
- Sensitive to outliers
- Assumes normality

**Spearman:** Monotonic relationship
- Measures: consistent ordering?
- Robust to outliers
- No distribution assumption

```python
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Perfect monotonic but non-linear
X = np.array([1, 2, 3, 4, 5])
Y = np.array([1, 4, 9, 16, 25])  # X squared

pearsonr(X, Y)   # 0.98 (not quite 1)
spearmanr(X, Y)  # 1.0 (perfect monotonic)
```

## When to use Spearman

- Ordinal data (ranks, ratings)
- Non-linear but monotonic relationships  
- Presence of outliers
- Distribution unknown or non-normal
- Ranking evaluation (ML models)

## In ML evaluation

Spearman useful for evaluating ranking quality:

```python
# Compare predicted rankings to true rankings
from scipy.stats import spearmanr

predicted_scores = model.predict(X)
true_relevance = y_test

correlation, _ = spearmanr(predicted_scores, true_relevance)
```

If your model ranks items correctly even if scores are off, Spearman will be high.

## Handling ties

When values are equal, average their ranks:

Values: [10, 20, 20, 30]
Ranks: [1, 2.5, 2.5, 4]

scipy handles this automatically.

```python
from scipy.stats import rankdata

ranks = rankdata([10, 20, 20, 30])  # [1, 2.5, 2.5, 4]
```

## Significance testing

Is the correlation significantly different from 0?

```python
correlation, p_value = spearmanr(X, Y)

if p_value < 0.05:
    print(f"Significant correlation: {correlation:.2f}")
```

Small p-value → unlikely to see this correlation by chance.

## Confidence intervals

Bootstrap for confidence intervals:

```python
from scipy.stats import bootstrap

def spearman_stat(x, y, axis):
    return spearmanr(x, y)[0]

result = bootstrap(
    (X, Y),
    spearman_stat,
    n_resamples=1000,
    paired=True
)
ci = result.confidence_interval
```

## Correlation matrix

For multiple variables:

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df.corr(method='spearman')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

## Partial Spearman correlation

Control for third variable:

```python
from scipy.stats import spearmanr
import numpy as np

def partial_spearman(X, Y, Z):
    # Rank transform
    rx = rankdata(X)
    ry = rankdata(Y)
    rz = rankdata(Z)
    
    # Residualize
    rx_res = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ry_res = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    
    return spearmanr(rx_res, ry_res)
```

## Common interpretation

| ρ | Interpretation |
|---|----------------|
| 0.9 - 1.0 | Very strong |
| 0.7 - 0.9 | Strong |
| 0.5 - 0.7 | Moderate |
| 0.3 - 0.5 | Weak |
| 0 - 0.3 | Very weak |

Same for negative values (inverse relationship).

The animation shows how ranking affects correlation: [Spearman Animation](https://danielsobrado.github.io/ml-animations/animation/spearman-correlation)

---

Related:
- [Cosine Similarity](/posts/cosine-similarity/)
- [Expected Value and Variance](/posts/expected-value-variance/)
