---
title: "Conditional Probability - P(A|B)"
date: 2024-11-04
draft: false
tags: ["probability", "bayes", "conditional"]
categories: ["Probability & Statistics"]
---

"Given that X happened, what's the probability of Y?" This is conditional probability. Foundation of Bayesian thinking.

## Definition

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

Probability of A given B = probability of both / probability of B.

![Conditional Probability](https://danielsobrado.github.io/ml-animations/animation/conditional-probability)

See the intuition: [Conditional Probability Animation](https://danielsobrado.github.io/ml-animations/animation/conditional-probability)

## Example

Deck of cards. 
- P(King) = 4/52
- P(Face card) = 12/52
- P(King | Face card) = ?

Given it's a face card, what's probability it's a King?
P(King | Face) = P(King and Face) / P(Face) = 4/52 / 12/52 = 4/12 = 1/3

Makes sense: 4 kings out of 12 face cards.

## Chain rule

$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

For multiple events:
$$P(A, B, C) = P(A) \cdot P(B|A) \cdot P(C|A,B)$$

Autoregressive language models use this:
$$P(w_1, w_2, ..., w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdots$$

## Bayes' theorem

The most important formula in probability:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Flip conditional around. Update beliefs with evidence.

- P(A): prior (before seeing B)
- P(A|B): posterior (after seeing B)
- P(B|A): likelihood
- P(B): evidence

## Bayes example

Test for disease:
- P(Disease) = 0.01 (1% have it)
- P(Positive | Disease) = 0.99 (99% sensitivity)
- P(Positive | No disease) = 0.05 (5% false positive)

Got positive result. What's P(Disease | Positive)?

$$P(D|+) = \frac{P(+|D) \cdot P(D)}{P(+)}$$

$$P(+) = P(+|D)P(D) + P(+|\neg D)P(\neg D) = 0.99(0.01) + 0.05(0.99) = 0.0594$$

$$P(D|+) = \frac{0.99 \times 0.01}{0.0594} = 0.167$$

Only 16.7%! Base rate matters a lot.

## Independence

A and B independent if:
$$P(A|B) = P(A)$$

Equivalently:
$$P(A \cap B) = P(A) \cdot P(B)$$

Knowing B tells you nothing about A.

## Conditional independence

A and B conditionally independent given C:
$$P(A|B,C) = P(A|C)$$

A and B aren't independent, but if you know C, B adds nothing.

Used in graphical models. Markov property.

## In machine learning

**Naive Bayes:**
Assume features independent given class:
$$P(y|x_1,...,x_n) \propto P(y) \prod_i P(x_i|y)$$

**Language models:**
$$P(\text{next word} | \text{previous words})$$

**Bayesian inference:**
Update model beliefs with data:
$$P(\theta|\text{data}) \propto P(\text{data}|\theta) P(\theta)$$

## Confusing cases

**Simpson's paradox:**
Relationship can reverse when conditioning on third variable.

Treatment works overall but appears to fail when you condition on hospital.

**Base rate neglect:**
People ignore P(A) when computing P(A|B). The disease example shows why that's wrong.

## Computing in practice

```python
# Joint probability table
import pandas as pd
import numpy as np

# Count occurrences
joint = pd.crosstab(df['A'], df['B'], normalize=True)

# Conditional probability
p_a_given_b = joint.loc['a1', 'b1'] / joint['b1'].sum()

# Using numpy
# P(A|B) = P(A,B) / P(B)
p_ab = np.mean((A == 1) & (B == 1))
p_b = np.mean(B == 1)
p_a_given_b = p_ab / p_b
```

The animation helps build intuition: [Conditional Probability Animation](https://danielsobrado.github.io/ml-animations/animation/conditional-probability)

---

Related:
- [Probability Distributions](/posts/probability-distributions/)
- [Entropy](/posts/entropy/)
- [Markov Chains](/posts/markov-chains/)
