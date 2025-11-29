---
title: "GloVe - Global Vectors for word representation"
date: 2024-11-25
draft: false
tags: ["glove", "embeddings", "nlp", "word-vectors"]
categories: ["NLP Fundamentals"]
---

Word2Vec uses local context windows. GloVe takes a different approach - use global word co-occurrence statistics. Came from Stanford in 2014.

## Word2Vec vs GloVe

Word2Vec: trains on (center, context) word pairs, processes corpus as stream
GloVe: first builds co-occurrence matrix, then factorizes it

Different philosophy but results are similar. Sometimes GloVe works better, sometimes Word2Vec.

![GloVe Training](https://danielsobrado.github.io/ml-animations/animation/glove)

Watch the process: [GloVe Animation](https://danielsobrado.github.io/ml-animations/animation/glove)

## The co-occurrence matrix

Count how often words appear together in a window.

```
         cat   dog   sat   mat
cat       -     5     3     2
dog       5     -     1     1
sat       3     1     -     4
mat       2     1     4     -
```

X[i,j] = how many times word i appears near word j

This matrix is huge. Vocabulary of 400K words = 160 billion entries. But it's very sparse.

## The objective

GloVe's insight: word vectors should encode the ratio of co-occurrence probabilities.

For words i and j:
$$w_i \cdot w_j + b_i + b_j = \log(X_{ij})$$

The loss function:

$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

Where f(x) is a weighting function that:
- Downweights very frequent pairs (they dominate otherwise)
- Handles X[i,j] = 0 cases

## Why ratios matter

Consider words: ice, steam, water, solid

P(solid|ice) should be high
P(solid|steam) should be low
Ratio P(solid|ice) / P(solid|steam) tells us something

Words with high ratio with ice and low with steam → related to ice specifically

GloVe tries to capture these ratios in the vector space.

## Building co-occurrence matrix

```python
import numpy as np
from collections import defaultdict

def build_cooccurrence(corpus, vocab, window=5):
    cooccurrence = defaultdict(float)
    
    for sentence in corpus:
        for i, center in enumerate(sentence):
            for j in range(max(0, i-window), min(len(sentence), i+window+1)):
                if i != j:
                    context = sentence[j]
                    distance = abs(i - j)
                    # weight by distance (closer = more weight)
                    cooccurrence[(center, context)] += 1.0 / distance
    
    return cooccurrence
```

## Training

GloVe uses AdaGrad or similar optimizer. Key things:

1. Initialize word vectors and biases randomly
2. Sample non-zero entries from co-occurrence matrix  
3. Compute loss for each pair
4. Update vectors via gradient descent

```python
# simplified training loop
for epoch in range(epochs):
    for (i, j), count in cooccurrence.items():
        # compute gradient
        diff = np.dot(w[i], w_context[j]) + b[i] + b_context[j] - np.log(count)
        weight = f(count)  # weighting function
        
        # update
        grad = weight * diff
        w[i] -= lr * grad * w_context[j]
        w_context[j] -= lr * grad * w[i]
        b[i] -= lr * grad
        b_context[j] -= lr * grad
```

Final vectors: w[i] + w_context[i] (they're symmetric so average them)

## Weighting function

$$f(x) = \begin{cases} (x/x_{max})^{0.75} & \text{if } x < x_{max} \\ 1 & \text{otherwise} \end{cases}$$

The 0.75 exponent is empirical. Could be different.

Prevents "the", "a", "is" from dominating training.

## Pretrained vectors

Stanford provides pretrained GloVe vectors:
- Wikipedia + Gigaword: 6B tokens
- Common Crawl: 42B and 840B tokens
- Twitter: 27B tokens (captures informal language)

Dimensions: 50, 100, 200, 300

```python
# loading pretrained
def load_glove(path):
    embeddings = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings
```

## When to use GloVe vs Word2Vec

**GloVe:**
- When you have fixed corpus
- Want reproducible results (deterministic given matrix)
- Global statistics matter for your task

**Word2Vec:**
- Streaming/online setting
- Very large corpus (building matrix expensive)
- Want to incrementally update

Honestly, differences are small. Try both if you're unsure.

## Limitations

Same as Word2Vec mostly:
- Static vectors (no context-dependent meaning)
- OOV words get nothing
- Biases from training data

Also:
- Memory for co-occurrence matrix
- Harder to train incrementally

The visualization shows how GloVe learns from co-occurrence patterns: [GloVe Animation](https://danielsobrado.github.io/ml-animations/animation/glove)

---

Related:
- [Word2Vec Explained](/posts/word2vec/)
- [FastText with Subwords](/posts/fasttext/)
