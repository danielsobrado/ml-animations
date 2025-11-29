---
title: "PageRank - how Google ranked the web"
date: 2024-10-24
draft: false
tags: ["pagerank", "graph-algorithms", "google", "markov-chains"]
categories: ["Algorithms"]
---

A page is important if important pages link to it. Circular definition but it works. This insight made Google.

## The idea

Imagine random web surfer. Keeps clicking random links. Where do they spend most time? That's PageRank.

Pages with many incoming links from important pages get visited more.

![PageRank](https://danielsobrado.github.io/ml-animations/animation/pagerank)

Watch scores converge: [PageRank Animation](https://danielsobrado.github.io/ml-animations/animation/pagerank)

## Mathematical formulation

PageRank of page i:

$$PR(i) = \frac{1-d}{N} + d \sum_{j \to i} \frac{PR(j)}{L(j)}$$

Where:
- d: damping factor (typically 0.85)
- N: total number of pages
- j → i: pages linking to i
- L(j): number of outgoing links from j

## Matrix form

Define transition matrix M:
$$M_{ij} = \begin{cases} 1/L(j) & \text{if } j \to i \\ 0 & \text{otherwise} \end{cases}$$

With damping:
$$\tilde{M} = d \cdot M + \frac{1-d}{N} \cdot \mathbf{1}$$

PageRank is stationary distribution:
$$\mathbf{PR} = \tilde{M} \cdot \mathbf{PR}$$

Dominant eigenvector of $\tilde{M}$.

## Computing PageRank

Power iteration:

```python
def pagerank(adj_matrix, d=0.85, max_iter=100, tol=1e-6):
    n = adj_matrix.shape[0]
    
    # Normalize to transition matrix
    out_degree = adj_matrix.sum(axis=0)
    out_degree[out_degree == 0] = 1  # handle dangling nodes
    M = adj_matrix / out_degree
    
    # Initial uniform distribution
    pr = np.ones(n) / n
    
    for _ in range(max_iter):
        pr_new = d * M @ pr + (1 - d) / n
        
        if np.abs(pr_new - pr).sum() < tol:
            break
        pr = pr_new
    
    return pr / pr.sum()  # normalize
```

## Why damping factor?

Without damping (d=1):
- Spider traps: cycles that accumulate all PageRank
- Dead ends: pages with no outgoing links leak PageRank

Damping adds random jumps: with probability (1-d), jump to random page.

Guarantees convergence and handles edge cases.

## Example

Three pages A, B, C:
- A links to B, C
- B links to C
- C links to A

```python
adj = np.array([
    [0, 0, 1],  # C links to A
    [1, 0, 0],  # A links to B
    [1, 1, 0],  # A,B link to C
])

pr = pagerank(adj.T)  # transpose for row-to-column
# C has highest PageRank (most incoming links)
```

## Personalized PageRank

Instead of uniform random jumps, jump to preferred pages:

$$\mathbf{PR} = d \cdot M \cdot \mathbf{PR} + (1-d) \cdot \mathbf{v}$$

Where v is preference vector.

Used for recommendations: personalized to each user's interests.

## Connection to Markov chains

PageRank is exactly the stationary distribution of a Markov chain where:
- States = web pages
- Transitions = following links (with random jumps)

See: [Markov Chains post](/posts/markov-chains/)

## Sparse implementation

Web graph is huge but sparse. Don't use dense matrices.

```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

# Sparse adjacency
adj_sparse = csr_matrix(adj)

# Power iteration on sparse
def sparse_pagerank(adj_sparse, d=0.85, max_iter=100):
    n = adj_sparse.shape[0]
    out_degree = np.array(adj_sparse.sum(axis=0)).flatten()
    out_degree[out_degree == 0] = 1
    
    pr = np.ones(n) / n
    
    for _ in range(max_iter):
        pr = d * adj_sparse.dot(pr / out_degree) + (1 - d) / n
    
    return pr / pr.sum()
```

## NetworkX

For convenience:

```python
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')])

pr = nx.pagerank(G, alpha=0.85)
```

## Beyond web search

PageRank applies to any graph:
- **Citation networks:** Important papers
- **Social networks:** Influential users
- **Recommendation:** Important items
- **Biology:** Important genes/proteins

Any "importance flows through connections" problem.

## Limitations

- Link spam (creating fake links)
- No content understanding
- Slow to update
- Doesn't consider user intent

Modern search uses hundreds of signals. PageRank was just the beginning.

## Historical impact

1998: Larry Page and Sergey Brin at Stanford
First effective way to rank the web
Google was born

Simple algorithm, revolutionary impact.

The animation shows PageRank flowing through nodes: [PageRank Animation](https://danielsobrado.github.io/ml-animations/animation/pagerank)

---

Related:
- [Markov Chains - the math behind PageRank](/posts/markov-chains/)
- [Eigenvalues - PageRank is an eigenvector](/posts/eigenvalue/)
- [Matrix Multiplication](/posts/matrix-multiplication/)
