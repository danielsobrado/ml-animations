---
title: "SVD - Singular Value Decomposition explained"
date: 2024-11-07
draft: false
tags: ["svd", "linear-algebra", "dimensionality-reduction"]
categories: ["Math Fundamentals"]
---

SVD factors any matrix into three pieces. Unlike eigendecomposition, works for non-square matrices. The Swiss Army knife of matrix methods.

## The decomposition

Any matrix A (m×n) can be written as:

$$A = U\Sigma V^T$$

Where:
- U is m×m orthogonal (columns are left singular vectors)
- Σ is m×n diagonal (singular values on diagonal)
- V is n×n orthogonal (columns are right singular vectors)

![SVD Decomposition](https://danielsobrado.github.io/ml-animations/animation/svd)

Visual breakdown: [SVD Animation](https://danielsobrado.github.io/ml-animations/animation/svd)

## Computing SVD

```python
import numpy as np

A = np.random.randn(5, 3)
U, s, Vt = np.linalg.svd(A)

# s is 1D array of singular values
# Full reconstruction
S = np.zeros((5, 3))
np.fill_diagonal(S, s)
A_reconstructed = U @ S @ Vt
```

Or for "economy" SVD (smaller matrices):
```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
```

## What singular values mean

Singular values are always non-negative, usually sorted largest to smallest.

They measure how much the matrix "stretches" in each direction.

- σ₁: maximum stretch
- σₙ: minimum stretch (or 0 for rank-deficient)

## Low-rank approximation

Key property: truncated SVD gives best rank-k approximation.

Keep only top k singular values:
$$A_k = U_k\Sigma_k V_k^T$$

This minimizes ||A - A_k|| among all rank-k matrices.

```python
k = 2
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

## Applications

**Image compression:**
```python
# Image as matrix
from PIL import Image
img = np.array(Image.open('photo.jpg').convert('L'))

U, s, Vt = np.linalg.svd(img)

# Keep top 50 components
k = 50
compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

**Dimensionality reduction (similar to PCA):**
```python
# Project data to k dimensions
X_reduced = X @ Vt[:k].T
```

**Recommender systems:**
```python
# User-item matrix
# Factor into user features and item features
# Handle missing values with specialized algorithms
```

**Pseudoinverse:**
$$A^+ = V\Sigma^{+}U^T$$

Where Σ⁺ inverts non-zero singular values.

## SVD vs Eigendecomposition

| Property | SVD | Eigendecomposition |
|----------|-----|-------------------|
| Matrix shape | Any | Square only |
| Values | Always real, non-negative | Can be complex |
| Vectors | Two sets (U, V) | One set |
| Exists | Always | Not always |

For symmetric A: singular values = |eigenvalues|

## Relation to PCA

PCA on centered data X:
1. Compute covariance X^TX
2. Eigendecomposition

Equivalently:
1. SVD of X = UΣV^T
2. Principal components are columns of V
3. Singular values² / (n-1) = eigenvalues of covariance

```python
# These give same result (up to sign)
U, s, Vt = np.linalg.svd(X_centered)
pca_components_svd = Vt.T

eigenvalues, eigenvectors = np.linalg.eigh(X_centered.T @ X_centered)
pca_components_eig = eigenvectors
```

## Truncated SVD for sparse matrices

Full SVD on huge sparse matrix? Expensive and dense result.

Use randomized/truncated SVD:
```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X_sparse)
```

Works directly on sparse matrices. Much faster.

## Numerical stability

SVD is numerically stable. Used to solve:
- Least squares problems
- Linear systems with ill-conditioned matrices
- Matrix rank determination

```python
# Numerical rank (how many singular values are "big enough")
rank = np.sum(s > threshold)
```

## Low-rank matrix completion

Netflix problem: fill missing entries in sparse user-item matrix.

Assume matrix is approximately low-rank. Find U, V such that:
- UV^T matches known entries
- Has low rank

SVD gives starting point, but need specialized algorithms for missing data.

## Eckart-Young Theorem

The truncated SVD is optimal:

$$\min_{\text{rank}(B)=k} ||A - B||_F = ||A - A_k||_F = \sqrt{\sum_{i>k}\sigma_i^2}$$

No other rank-k matrix is closer (in Frobenius norm).

Watch the decomposition process: [SVD Animation](https://danielsobrado.github.io/ml-animations/animation/svd)

---

Related:
- [Eigenvalues](/posts/eigenvalue/)
- [QR Decomposition](/posts/qr-decomposition/)
- [Matrix Multiplication](/posts/matrix-multiplication/)
