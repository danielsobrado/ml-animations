---
title: "Matrix Multiplication - the operation behind everything"
date: 2024-11-09
draft: false
tags: ["matrix-multiplication", "linear-algebra", "fundamentals"]
categories: ["Math Fundamentals"]
---

Neural networks are mostly matrix multiplication. Understanding this operation deeply helps understand everything else.

## The basic operation

Matrix A (m×n) times matrix B (n×p) gives matrix C (m×p).

Each element C[i,j] is dot product of row i from A and column j from B.

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

```python
# Manual implementation
def matmul(A, B):
    m, n = A.shape
    n, p = B.shape
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]
    return C
```

Never use this - O(n³) and slow. Use numpy/torch.

![Matrix Multiplication](https://danielsobrado.github.io/ml-animations/animation/matrix-multiplication)

Visualized step by step: [Matrix Multiplication Animation](https://danielsobrado.github.io/ml-animations/animation/matrix-multiplication)

## Dimension rules

**Inner dimensions must match:**
(m × **n**) × (**n** × p) = (m × p)

If they don't match, multiplication is undefined.

```python
A = np.random.randn(3, 4)  # 3×4
B = np.random.randn(4, 2)  # 4×2
C = A @ B                   # 3×2 ✓

B = np.random.randn(5, 2)  # 5×2
C = A @ B                   # Error! 4 ≠ 5
```

## In neural networks

Linear layer: y = Wx + b

```python
# Batch of inputs: x is (batch_size, input_dim)
# Weights: W is (input_dim, output_dim)
# Output: y is (batch_size, output_dim)

x = torch.randn(32, 100)   # 32 samples, 100 features
W = torch.randn(100, 50)    # project to 50 dims
y = x @ W                   # 32×50 output
```

Each sample gets its own output through same weights.

## Why it's associative

(AB)C = A(BC)

Order of multiplication matters. Choose wisely.

```python
# A is 1000×10, B is 10×1000, C is 1000×1

# (AB)C: (1000×10)(10×1000) = 1000×1000, then (1000×1000)(1000×1) = expensive!
# A(BC): (10×1000)(1000×1) = 10×1, then (1000×10)(10×1) = cheap!
```

Same result, vastly different compute.

## Transpose properties

$(AB)^T = B^T A^T$

Order reverses when transposing product.

```python
(A @ B).T == B.T @ A.T  # True (up to floating point)
```

## Batched matrix multiplication

Modern ML uses batched operations:

```python
# A is (batch, m, n)
# B is (batch, n, p)
# Result is (batch, m, p)

A = torch.randn(64, 10, 20)
B = torch.randn(64, 20, 30)
C = torch.bmm(A, B)  # batch matrix multiply
# C is (64, 10, 30)
```

Each batch element multiplied independently.

## Einsum - flexible notation

Einstein summation notation handles complex cases:

```python
# Regular matmul
C = torch.einsum('ij,jk->ik', A, B)

# Batched matmul
C = torch.einsum('bij,bjk->bik', A, B)

# Dot product
c = torch.einsum('i,i->', a, b)

# Outer product
C = torch.einsum('i,j->ij', a, b)
```

## Attention is matmul

Self-attention core operation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Two matrix multiplications: Q @ K.T and result @ V

```python
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # matmul
attn = F.softmax(scores, dim=-1)
output = attn @ V  # matmul
```

## GPU efficiency

GPUs are optimized for matrix multiplication. Key to making it fast:

1. **Memory alignment:** Matrices should be contiguous
2. **Batch operations:** More parallelism
3. **Appropriate sizes:** Powers of 2 often faster
4. **FP16/BF16:** Half precision for speed

```python
# Slow: many small matmuls
for i in range(1000):
    y = x[i] @ W

# Fast: one batched matmul
y = x @ W  # x is (1000, d)
```

## Sparse matrices

When matrix mostly zeros, use sparse formats:

```python
from scipy import sparse

# CSR format
A_sparse = sparse.csr_matrix(A)
C = A_sparse @ B  # efficient for sparse A
```

Attention masks, graph adjacency matrices often sparse.

## Common pitfalls

**Shape mismatch:** Most common error
```python
# Debug shapes
print(A.shape, B.shape)  # before every matmul when debugging
```

**Broadcasting confusion:**
```python
# These are different!
A @ B      # matrix multiply
A * B      # element-wise multiply (with broadcasting)
```

**Memory:** Matrix multiplication creates new tensor
```python
C = A @ B  # allocates new memory
# For in-place: use torch.mm(A, B, out=C)
```

The animation makes the operation intuitive: [Matrix Multiplication Animation](https://danielsobrado.github.io/ml-animations/animation/matrix-multiplication)

---

Related:
- [Attention uses matmul](/posts/attention-mechanism-part1/)
- [Eigenvalues of matrices](/posts/eigenvalue/)
- [SVD decomposition](/posts/svd/)
