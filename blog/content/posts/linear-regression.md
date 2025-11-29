---
title: "Linear Regression - the foundation of ML"
date: 2024-11-10
draft: false
tags: ["linear-regression", "regression", "ml-basics", "statistics"]
categories: ["Math Fundamentals"]
---

Before deep learning there was linear regression. Still the most useful model for many problems. And understanding it helps understand everything else.

## What is it?

Predict output y from input x using linear function:

$$y = wx + b$$

For multiple features:
$$y = w_1x_1 + w_2x_2 + ... + w_nx_n + b = \mathbf{w}^T\mathbf{x} + b$$

Find w and b that best fit the data.

![Linear Regression](https://danielsobrado.github.io/ml-animations/animation/linear-regression)

See the fitting process: [Linear Regression Animation](https://danielsobrado.github.io/ml-animations/animation/linear-regression)

## Loss function

How do you define "best fit"? Minimize squared errors.

$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i - b)^2$$

Called Mean Squared Error (MSE).

Why squared? 
- Penalizes big errors more
- Differentiable everywhere
- Has nice mathematical properties

## Closed-form solution

Unlike neural networks, linear regression has analytical solution.

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

This is called the Normal Equation.

```python
import numpy as np

# Add bias column
X_b = np.c_[np.ones(len(X)), X]
# Solve
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

No iteration needed. But matrix inversion is O(n³), expensive for large n.

## Gradient descent solution

For large datasets, use gradient descent:

$$w = w - \alpha \frac{\partial L}{\partial w}$$

Gradient:
$$\frac{\partial L}{\partial w} = -\frac{2}{n}\sum(y_i - \hat{y}_i)x_i$$

```python
def linear_regression_gd(X, y, lr=0.01, epochs=1000):
    n = len(X)
    w = np.zeros(X.shape[1])
    b = 0
    
    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y
        
        w -= lr * (2/n) * (X.T @ error)
        b -= lr * (2/n) * error.sum()
    
    return w, b
```

## Using sklearn

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Coefficients
print(model.coef_)      # weights
print(model.intercept_) # bias
```

## Assumptions

Linear regression assumes:
1. **Linearity:** Relationship is actually linear
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant variance of errors
4. **Normality:** Errors are normally distributed

Violating these doesn't break the model but predictions/confidence intervals may be off.

## Feature scaling

Features with different scales cause problems for gradient descent.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Or normalize to [0, 1]:
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

## Regularization

**Ridge regression (L2):**

$$L = MSE + \lambda\sum w_i^2$$

Penalizes large weights. Handles multicollinearity.

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
```

**Lasso (L1):**

$$L = MSE + \lambda\sum |w_i|$$

Can zero out weights. Feature selection built in.

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=1.0)
```

**Elastic Net:**

Combines both.

## Polynomial regression

Linear in parameters, not features. Can fit curves.

$$y = w_0 + w_1x + w_2x^2 + w_3x^3$$

Still linear regression - just transform features first.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model.fit(X_poly, y)
```

## Evaluation metrics

**R² (coefficient of determination):**
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

1 = perfect, 0 = no better than mean, negative = terrible

**RMSE:** √MSE - same units as target

**MAE:** Mean absolute error - less sensitive to outliers

## When to use linear regression

Good for:
- Baseline model
- Interpretability needed
- Few features
- Linear relationships
- Small to medium data

Not good for:
- Complex nonlinear patterns
- High-dimensional sparse data
- When interactions matter a lot

But always try linear first. Surprisingly often it's enough.

The animation shows how the line fits to data: [Linear Regression Animation](https://danielsobrado.github.io/ml-animations/animation/linear-regression)

---

Related:
- [Gradient Descent optimization](/posts/gradient-descent/)
- [Cross-Entropy for classification](/posts/cross-entropy/)
