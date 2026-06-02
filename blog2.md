# Whitening Inputs, Explained Simply

Think of a dataset as a table. Each row is one example, and each column is one feature. In a housing dataset, one row might be one house, and the columns might be size, number of rooms, age, and price.

In machine learning, we often call the input table `X`.

## Mean, Variance, and Covariance

The mean is the average value of a feature. If house sizes are `1000`, `1500`, and `2000`, the mean is `1500`.

Variance tells us how spread out one feature is. If most houses have similar sizes, the variance of size is small. If some houses are tiny and others are huge, the variance is large.

Covariance tells us whether two features move together. If bigger houses usually have more rooms, size and room count have positive covariance. If one feature goes up while another goes down, their covariance is negative. If there is no clear relationship, the covariance is near zero.

A covariance matrix stores these values:

```text
diagonal entries: variance of each feature
off-diagonal entries: covariance between pairs of features
```

## What Whitening Does

Whitening transforms the input data so the features are easier to optimize over.

After whitening:

```text
each feature has mean 0
each feature has variance 1
different features have covariance 0
```

In matrix form:

```text
Cov(X_white) = I
```

`I` is the identity matrix: ones on the diagonal and zeros elsewhere.

Geometrically, whitening turns a stretched, tilted data cloud into a round one. It removes correlations between features and gives every direction the same scale.

## Whitening With SVD

First, center the data:

```text
Xc = X - mean(X)
```

Then take the singular value decomposition:

```text
Xc = U S V^T
```

The pieces have useful meanings:

```text
V: directions in feature space
S: how stretched the data is along each direction
U: the data written in those directions
```

Whitening removes the uneven stretching stored in `S`.

PCA whitening rotates the data into the directions `V`, then rescales by the inverse of `S`:

```text
X_pca_white = Xc V sqrt(n) S^(-1)
            = sqrt(n) U
```

ZCA whitening does the same whitening, then rotates back toward the original feature axes:

```text
X_zca_white = sqrt(n) U V^T
```

So PCA whitening gives principal-component coordinates. ZCA whitening gives whitened data that still resembles the original centered data.

If you set all singular values to the same number, the data becomes round. To make the covariance exactly `I`, that number should be `sqrt(n)`, not `1`, assuming covariance is computed with division by `n`.

## Why Whitening Helps Linear Regression

Linear regression predicts:

```text
y_hat = Xw
```

where `w` is the vector of learned weights.

With squared error loss, the optimal weights have a closed-form solution:

```text
w* = (X^T X)^(-1) X^T y
```

If the input is whitened, then:

```text
X^T X / n = I
```

For the usual squared loss, the full-batch gradient becomes:

```text
grad L(w) = w - w*
```

That means the gradient points directly from the current weights to the optimal weights. With learning rate `1`, full-batch gradient descent reaches the optimum in one step.

This is not usually true for SGD. SGD uses one example or a small batch at a time. The whole dataset may be whitened, but one small batch usually is not perfectly whitened, so SGD steps are still noisy.

## Why Classification Is Different

For multiclass linear classification with softmax cross-entropy:

```text
P = softmax(XW)
```

The gradient is:

```text
X^T (P - Y)
```

The problem is that `P` depends nonlinearly on `W` because of the softmax. So we cannot rearrange the equation into a closed-form answer like linear regression.

That is the main split: linear regression with squared loss has an analytical solution, while softmax classification with cross-entropy is solved iteratively.

