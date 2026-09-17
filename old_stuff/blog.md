# Whitening Inputs, Linear Regression, and the Shape of Optimization

Whitening is one of those linear algebra tricks that looks like a preprocessing detail at first, but it reveals a lot about why some optimization problems are easy and others are not. In a short chain of questions, we can go from "what does it mean to whiten inputs?" to "why can linear regression be solved in one step under the right conditions?" and then to "why does softmax classification refuse to give us the same closed-form comfort?"

This post walks through that chain.

## What It Means to Whiten Input Data

Suppose we have a data matrix

```text
X in R^(n x d)
```

where rows are samples and columns are features. Before whitening, the features might have different scales and might be correlated. For example, in a housing dataset, square footage and number of rooms are probably strongly correlated.

Whitening transforms the input into a new matrix `Z` whose features are:

```text
mean(Z) = 0
Cov(Z) = I
```

That means each feature has variance `1`, and different features have covariance `0`.

If `Xc` is the centered data matrix,

```text
Xc = X - mean(X)
```

then the empirical covariance is often written as

```text
Sigma = Xc^T Xc / n
```

or, with the unbiased convention,

```text
Sigma = Xc^T Xc / (n - 1)
```

Whitening finds a linear transformation that makes the covariance matrix equal to the identity. Geometrically, it turns a tilted, stretched cloud of points into a round cloud.

## Linear Regression Has a Closed-Form Solution

For ordinary least squares linear regression, we model

```text
y_hat = Xw
```

and minimize the squared loss

```text
L(w) = 1/(2n) ||Xw - y||^2
```

The analytical solution is

```text
w* = (X^T X)^(-1) X^T y
```

assuming `X^T X` is invertible.

If `X^T X` is singular, the Moore-Penrose pseudoinverse gives the minimum-norm solution:

```text
w* = X^+ y
```

If we include an intercept, we can either add a column of ones to `X`, or center `X` and `y`, solve for `w`, and recover the intercept as

```text
b* = mean(y) - mean(X)^T w*
```

For ridge regression, the solution becomes

```text
w* = (X^T X + lambda I)^(-1) X^T y
```

The key reason ordinary least squares has this clean answer is that the squared loss produces linear normal equations. Setting the gradient equal to zero gives an equation we can rearrange directly.

## Why Whitening Makes Linear Regression Especially Simple

Now suppose `X` is centered and whitened so that

```text
X^T X / n = I
```

For the squared loss

```text
L(w) = 1/(2n) ||Xw - y||^2
```

the gradient is

```text
grad L(w) = X^T (Xw - y) / n
          = (X^T X / n) w - X^T y / n
```

Since whitening gives `X^T X / n = I`, this simplifies to

```text
grad L(w) = w - X^T y / n
```

But under this same assumption,

```text
w* = (X^T X)^(-1) X^T y
   = (n I)^(-1) X^T y
   = X^T y / n
```

So the gradient becomes

```text
grad L(w) = w - w*
```

That is beautifully direct: the gradient points exactly from the optimum to the current weights.

A full-batch gradient descent update is

```text
w_1 = w_0 - eta grad L(w_0)
```

Substituting the simplified gradient:

```text
w_1 = w_0 - eta (w_0 - w*)
```

If the learning rate is `eta = 1`, then

```text
w_1 = w*
```

So yes: with whitened input and the standard `1/(2n)` loss scaling, full-batch gradient descent can reach the least-squares optimum in one step.

But this does not usually hold for true SGD. A stochastic gradient based on one sample is

```text
g_i = x_i (x_i^T w - y_i)
```

Even if the whole dataset is whitened, an individual outer product `x_i x_i^T` is not usually the identity matrix. Whitening removes covariance at the dataset level, not at every individual sample. So SGD gradients are still noisy and generally do not land at the optimum in a single step.

## Whitening Through SVD

The singular value decomposition gives a very clean view of whitening.

For centered data,

```text
Xc = U S V^T
```

where:

```text
U: left singular vectors, sample-side coordinates
S: singular values
V: right singular vectors, feature-space directions
```

The covariance matrix is

```text
Sigma = Xc^T Xc / n
```

Using the SVD:

```text
Sigma = V (S^2 / n) V^T
```

So the principal directions are the columns of `V`, and the variances along those directions are

```text
lambda_i = s_i^2 / n
```

To whiten using PCA whitening, rotate into the principal-component basis and divide by the standard deviation in each principal direction:

```text
Z_pca = Xc V (S^2 / n)^(-1/2)
      = Xc V sqrt(n) S^(-1)
```

Since

```text
Xc V = U S
```

we get

```text
Z_pca = sqrt(n) U
```

assuming full rank. With the unbiased covariance convention, replace `sqrt(n)` with `sqrt(n - 1)`.

In practice, small singular values can cause numerical instability. A common stabilized version is

```text
Z = Xc V sqrt(n) (S + epsilon)^(-1)
```

where `epsilon` is small.

## What Diagonal and Orthogonal Transforms Do to Covariance

A useful general rule is this:

If

```text
Z = X A
```

then

```text
Cov(Z) = A^T Cov(X) A
```

Using `Sigma = Cov(X)`, this is

```text
Cov(XA) = A^T Sigma A
```

### Diagonal Transforms

Let

```text
A = D = diag(d_1, d_2, ..., d_d)
```

Then

```text
Cov(XD) = D Sigma D
```

Each covariance entry changes as

```text
Cov(XD)_ij = d_i d_j Sigma_ij
```

So a diagonal transform rescales each feature independently:

```text
variance_i -> d_i^2 variance_i
covariance_ij -> d_i d_j covariance_ij
```

It changes feature scales, but it does not rotate the coordinate system.

### Orthogonal Transforms

Let

```text
A = Q
Q^T Q = I
```

Then

```text
Cov(XQ) = Q^T Sigma Q
```

An orthogonal transform rotates or reflects the feature space. It preserves total variance:

```text
trace(Q^T Sigma Q) = trace(Sigma)
```

and it preserves the eigenvalues of the covariance matrix. It changes the coordinates used to describe the variance, not the amount of variance itself.

PCA uses exactly this idea. If `V` contains the eigenvectors of `Sigma`, then

```text
V^T Sigma V = Lambda
```

which is diagonal. That decorrelates the features. Then whitening adds a diagonal rescaling:

```text
Z = Xc V Lambda^(-1/2)
```

So whitening can be understood as:

```text
orthogonal rotation + diagonal scaling
```

The rotation decorrelates the features. The scaling makes each resulting feature have variance `1`.

## What If We Set All Singular Values to 1?

Suppose

```text
Xc = U S V^T
```

If we replace all nonzero singular values with the same constant, then the resulting matrix has spherical covariance. But the constant determines the scale.

If we literally set all singular values to `1`, we get

```text
X_new = U I V^T
```

Then

```text
X_new^T X_new / n = I / n
```

So the data is decorrelated and isotropic, but the variance is `1/n`, not `1`.

To make the covariance exactly `I`, we set the singular values to `sqrt(n)`:

```text
X_white = U sqrt(n) I V^T
        = sqrt(n) U V^T
```

Then

```text
X_white^T X_white / n = I
```

again assuming full rank.

This is closely related to ZCA whitening.

## PCA Whitening vs ZCA Whitening

There are many whitening transforms. PCA whitening and ZCA whitening are two common ones.

PCA whitening is

```text
Z_pca = Xc V sqrt(n) S^(-1)
      = sqrt(n) U
```

It rotates the data into the principal-component basis and rescales each principal direction to unit variance. The output coordinates are principal-component coordinates. They are whitened, but they are no longer aligned with the original feature axes.

ZCA whitening is

```text
Z_zca = Xc V sqrt(n) S^(-1) V^T
      = sqrt(n) U V^T
```

It also whitens the data, but then rotates it back toward the original feature basis. Among whitening transforms, ZCA is often described as the whitening transform that keeps the transformed data closest to the original centered data in squared-error distance.

So if we "set all singular values to `sqrt(n)`" while keeping both `U` and `V^T`,

```text
Xc = U S V^T
```

becomes

```text
X_white = U sqrt(n) I V^T
```

That is ZCA-style whitening.

In short:

```text
Set S -> 1:        isotropic covariance I / n
Set S -> sqrt(n):  whitened covariance I
sqrt(n) U:         PCA whitening
sqrt(n) U V^T:     ZCA whitening
```

## Why Softmax Classification Does Not Have the Same Closed Form

Linear regression with squared error has a closed-form solution because the optimality condition is linear in the weights.

Multiclass linear classification with softmax cross-entropy looks similar at first:

```text
logits = XW
P = softmax(XW)
```

For one-hot labels `Y`, the cross-entropy loss is

```text
L(W) = - sum_i sum_k Y_ik log P_ik
```

The gradient is

```text
grad_W L = X^T (P - Y)
```

The optimum satisfies

```text
X^T (P - Y) = 0
```

But

```text
P = softmax(XW)
```

so this equation is nonlinear in `W`. We cannot rearrange it into a direct formula like

```text
W* = (X^T X)^(-1) X^T Y
```

That formula belongs to least squares, not softmax cross-entropy.

Plain multinomial logistic regression is convex, so iterative methods can find a global optimum when it exists. Common solvers include:

```text
gradient descent
SGD
Newton's method
IRLS
L-BFGS
```

There are two more details worth remembering.

First, if the data is perfectly linearly separable, the unregularized softmax cross-entropy optimum may not be finite. The weights can grow without bound while the loss approaches zero.

Second, softmax has a redundancy: adding the same vector to every class's weight column does not change the probabilities. This is often handled by choosing a reference class or by adding regularization.

With L2 regularization, the solution is typically finite and well behaved, but still not closed form.

## The Big Picture

Whitening changes the coordinate system of the inputs so that the covariance becomes the identity. For unregularized linear regression, this does not add information, but it makes the geometry of the objective extremely simple. In the perfectly whitened case, full-batch gradient descent with the right learning rate can jump to the optimum in one step.

SVD makes the mechanics transparent: singular vectors rotate the data into principal directions, singular values encode the scale along those directions, and whitening replaces those scales with a common value.

But this simplicity depends strongly on the squared loss. Once softmax cross-entropy enters the picture, the model output depends nonlinearly on the weights. The optimization problem can still be convex, but the closed-form least-squares miracle is gone.

That is the useful mental split:

```text
linear regression + squared loss:
    closed form, and whitening can make optimization trivial

linear classifier + softmax cross-entropy:
    no closed form, solved iteratively
```

