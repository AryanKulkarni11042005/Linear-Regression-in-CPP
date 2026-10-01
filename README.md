# Linear Regression from Scratch in C++

Linear regression trained with gradient descent, built on a hand-written C++ matrix library (no Eigen, no BLAS).

## Model

Predict `y` from `x` with a weight (slope) and a bias (intercept):

```
y_hat = w0 * x + w1
```

Add a column of 1s to `X` so the bias becomes part of the matrix multiply:

```
X (n x 2)  =  [ x_1  1 ]        w (2 x 1)  =  [ w0 ]
              [ x_2  1 ]                      [ w1 ]
              [ ...    ]
              [ x_n  1 ]

y_hat = X * w          (n x 1)
```

## Matrix multiplication

For `A (m x k)` and `B (k x n)`, the result `C` is `m x n`:

```
C(i, j) = sum over p of  A(i, p) * B(p, j)
```

The inner dimensions (`k`) must match.

## Error and cost

```
e = y_hat - y                       (n x 1)

MSE = (1 / n) * sum(e_i ^ 2)  =  (1 / n) * e^T * e
```

Squaring keeps errors positive, so they can't cancel, and penalizes large misses more.

## Gradient

By the chain rule, the derivative of the MSE with respect to `w` is:

```
grad = (2 / n) * X^T * e            (2 x 1, same shape as w)
```

Andrew Ng's version uses the cost `(1 / 2n) * sum(e^2)`, which gives `(1 / n) * X^T * e`. The two differ only by a constant factor of 2, which acts like a different learning rate.

## Gradient descent

Start with `w = 0` and repeat:

```
1. y_hat = X * w
2. e     = y_hat - y
3. grad  = (2 / n) * X^T * e
4. w     = w - lr * grad
```

`lr` is the learning rate. If it is too large, the loss grows every step and overflows to `inf` and then `NaN`.

## Normalization

Features on very different scales make gradient descent slow or unstable. Standardize `x` before training:

```
x_norm = (x - mean) / std
```

- `mean` and `std` are computed from the **training data only**.
- New data must be scaled with the **same** `mean` and `std` before predicting.

To convert the learned weights back to the original units:

```
slope = w0 / std
bias  = w1 - w0 * mean / std
```

## Results

Data: 10 points, `x = 2 ... 18`, roughly `y = 2x + 1` with noise.

| | Without normalization | With normalization |
|---|---|---|
| Max stable learning rate | about 0.0084 | about 1.0 |
| Learned slope / bias | 1.839 / 1.743 | 1.839 / 1.743 (after converting back) |
| Final MSE | 0.098 | 0.098 |

Prediction for `x = 20` is about `38.53`.

The MSE stops at about 0.098 and not 0, because no straight line passes through all 10 points.

## Build

```
g++ -std=c++17 -O2 main.cpp matrix.cpp -o linreg
```

Use a Release build when timing. Debug builds are much slower.
