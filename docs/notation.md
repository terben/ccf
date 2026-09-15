# Notation: paper vs. code

The package follows the paper's notation and sign conventions throughout.
Python arrays are zero-based, so every array position is offset by one
from the paper's subscript. This table fixes that correspondence once,
rather than repeating it in individual docstrings.

| Paper           | Code                                                                                 |
| --------------- | ------------------------------------------------------------------------------------ |
| `r_n`           | `r[n - 1]` (1-D array, 0-indexed)                                                     |
| `alpha_n`       | `alpha[n - 1]`                                                                        |
| `sigma_n^2`     | `sigma2[n - 1]` -- see "The `sigma2` array" below                                     |
| `phi_j^{(n)}`   | `PrefixResult.predictor[j - 1]` at `n = order + 1` -- see "Predictor coefficients" below |

## Zero-based array positions

Every array above is an ordinary Python sequence, indexed from 0. Array
position `k` always corresponds to the paper subscript `k + 1`. This is
just the usual offset between a 1-based mathematical subscript and a
0-based array position, not a second mathematical indexing convention.

## The `sigma2` array

`sigma2`, as returned by `innovation_variances` and stored in
`PrefixResult.sigma2`, holds the paper's residual variance and has one
more entry than `alpha`. The paper's recursion starts from
`sigma_1^2 = 1` and proceeds via `sigma_(n+1)^2 = sigma_n^2 * (1 -
alpha_n^2)`. Array position `j` stores

```
sigma2[j]  <->  sigma_(j+1)^2.
```

So `sigma2[0]` is `sigma_1^2 = 1`, `sigma2[1]` is `sigma_2^2`, and for an
array of length `N + 1`, `sigma2[N]` is `sigma_(N+1)^2`. The paper does
not define a quantity `sigma_0^2`, and this array position should not be
read as one -- the leading zero-based array position simply stores
`sigma_1^2`.

## Predictor coefficients

`PrefixResult.predictor` holds the terminal Levinson--Durbin predictor
coefficients at a reached boundary. In the paper's boundary convention,
`m` is the index of the first singular Toeplitz matrix `A_m`, so
`alpha_(m-1)` is the last well-defined PACF and `PrefixResult.order =
m - 1`. The stored predictor is the order-`m` predictor:

```
predictor[j - 1]  <->  phi_j^{(m)},   j = 1, ..., m - 1,
```

i.e. `phi^{(order + 1)}`, not `phi^{(order)}`. These are exactly the
coefficients of

```
p_m = sum_{j=1}^{m-1} phi_j^{(m)} r_{m-j},
```

the linear prediction of `r_m`. Since `sigma_m^2 = 0` at the boundary,
the admissible interval for `r_m` has collapsed to a point, so
`r_m = p_m` -- the first coefficient of the Toeplitz-forced continuation.
`predictor` is `None` unless the boundary was reached.

For the boundary and admissibility semantics of `r`, `alpha`, and
`sigma2` -- including how `m` and the forced continuation past the
boundary are defined -- see `docs/boundary_semantics.md`.
