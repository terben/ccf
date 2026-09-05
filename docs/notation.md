# Notation: paper vs. code

The package follows the paper's notation and sign conventions throughout.
Two points need a fixed index correspondence, documented here once rather
than repeated in individual docstrings.

| Paper           | Code                                                    |
| --------------- | -------------------------------------------------------- |
| `r_n`           | `r[n - 1]` (1-D array, 0-indexed)                         |
| `alpha_n`       | `alpha[n - 1]`                                            |
| `sigma_n^2`     | `sigma2[n - 1]`, where `sigma2` is the array returned by `innovation_variances` or `PrefixResult.sigma2` |
| `phi_j^{(n)}`   | not addressable directly; `PrefixResult.predictor` / the terminal predictor used by `extend_at_boundary` is `phi^{(m)}` at the order `m` where the boundary is reached |

## The `sigma2` off-by-one

`sigma2` arrays in the code are documented as `(sigma_0^2, ..., sigma_N^2)`
-- one entry longer than `alpha`, with `sigma2[0] = 1.0` as a fixed prior
value. The companion paper instead starts its recursion at
`sigma_1^2 = 1` and gives the residual-variance recursion in Eq. (19d).
These are the same quantities at the same array positions, just named one
index apart:

```
code sigma2[k]  ==  paper sigma_(k+1)^2   for k = 0, ..., N
```

Concretely, `sigma2[0]` is `1.0` in both conventions, but `sigma2[1]` is
the paper's `sigma_2^2`, not `sigma_1^2`. This affects only the *label*
used for the array position, not any formula or numerical value.

For the boundary and admissibility semantics of `r`, `alpha`, and
`sigma2`, see `docs/boundary_semantics.md`.
