# Degenerate boundary semantics

## Overview

A correlation sequence `r` is admissible if its Toeplitz matrix is positive
semidefinite at every order. The Levinson--Durbin recursion that maps `r` to
partial autocorrelations `alpha` divides by the innovation variance
`sigma_n^2` at each step, so it is a genuine bijection only where
`sigma_n^2 > 0`. `sigma_n^2` can reach exactly zero for an otherwise
perfectly admissible `r`: the sequence has simply run out of independent
degrees of freedom. This is a legitimate, *admissible* state, not an error,
and the package treats it as a distinct case throughout rather than folding
it into either "valid interior point" or "invalid input."

Four cases need to be told apart:

1. admissible interior sequences,
2. admissible degenerate boundary sequences,
3. inadmissible sequences,
4. finite-precision ambiguity between the above.

All four are read off one quantity: the innovation-variance trace
`sigma_0^2 = 1, sigma_1^2, ..., sigma_N^2` produced by the single Levinson
recursion (`ccf.levinson._levinson_correlations_batch`) shared by
`pacf`, `pacf_status`, `pacf_prefix`, and `ccf.bounds`. The sections
below are different readings of that one recursion's output, not different
algorithms.

That recursion always stops at the first terminal event -- a reached
boundary or an inadmissibility -- and never inspects coefficients supplied
beyond it. This gives the API three layers, from cheapest to most
complete:

- `pacf_status` / `pacf_prefix` describe the recursion up to its first
  terminal event only.
- `pacf` is the strict `r <-> alpha` coordinate transformation: boundary or
  invalid both raise.
- `check_admissibility` / `admissible_bounds` validate the complete
  supplied `r`, including any continuation supplied past a reached
  boundary.

Consequently, whether a `ValueError` on a bad supplied tail comes from the
independent recursion or from tail validation depends on where the problem
sits:

- **Inadmissibility encountered before a boundary** (`abs(alpha_n) > 1` for
  some `n`, reached without ever hitting `sigma_n^2 = 0`): `pacf`,
  `pacf_prefix`, and `admissible_bounds` all raise `ValueError`, and
  `pacf_status` reports `invalid`.
- **Inconsistent continuation supplied after a boundary already reached**:
  the independent recursion has already stopped at a valid boundary, so
  `pacf_status` reports `boundary` and `pacf_prefix` returns the
  boundary-inclusive independent prefix regardless of what follows in `r`.
  `check_admissibility`, `admissible_bounds`, and `extend_at_boundary`
  instead inspect that supplied tail and raise/report failure if it is
  inconsistent with the forced continuation.

## 1. Interior sequences

**Meaning.** `sigma_n^2 > 0` for every `n = 1, ..., N` (equivalently
`abs(alpha_n) < 1` throughout). The map `r <-> alpha` is a genuine bijection
at this order.

**Behavior.** `pacf(r)` and `from_pacf(alpha)` return the full vector
unconditionally -- no exception, no warning.

## 2. Degenerate boundary sequences

**Meaning.** `sigma_m^2 = 0` at some order `m <= N` (equivalently
`abs(alpha_m) = 1`), reached from a strictly positive-definite prefix
(`sigma_n^2 > 0` for `n < m`). The Toeplitz matrix built from `r_1, ...,
r_m` is positive semidefinite but singular. If further coefficients
`r_{m+1}, ..., r_N` are supplied, they are not free: they are uniquely
forced by the null vector of that singular matrix, and a valid admissible
sequence must equal the forced continuation (see `SH_research_note.pdf` and
the `extend_at_boundary` docstring for the derivation).

The consequence for the API: `alpha_1, ..., alpha_{m-1}`, together with the
forced `alpha_m = +-1`, remain well defined, but there is no `r <-> alpha`
bijection beyond order `m` -- `alpha_{m+1}, ..., alpha_N` simply do not
exist as independent coordinates. Three distinct operations apply here:

- **Coordinate transformation -- `pacf(r)`.** Raises `SingularToeplitzError`
  the moment it hits this case. There is no mode parameter to suppress it;
  raising is the only behavior.
- **Boundary analysis -- `pacf_prefix(r)`.** Never raises on a
  degenerate-but-admissible input. Returns a `PrefixResult` with the maximal
  independent PACF prefix (`alpha`, its last entry exactly `+1` or `-1`),
  the order `m` at which the boundary was reached, the innovation-variance
  trace, and the terminal predictor `phi^{(m)}` needed to continue past
  the boundary.
- **Deterministic continuation -- `extend_at_boundary(r, n_extra)`.**
  Appends the uniquely forced continuation past the boundary. Coefficients
  already supplied past the boundary are validated against that forced
  continuation rather than overwritten; an inconsistent value raises
  `ValueError`. `admissible_bounds(r)` performs the same validation when
  coefficients past the boundary are part of its input, and always appends
  one further interval for the next, unsupplied coefficient, which also
  collapses to the forced continuation past the boundary. Both functions
  read the boundary state from the same recursion `pacf_prefix` uses,
  rather than duplicating it.

## 3. Inadmissible sequences

**Meaning.** `abs(alpha_n) > 1` for some `n` -- equivalently, a leading
principal submatrix of the Toeplitz matrix has a negative eigenvalue. This
is a plain input error: `r` does not correspond to any nonnegative power
spectrum. It is unrelated to the boundary case above even though both are
detected during the same recursion pass -- the boundary is `sigma_n^2 = 0`
exactly; this is the recursion becoming mathematically impossible to
continue at all.

**Behavior.** `pacf`, `pacf_prefix`, `from_pacf`, and `admissible_bounds`
all raise `ValueError`, distinct from `SingularToeplitzError`.
`check_admissibility(r)` returns `False` instead of raising, unless called
with `raise_error=True`.

## 4. Finite precision

Not a distinct mathematical category, but a finite-precision artifact of
the other three:

- `sigma_n^2` can compute as a small negative number (e.g. `-1e-15`) at a
  point that is mathematically strictly interior (`sigma_n^2 > 0` exactly),
  purely from float64 roundoff accumulated over the recursion.
- At high order, or for correlations close to the boundary, float64 may be
  unable to resolve whether a given `r` is exactly at the boundary,
  strictly interior but very close to it, or (rarely) very slightly
  inadmissible.

The float64 path (`ccf/levinson.py`) absorbs this with a fixed
tolerance, `_ROUNDING_TOL = 1e-12`: a computed `sigma_n^2` in
`(-_ROUNDING_TOL, 0)` is clamped to exactly `0.0`, treated as case 2 rather
than case 3. `ccf.bounds` uses a second, looser tolerance,
`_BOUNDARY_CONTINUATION_TOL`, for comparing a *supplied* coefficient
against its forced continuation -- an order-dependent chain of products
that amplifies roundoff faster than the single-step comparisons
`_ROUNDING_TOL` guards. See `DOCUMENTATION.md` ("Tolerances") for why the
two are kept separate.

The arbitrary-precision path (`ccf/precision.py`) lets a caller
resolve this ambiguity directly by increasing the working precision
(`recommended_dps`) rather than relying on a fixed tolerance.

## Public API summary

| Function | Behavior at the boundary |
| --- | --- |
| `pacf(r)` | raises `SingularToeplitzError` |
| `from_pacf(alpha)` | boundary unreachable by construction (`abs(alpha_n) < 1` required) |
| `pacf_prefix(r)` | returns the maximal independent prefix, reached through the first boundary; never raises for a degenerate-but-admissible `r`, and does not validate coefficients supplied beyond a reached boundary |
| `extend_at_boundary(r, n_extra)` | appends the forced continuation |
| `admissible_bounds(r)` | returns bounds for `r` plus the next coefficient; collapse to the forced continuation past the boundary |
| `check_admissibility(r)` | `True` for an admissible boundary sequence |
| `pacf_status(r)` | reports the recursion state up to the first boundary/invalid order; does not validate coefficients supplied beyond a reached boundary |
| `pacf_mp(r, ...)` / `from_pacf_mp(alpha, ...)` | arbitrary-precision counterparts; see their docstrings for `at_boundary` |
