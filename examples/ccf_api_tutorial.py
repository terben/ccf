# %% [markdown]
"""
# ccf: a practical API tutorial

This file is a VS Code / Spyder notebook-style Python script. Each `# %%`
marker defines a cell.

The goal is to give a paper reader a fast, executable tour of the public API:

- correlation coefficients `r` <-> partial autocorrelations `alpha`,
- batch transforms,
- diagnostics for interior / boundary / invalid sequences,
- boundary analysis and deterministic continuation,
- admissible intervals,
- Fisher coordinates,
- innovation variances and Jacobians,
- admissible-region volume,
- arbitrary-precision roundtrips,
- the didactic reference implementation,
- selected symbolic helpers.

The notation follows the paper:

    r       correlation coefficients (r_1, ..., r_N)
    alpha   partial autocorrelations (alpha_1, ..., alpha_N)
    y       Fisher coordinates
    sigma2  innovation variances

The core package depends only on NumPy. Arbitrary precision requires `mpmath`;
the symbolic examples require `sympy`.
"""

# %%
"""
## Suggested reading order for a new user

If you only have a few minutes:

1. run Sections 1--4 to understand `r <-> alpha` and `pacf_status`;
2. run Sections 5--9 to understand boundary and invalid inputs;
3. run Sections 11--14 for the coordinate geometry used in the paper;
4. use Section 15 only when float64 conditioning becomes limiting;
5. inspect `ccf.reference` alongside the paper equations when you want
   the most transparent implementation.

That is enough to start using the numerical API without first reading the
implementation details.
"""

# %%
import numpy as np
import ccf

np.set_printoptions(precision=6, suppress=True)

print("ccf version:", ccf.__version__)
print("Public top-level names:")
print(ccf.__all__)

# %% [markdown]
"""
## 1. The central map: `r <-> alpha`

For an admissible interior correlation sequence, the Levinson--Durbin
recursion gives a bijection

    r = (r_1, ..., r_N)  <->  alpha = (alpha_1, ..., alpha_N),

with every interior PACF satisfying `abs(alpha_n) < 1`.

Use

    ccf.pacf(r)         # r -> alpha
    ccf.from_pacf(alpha)  # alpha -> r

These are the central numerical transformations.
"""

# %%
alpha = np.array([0.50, -0.30, 0.20])

r = ccf.from_pacf(alpha)
alpha_recovered = ccf.pacf(r)

print("alpha:          ", alpha)
print("r:              ", r)
print("alpha recovered:", alpha_recovered)

np.testing.assert_allclose(alpha_recovered, alpha, rtol=1e-12, atol=1e-12)

# %% [markdown]
"""
## 2. Batch transforms

The NumPy implementation accepts either

    (N,)       one sequence
    (M, N)     M independent sequences

and uses the same batched recursion internally. For Monte-Carlo work, pass
the whole ensemble rather than looping over rows in Python.
"""

# %%
rng = np.random.default_rng(42)

alpha_batch = rng.uniform(-0.6, 0.6, size=(5, 6))
r_batch = ccf.from_pacf(alpha_batch)
alpha_batch_recovered = ccf.pacf(r_batch)

print("alpha_batch shape:", alpha_batch.shape)
print("r_batch shape:    ", r_batch.shape)

np.testing.assert_allclose(
    alpha_batch_recovered,
    alpha_batch,
    rtol=1e-11,
    atol=1e-11,
)

# %% [markdown]
"""
## 3. A useful mental model: three states

For a supplied correlation sequence `r`, the float64 recursion distinguishes:

1. **interior**:
   every relevant innovation variance is positive and `abs(alpha_n) < 1`;

2. **boundary**:
   the sequence is still admissible, but the Toeplitz matrix becomes
   singular. The last independent PACF is `+1` or `-1`, and later
   correlations are no longer free;

3. **invalid**:
   the sequence is inadmissible; equivalently the recursion encounters a PACF
   outside the admissible range.

For direct transformation, `ccf.pacf(r)` is intentionally strict:

- interior -> returns the complete PACF vector;
- boundary -> raises `ccf.SingularToeplitzError`;
- invalid -> raises `ValueError`.

If you do not know in advance which case you have, use `ccf.pacf_status(r)`.
It reports the state without raising for boundary or invalid sequences.
"""

# %%
r_interior = ccf.from_pacf(np.array([0.2, -0.3, 0.4]))
r_boundary = np.array([1.0, 0.5, 0.3])
r_invalid = np.array([0.9, 0.95, 0.99])

for name, r_test in [
    ("interior", r_interior),
    ("boundary", r_boundary),
    ("invalid", r_invalid),
]:
    status = ccf.pacf_status(r_test)
    print(
        f"{name:8s}: "
        f"interior={status.interior}, "
        f"boundary={status.boundary}, "
        f"invalid={status.invalid}, "
        f"order={status.order}"
    )

# %% [markdown]
"""
`PACFStatus.order` uses the 1-based Levinson order:

- for an interior sequence, `order == N`;
- for a boundary sequence, it is the order where the boundary is reached;
- for an invalid sequence, it is the first order where admissibility fails.

For a 2-D input, the status fields are 1-D arrays with one entry per row.
"""

# %%
r_mixed = np.stack(
    [
        r_interior,
        r_boundary,
        r_invalid,
    ]
)

status = ccf.pacf_status(r_mixed)

print("interior mask:", status.interior)
print("boundary mask:", status.boundary)
print("invalid mask: ", status.invalid)
print("orders:       ", status.order)

# %% [markdown]
"""
## 4. Recommended diagnostic workflow

If `r` is known to be a regular admissible sequence, simply call

    alpha = ccf.pacf(r)

If `r` comes from data, a fit, a file, or another source of uncertain
quality, the more informative workflow is:

    status = ccf.pacf_status(r)

and then branch by mathematical state.

This avoids using exceptions as the primary diagnostic mechanism.
"""

# %%
def inspect_r(r):
    """Small tutorial helper showing the intended public-API workflow."""
    r = np.asarray(r, dtype=float)
    status = ccf.pacf_status(r)

    if status.interior:
        return {
            "state": "interior",
            "order": status.order,
            "alpha": ccf.pacf(r),
        }

    if status.boundary:
        prefix = ccf.pacf_prefix(r)
        return {
            "state": "boundary",
            "order": status.order,
            "alpha_prefix": prefix.alpha,
            "sigma2": prefix.sigma2,
            "predictor": prefix.predictor,
        }

    # invalid
    valid_prefix = r[: status.order - 1]
    return {
        "state": "invalid",
        "order": status.order,
        "valid_prefix": valid_prefix,
    }


for name, r_test in [
    ("interior", r_interior),
    ("boundary", r_boundary),
    ("invalid", r_invalid),
]:
    print(name, "->", inspect_r(r_test))

# %% [markdown]
"""
## 5. Boundary analysis with `pacf_prefix`

A boundary is *admissible but degenerate*. It is not the same as an invalid
sequence.

At a boundary, the full `r <-> alpha` bijection stops because the innovation
variance reaches zero. `ccf.pacf_prefix(r)` returns the maximal independent
PACF prefix and the recursion state needed to understand that boundary.

The returned `PrefixResult` contains:

    alpha             maximal PACF prefix
    order             number of independent PACF coefficients
    reached_boundary  whether a boundary was reached
    sigma2            innovation-variance trace
    predictor         terminal predictor at the boundary, else None
"""

# %%
prefix = ccf.pacf_prefix(r_boundary)

print("alpha prefix:     ", prefix.alpha)
print("order:            ", prefix.order)
print("reached boundary: ", prefix.reached_boundary)
print("sigma2 trace:     ", prefix.sigma2)
print("terminal predictor:", prefix.predictor)

assert prefix.reached_boundary
assert abs(prefix.alpha[-1]) == 1.0

# %% [markdown]
"""
## 6. Deterministic continuation with `extend_at_boundary`

Once a sequence reaches a degenerate boundary, later correlation
coefficients are not free. They are uniquely determined by the terminal
predictor / null-vector recurrence.

Use

    ccf.extend_at_boundary(r, n_extra)

to append that forced continuation.

This function is *not* a repair function for a genuinely invalid sequence.
It is specifically for an admissible sequence that has reached the singular
boundary.
"""

# %%
r_at_boundary = np.array([1.0])

r_extended = ccf.extend_at_boundary(r_at_boundary, n_extra=5)

print("boundary sequence:", r_at_boundary)
print("forced extension: ", r_extended)

# The extension remains admissible, although degenerate.
assert ccf.check_admissibility(r_extended)

# %% [markdown]
"""
## 7. `check_admissibility`

`ccf.check_admissibility(r)` is a simple boolean check.

Important: a correctly continued boundary sequence is admissible, so this
function returns `True` for both

- strict interior sequences, and
- admissible degenerate boundary sequences.

It returns `False` for an inadmissible sequence.

With `raise_error=True`, inadmissibility is reported as `ValueError`.
"""

# %%
print("interior admissible:", ccf.check_admissibility(r_interior))
print("boundary admissible:", ccf.check_admissibility(r_extended))
print("invalid admissible: ", ccf.check_admissibility(r_invalid))

assert ccf.check_admissibility(r_interior)
assert ccf.check_admissibility(r_extended)
assert not ccf.check_admissibility(r_invalid)

# %% [markdown]
"""
## 8. Admissible intervals with `admissible_bounds`

For each correlation coefficient, the admissible interval is centered on the
Levinson linear prediction and has a half-width given by the corresponding
innovation variance.

The attached code snapshot returns bounds for the *supplied* coefficients:

    lower, upper = ccf.admissible_bounds(r)

with `len(lower) == len(r)`.

A planned API extension discussed during development is to also append the
interval for the next coefficient, giving arrays of length `N + 1`.
The compatibility helper below works with either convention.

At a degenerate boundary, later bounds collapse to the uniquely forced
continuation.
"""

# %%
r = np.array([0.1, 0.2, 0.3, 0.9])

lower, upper = ccf.admissible_bounds(r)

print("r:    ", r)
print("lower:", lower)
print("upper:", upper)

# Each supplied coefficient must lie within the interval implied by its prefix.
assert np.all(r >= lower[: len(r)] - 1e-12)
assert np.all(r <= upper[: len(r)] + 1e-12)

# %% [markdown]
"""
### The admissible interval for the *next* coefficient

If your installed version already implements the `N+1` return convention,
the next interval is simply

    lower[-1], upper[-1]

For the attached snapshot, the following helper obtains the same information
using only public functions. It chooses `alpha_(N+1)=0`, which places the
next correlation coefficient at the center of its admissible interval, and
then asks `admissible_bounds` for that extended sequence.
"""

# %%
def next_admissible_interval(r):
    """Compatibility helper for the current and planned bounds API."""
    r = np.asarray(r, dtype=float)
    lower, upper = ccf.admissible_bounds(r)

    # Newer N+1 convention, if present.
    if len(lower) == len(r) + 1:
        return float(lower[-1]), float(upper[-1])

    # Current attached snapshot: extend through alpha_(N+1) = 0.
    alpha = ccf.pacf(r)
    r_center = ccf.from_pacf(np.append(alpha, 0.0))
    lower_ext, upper_ext = ccf.admissible_bounds(r_center)
    return float(lower_ext[-1]), float(upper_ext[-1])


next_lo, next_hi = next_admissible_interval(r)

print("next admissible interval:", (next_lo, next_hi))
print("center:", 0.5 * (next_lo + next_hi))

# %% [markdown]
"""
The interval has a useful affine interpretation. If

    center = (lower + upper) / 2
    half_width = (upper - lower) / 2,

then choosing a PACF value `alpha_next` in `(-1, 1)` gives

    r_next = center + alpha_next * half_width.

Thus `alpha_next = 0` is the interval center, while `alpha_next -> +/-1`
approaches the admissible boundaries.
"""

# %%
alpha_next = 0.37

center = 0.5 * (next_lo + next_hi)
half_width = 0.5 * (next_hi - next_lo)
r_next = center + alpha_next * half_width

r_augmented = np.append(r, r_next)
alpha_augmented = ccf.pacf(r_augmented)

print("chosen alpha_next:   ", alpha_next)
print("constructed r_next:  ", r_next)
print("recovered alpha_next:", alpha_augmented[-1])

np.testing.assert_allclose(alpha_augmented[-1], alpha_next, atol=1e-11)

# %% [markdown]
"""
## 9. Diagnosing an invalid sequence and locating the first bad coefficient

A useful workflow for an invalid sequence is:

1. call `pacf_status(r)`;
2. read `status.order` to find the first failing order;
3. keep the preceding valid prefix;
4. ask for the admissible interval of the next coefficient.

This tells you where the first invalid coefficient *would have had to lie*.

There is deliberately no automatic "repair": choosing a replacement inside
that interval is an application-level decision.
"""

# %%
r_bad = np.array([0.2, 0.1, 0.95])

status = ccf.pacf_status(r_bad)

if status.invalid:
    bad_order = status.order
    valid_prefix = r_bad[: bad_order - 1]
    lo, hi = next_admissible_interval(valid_prefix)

    print("first invalid order:", bad_order)
    print("supplied value:     ", r_bad[bad_order - 1])
    print("allowed interval:   ", (lo, hi))

# %% [markdown]
"""
## 10. Schneider--Hartlap coordinates

`ccf.sh_coordinates(r)` is an alias of `ccf.pacf(r)`.

It exists because the Schneider--Hartlap normalized coordinate `x_n` is
identical to the partial autocorrelation `alpha_n`:

    x_n = alpha_n.
"""

# %%
r = ccf.from_pacf(np.array([0.2, -0.4, 0.1]))

np.testing.assert_allclose(ccf.sh_coordinates(r), ccf.pacf(r))

print("SH coordinates:", ccf.sh_coordinates(r))

# %% [markdown]
"""
## 11. Fisher coordinates

PACFs live in the open hypercube `(-1, 1)^N`. Fisher coordinates remove
those box constraints elementwise:

    y_n = arctanh(alpha_n)
    alpha_n = tanh(y_n)

Use

    ccf.fisher(alpha)
    ccf.inverse_fisher(y)

Both functions support 1-D arrays and 2-D batches.
"""

# %%
alpha = np.array([0.2, -0.5, 0.8])

y = ccf.fisher(alpha)
alpha_back = ccf.inverse_fisher(y)

print("alpha:", alpha)
print("y:    ", y)

np.testing.assert_allclose(alpha_back, alpha)

# %% [markdown]
"""
A convenient unconstrained construction workflow is therefore

    y in R^N
      -> inverse_fisher
    alpha in (-1, 1)^N
      -> from_pacf
    admissible r.

For finite `y`, this constructs a strict interior sequence automatically.
"""

# %%
y = np.array([0.5, -1.0, 0.2, 1.5])

alpha = ccf.inverse_fisher(y)
r = ccf.from_pacf(alpha)

print("y:    ", y)
print("alpha:", alpha)
print("r:    ", r)
print("status:", ccf.pacf_status(r))

assert ccf.pacf_status(r).interior

# %% [markdown]
"""
## 12. Innovation variances

Use

    ccf.innovation_variances(alpha)

to compute

    sigma_0^2, sigma_1^2, ..., sigma_N^2,

where

    sigma_0^2 = 1,
    sigma_n^2 = sigma_(n-1)^2 * (1 - alpha_n^2).

The result has length `N+1`, or shape `(M, N+1)` for a batch.
"""

# %%
alpha = np.array([0.2, -0.3, 0.4])
sigma2 = ccf.innovation_variances(alpha)

print("alpha: ", alpha)
print("sigma2:", sigma2)

assert sigma2.shape == (len(alpha) + 1,)
assert sigma2[0] == 1.0

# %%
alpha_batch = np.array(
    [
        [0.2, -0.3, 0.4],
        [0.5, 0.1, -0.2],
    ]
)

sigma2_batch = ccf.innovation_variances(alpha_batch)

print("batch sigma2:\n", sigma2_batch)
print("shape:", sigma2_batch.shape)

# %% [markdown]
"""
## 13. Jacobian of `alpha -> r`

The change of variables from PACFs to correlations has a triangular
Jacobian. The package exposes

    ccf.jacobian(alpha)
    ccf.log_jacobian(alpha)

The log form is preferable at high order because the determinant can become
very small.
"""

# %%
alpha = np.array([0.2, -0.3, 0.4, 0.1])

J = ccf.jacobian(alpha)
logJ = ccf.log_jacobian(alpha)

print("Jacobian:    ", J)
print("log Jacobian:", logJ)

np.testing.assert_allclose(np.log(J), logJ, rtol=1e-12, atol=1e-12)

# %% [markdown]
"""
## 14. Volume of the admissible region

The package provides the Lebesgue volume of the admissible region in
correlation-coefficient space:

    ccf.admissible_volume(N)
    ccf.log_admissible_volume(N)

Again, prefer the logarithmic form at high order because the volume shrinks
rapidly and can underflow in float64.
"""

# %%
for N in [1, 2, 5, 10, 50]:
    volume = ccf.admissible_volume(N)
    log_volume = ccf.log_admissible_volume(N)
    print(f"N={N:2d}: V_N={volume:.6e}, log(V_N)={log_volume:.6f}")

# %% [markdown]
"""
## 15. Arbitrary precision

High-order or strongly ill-conditioned roundtrips can exceed float64's
useful precision. If `mpmath` is installed, use

    ccf.recommended_dps(N)
    ccf.from_pacf_mp(alpha, dps=...)
    ccf.pacf_mp(r, dps=...)

The arbitrary-precision functions are 1-D and return lists of `mpmath.mpf`
objects, intentionally preserving high precision rather than converting back
to float64.

`recommended_dps` is an empirical recommendation for roundtrip calculations,
not a mathematical guarantee.
"""

# %%
if hasattr(ccf, "recommended_dps"):
    N = 32
    dps = ccf.recommended_dps(N)
    print(f"recommended dps for N={N}: {dps}")

    rng = np.random.default_rng(3)
    alpha = rng.uniform(-0.8, 0.8, size=N)

    r_mp = ccf.from_pacf_mp(alpha, dps=dps)
    alpha_mp = ccf.pacf_mp(r_mp, dps=dps)

    # Compare in ordinary float only for this small tutorial demonstration.
    alpha_back = np.array([float(a) for a in alpha_mp])

    print("max float-converted roundtrip error:",
          np.max(np.abs(alpha_back - alpha)))
else:
    print("mpmath support is not installed.")
    print('Install with: pip install -e ".[precision]"')

# %% [markdown]
"""
### Boundary behavior in `pacf_mp`

The arbitrary-precision API differs slightly from the float64 API:
`pacf_mp` retains an `at_boundary` mode.

    at_boundary="raise"  -> raise on a degenerate boundary
    at_boundary="warn"   -> warn and return the non-degenerate part

There is no arbitrary-precision `extend` mode.
"""

# %% [markdown]
"""
## 16. Didactic reference implementation

For readers who want to compare the code directly with the equations in the
paper, `ccf.reference` provides deliberately simple, single-sequence
implementations:

    pacf_reference
    from_pacf_reference

They are not used internally by the robust implementation. Their purpose is
to make the Levinson--Durbin equations easy to inspect.
"""

# %%
from ccf.reference import pacf_reference, from_pacf_reference

alpha = np.array([0.3, -0.2, 0.4])
r_ref = from_pacf_reference(alpha)
alpha_ref = pacf_reference(r_ref)

print("reference r:    ", r_ref)
print("reference alpha:", alpha_ref)

np.testing.assert_allclose(r_ref, ccf.from_pacf(alpha), rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(alpha_ref, alpha, rtol=1e-12, atol=1e-12)

# %% [markdown]
"""
## 17. Optional symbolic helpers

With the `symbolic` extra installed, `ccf.symbolic` can generate and
verify low-order formulas. This is useful for checking the paper's algebra,
not for large numerical calculations.

Representative helpers include:

    pacf_symbolic(order)
    admissible_bounds_symbolic(order)
    predictor_symbolic(order)
    sh_coordinate_symbolic(order)
    verify_x_equals_alpha(order)

For example, `verify_x_equals_alpha(order)` simplifies the symbolic
difference between the Schneider--Hartlap coordinate and the PACF. A
successful verification returns exactly zero.
"""

# %%
try:
    from ccf import symbolic

    print("alpha_3 =", symbolic.pacf_symbolic(3))

    lo3, hi3 = symbolic.admissible_bounds_symbolic(3)
    print("r_3 lower bound =", lo3)
    print("r_3 upper bound =", hi3)

    print("x_3 - alpha_3 =", symbolic.verify_x_equals_alpha(3))
except ImportError:
    print("SymPy support is not installed.")
    print('Install with: pip install -e ".[symbolic]"')

# %% [markdown]
"""
## 18. Error handling: what each function is for

A useful summary:

### `pacf(r)`
Use when you want the actual `r -> alpha` transformation.

- interior: returns all PACFs;
- boundary: raises `SingularToeplitzError`;
- invalid: raises `ValueError`.

### `pacf_status(r)`
Use when the mathematical state itself is the question.

- does not raise merely because a sequence is boundary or invalid;
- returns `interior`, `boundary`, `invalid`, and `order`;
- supports batches.

### `pacf_prefix(r)`
Use for detailed analysis of a degenerate boundary.

- returns the maximal boundary-inclusive PACF prefix;
- exposes the innovation-variance trace and terminal predictor;
- invalid input still raises.

### `extend_at_boundary(r, n_extra)`
Use only after an admissible boundary has been reached.

- appends the uniquely forced correlation continuation;
- it is not a general repair function.

### `admissible_bounds(r)`
Use to inspect the Schneider--Hartlap intervals implied by a prefix.

### `check_admissibility(r)`
Use for a simple boolean admissibility check.
"""

# %% [markdown]
"""
## 19. End-to-end examples

### A. Constructing a valid sequence from unconstrained parameters

A common generative workflow is

    unconstrained y
        -> inverse_fisher
    alpha in (-1, 1)^N
        -> from_pacf
    admissible interior r.
"""

# %%
y = np.array([0.2, -0.8, 1.1, 0.0, 0.5])

alpha = ccf.inverse_fisher(y)
r = ccf.from_pacf(alpha)

print("y:", y)
print("alpha:", alpha)
print("r:", r)
print("admissible:", ccf.check_admissibility(r))

# %% [markdown]
"""
### B. Inspecting an unknown correlation sequence

Use status first when the input may be problematic.
"""

# %%
r_unknown = np.array([0.2, 0.1, 0.95])

status = ccf.pacf_status(r_unknown)
print("status:", status)

if status.interior:
    alpha = ccf.pacf(r_unknown)
    print("PACFs:", alpha)

elif status.boundary:
    prefix = ccf.pacf_prefix(r_unknown)
    print("boundary PACF prefix:", prefix.alpha)
    print("forced future:",
          ccf.extend_at_boundary(r_unknown, n_extra=3))

else:
    valid_prefix = r_unknown[: status.order - 1]
    lo, hi = next_admissible_interval(valid_prefix)

    print("first invalid order:", status.order)
    print("valid prefix:", valid_prefix)
    print(
        "the first invalid coefficient would need to lie in:",
        (lo, hi),
    )

# %% [markdown]
"""
### C. Batch classification

For simulation work, classify many sequences at once. This is especially
useful when some rows may reach the boundary or become invalid, because
`pacf_status` returns a status for every row instead of stopping at the first
problematic sample.
"""

# %%
alpha_good = rng.uniform(-0.5, 0.5, size=(4, 5))
r_good = ccf.from_pacf(alpha_good)

r_test = np.vstack(
    [
        r_good,
        np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]),
        np.array([[0.9, 0.95, 0.99, 0.99, 0.99]]),
    ]
)

status = ccf.pacf_status(r_test)

print("interior:", status.interior)
print("boundary:", status.boundary)
print("invalid: ", status.invalid)
print("order:   ", status.order)

# %% [markdown]
"""
## 20. Public API cheat sheet

### Core transformations
- `ccf.pacf(r)`
- `ccf.from_pacf(alpha)`

### Diagnostics and boundary handling
- `ccf.pacf_status(r)` -> `PACFStatus`
- `ccf.pacf_prefix(r)` -> `PrefixResult`
- `ccf.check_admissibility(r)`
- `ccf.admissible_bounds(r)`
- `ccf.extend_at_boundary(r, n_extra)`
- `ccf.SingularToeplitzError`

### Alternative coordinates and derived quantities
- `ccf.sh_coordinates(r)`
- `ccf.fisher(alpha)`
- `ccf.inverse_fisher(y)`
- `ccf.innovation_variances(alpha)`
- `ccf.jacobian(alpha)`
- `ccf.log_jacobian(alpha)`

### Admissible-region geometry
- `ccf.admissible_volume(N)`
- `ccf.log_admissible_volume(N)`

### Arbitrary precision (optional `mpmath`)
- `ccf.recommended_dps(N)`
- `ccf.from_pacf_mp(alpha, dps=...)`
- `ccf.pacf_mp(r, dps=...)`

### Paper-facing auxiliary modules
- `ccf.reference`: simple line-by-line numerical recursion
- `ccf.symbolic`: low-order symbolic verification

For mathematical derivations, notation, and the exact boundary semantics,
read the paper together with `docs/notation.md` and
`docs/boundary_semantics.md`.
"""

# %% [markdown]
