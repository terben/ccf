# Degenerate boundary semantics — design note

Status: **implemented.** This note was written as Phase 1b of the migration
described in `CLAUDE.md`, proposing the public API that Phases 2–6 then
built: `pacf`/`from_pacf`/`pacf_prefix` in `schurcorr/levinson.py` as
described below, `extend_at_boundary`/`admissible_bounds` in
`schurcorr/bounds.py`, and `pacf_mp`/`from_pacf_mp` in
`schurcorr/precision.py`. It remains the reference for *why* the boundary
handling is split the way it is.

## Why this note exists

`CLAUDE.md` requires four cases to be told apart explicitly before the
numerical core is rewritten:

1. admissible interior points,
2. mathematically degenerate boundary points,
3. inadmissible correlation sequences,
4. numerical breakdown caused by finite precision.

The current implementation (`schurcorr/levinson.py`, `schurcorr/sh_bounds.py`,
`schurcorr/_levinson_mp.py`) already distinguishes these correctly at the
*mathematical* level, but exposes the distinction through a single function,
`pacf(r, at_boundary=...)`, using mode strings (`'raise'`, `'warn'`,
`'extend'`) plus a separate `backend=` dispatch parameter. `CLAUDE.md` names
both of these patterns as things to avoid. This note works out what the
resulting three-function split (coordinate transformation / boundary analysis
/ deterministic continuation) should look like.

## The recursion state, once per category

All four categories are read off the same quantity: the innovation-variance
sequence `sigma_0^2 = 1, sigma_1^2, ..., sigma_N^2` produced by the Levinson
recursion (`_LevinsonState` in `levinson.py`), together with `alpha_n` at
each order. There is and should remain exactly one recursion that computes
this trace; the categories below are different *readings* of its output, not
different algorithms.

## 1. Admissible interior point

**Mathematical meaning.** `sigma_n^2 > 0` for every `n = 1, ..., N`
(equivalently `abs(alpha_n) < 1` throughout). The map `r <-> alpha` is a
genuine bijection at this order.

**Desired public behavior.** `pacf(r)` and `from_pacf(alpha)` return the full
vector, unconditionally — no exception, no warning. This is already exactly
today's behavior at interior points and does not change.

**Required internal state.** The complete recursion trace
(`alpha[1..N]`, `sigma2[0..N]`, predictor `phi` at each order).

**Intended public API.** `pacf(r) -> alpha`, `from_pacf(alpha) -> r`.

## 2. Mathematically degenerate boundary point

**Mathematical meaning.** `sigma_m^2 = 0` at some order `m <= N`
(equivalently `abs(alpha_m) = 1`), reached from a strictly positive-definite
prefix (`sigma_n^2 > 0` for `n < m`). The Toeplitz matrix `A_{m+1}` built
from `r_1, ..., r_m` is positive *semi*definite but singular. This is an
**admissible, not an error**, state — a legitimate correlation sequence that
has simply run out of independent degrees of freedom. If further coefficients
`r_{m+1}, ..., r_N` are supplied, they are not free: they are uniquely forced
by the null vector of `A_{m+1}` (see `SH_research_note.pdf`, `extend_at_boundary`
docstring), and a valid admissible sequence must equal that forced
continuation.

The key consequence for API design: `alpha_1, ..., alpha_{m-1}` (together
with the forced `alpha_m = ±1`) remain well defined, but there is no
`r <-> alpha` bijection beyond order `m` — `alpha_{m+1}, ..., alpha_N` simply
do not exist as independent coordinates. `CLAUDE.md` names three distinct
mathematical operations here; the current code already performs all three
internally, just not as three separate entry points:

| CLAUDE.md operation | Current implementation | Proposed public API |
|---|---|---|
| Coordinate transformation | `pacf(r)` default (`at_boundary='raise'`) | `pacf(r)` — unconditional, no `at_boundary` parameter |
| Boundary analysis | `pacf(r, at_boundary='warn')` | new: `pacf_prefix(r)` |
| Deterministic continuation | `extend_at_boundary(r, n_extra)` (`schurcorr/sh_bounds.py`) | unchanged: `extend_at_boundary(r, n_extra)` |

**Coordinate transformation — `pacf(r)`.** Should terminate with a clear
exception when it hits this case, full stop. No mode parameter: the default
*is* the only behavior. Raises `SingularToeplitzError`, as today's default
already does.

**Boundary analysis — `pacf_prefix(r)` (new).** Never raises on a
degenerate-but-admissible input. Returns the maximal independent PACF prefix
together with the recursion state needed by a caller that wants to continue
the sequence: the order `m` at which the boundary was reached, `alpha[1..m]`
(the last entry is exactly `+1` or `-1`), the boundary innovation variance
trace, and the terminal predictor `phi^{(m-1)}` (the quantity
`extend_at_boundary` needs). Concretely, a small immutable result — a
`@dataclass` or `NamedTuple`, matching the "small trace/result dataclass"
`CLAUDE.md` explicitly permits in `levinson.py` — replacing today's
`at_boundary='warn'` return value (which silently truncates the array and
buries the boundary order in `len(alpha)`, or NaN-pads it for a 2-D batch).
For 2-D batched input, this should report the per-row boundary order
explicitly (e.g. as a field on the result) rather than encoding it through
truncation length or NaN-padding.

*Naming.* `CLAUDE.md` offers `pacf_prefix(r)` and `analyze_correlations(r)`
as examples. This note recommends **`pacf_prefix`**: it names the returned
mathematical object directly (the maximal independent PACF prefix) and
follows the existing `pacf`/`from_pacf` naming convention, whereas
`analyze_correlations` reads more like a generic diagnostic entry point. This
is the one open naming choice in this design and should be confirmed before
Phase 2 implements it.

**Deterministic continuation — `extend_at_boundary(r, n_extra)`.** Already
matches the target shape described in `CLAUDE.md`: a separate, explicitly
named function, not a mode of `pacf`. Its signature does not need to change.
What should change internally (Phase 2, not this note) is that it currently
constructs its own boundary state directly
(`_LevinsonState.from_correlations(r, at_boundary="warn")`, in
`sh_bounds.py:241`) instead of calling the new `pacf_prefix`; once
`pacf_prefix` exists, `extend_at_boundary` and `admissible_bounds` (which
does the same thing at `sh_bounds.py:115`) should both be rewritten to call
it, so there is exactly one code path that reads "boundary reached" out of
the recursion trace instead of three (`levinson.py:609`, `sh_bounds.py:115`,
`sh_bounds.py:241`, all currently hard-coding `at_boundary="warn"`
independently).

## 3. Inadmissible correlation sequence

**Mathematical meaning.** `abs(alpha_n) > 1` for some `n` — equivalently, a
leading principal submatrix of the Toeplitz matrix has a negative eigenvalue.
This is a plain input error: `r` does not correspond to any nonnegative power
spectrum. It is unrelated to the boundary case above (category 2) even
though both are detected during the same recursion pass — category 2 is
`sigma_n^2 = 0` exactly; this category is the recursion becoming
mathematically impossible to continue at all.

**Desired public behavior.** Unchanged from today: every entry point
(`pacf`, `pacf_prefix`, `from_pacf`, `admissible_bounds`) raises `ValueError`
distinct from `SingularToeplitzError`. `check_admissibility(r) -> bool`
remains the boolean query that returns `False` instead of raising (with
`raise_error=True` available for callers that want the exception).

**Required internal state.** None beyond the point of failure — the
recursion aborts at the first `n` with `abs(alpha_n) > 1 + tol`.

**Intended public API.** No change.

## 4. Numerical breakdown from finite precision

**Mathematical meaning.** Not a distinct mathematical category — a *finite-precision
artifact* of the other three. Two sub-cases occur in the current code:

- `sigma_n^2` computed as a small negative number (e.g. `-1e-15`) at a point
  that is mathematically strictly interior (`sigma_n^2 > 0` exactly), purely
  from float64 roundoff accumulated over the recursion.
- At high order or with correlations close to the boundary, float64 may be
  unable to resolve whether a given `r` is exactly at the boundary, strictly
  interior but extremely close to it, or (rarely) very slightly inadmissible
  — the three categories above become numerically indistinguishable at
  working precision.

**Current handling.** The float64 path (`levinson.py`) uses a fixed
tolerance `_TOL = 1e-12`: a computed `sigma_n^2 \in (-\_TOL, 0)` is clamped to
exactly `0.0` with a `RuntimeWarning` (`levinson.py:291-298`, `:370-377`,
`:861-868`), i.e. treated as category 2, not category 3. This is a
deliberate, existing numerical-conditioning choice being *documented* here,
not changed.

**Gap identified, not fixed in this phase.** The `mpmath` arbitrary-precision
path (`_levinson_mp.py`) exists precisely so a caller can resolve this
ambiguity by increasing working precision (`recommended_dps`), but it
currently detects the boundary purely via `sigma_sq[-1] <= 0` at whatever
`dps` was requested, without distinguishing "boundary genuinely reached,"
"boundary artifact of insufficient `dps`," and "input is actually
inadmissible" — all three currently produce the same generic message
referencing `dps` (see `_pacf_mp`, `_levinson_mp.py:162-181`). Recommendation:
when `_levinson_mp.py` is promoted to `precision.py` (`CLAUDE.md` Phase 5),
its error/warning messages should be tightened to name which of the three
this is, since the whole point of the arbitrary-precision path is to let a
caller tell them apart. Not required before Phase 2.

**Required internal state.** The same recursion trace plus the active
tolerance (`_TOL` for float64, working `dps` for mpmath).

**Intended public API.** No new function. `pacf`/`pacf_prefix`/`from_pacf`
keep clamping silently within `_TOL` for float64; `recommended_dps` remains
the documented way to resolve ambiguity via the arbitrary-precision path.

## Resulting signatures (as implemented)

```python
pacf(r) -> alpha                      # raises SingularToeplitzError at a degenerate boundary
from_pacf(alpha) -> r
pacf_prefix(r) -> PrefixResult        # boundary analysis
extend_at_boundary(r, n_extra) -> r   # deterministic continuation, in schurcorr/bounds.py
```

`backend='mpmath'` / `dps=` were dropped from `pacf`/`from_pacf` entirely;
the arbitrary-precision path is the standalone `pacf_mp` / `from_pacf_mp` in
`schurcorr/precision.py`. The old `backend=`/`at_boundary=` implementation
was kept alongside the new one during the migration for comparison and was
removed once the comparison confirmed no meaningful numerical difference
(CLAUDE.md Phase 6).
