# Development notes

Investigation results that shaped the figure scripts but do not belong in
their module docstrings or in code comments.

## `figure_roundtrip.py`: the dropped O(N^2) timing panel

An earlier version of this figure included a third panel timing `pacf`
over a log-log grid up to `N=8192`, intended to show the theoretical
`O(N^2)` complexity of the Levinson--Durbin recursion. It was dropped:
the measured local slope climbed only from about 1.0 to 1.4, nowhere
near the theoretical reference slope of 2, even at `N=8192`.

Root cause: the recursion is a Python-level loop calling NumPy once per
step. At these `N`, the per-call NumPy/Python dispatch overhead
(microseconds) exceeds the actual `O(n)` floating-point work per step
(nanoseconds), so an `O(N)` overhead term dominates the `O(N^2)`
arithmetic term across the entire practically testable range. This is a
genuine property of this implementation, not an artifact of the `alpha`
range used elsewhere in the figure. Showing it would need a lengthy
methodological caveat to avoid being misread as evidence against
`O(N^2)` complexity, so the panel was removed; the complexity itself is
argued analytically in the paper text, not demonstrated by this figure.

## `figure_roundtrip.py`: Panel A is an adversarial stress test

Panel A draws `alpha` independently at *every* lag from a wide, fixed
interval -- exactly the regime in which the Jacobian
`det(dr/dalpha) = prod(sigma_n^2)` vanishes fastest for a generic draw.
Realistic correlation functions have partial autocorrelations that decay
with lag, not i.i.d. draws across all lags at once, and are expected to
stay far more benign in `float64` at the same `N`. The float64 failure
rates shown are a worst-case characterization of the recursion, not a
claim about typical astrophysical use.

## `levinson.py`: one kernel per direction, 1-D as a batch of one

`_levinson_correlations_batch` (`r -> alpha`) and `_run_levinson_from_pacf`
(`alpha -> r`) are the sole NumPy implementations of each recursion
direction; a 1-D sequence is handled by reshaping to a batch of one row
rather than by a separate scalar code path. Because a single row's
`sum(phi * r, axis=1)` term does not depend on how many other rows are
present in the batch, `pacf`/`from_pacf` on a batch are bit-identical to
calling them row by row -- `tests/test_schurcorr.py`'s batch-vs-loop tests
still compare with a tolerance mainly as a guard against future changes to
the summation strategy, not because a difference is currently expected.

`pacf_prefix` and `schurcorr.bounds` reach the same `r -> alpha` kernel
through `_run_levinson_from_correlations`, a single-row adapter that raises
immediately for an inadmissible sequence; `pacf` calls the batch kernel
directly so it can report the smallest offending row across the whole
batch (row order takes priority over error type: a row that is merely at
the boundary is reported before a later row that is outright inadmissible,
matching the row-by-row loop this replaced).

## `figure_roundtrip.py`: Panel A is batched

Panel A calls `sc.from_pacf` and `sc.pacf_status` on chunks of
`TRIAL_BATCH_SIZE` trials at once, rather than once per trial. The
recursion is a Python loop over lags calling NumPy once per step;
batching amortizes that per-call dispatch overhead across many trials
instead of paying it per trial, the same reasoning as "one kernel per
direction, 1-D as a batch of one" above.

`pacf_status` exposes the per-row interior/boundary/invalid
classification and terminal order that `_levinson_correlations_batch`
already computes internally, without raising the way `pacf`'s batch mode
does (which intentionally reports only the smallest offending row --
the right contract for library users, but the wrong shape of answer for
a full pass/fail count over thousands of trials). Panel A's failure
count is `status.boundary | status.invalid`, matching `pacf`'s own
`SingularToeplitzError`-or-`ValueError` failure criterion.

`numpy.random.Generator.uniform(..., size=(m, N))` was confirmed to draw
the exact same underlying sequence as `m` calls of `size=(N,)`,
regardless of how the `m` trials are grouped into chunks, so batching
does not change the Monte Carlo sample.

## `figure_roundtrip.py`: Panel B trials run in parallel processes

Panel B's arbitrary-precision roundtrips (`sc.from_pacf_mp` /
`sc.pacf_mp` via `mp_roundtrip_error`) are independent per trial, unlike
Panel A they are not batchable through a vectorized NumPy kernel --
`precision.py` works one `mpmath` value at a time -- so the available
parallelism is across trials, not within one. `precision_scan` draws all
`alpha` samples for a given `(bound, N)` in the parent process first
(`rng.uniform(..., size=(trials, N))`), then hands fixed-size chunks of
already-drawn samples to a reused `ProcessPoolExecutor`. Because the
random draws never depend on the job count, `--jobs 1` (serial) and
`--jobs N` (parallel) consume the exact same underlying RNG sequence and
produce numerically identical median/worst statistics; this is checked
directly in `tests/test_figure_roundtrip_panel_b.py`. `--quick` defaults
to `--jobs 1` to avoid paying process-startup cost on a workload that is
already fast; `--paper` defaults to `min(8, os.cpu_count())` workers.

## `precision.py`: `recommended_dps` calibration

Empirically, the recursion loses about 0.45 decimal digits of precision
per lag for `alpha` drawn from `(-0.95, 0.95)` (the wider of the two
bounds used in the paper's roundtrip figure), independent of the working
precision itself -- this is conditioning loss, not a property of
`float64` specifically. `recommended_dps`'s formula is a conservative,
deliberately simple affine fit to that empirical rate; it is calibrated
for `b=0.95` and therefore also safe (with some margin to spare) for the
narrower `b=0.9` case used alongside it in the same figure.
Reference: Schneider, P. and Hartlap, J. 2009, A&A 504, 705.

## `precision.py`: why `pacf_mp` keeps an `at_boundary` mode parameter

The `float64` path splits boundary handling into separate functions
(`pacf` vs. `pacf_prefix`) because the boundary there is usually a
genuine, admissible degenerate sequence. At arbitrary precision, reaching
the boundary almost always means `dps` was insufficient rather than a
genuine boundary sequence, so the two cases this module actually needs to
distinguish are "raise" and "return the reliable prefix" -- a third,
boundary-aware entry point (`pacf_prefix_mp`) would add API surface
without a proportionate clarity gain for this comparatively rare path.
This is a deliberate, provisional choice, not an oversight.

Complexity is `O(N^2)` recursion steps, as in the `float64` version, but
each arithmetic operation costs more than a `float64` operation, and that
cost grows with `dps` (which in turn grows with `N` via
`recommended_dps`), so wall-clock time scales worse than `O(N^2)` in
practice.
