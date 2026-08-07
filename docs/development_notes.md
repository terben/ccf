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

## `levinson.py`: batched vs. scalar path is not bit-identical

`_pacf_2d_fast` and `_from_pacf_2d` compute the same recursion as the
per-row scalar path (`_run_levinson_from_correlations` /
`_run_levinson_from_pacf`), vectorized across the batch dimension instead
of looped in Python. The prediction step there, `sum(phi * r, axis=1)`,
uses a different floating-point summation order than the scalar path's
`np.dot`, which can differ by up to a few ULP. This is a summation-order
effect, not a different algorithm, and is why batch-vs-loop tests in
`tests/test_schurcorr.py` compare with a tolerance rather than exact
equality.

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
