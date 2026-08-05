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
