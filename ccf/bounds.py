"""Admissible correlation bounds and boundary continuation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .levinson import (
    FloatArray,
    _asarray1d,
    _run_levinson_from_correlations,
    pacf,
)

# Looser than ccf.levinson._ROUNDING_TOL by design (see
# DOCUMENTATION.md, "Tolerances"): this compares a *supplied* coefficient
# against the Toeplitz-forced continuation computed from it, an
# order-dependent chain of products that amplifies roundoff faster than
# the single-step comparisons _ROUNDING_TOL is used for.
_BOUNDARY_CONTINUATION_TOL = 1.0e-8


def _forced_continuation(phi: FloatArray, window: list[float]) -> float:
    """
    Evaluate the Toeplitz-forced linear recurrence one step ahead.

    Parameters
    ----------
    phi
        Terminal Levinson--Durbin predictor coefficients at the
        singular boundary (see :func:`extend_at_boundary`).
    window
        The ``len(phi)`` most recent correlation coefficients, oldest
        first.

    Returns
    -------
    float
        The uniquely forced next coefficient.
    """
    reversed_window = np.asarray(window[::-1], dtype=np.float64)
    return float(np.dot(phi, reversed_window))


def admissible_bounds(
    r: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """
    Compute admissible bounds for a correlation sequence and its successor.

    For a supplied prefix ``r_1, ..., r_N``, returns the admissible
    interval for each of ``r_1, ..., r_N`` and, appended as one extra
    entry, the admissible interval for the next coefficient ``r_(N+1)``.
    For every supplied coefficient ``r_n``, its interval is implied by the
    preceding coefficients ``r_1, ..., r_(n-1)`` alone.

    Parameters
    ----------
    r
        One-dimensional correlation sequence.

    Returns
    -------
    r_lower, r_upper
        Arrays of length ``N + 1``. Entry ``n - 1`` gives the admissible
        interval for ``r_n``; the final entry gives the interval for the
        next coefficient ``r_(N+1)``.

    Raises
    ------
    ValueError
        If ``r`` is not admissible.

    Notes
    -----
    At order ``n``, the interval is ``p_n -+ sigma_(n-1)^2``, centred on
    the linear prediction ``p_n``. At a degenerate boundary, subsequent
    bounds -- including the appended next-coefficient interval -- collapse
    to the uniquely determined continuation (see :func:`extend_at_boundary`);
    coefficients supplied past that point are validated against it, not
    merely checked for presence.
    """
    r_array = _asarray1d(r, name="r")
    result = _run_levinson_from_correlations(r_array)

    number_computed = result.r.size

    # r_(n, lower/upper) = p_n -+ sigma_(n-1)^2; result.sigma2[:number_computed]
    # holds exactly sigma_0^2, ..., sigma_(m-1)^2, the half-width at each step.
    half_width = result.sigma2[:number_computed]
    r_lower = result.prediction - half_width
    r_upper = result.prediction + half_width

    phi = result.terminal_phi
    window = list(result.r)

    if result.reached_boundary:
        excess = r_array[number_computed:]
        extra_bounds = np.empty(excess.size, dtype=np.float64)

        for i, supplied_raw in enumerate(excess):
            forced = _forced_continuation(phi, window)
            supplied = float(supplied_raw)

            if not np.isclose(
                forced,
                supplied,
                rtol=_BOUNDARY_CONTINUATION_TOL,
                atol=_BOUNDARY_CONTINUATION_TOL,
            ):
                raise ValueError(
                    f"r_{number_computed + i + 1} = {supplied!r} is "
                    "inconsistent with the Toeplitz-forced continuation "
                    f"r_{number_computed + i + 1} = {forced!r} past the "
                    f"singular boundary (order {number_computed}); r "
                    "does not define an admissible correlation sequence."
                )

            extra_bounds[i] = supplied
            window = window[1:] + [supplied]

        r_lower = np.concatenate([r_lower, extra_bounds])
        r_upper = np.concatenate([r_upper, extra_bounds])

    # Next-coefficient interval, from the same terminal predictor/window
    # used above for the boundary continuation (or, in the interior case,
    # the linear prediction p_(N+1) -+ sigma_N^2).
    next_center = _forced_continuation(phi, window)
    next_half_width = float(result.sigma2[number_computed])

    return (
        np.concatenate([r_lower, [next_center - next_half_width]]),
        np.concatenate([r_upper, [next_center + next_half_width]]),
    )


def extend_at_boundary(r: ArrayLike, n_extra: int) -> FloatArray:
    """
    Extend a sequence from a degenerate boundary.

    Parameters
    ----------
    r
        Admissible sequence ending at a degenerate boundary
        (``sigma_m^2 = 0`` for some order ``m``). Entries already
        supplied past the boundary are validated against the forced
        continuation rather than overwritten.
    n_extra
        Total number of coefficients after the boundary to include in the
        returned sequence. Coefficients already supplied beyond the
        boundary count toward this number and are validated against the
        forced continuation; must be at least the number of such entries
        ``r`` already supplies.

    Returns
    -------
    numpy.ndarray
        The extended correlation sequence.

    Raises
    ------
    ValueError
        If ``r`` does not end at a valid degenerate boundary, if
        ``n_extra`` is negative or smaller than the number of entries
        ``r`` already supplies past the boundary, or if those entries
        are inconsistent with the forced continuation.

    Notes
    -----
    The continuation is the unique linear recurrence forced by the null
    vector of the singular Toeplitz matrix at the boundary, given by the
    terminal Levinson--Durbin predictor; see Sect. 5 of the paper and
    ``SH_research_note.pdf`` for the derivation.
    """
    if n_extra < 0:
        raise ValueError("n_extra must be non-negative.")

    r_array = _asarray1d(r, name="r")
    result = _run_levinson_from_correlations(r_array)

    if not result.reached_boundary:
        raise ValueError(
            "r does not reach the singular boundary of the admissible "
            "region within its given length; extend_at_boundary is "
            "only meaningful once sigma_n^2 = 0 has actually been "
            "reached. Check with pacf_prefix(r) first."
        )

    prefix_len = result.r.size
    phi = result.terminal_phi

    existing_excess = r_array[prefix_len:]

    if existing_excess.size > n_extra:
        raise ValueError(
            f"r already contains {existing_excess.size} entries past "
            f"the boundary (order {prefix_len}), but "
            f"n_extra={n_extra} was requested; pass n_extra >= "
            f"{existing_excess.size}, or pass only the admissible "
            "prefix of r."
        )

    window = list(result.r)
    generated: list[float] = []

    for s in range(n_extra):
        r_next = _forced_continuation(phi, window)

        if s < existing_excess.size:
            supplied = float(existing_excess[s])

            if not np.isclose(
                r_next,
                supplied,
                rtol=_BOUNDARY_CONTINUATION_TOL,
                atol=_BOUNDARY_CONTINUATION_TOL,
            ):
                raise ValueError(
                    f"Supplied r_{prefix_len + s + 1} = {supplied!r} is "
                    "inconsistent with the Toeplitz-forced continuation "
                    f"r_{prefix_len + s + 1} = {r_next!r}; r is not "
                    "realizable by any nonnegative power spectrum past "
                    "the boundary."
                )

            r_next = supplied

        generated.append(r_next)
        window = window[1:] + [r_next]

    return np.concatenate(
        [result.r, np.asarray(generated, dtype=np.float64)]
    )


def sh_coordinates(r: ArrayLike) -> FloatArray:
    """Alias of :func:`pacf`.

    The Schneider--Hartlap coordinate ``x_n`` and the partial
    autocorrelation ``alpha_n`` are the same quantity (``x_n = alpha_n``);
    kept for readers coming from the Schneider--Hartlap notation.
    """
    return pacf(r)


def check_admissibility(
    r: ArrayLike,
    *,
    atol: float = 1.0e-12,
    raise_error: bool = False,
) -> bool:
    """
    Check whether correlations define an admissible Toeplitz sequence.

    Parameters
    ----------
    r
        Normalized correlation coefficients.
    atol
        Absolute tolerance used when comparing coefficients with their
        admissible interval boundaries.
    raise_error
        If ``True``, raise ``ValueError`` instead of returning ``False``
        for an inadmissible input.

    Returns
    -------
    bool
        ``True`` if the complete sequence is admissible, otherwise
        ``False``.

    Raises
    ------
    ValueError
        If ``raise_error=True`` and the supplied sequence is not
        admissible.

    Notes
    -----
    A sequence that reaches the singular boundary of the admissible
    region (see :func:`extend_at_boundary`) is admissible -- just
    degenerate -- provided every coefficient past the boundary equals
    the uniquely forced continuation; ``admissible_bounds`` validates
    this rather than treating a boundary sequence as automatically
    inadmissible.
    """
    if atol < 0.0:
        raise ValueError("atol must be non-negative.")

    try:
        r_array = _asarray1d(r, name="r")
        r_lower, r_upper = admissible_bounds(r_array)
        # admissible_bounds appends one extra entry for the next, unsupplied
        # coefficient; only the first len(r_array) entries apply to r itself.
        r_lower, r_upper = r_lower[: r_array.size], r_upper[: r_array.size]

        admissible = bool(
            np.all(r_array >= r_lower - atol)
            and np.all(r_array <= r_upper + atol)
        )

    except (TypeError, ValueError):
        admissible = False

    if raise_error and not admissible:
        raise ValueError(
            "The supplied coefficients do not define an admissible "
            "positive semidefinite Toeplitz correlation matrix."
        )

    return admissible


def log_admissible_volume(N: int) -> float:
    """
    Natural logarithm of the volume of the admissible region.

    Parameters
    ----------
    N
        Order of the Toeplitz correlation sequence (number of lags).
        Must be a positive integer.

    Returns
    -------
    float
        ``log(V_N)``, the natural logarithm of the Lebesgue volume of
        the admissible region in ``r``-space (Eq. (volume), Sect. 6.1
        of the paper).

    Raises
    ------
    ValueError
        If ``N`` is not a positive integer.

    Notes
    -----
    Computed in log-space via ``math.lgamma``, since the underlying
    product

    ``V_N = 2 * prod_(j=1)^(N-1) sqrt(pi) * j! / Gamma(j + 3/2)``

    underflows ``float64`` for large ``N`` (see :func:`admissible_volume`).

    References
    ----------
    Erben (in preparation), Eq. (volume), Sect. 6.1; the SH-normalized
    volume ``V_N^SH = V_N / 2^N`` (Eq. (volume_SH)) is cross-checked
    in the test suite.
    """
    if N < 1 or N != int(N):
        raise ValueError(f"N must be a positive integer; got {N!r}.")

    if N == 1:
        return math.log(2.0)

    half_log_pi = 0.5 * math.log(math.pi)
    log_terms = (
        half_log_pi + math.lgamma(j + 1.0) - math.lgamma(j + 1.5)
        for j in range(1, N)
    )

    return math.log(2.0) + math.fsum(log_terms)


def admissible_volume(N: int) -> float:
    """
    Lebesgue volume of the admissible region in ``r``-space.

    Parameters
    ----------
    N
        Order of the Toeplitz correlation sequence (number of lags).
        Must be a positive integer.

    Returns
    -------
    float
        ``V_N`` (Eq. (volume), Sect. 6.1 of the paper),

        ``V_N = 2 * prod_(j=1)^(N-1) sqrt(pi) * j! / Gamma(j + 3/2)``.

    Raises
    ------
    ValueError
        If ``N`` is not a positive integer.

    Notes
    -----
    ``V_N`` shrinks rapidly with ``N`` (the admissible region is a
    vanishing fraction of the enclosing hypercube ``(-1, 1)^N``) and
    underflows to exactly ``0.0`` in ``float64`` for large ``N``; use
    :func:`log_admissible_volume` instead in that regime, following
    the numerical-robustness convention used elsewhere in the package
    for products of many factors (see :func:`ccf.coordinates.jacobian`
    / :func:`ccf.coordinates.log_jacobian`).
    """
    return float(np.exp(log_admissible_volume(N)))
