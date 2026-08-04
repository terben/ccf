"""
Schneider--Hartlap admissible bounds and coordinates.

This module connects the admissible intervals introduced by
Schneider and Hartlap with the Levinson--Durbin recursion.

For fixed preceding correlation coefficients

    r_1, ..., r_(n-1),

the next coefficient satisfies

    r_(n, lower) <= r_n <= r_(n, upper).

The centre of this interval is the linear prediction p_n and its
half-width is the preceding innovation variance sigma_(n-1)^2.

The normalized Schneider--Hartlap coordinate is identical to the
partial autocorrelation coefficient,

    x_n = alpha_n.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

from .levinson import (
    _TOL,
    FloatArray,
    _LevinsonState,
    _asarray1d,
    pacf,
)

_CONSISTENCY_TOL = 1.0e-8


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
    Compute successive Schneider--Hartlap admissible bounds.

    For every supplied coefficient ``r_n``, the function computes the
    lower and upper bounds implied by the preceding coefficients
    ``r_1, ..., r_(n-1)``.

    Parameters
    ----------
    r
        Normalized correlation coefficients
        ``(r_1, ..., r_N)``.

    Returns
    -------
    r_lower
        Successive lower bounds
        ``(r_(1, lower), ..., r_(N, lower))``.
    r_upper
        Successive upper bounds
        ``(r_(1, upper), ..., r_(N, upper))``.

    Raises
    ------
    ValueError
        If ``r`` is not one-dimensional or does not define an
        admissible Toeplitz correlation sequence.

    Notes
    -----
    At order ``n``, the admissible interval has the form

    ``p_n - sigma_(n-1)^2 <= r_n <= p_n + sigma_(n-1)^2``,

    where ``p_n`` is the linear prediction from the preceding
    correlation coefficients.

    If ``r`` reaches the singular boundary of the admissible region
    (``sigma_(m-1)^2 = 0`` for some order ``m <= N``) within its given
    length, every further coefficient is uniquely forced (see
    :func:`extend_at_boundary`) and has no freedom left, so its bounds
    collapse to a single point, ``r_(n, lower) = r_(n, upper) = r_n``.
    Entries past the boundary are validated against that forced
    continuation, not just checked for presence; a ``ValueError`` is
    raised if they are inconsistent with it.
    """
    r_array = _asarray1d(r, name="r")
    state = _LevinsonState.from_correlations(r_array, at_boundary="warn")

    number_computed = state.r.size

    r_lower = np.empty(
        number_computed,
        dtype=np.float64,
    )
    r_upper = np.empty(
        number_computed,
        dtype=np.float64,
    )

    for n in range(1, number_computed + 1):
        if n == 1:
            prediction = 0.0
            half_width = 1.0
        else:
            predictor = state.predictor_coefficients[n - 2]

            prediction = float(
                np.dot(
                    predictor,
                    state.r[n - 2 :: -1],
                )
            )
            half_width = float(state.sigma2[n - 1])

        r_lower[n - 1] = prediction - half_width
        r_upper[n - 1] = prediction + half_width

    if number_computed == r_array.size:
        return r_lower, r_upper

    phi = state.predictor_coefficients[-1]
    window = list(state.r)
    excess = r_array[number_computed:]
    extra_bounds = np.empty(excess.size, dtype=np.float64)

    for i, supplied_raw in enumerate(excess):
        forced = _forced_continuation(phi, window)
        supplied = float(supplied_raw)

        if not np.isclose(
            forced, supplied, rtol=_CONSISTENCY_TOL, atol=_CONSISTENCY_TOL
        ):
            raise ValueError(
                f"r_{number_computed + i + 1} = {supplied!r} is "
                "inconsistent with the Toeplitz-forced continuation "
                f"r_{number_computed + i + 1} = {forced!r} past the "
                f"singular boundary (order {number_computed + 1}); r "
                "does not define an admissible correlation sequence."
            )

        extra_bounds[i] = supplied
        window = window[1:] + [supplied]

    return (
        np.concatenate([r_lower, extra_bounds]),
        np.concatenate([r_upper, extra_bounds]),
    )


def extend_at_boundary(r: ArrayLike, n_extra: int) -> FloatArray:
    """
    Extend an admissible correlation sequence past its singular
    boundary using the unique Toeplitz-forced continuation.

    Parameters
    ----------
    r
        Normalized correlation coefficients ``(r_1, ..., r_N)`` that
        reach the singular boundary of the admissible region
        (``sigma_(m-1)^2 = 0`` for some order ``m <= N``) within the
        supplied length. Entries already supplied past the boundary
        are validated against the forced continuation rather than
        overwritten.
    n_extra
        Number of further coefficients to generate past the boundary.
        Must be at least the number of entries ``r`` already supplies
        past the boundary.

    Returns
    -------
    numpy.ndarray
        The extended correlation sequence, of length
        ``(m - 1) + n_extra``, where ``m`` is the first order at
        which the Toeplitz matrix is singular.

    Raises
    ------
    ValueError
        If ``n_extra`` is negative; if ``r`` does not reach the
        boundary within its given length (this function is only
        meaningful once ``sigma_n^2 = 0`` has actually been reached,
        e.g. as checked via ``pacf(r, at_boundary='warn')``); if
        ``n_extra`` is smaller than the number of entries ``r``
        already supplies past the boundary; or if those entries are
        inconsistent with the forced continuation (i.e. ``r`` is not
        realizable by any nonnegative power spectrum past that
        point).

    Notes
    -----
    See Sect. 5 of the CCF paper and ``SH_research_note.pdf`` (Erben
    2026), "Toeplitz Determinants and Admissible Correlation
    Intervals". Let ``m`` be the first order at which the Toeplitz
    correlation matrix ``A_m`` (built from ``r_1, ..., r_(m-1)``) is
    singular. Because ``A_m`` is singular, it has a null vector
    ``w = (w_0, ..., w_(m-1))`` with ``w_0 != 0``; normalizing
    ``w_0 = 1`` identifies ``w`` with the terminal Levinson--Durbin
    predictor coefficients, ``w = (1, -phi_1, ..., -phi_(m-1))``,
    where ``phi = phi^{(m-1)}`` is the predictor at the point the
    recursion reaches the boundary. Every further coefficient is then
    forced by the linear recurrence

    ``r_(m+s) = sum_(j=1)^(m-1) phi_j * r_(m+s-j)``, ``s >= 0``,

    i.e. repeated application of the same fixed-order linear
    predictor that produced the boundary -- there is no further
    freedom.
    """
    if n_extra < 0:
        raise ValueError("n_extra must be non-negative.")

    r_array = _asarray1d(r, name="r")
    state = _LevinsonState.from_correlations(r_array, at_boundary="warn")

    if state.sigma2[-1] > _TOL:
        raise ValueError(
            "r does not reach the singular boundary of the admissible "
            "region within its given length; extend_at_boundary is "
            "only meaningful once sigma_n^2 = 0 has actually been "
            "reached. Check with pacf(r, at_boundary='warn') first."
        )

    prefix_len = state.r.size
    phi = state.predictor_coefficients[-1]

    existing_excess = r_array[prefix_len:]

    if existing_excess.size > n_extra:
        raise ValueError(
            f"r already contains {existing_excess.size} entries past "
            f"the boundary (order {prefix_len + 1}), but "
            f"n_extra={n_extra} was requested; pass n_extra >= "
            f"{existing_excess.size}, or pass only the admissible "
            "prefix of r."
        )

    window = list(state.r)
    generated: list[float] = []

    for s in range(n_extra):
        r_next = _forced_continuation(phi, window)

        if s < existing_excess.size:
            supplied = float(existing_excess[s])

            if not np.isclose(
                r_next, supplied, rtol=_CONSISTENCY_TOL, atol=_CONSISTENCY_TOL
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
        [state.r, np.asarray(generated, dtype=np.float64)]
    )


def sh_coordinates(r: ArrayLike) -> FloatArray:
    """
    Compute Schneider--Hartlap coordinates.

    Parameters
    ----------
    r
        Normalized correlation coefficients
        ``(r_1, ..., r_N)``.

    Returns
    -------
    numpy.ndarray
        Schneider--Hartlap coordinates
        ``(x_1, ..., x_N)``.

    Notes
    -----
    The Schneider--Hartlap coordinates are identical to the partial
    autocorrelation coefficients,

    ``x_n = alpha_n``.

    The implementation therefore delegates directly to :func:`pacf`.
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
    Computed in log-space via ``scipy.special.gammaln``, since the
    underlying product

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

    j = np.arange(1, N, dtype=np.float64)
    log_terms = 0.5 * math.log(math.pi) + gammaln(j + 1.0) - gammaln(j + 1.5)

    return math.log(2.0) + float(np.sum(log_terms))


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
    for products of many factors (see :func:`schurcorr.levinson.jacobian`
    / :func:`schurcorr.levinson.log_jacobian`).
    """
    return float(np.exp(log_admissible_volume(N)))
