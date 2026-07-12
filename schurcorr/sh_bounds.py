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

import numpy as np
from numpy.typing import ArrayLike

from .levinson import (
    FloatArray,
    _LevinsonState,
    _asarray1d,
    pacf,
)


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
    """
    r_array = _asarray1d(r, name="r")
    state = _LevinsonState.from_correlations(r_array)

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

    return r_lower, r_upper


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
    """
    if atol < 0.0:
        raise ValueError("atol must be non-negative.")

    try:
        r_array = _asarray1d(r, name="r")
        r_lower, r_upper = admissible_bounds(r_array)

        complete_sequence = (
            r_lower.size == r_array.size
            and r_upper.size == r_array.size
        )

        if complete_sequence:
            admissible = bool(
                np.all(r_array >= r_lower - atol)
                and np.all(r_array <= r_upper + atol)
            )
        else:
            admissible = False

    except (TypeError, ValueError):
        admissible = False

    if raise_error and not admissible:
        raise ValueError(
            "The supplied coefficients do not define an admissible "
            "positive semidefinite Toeplitz correlation matrix."
        )

    return admissible
