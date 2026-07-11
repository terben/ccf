"""
Schneider-Hartlap admissible bounds.

This module connects the Schneider-Hartlap bounds to the PACF
coordinates computed by the Levinson-Durbin recursion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .levinson import _LevinsonState, pacf


def admissible_bounds(r: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Schneider-Hartlap admissible bounds for a correlation sequence.
    """
    state = _LevinsonState.from_correlations(r)

    r_arr = state.r
    n_max = r_arr.size

    r_lower = np.empty(n_max, dtype=float)
    r_upper = np.empty(n_max, dtype=float)

    for n in range(1, n_max + 1):
        if n == 1:
            prediction = 0.0
            sigma = 1.0
        else:
            phi = state.predictor_coefficients[n - 2]
            prediction = float(np.dot(phi, r_arr[n - 2 :: -1]))
            sigma = float(state.sigma2[n - 1])

        r_lower[n - 1] = prediction - sigma
        r_upper[n - 1] = prediction + sigma

    return r_lower, r_upper


def sh_coordinates(r: ArrayLike) -> np.ndarray:
    """
    Compute the Schneider-Hartlap coordinates.

    In the notation of the accompanying paper these coordinates are
    identical to the partial autocorrelation coefficients.
    """
    return pacf(r)


def check_admissibility(
    r: ArrayLike,
    *,
    atol: float = 1.0e-12,
    raise_error: bool = False,
) -> bool:
    """
    Check whether a sequence of correlation coefficients is admissible.
    """
    try:
        r_arr = np.asarray(r, dtype=float)

        if r_arr.ndim != 1:
            raise ValueError("r must be a one-dimensional array.")

        r_lower, r_upper = admissible_bounds(r_arr)

        ok = bool(
            np.all(r_arr >= r_lower - atol)
            and np.all(r_arr <= r_upper + atol)
        )

    except ValueError:
        ok = False

    if raise_error and not ok:
        raise ValueError(
            "The supplied coefficients do not define an admissible "
            "positive semidefinite Toeplitz correlation matrix."
        )

    return ok
