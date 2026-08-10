"""Didactic reference implementation of the r <-> alpha bijection.

Follows Eqs. (ld_p)-(ld_sigma) of the paper line by line, for a single
one-dimensional sequence. See :mod:`ccf.levinson` for the batched,
boundary-aware implementation used by the rest of the package; see
``docs/notation.md`` for the index correspondence between ``sigma2`` here
and the paper's ``sigma_n^2``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_reference_array(x: ArrayLike, *, name: str) -> FloatArray:
    """Return input as a contiguous one-dimensional float64 array."""
    array = np.asarray(x, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    return array


def pacf_reference(r: ArrayLike) -> FloatArray:
    """Convert correlation coefficients to partial autocorrelations.

    Direct, unoptimized implementation of the forward Levinson--Durbin
    recursion (Eqs. ld_p-ld_sigma) for an admissible interior sequence.

    Parameters
    ----------
    r
        One-dimensional correlation coefficients ``(r_1, ..., r_N)``.

    Returns
    -------
    numpy.ndarray
        Partial autocorrelations ``(alpha_1, ..., alpha_N)``.

    Raises
    ------
    ValueError
        If ``r`` is not one-dimensional, or if the sequence is not
        admissible (interior).
    """
    r = _as_reference_array(r, name="r")
    n_max = r.size

    alpha = [0.0] * n_max
    phi: list[float] = []
    sigma2 = 1.0

    for n in range(1, n_max + 1):
        if n == 1:
            prediction = 0.0
        else:
            prediction = sum(
                phi[j - 1] * r[n - j - 1] for j in range(1, n)
            )

        alpha_n = (float(r[n - 1]) - prediction) / sigma2

        if abs(alpha_n) >= 1.0:
            raise ValueError(
                f"r is not an admissible interior sequence: "
                f"abs(alpha_{n}) = {abs(alpha_n)!r} >= 1."
            )

        alpha[n - 1] = alpha_n

        phi_next = [0.0] * n
        for j in range(1, n):
            phi_next[j - 1] = phi[j - 1] - alpha_n * phi[n - j - 1]
        phi_next[n - 1] = alpha_n
        phi = phi_next

        sigma2 = sigma2 * (1.0 - alpha_n * alpha_n)

    return np.array(alpha, dtype=np.float64)


def from_pacf_reference(alpha: ArrayLike) -> FloatArray:
    """Reconstruct correlation coefficients from partial autocorrelations.

    Direct, unoptimized implementation of the inverse Levinson--Durbin
    recursion, using ``r_n = prediction + alpha_n * sigma2`` explicitly at
    each step.

    Parameters
    ----------
    alpha
        One-dimensional partial autocorrelations ``(alpha_1, ..., alpha_N)``.
        Entries must lie strictly between -1 and 1.

    Returns
    -------
    numpy.ndarray
        Correlation coefficients ``(r_1, ..., r_N)``.

    Raises
    ------
    ValueError
        If ``alpha`` is not one-dimensional, or an entry lies outside
        ``(-1, 1)``.
    """
    alpha = _as_reference_array(alpha, name="alpha")
    n_max = alpha.size

    if np.any(np.abs(alpha) >= 1.0):
        raise ValueError(
            "All PACF coefficients must satisfy abs(alpha_n) < 1."
        )

    r = [0.0] * n_max
    phi: list[float] = []
    sigma2 = 1.0

    for n in range(1, n_max + 1):
        alpha_n = float(alpha[n - 1])

        if n == 1:
            prediction = 0.0
        else:
            prediction = sum(
                phi[j - 1] * r[n - j - 1] for j in range(1, n)
            )

        r[n - 1] = prediction + alpha_n * sigma2

        phi_next = [0.0] * n
        for j in range(1, n):
            phi_next[j - 1] = phi[j - 1] - alpha_n * phi[n - j - 1]
        phi_next[n - 1] = alpha_n
        phi = phi_next

        sigma2 = sigma2 * (1.0 - alpha_n * alpha_n)

    return np.array(r, dtype=np.float64)
