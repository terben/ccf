"""Coordinate transforms, residual variances, and Jacobians."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from .levinson import _ROUNDING_TOL, FloatArray, _asarray1d, _asarray_batchable


def fisher(alpha: ArrayLike) -> FloatArray:
    """
    Map partial autocorrelations to Fisher coordinates.

    The transformation is

    ``y_n = arctanh(alpha_n)``.

    Since the paper identifies ``x_n = alpha_n``, this is the
    Schneider--Hartlap Fisher coordinate
    ``y_n = atanh(x_n) = atanh(alpha_n)``.

    Parameters
    ----------
    alpha
        Partial autocorrelation coefficients, either a 1-D array
        ``(N,)`` or a 2-D batch ``(n_samples, N)``. The map is
        applied elementwise, so the batch case is not a distinct
        code path.

    Returns
    -------
    numpy.ndarray
        Fisher coordinates, with the same shape as ``alpha``.
    """
    alpha_array = _asarray_batchable(alpha, name="alpha")

    if np.any(np.abs(alpha_array) >= 1.0):
        raise ValueError(
            "All PACF coefficients must satisfy abs(alpha_n) < 1."
        )

    return np.arctanh(alpha_array)


def inverse_fisher(y: ArrayLike) -> FloatArray:
    """
    Map Fisher coordinates back to partial autocorrelations.

    The inverse transformation is

    ``alpha_n = tanh(y_n)``.

    Parameters
    ----------
    y
        Fisher coordinates, either a 1-D array ``(N,)`` or a 2-D
        batch ``(n_samples, N)``. The map is applied elementwise, so
        the batch case is not a distinct code path.

    Returns
    -------
    numpy.ndarray
        Partial autocorrelation coefficients, with the same shape as
        ``y``.
    """
    y_array = _asarray_batchable(y, name="y")
    return np.tanh(y_array)


def innovation_variances(alpha: ArrayLike) -> FloatArray:
    """
    Compute residual variances from partial autocorrelations.

    The paper's Levinson--Durbin residual-variance recursion (Sect. 4.4) is

    ``sigma_(n+1)^2 = sigma_n^2 * (1 - alpha_n^2)``

    with ``sigma_1^2 = 1``.

    Parameters
    ----------
    alpha
        Partial autocorrelation coefficients, either a 1-D array
        ``(N,)`` or a 2-D batch ``(n_samples, N)``. For a 2-D batch,
        the recursion runs independently, row by row, over the same
        per-order loop as the 1-D case, vectorized across samples.

    Returns
    -------
    numpy.ndarray
        Residual variances stored at implementation array positions:
        array position ``j`` contains the paper quantity
        ``sigma_(j+1)^2``. For a 1-D input with ``N`` PACFs, the
        returned positions 0 through ``N`` therefore contain
        ``sigma_1^2, ..., sigma_(N+1)^2`` in the paper notation.
        For a 2-D input batch of shape ``(n_samples, N)``, the returned
        array has shape ``(n_samples, N + 1)``, with the same mapping
        along the second axis.
    """
    alpha_array = _asarray_batchable(alpha, name="alpha")

    if np.any(np.abs(alpha_array) > 1.0):
        raise ValueError(
            "All PACF coefficients must satisfy abs(alpha_n) <= 1."
        )

    if alpha_array.ndim == 1:
        sigma2 = np.empty(
            alpha_array.size + 1,
            dtype=np.float64,
        )
        # sigma2[j] stores the paper quantity sigma_(j+1)^2; sigma2[0] = 1
        # is the paper initialization sigma_1^2 = 1.
        sigma2[0] = 1.0

        for n, alpha_n in enumerate(alpha_array, start=1):
            sigma2[n] = (
                sigma2[n - 1]
                * (1.0 - alpha_n * alpha_n)
            )

            if sigma2[n] < 0.0 and sigma2[n] > -_ROUNDING_TOL:
                warnings.warn(
                    "Innovation variance became slightly negative due "
                    "to roundoff and was clamped to zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                sigma2[n] = 0.0

        return sigma2

    n_samples, n_max = alpha_array.shape
    sigma2 = np.empty((n_samples, n_max + 1), dtype=np.float64)
    sigma2[:, 0] = 1.0

    for n in range(1, n_max + 1):
        alpha_n = alpha_array[:, n - 1]
        sigma2[:, n] = sigma2[:, n - 1] * (1.0 - alpha_n * alpha_n)

        clamp_mask = (sigma2[:, n] < 0.0) & (sigma2[:, n] > -_ROUNDING_TOL)
        if np.any(clamp_mask):
            warnings.warn(
                "Innovation variance became slightly negative due "
                "to roundoff and was clamped to zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            sigma2[clamp_mask, n] = 0.0

    return sigma2


def log_jacobian(alpha: ArrayLike) -> float:
    """
    Compute the logarithm of the Jacobian determinant of ``alpha -> r``.

    For ``N`` partial autocorrelations, the determinant is

    ``prod_{k=1}^{N-1} (1 - alpha_k^2)^(N-k)``,

    the Jacobian ``det(d r / d alpha)`` of Sect. 4.4 (Eq. (20)) of the
    companion paper, equivalently ``prod_{n=1}^{N} sigma_n^2``.

    Parameters
    ----------
    alpha
        Partial autocorrelation coefficients.

    Returns
    -------
    float
        Natural logarithm of the Jacobian determinant.
    """
    alpha_array = _asarray1d(alpha, name="alpha")

    if np.any(np.abs(alpha_array) >= 1.0):
        raise ValueError(
            "All PACF coefficients must satisfy abs(alpha_n) < 1."
        )

    n = alpha_array.size

    if n <= 1:
        return 0.0

    weights = np.arange(
        n - 1,
        0,
        -1,
        dtype=np.float64,
    )

    return float(
        np.sum(
            weights
            * np.log1p(-alpha_array[:-1] ** 2)
        )
    )


def jacobian(alpha: ArrayLike) -> float:
    """
    Compute the Jacobian determinant of the map ``alpha -> r``.

    See :func:`log_jacobian` for the Sect. 4.4 / Eq. (20) connection to
    the companion paper.

    Parameters
    ----------
    alpha
        Partial autocorrelation coefficients.

    Returns
    -------
    float
        Jacobian determinant.
    """
    return float(np.exp(log_jacobian(alpha)))
