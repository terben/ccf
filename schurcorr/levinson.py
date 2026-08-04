"""
Levinson--Durbin recursions for Toeplitz correlation matrices.

This module provides the numerical core of the :mod:`schurcorr` package.
It implements the bijection between normalized correlation coefficients

    r = (r_1, ..., r_N)

and partial autocorrelation coefficients

    alpha = (alpha_1, ..., alpha_N).

The notation follows the accompanying paper.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

_TOL = 1.0e-12


def _asarray1d(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Convert an array-like input to a contiguous one-dimensional float array.

    Parameters
    ----------
    x
        Input values.
    name
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        Contiguous one-dimensional array with double-precision floating
        point entries.

    Raises
    ------
    ValueError
        If the input is not one-dimensional.
    """
    array = np.asarray(x, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    return np.ascontiguousarray(array)


class _LevinsonState:
    """
    Internal representation of a Levinson--Durbin recursion.

    The state may be constructed either from normalized correlation
    coefficients or from partial autocorrelation coefficients.

    Parameters
    ----------
    r
        Normalized correlation coefficients ``(r_1, ..., r_N)``.
    alpha
        Partial autocorrelation coefficients
        ``(alpha_1, ..., alpha_N)``.
    sigma2
        Innovation variances
        ``(sigma_0^2, ..., sigma_N^2)``.
    predictor_coefficients
        Predictor coefficients at each recursion order.
    """

    def __init__(
        self,
        *,
        r: FloatArray,
        alpha: FloatArray,
        sigma2: FloatArray,
        predictor_coefficients: list[FloatArray],
    ) -> None:
        self.r = r
        self.alpha = alpha
        self.sigma2 = sigma2
        self.predictor_coefficients = predictor_coefficients

    @classmethod
    def from_correlations(cls, r: ArrayLike) -> "_LevinsonState":
        """
        Construct the recursion state from correlation coefficients.
        """
        r_array = _asarray1d(r, name="r")
        n_max = r_array.size

        alpha_values: list[float] = []
        sigma2_values: list[float] = [1.0]
        predictors: list[FloatArray] = []

        phi = np.empty(0, dtype=np.float64)

        for n in range(1, n_max + 1):
            if sigma2_values[-1] <= _TOL:
                warnings.warn(
                    "Degenerate Toeplitz matrix encountered; "
                    "returning the non-degenerate part only.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            if n == 1:
                alpha_n = float(r_array[0])
            else:
                prediction = float(
                    np.dot(
                        phi,
                        r_array[n - 2 :: -1],
                    )
                )
                alpha_n = (
                    float(r_array[n - 1]) - prediction
                ) / sigma2_values[-1]

            if abs(alpha_n) > 1.0 + _TOL:
                raise ValueError(
                    "Input correlations are not admissible: "
                    f"computed abs(alpha_{n}) = "
                    f"{abs(alpha_n):.6g} > 1."
                )

            if abs(alpha_n) >= 1.0:
                alpha_n = float(np.sign(alpha_n))
                alpha_values.append(alpha_n)

                phi_boundary = np.append(phi, alpha_n)
                predictors.append(
                    np.asarray(phi_boundary, dtype=np.float64)
                )

                sigma2_values.append(0.0)

                warnings.warn(
                    "Boundary point encountered with abs(alpha_n) = 1; "
                    "subsequent coefficients are uniquely determined.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            if n == 1:
                phi_new = np.array(
                    [alpha_n],
                    dtype=np.float64,
                )
            else:
                phi_new = np.empty(n, dtype=np.float64)
                phi_new[:-1] = phi - alpha_n * phi[::-1]
                phi_new[-1] = alpha_n

            sigma2_next = (
                sigma2_values[-1]
                * (1.0 - alpha_n * alpha_n)
            )

            if sigma2_next < 0.0 and sigma2_next > -_TOL:
                warnings.warn(
                    "Innovation variance became slightly negative due "
                    "to roundoff and was clamped to zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                sigma2_next = 0.0

            alpha_values.append(float(alpha_n))
            sigma2_values.append(float(sigma2_next))
            predictors.append(phi_new.copy())

            phi = phi_new

        number_computed = len(alpha_values)

        return cls(
            r=r_array[:number_computed].copy(),
            alpha=np.asarray(
                alpha_values,
                dtype=np.float64,
            ),
            sigma2=np.asarray(
                sigma2_values,
                dtype=np.float64,
            ),
            predictor_coefficients=predictors,
        )

    @classmethod
    def from_pacf(cls, alpha: ArrayLike) -> "_LevinsonState":
        """
        Construct the recursion state from partial autocorrelations.
        """
        alpha_array = _asarray1d(alpha, name="alpha")

        if np.any(np.abs(alpha_array) >= 1.0):
            raise ValueError(
                "All PACF coefficients must satisfy abs(alpha_n) < 1."
            )

        r_values: list[float] = []
        sigma2_values: list[float] = [1.0]
        predictors: list[FloatArray] = []

        phi = np.empty(0, dtype=np.float64)

        for n, alpha_n in enumerate(alpha_array, start=1):
            alpha_n = float(alpha_n)

            if n == 1:
                phi_new = np.array(
                    [alpha_n],
                    dtype=np.float64,
                )
                r_n = alpha_n
            else:
                phi_new = np.empty(n, dtype=np.float64)
                phi_new[:-1] = phi - alpha_n * phi[::-1]
                phi_new[-1] = alpha_n

                r_with_zero = np.concatenate(
                    (
                        np.array([1.0], dtype=np.float64),
                        np.asarray(
                            r_values,
                            dtype=np.float64,
                        ),
                    )
                )

                r_n = float(
                    np.dot(
                        phi_new,
                        r_with_zero[n - 1 :: -1],
                    )
                )

            sigma2_next = (
                sigma2_values[-1]
                * (1.0 - alpha_n * alpha_n)
            )

            if sigma2_next < 0.0 and sigma2_next > -_TOL:
                warnings.warn(
                    "Innovation variance became slightly negative due "
                    "to roundoff and was clamped to zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                sigma2_next = 0.0

            r_values.append(float(r_n))
            sigma2_values.append(float(sigma2_next))
            predictors.append(phi_new.copy())

            phi = phi_new

        return cls(
            r=np.asarray(
                r_values,
                dtype=np.float64,
            ),
            alpha=alpha_array.copy(),
            sigma2=np.asarray(
                sigma2_values,
                dtype=np.float64,
            ),
            predictor_coefficients=predictors,
        )


def pacf(r: ArrayLike) -> FloatArray:
    """
    Compute partial autocorrelations from correlation coefficients.

    Parameters
    ----------
    r
        Normalized correlation coefficients
        ``(r_1, ..., r_N)``.

    Returns
    -------
    alpha
        Partial autocorrelation coefficients
        ``(alpha_1, ..., alpha_N)``.
    """
    state = _LevinsonState.from_correlations(r)
    return state.alpha


def from_pacf(alpha: ArrayLike) -> FloatArray:
    """
    Reconstruct correlations from partial autocorrelations.

    Parameters
    ----------
    alpha
        Partial autocorrelation coefficients
        ``(alpha_1, ..., alpha_N)``.

    Returns
    -------
    r
        Normalized correlation coefficients
        ``(r_1, ..., r_N)``.
    """
    state = _LevinsonState.from_pacf(alpha)
    return state.r


def fisher(alpha: ArrayLike) -> FloatArray:
    """
    Map partial autocorrelations to Fisher coordinates.

    The transformation is

    ``y_n = arctanh(alpha_n)``.

    Parameters
    ----------
    alpha
        Partial autocorrelation coefficients.

    Returns
    -------
    numpy.ndarray
        Fisher coordinates.
    """
    alpha_array = _asarray1d(alpha, name="alpha")

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
        Fisher coordinates.

    Returns
    -------
    numpy.ndarray
        Partial autocorrelation coefficients.
    """
    y_array = _asarray1d(y, name="y")
    return np.tanh(y_array)


def innovation_variances(alpha: ArrayLike) -> FloatArray:
    """
    Compute innovation variances from partial autocorrelations.

    The recursion is

    ``sigma_n^2 = sigma_(n-1)^2 * (1 - alpha_n^2)``

    with ``sigma_0^2 = 1``.

    Parameters
    ----------
    alpha
        Partial autocorrelation coefficients.

    Returns
    -------
    numpy.ndarray
        Innovation variances
        ``(sigma_0^2, ..., sigma_N^2)``.
    """
    alpha_array = _asarray1d(alpha, name="alpha")

    if np.any(np.abs(alpha_array) > 1.0):
        raise ValueError(
            "All PACF coefficients must satisfy abs(alpha_n) <= 1."
        )

    sigma2 = np.empty(
        alpha_array.size + 1,
        dtype=np.float64,
    )
    sigma2[0] = 1.0

    for n, alpha_n in enumerate(alpha_array, start=1):
        sigma2[n] = (
            sigma2[n - 1]
            * (1.0 - alpha_n * alpha_n)
        )

        if sigma2[n] < 0.0 and sigma2[n] > -_TOL:
            warnings.warn(
                "Innovation variance became slightly negative due "
                "to roundoff and was clamped to zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            sigma2[n] = 0.0

    return sigma2


def log_jacobian(alpha: ArrayLike) -> float:
    """
    Compute the logarithm of the Jacobian determinant of ``alpha -> r``.

    For ``N`` partial autocorrelations, the determinant is

    ``prod_{k=1}^{N-1} (1 - alpha_k^2)^(N-k)``.

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
