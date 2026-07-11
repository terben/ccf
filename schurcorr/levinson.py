"""
Levinson-Durbin recursions for Toeplitz correlation matrices.

This module provides the numerical core of the schurcorr package.
It implements the bijection between normalized correlation coefficients
and partial autocorrelation coefficients (PACFs).

The notation follows Section 4 of the accompanying paper.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from .types import LevinsonInfo


_TOL = 1.0e-12


def _as_1d_float_array(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    return arr


class _LevinsonState:
    """
    Internal representation of a positive Toeplitz correlation matrix.

    The state may be constructed either from correlation coefficients
    or from partial autocorrelations.
    """

    def __init__(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        sigma2: np.ndarray,
        predictor_coefficients: list[np.ndarray],
    ) -> None:
        self.r = r
        self.alpha = alpha
        self.sigma2 = sigma2
        self.predictor_coefficients = predictor_coefficients

    @classmethod
    def from_correlations(cls, r: ArrayLike) -> "_LevinsonState":
        r = _as_1d_float_array(r, "r")
        n_max = r.size

        alpha: list[float] = []
        sigma2: list[float] = [1.0]
        predictors: list[np.ndarray] = []

        phi = np.empty(0, dtype=float)

        for n in range(1, n_max + 1):
            if sigma2[-1] <= _TOL:
                warnings.warn(
                    "Degenerate Toeplitz matrix encountered; "
                    "returning the non-degenerate part only.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            if n == 1:
                a_n = r[0]
            else:
                prediction = np.dot(phi, r[n - 2 :: -1])
                a_n = (r[n - 1] - prediction) / sigma2[-1]

            if abs(a_n) > 1.0 + _TOL:
                raise ValueError(
                    "Input correlations are not admissible: "
                    f"computed abs(alpha_{n}) = {abs(a_n):.6g} > 1."
                )

            if abs(a_n) >= 1.0:
                a_n = float(np.sign(a_n))
                alpha.append(a_n)
                predictors.append(np.append(phi, a_n))

                sigma_next = 0.0
                sigma2.append(sigma_next)

                warnings.warn(
                    "Boundary point encountered with abs(alpha_n) = 1; "
                    "subsequent coefficients are uniquely determined.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            if n == 1:
                phi_new = np.array([a_n], dtype=float)
            else:
                phi_new = np.empty(n, dtype=float)
                phi_new[:-1] = phi - a_n * phi[::-1]
                phi_new[-1] = a_n

            sigma_next = sigma2[-1] * (1.0 - a_n * a_n)
            if sigma_next < 0.0 and sigma_next > -_TOL:
                warnings.warn(
                    "Innovation variance became slightly negative due to "
                    "roundoff and was clamped to zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                sigma_next = 0.0

            alpha.append(float(a_n))
            sigma2.append(float(sigma_next))
            predictors.append(phi_new.copy())
            phi = phi_new

        return cls(
            r=r[: len(alpha)].copy(),
            alpha=np.asarray(alpha, dtype=float),
            sigma2=np.asarray(sigma2, dtype=float),
            predictor_coefficients=predictors,
        )

    @classmethod
    def from_pacf(cls, alpha: ArrayLike) -> "_LevinsonState":
        alpha = _as_1d_float_array(alpha, "alpha")

        if np.any(np.abs(alpha) >= 1.0):
            raise ValueError("All PACF coefficients must satisfy abs(alpha_n) < 1.")

        r_values: list[float] = []
        sigma2: list[float] = [1.0]
        predictors: list[np.ndarray] = []

        phi = np.empty(0, dtype=float)

        for n, a_n in enumerate(alpha, start=1):
            if n == 1:
                phi_new = np.array([a_n], dtype=float)
                r_n = a_n
            else:
                phi_new = np.empty(n, dtype=float)
                phi_new[:-1] = phi - a_n * phi[::-1]
                phi_new[-1] = a_n

                r_with_zero = np.concatenate(([1.0], np.asarray(r_values)))
                r_n = np.dot(phi_new, r_with_zero[n - 1 :: -1])

            sigma_next = sigma2[-1] * (1.0 - a_n * a_n)
            if sigma_next < 0.0 and sigma_next > -_TOL:
                warnings.warn(
                    "Innovation variance became slightly negative due to "
                    "roundoff and was clamped to zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                sigma_next = 0.0

            r_values.append(float(r_n))
            sigma2.append(float(sigma_next))
            predictors.append(phi_new.copy())
            phi = phi_new

        return cls(
            r=np.asarray(r_values, dtype=float),
            alpha=alpha.copy(),
            sigma2=np.asarray(sigma2, dtype=float),
            predictor_coefficients=predictors,
        )


def pacf(
    r: ArrayLike,
    *,
    return_info: bool = False,
):
    """
    Compute partial autocorrelation coefficients from correlation coefficients.
    """
    state = _LevinsonState.from_correlations(r)

    if return_info:
        return state.alpha, LevinsonInfo(
            sigma2=state.sigma2,
            predictor_coefficients=state.predictor_coefficients,
        )

    return state.alpha


def from_pacf(
    alpha: ArrayLike,
    *,
    return_info: bool = False,
):
    """
    Reconstruct correlation coefficients from partial autocorrelations.
    """
    state = _LevinsonState.from_pacf(alpha)

    if return_info:
        return state.r, LevinsonInfo(
            sigma2=state.sigma2,
            predictor_coefficients=state.predictor_coefficients,
        )

    return state.r


def fisher(alpha: ArrayLike) -> np.ndarray:
    """
    Map partial autocorrelations to Fisher coordinates.
    """
    alpha = _as_1d_float_array(alpha, "alpha")

    if np.any(np.abs(alpha) >= 1.0):
        raise ValueError("All PACF coefficients must satisfy abs(alpha_n) < 1.")

    return np.arctanh(alpha)


def inverse_fisher(y: ArrayLike) -> np.ndarray:
    """
    Map Fisher coordinates back to partial autocorrelations.
    """
    y = _as_1d_float_array(y, "y")
    return np.tanh(y)


def innovation_variances(alpha: ArrayLike) -> np.ndarray:
    """
    Compute innovation variances from partial autocorrelations.
    """
    alpha = _as_1d_float_array(alpha, "alpha")

    if np.any(np.abs(alpha) > 1.0):
        raise ValueError("All PACF coefficients must satisfy abs(alpha_n) <= 1.")

    sigma2 = np.empty(alpha.size + 1, dtype=float)
    sigma2[0] = 1.0

    for n, a_n in enumerate(alpha, start=1):
        sigma2[n] = sigma2[n - 1] * (1.0 - a_n * a_n)

        if sigma2[n] < 0.0 and sigma2[n] > -_TOL:
            warnings.warn(
                "Innovation variance became slightly negative due to "
                "roundoff and was clamped to zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            sigma2[n] = 0.0

    return sigma2


def log_jacobian(alpha: ArrayLike) -> float:
    """
    Compute the logarithm of the Jacobian determinant of alpha -> r.
    """
    alpha = _as_1d_float_array(alpha, "alpha")

    if np.any(np.abs(alpha) >= 1.0):
        raise ValueError("All PACF coefficients must satisfy abs(alpha_n) < 1.")

    n = alpha.size

    if n <= 1:
        return 0.0

    weights = np.arange(n - 1, 0, -1, dtype=float)

    return float(np.sum(weights * np.log1p(-alpha[:-1] ** 2)))


def jacobian(alpha: ArrayLike) -> float:
    """
    Compute the Jacobian determinant of the map alpha -> r.
    """
    return float(np.exp(log_jacobian(alpha)))
