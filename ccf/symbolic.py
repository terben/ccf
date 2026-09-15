"""Low-order symbolic expressions and identity checks for the companion paper.

These utilities build finite-order symbolic Toeplitz correlation
matrices and Levinson-Durbin quantities used to verify, at low order,
the relations of "Natural Coordinates for Constrained Correlation
Functions: Partial Autocorrelations and the Geometry of Positive Power
Spectra".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp


_MAX_SYMBOLIC_ORDER = 6


@dataclass(frozen=True, slots=True)
class _SymbolicLevinsonState:
    """
    Internal state of the symbolic Levinson recursion.

    Attributes
    ----------
    r
        Symbolic normalized correlation coefficients ``(r_1, ..., r_N)``.
    alpha
        Symbolic partial autocorrelations ``(alpha_1, ..., alpha_N)``.
    sigma2
        Symbolic residual variances. Tuple position ``j`` stores the
        paper quantity ``sigma_(j+1)^2``, so ``sigma2[0] = sigma_1^2``
        (equal to 1) and, in general, ``sigma2[j] <-> sigma_(j+1)^2``.
        There is no mathematical ``sigma_0^2``; this is a storage
        convention only.
    predictor_coefficients
        Symbolic Levinson-Durbin predictor coefficients for each order
        computed so far.
    """

    r: tuple[sp.Symbol, ...]
    alpha: tuple[sp.Expr, ...]
    sigma2: tuple[sp.Expr, ...]
    predictor_coefficients: tuple[tuple[sp.Expr, ...], ...]


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate a positive integer argument."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")

    if value < 1:
        raise ValueError(f"{name} must be at least 1.")


def _validate_symbolic_order(order: int) -> None:
    """Validate an order intended for symbolic recursion."""
    _validate_positive_integer(order, "order")

    if order > _MAX_SYMBOLIC_ORDER:
        raise ValueError(
            f"Symbolic calculations are restricted to order "
            f"{_MAX_SYMBOLIC_ORDER} or lower because expression sizes "
            "grow rapidly."
        )


def correlation_symbols(order: int) -> tuple[sp.Symbol, ...]:
    """
    Create symbolic normalized correlation coefficients.

    Parameters
    ----------
    order
        Highest paper lag; symbols are returned for ``r_1`` through
        ``r_order``.

    Returns
    -------
    tuple of Symbol
        Symbols ``(r1, ..., r_order)``.

    Examples
    --------
    >>> correlation_symbols(3)
    (r1, r2, r3)
    """
    _validate_positive_integer(order, "order")
    return tuple(sp.symbols(f"r1:{order + 1}", real=True))


def toeplitz_matrix(size: int) -> sp.Matrix:
    """
    Construct a symbolic Toeplitz correlation matrix.

    Parameters
    ----------
    size
        Matrix dimension. The resulting matrix is ``A_size`` and contains
        the correlation coefficients ``r1, ..., r_(size-1)``.

    Returns
    -------
    Matrix
        Symbolic Toeplitz correlation matrix with unit diagonal.

    Examples
    --------
    >>> toeplitz_matrix(3)
    Matrix([
    [ 1, r1, r2],
    [r1,  1, r1],
    [r2, r1,  1]])
    """
    _validate_positive_integer(size, "size")

    if size == 1:
        return sp.Matrix([[1]])

    r = correlation_symbols(size - 1)

    return sp.Matrix(
        size,
        size,
        lambda i, j: sp.Integer(1) if i == j else r[abs(i - j) - 1],
    )


def toeplitz_determinant(size: int) -> sp.Expr:
    """
    Compute a symbolic Toeplitz determinant.

    Parameters
    ----------
    size
        Matrix dimension.

    Returns
    -------
    Expr
        Factored expression for ``det(A_size)``.
    """
    return sp.factor(toeplitz_matrix(size).det())


@lru_cache(maxsize=None)
def _symbolic_state(order: int) -> _SymbolicLevinsonState:
    """
    Return the cached symbolic Levinson state through the requested order.

    Notes
    -----
    Following the companion paper's Levinson-Durbin relations,

    ``alpha_n = (r_n - p_n) / sigma_n^2``

    and

    ``sigma_(n+1)^2 = sigma_n^2 * (1 - alpha_n^2)``,

    with ``sigma_1^2 = 1``. See ``_SymbolicLevinsonState.sigma2`` for
    the tuple-position convention used to store the resulting
    ``sigma_n^2`` sequence.
    """
    _validate_symbolic_order(order)

    r = correlation_symbols(order)

    alpha_values: list[sp.Expr] = []
    sigma2_values: list[sp.Expr] = [sp.Integer(1)]
    predictors: list[tuple[sp.Expr, ...]] = []

    phi: tuple[sp.Expr, ...] = ()

    for n in range(1, order + 1):
        if n == 1:
            prediction = sp.Integer(0)
        else:
            reversed_correlations = tuple(reversed(r[: n - 1]))
            prediction = sum(
                coefficient * correlation
                for coefficient, correlation in zip(
                    phi, reversed_correlations, strict=True
                )
            )

        alpha_n = sp.cancel(
            (r[n - 1] - prediction) / sigma2_values[-1]
        )

        if n == 1:
            phi_new = (alpha_n,)
        else:
            phi_new = tuple(
                sp.cancel(phi[j] - alpha_n * phi[-j - 1])
                for j in range(n - 1)
            ) + (alpha_n,)

        sigma2_next = sp.factor(
            sigma2_values[-1] * (1 - alpha_n**2)
        )

        alpha_values.append(alpha_n)
        sigma2_values.append(sigma2_next)
        predictors.append(phi_new)
        phi = phi_new

    return _SymbolicLevinsonState(
        r=r,
        alpha=tuple(alpha_values),
        sigma2=tuple(sigma2_values),
        predictor_coefficients=tuple(predictors),
    )


def pacf_sequence_symbolic(order: int) -> tuple[sp.Expr, ...]:
    """
    Compute symbolic PACFs through a given order.

    Parameters
    ----------
    order
        Highest partial autocorrelation order.

    Returns
    -------
    tuple of Expr
        Symbolic expressions ``(alpha1, ..., alpha_order)``.
    """
    return _symbolic_state(order).alpha


def pacf_symbolic(order: int) -> sp.Expr:
    """
    Compute one symbolic partial autocorrelation coefficient.

    Parameters
    ----------
    order
        Partial autocorrelation order.

    Returns
    -------
    Expr
        Symbolic expression for ``alpha_order``.

    Examples
    --------
    >>> pacf_symbolic(1)
    r1
    >>> pacf_symbolic(2)
    (r1**2 - r2)/((r1 - 1)*(r1 + 1))
    """
    return sp.factor(_symbolic_state(order).alpha[-1])


def innovation_variances_symbolic(
    order: int,
) -> tuple[sp.Expr, ...]:
    """
    Compute symbolic residual variances.

    Parameters
    ----------
    order
        Highest recursion order.

    Returns
    -------
    tuple of Expr
        Residual variances where tuple position ``j`` holds the paper
        quantity ``sigma_(j+1)^2``, i.e. the tuple represents
        ``(sigma_1^2, ..., sigma_(order+1)^2)``. There is no
        mathematical ``sigma_0^2``.
    """
    return _symbolic_state(order).sigma2


def predictor_symbolic(order: int) -> sp.Expr:
    """
    Compute the linear prediction for ``r_order``.

    The prediction depends only on the preceding coefficients
    ``r1, ..., r_(order-1)``.

    Parameters
    ----------
    order
        Correlation order to be predicted.

    Returns
    -------
    Expr
        Symbolic linear prediction ``p_order``.
    """
    _validate_symbolic_order(order)

    if order == 1:
        return sp.Integer(0)

    previous_state = _symbolic_state(order - 1)
    phi = previous_state.predictor_coefficients[-1]
    reversed_correlations = tuple(reversed(previous_state.r))

    prediction = sum(
        coefficient * correlation
        for coefficient, correlation in zip(
            phi, reversed_correlations, strict=True
        )
    )

    return sp.factor(prediction)


def admissible_bounds_symbolic(
    order: int,
) -> tuple[sp.Expr, sp.Expr]:
    """
    Compute symbolic Schneider-Hartlap bounds.

    For fixed ``r1, ..., r_(order-1)``, the next coefficient satisfies

    ``r_lower <= r_order <= r_upper``.

    Parameters
    ----------
    order
        Correlation coefficient whose bounds are requested.

    Returns
    -------
    r_lower
        Symbolic lower bound.
    r_upper
        Symbolic upper bound.

    Notes
    -----
    The interval is centered on the linear prediction ``p_order`` and
    has half-width ``sigma_order^2``:

    ``p_order - sigma_order^2 <= r_order <= p_order + sigma_order^2``.

    The half-width is obtained from ``previous_state.sigma2[-1]``,
    i.e. tuple position ``order - 1`` of the state for ``order - 1``.
    Under the ``sigma2[j] <-> sigma_(j+1)^2`` storage convention (see
    ``_SymbolicLevinsonState.sigma2``), that position holds exactly
    the paper quantity ``sigma_order^2``.
    """
    _validate_symbolic_order(order)

    if order == 1:
        return sp.Integer(-1), sp.Integer(1)

    previous_state = _symbolic_state(order - 1)

    prediction = predictor_symbolic(order)
    half_width = previous_state.sigma2[-1]

    r_lower = sp.factor(prediction - half_width)
    r_upper = sp.factor(prediction + half_width)

    return r_lower, r_upper


def sh_coordinate_symbolic(order: int) -> sp.Expr:
    """
    Compute a symbolic Schneider-Hartlap coordinate.

    Parameters
    ----------
    order
        Coordinate order.

    Returns
    -------
    Expr
        Symbolic expression for ``x_order`` constructed directly from
        the admissible interval boundaries.

    Notes
    -----
    Following the companion paper,

    ``x_n = (2 r_n - r_(n,u) - r_(n,l)) / (r_(n,u) - r_(n,l))``,

    where ``r_(n,l)`` and ``r_(n,u)`` are the admissible-interval
    bounds returned by ``admissible_bounds_symbolic``.
    """
    _validate_symbolic_order(order)

    r_n = correlation_symbols(order)[-1]
    r_lower, r_upper = admissible_bounds_symbolic(order)

    x_n = (
        2 * r_n - r_upper - r_lower
    ) / (
        r_upper - r_lower
    )

    return sp.factor(sp.cancel(x_n))


def verify_x_equals_alpha(order: int) -> sp.Expr:
    """
    Verify symbolically that ``x_order = alpha_order``.

    Checks, at the requested low order, the companion paper's main
    identification ``x_n = alpha_n`` between the Schneider-Hartlap
    coordinate and the partial autocorrelation. This is only a
    symbolic consistency check at finite order; it does not constitute
    the paper's (geometric) proof of the identification.

    Parameters
    ----------
    order
        Recursion order to verify.

    Returns
    -------
    Expr
        Simplified difference ``x_order - alpha_order``. A successful
        verification returns exactly zero.

    Examples
    --------
    >>> verify_x_equals_alpha(3)
    0
    """
    difference = sh_coordinate_symbolic(order) - pacf_symbolic(order)
    return sp.factor(sp.cancel(difference))
