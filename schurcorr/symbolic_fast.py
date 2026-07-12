"""
Efficient exact symbolic calculations for Toeplitz correlation matrices.

This module provides an experimental high-order symbolic backend for
``schurcorr``. In contrast to :mod:`schurcorr.symbolic`, the Levinson
recursion is performed in a rational function field over the correlation
coefficients,

    QQ(r1, ..., rN),

rather than on general SymPy expression trees.

This representation keeps rational functions in a canonical algebraic
form during the recursion and substantially reduces expression swell.
Conversion to ordinary SymPy expressions is delayed until a result is
requested.

The module is intended for exploratory exact calculations at orders for
which the didactic implementation in :mod:`schurcorr.symbolic` becomes
too expensive.

Notes
-----
The size of explicit symbolic expressions still grows rapidly with the
recursion order. This implementation improves exact arithmetic and avoids
unnecessary intermediate simplification, but it cannot eliminate the
intrinsic growth of the final formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import sympy as sp
from sympy.polys.domains import QQ
from sympy.polys.fields import field


SimplificationMode = Literal[
    "none",
    "cancel",
    "factor",
    "together",
    "simplify",
]


@dataclass(frozen=True, slots=True)
class _RationalFunctionContext:
    """
    Rational-function field used for one symbolic recursion.

    Attributes
    ----------
    domain
        Fraction field ``QQ(r1, ..., rN)``.
    generators
        Field generators corresponding to the correlation coefficients.
    symbols
        Ordinary real-valued SymPy symbols used in returned expressions.
    """

    domain: Any
    generators: tuple[Any, ...]
    symbols: tuple[sp.Symbol, ...]


@dataclass(frozen=True, slots=True)
class _FastSymbolicState:
    """
    Exact symbolic Levinson state in a rational-function field.

    Attributes
    ----------
    context
        Rational-function context of the calculation.
    alpha
        Partial autocorrelation coefficients represented as fraction-field
        elements.
    sigma2
        Innovation variances, including ``sigma_0^2 = 1``.
    predictor_coefficients
        Prediction coefficients at every recursion order.
    """

    context: _RationalFunctionContext
    alpha: tuple[Any, ...]
    sigma2: tuple[Any, ...]
    predictor_coefficients: tuple[tuple[Any, ...], ...]


def _validate_positive_integer(value: int, name: str) -> None:
    """
    Validate that an argument is a strictly positive integer.

    Parameters
    ----------
    value
        Value to validate.
    name
        Argument name used in error messages.

    Raises
    ------
    TypeError
        If ``value`` is not an integer.
    ValueError
        If ``value`` is smaller than one.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")

    if value < 1:
        raise ValueError(f"{name} must be at least 1.")


def _validate_simplification_mode(mode: SimplificationMode) -> None:
    """Validate a requested final simplification strategy."""
    valid_modes = {
        "none",
        "cancel",
        "factor",
        "together",
        "simplify",
    }

    if mode not in valid_modes:
        raise ValueError(
            "simplify must be one of "
            "'none', 'cancel', 'factor', 'together', or 'simplify'."
        )


def _apply_simplification(
    expression: sp.Expr,
    mode: SimplificationMode,
) -> sp.Expr:
    """
    Apply a final simplification operation to a SymPy expression.

    Parameters
    ----------
    expression
        Expression to process.
    mode
        Requested simplification strategy.

    Returns
    -------
    Expr
        Processed expression.
    """
    _validate_simplification_mode(mode)

    if mode == "none":
        return expression
    if mode == "cancel":
        return sp.cancel(expression)
    if mode == "factor":
        return sp.factor(expression)
    if mode == "together":
        return sp.together(expression)

    return sp.simplify(expression)


@lru_cache(maxsize=None)
def correlation_symbols(order: int) -> tuple[sp.Symbol, ...]:
    """
    Create real-valued symbolic correlation coefficients.

    Parameters
    ----------
    order
        Highest correlation order.

    Returns
    -------
    tuple of Symbol
        Symbols ``(r1, ..., r_order)``.
    """
    _validate_positive_integer(order, "order")

    return tuple(
        sp.symbols(
            f"r1:{order + 1}",
            real=True,
        )
    )


@lru_cache(maxsize=None)
def _rational_function_context(
    order: int,
) -> _RationalFunctionContext:
    """
    Construct the fraction field ``QQ(r1, ..., r_order)``.

    Parameters
    ----------
    order
        Number of field generators.

    Returns
    -------
    _RationalFunctionContext
        Cached field, generators, and output symbols.
    """
    _validate_positive_integer(order, "order")

    names = ",".join(f"r{index}" for index in range(1, order + 1))
    result = field(names, QQ)

    domain = result[0]
    generators = tuple(result[1:])

    return _RationalFunctionContext(
        domain=domain,
        generators=generators,
        symbols=correlation_symbols(order),
    )


def _field_element_to_expression(
    element: Any,
    context: _RationalFunctionContext,
    *,
    simplify: SimplificationMode = "none",
) -> sp.Expr:
    """
    Convert a fraction-field element to an ordinary SymPy expression.

    Parameters
    ----------
    element
        Element of ``QQ(r1, ..., rN)``.
    context
        Field context containing the corresponding output symbols.
    simplify
        Optional final simplification strategy.

    Returns
    -------
    Expr
        Ordinary SymPy expression using real-valued correlation symbols.
    """
    expression = element.as_expr()

    # ``field`` creates generators without the ``real=True`` assumption.
    # Replace them by the public symbols used by the remaining package.
    replacements = {
        sp.Symbol(str(generator)): symbol
        for generator, symbol in zip(
            context.generators,
            context.symbols,
            strict=True,
        )
    }

    expression = expression.xreplace(replacements)

    return _apply_simplification(expression, simplify)


@lru_cache(maxsize=None)
def toeplitz_matrix(size: int) -> sp.ImmutableDenseMatrix:
    """
    Construct a symbolic Toeplitz correlation matrix.

    Parameters
    ----------
    size
        Matrix dimension.

    Returns
    -------
    ImmutableDenseMatrix
        Toeplitz correlation matrix ``A_size`` with unit diagonal.
    """
    _validate_positive_integer(size, "size")

    if size == 1:
        return sp.ImmutableDenseMatrix([[1]])

    r = correlation_symbols(size - 1)

    return sp.ImmutableDenseMatrix(
        size,
        size,
        lambda i, j: (
            sp.Integer(1)
            if i == j
            else r[abs(i - j) - 1]
        ),
    )


@lru_cache(maxsize=None)
def _toeplitz_determinant_raw(size: int) -> sp.Expr:
    """
    Compute and cache an unfactored Toeplitz determinant.

    The domain-based Gaussian elimination backend is generally more
    efficient for exact polynomial matrices than generic expression
    expansion.
    """
    _validate_positive_integer(size, "size")

    return toeplitz_matrix(size).det(method="domain-ge")


def toeplitz_determinant(
    size: int,
    *,
    simplify: SimplificationMode = "none",
) -> sp.Expr:
    """
    Compute a symbolic Toeplitz determinant.

    Parameters
    ----------
    size
        Matrix dimension.
    simplify
        Final simplification strategy. ``"none"`` avoids potentially
        expensive factorization of high-order determinants.

    Returns
    -------
    Expr
        Symbolic expression for ``det(A_size)``.
    """
    determinant = _toeplitz_determinant_raw(size)

    return _apply_simplification(
        determinant,
        simplify,
    )


@lru_cache(maxsize=None)
def _symbolic_state_raw(order: int) -> _FastSymbolicState:
    """
    Compute an exact Levinson state in ``QQ(r1, ..., r_order)``.

    No conversion to general SymPy expressions takes place inside the
    recursion. Fraction-field arithmetic automatically maintains
    canonical rational representations.
    """
    _validate_positive_integer(order, "order")

    context = _rational_function_context(order)
    r = context.generators

    alpha_values: list[Any] = []
    sigma2_values: list[Any] = [context.domain.one]
    predictors: list[tuple[Any, ...]] = []

    phi: tuple[Any, ...] = ()

    for n in range(1, order + 1):
        if n == 1:
            prediction = context.domain.zero
        else:
            prediction = sum(
                (
                    phi[j] * r[n - 2 - j]
                    for j in range(n - 1)
                ),
                context.domain.zero,
            )

        alpha_n = (
            r[n - 1] - prediction
        ) / sigma2_values[-1]

        if n == 1:
            phi_new = (alpha_n,)
        else:
            phi_new = tuple(
                phi[j] - alpha_n * phi[n - 2 - j]
                for j in range(n - 1)
            ) + (alpha_n,)

        sigma2_next = (
            sigma2_values[-1]
            * (context.domain.one - alpha_n**2)
        )

        alpha_values.append(alpha_n)
        sigma2_values.append(sigma2_next)
        predictors.append(phi_new)

        phi = phi_new

    return _FastSymbolicState(
        context=context,
        alpha=tuple(alpha_values),
        sigma2=tuple(sigma2_values),
        predictor_coefficients=tuple(predictors),
    )


def pacf_sequence_symbolic(
    order: int,
    *,
    simplify: SimplificationMode = "none",
) -> tuple[sp.Expr, ...]:
    """
    Compute symbolic PACFs through a given order.

    Parameters
    ----------
    order
        Highest partial autocorrelation order.
    simplify
        Final simplification applied independently to each expression.

    Returns
    -------
    tuple of Expr
        Expressions ``(alpha1, ..., alpha_order)``.
    """
    state = _symbolic_state_raw(order)

    return tuple(
        _field_element_to_expression(
            element,
            state.context,
            simplify=simplify,
        )
        for element in state.alpha
    )


def pacf_symbolic(
    order: int,
    *,
    simplify: SimplificationMode = "none",
) -> sp.Expr:
    """
    Compute one symbolic partial autocorrelation coefficient.

    Parameters
    ----------
    order
        Partial autocorrelation order.
    simplify
        Final simplification strategy.

    Returns
    -------
    Expr
        Exact symbolic expression for ``alpha_order``.

    Notes
    -----
    The result is already a canonical rational function before conversion
    to a SymPy expression. Additional calls to ``cancel`` are therefore
    usually unnecessary.
    """
    state = _symbolic_state_raw(order)

    return _field_element_to_expression(
        state.alpha[-1],
        state.context,
        simplify=simplify,
    )


def innovation_variances_symbolic(
    order: int,
    *,
    simplify: SimplificationMode = "none",
) -> tuple[sp.Expr, ...]:
    """
    Compute symbolic innovation variances.

    Parameters
    ----------
    order
        Highest recursion order.
    simplify
        Final simplification applied independently to each expression.

    Returns
    -------
    tuple of Expr
        Expressions ``(sigma_0^2, ..., sigma_order^2)``.
    """
    state = _symbolic_state_raw(order)

    return tuple(
        _field_element_to_expression(
            element,
            state.context,
            simplify=simplify,
        )
        for element in state.sigma2
    )


@lru_cache(maxsize=None)
def _predictor_field_element(order: int) -> tuple[Any, _RationalFunctionContext]:
    """
    Compute the prediction for ``r_order`` as a field element.
    """
    _validate_positive_integer(order, "order")

    if order == 1:
        context = _rational_function_context(1)
        return context.domain.zero, context

    state = _symbolic_state_raw(order - 1)
    phi = state.predictor_coefficients[-1]
    r = state.context.generators

    prediction = sum(
        (
            phi[j] * r[order - 2 - j]
            for j in range(order - 1)
        ),
        state.context.domain.zero,
    )

    return prediction, state.context


def predictor_symbolic(
    order: int,
    *,
    simplify: SimplificationMode = "none",
) -> sp.Expr:
    """
    Compute the linear prediction for ``r_order``.

    Parameters
    ----------
    order
        Correlation coefficient to be predicted.
    simplify
        Final simplification strategy.

    Returns
    -------
    Expr
        Exact symbolic prediction ``p_order``.
    """
    prediction, context = _predictor_field_element(order)

    return _field_element_to_expression(
        prediction,
        context,
        simplify=simplify,
    )


@lru_cache(maxsize=None)
def _bounds_field_elements(
    order: int,
) -> tuple[Any, Any, _RationalFunctionContext]:
    """
    Compute exact admissible bounds as fraction-field elements.
    """
    _validate_positive_integer(order, "order")

    if order == 1:
        context = _rational_function_context(1)
        return (
            -context.domain.one,
            context.domain.one,
            context,
        )

    previous_state = _symbolic_state_raw(order - 1)
    prediction, _ = _predictor_field_element(order)
    half_width = previous_state.sigma2[-1]

    return (
        prediction - half_width,
        prediction + half_width,
        previous_state.context,
    )


def admissible_bounds_symbolic(
    order: int,
    *,
    simplify: SimplificationMode = "none",
) -> tuple[sp.Expr, sp.Expr]:
    """
    Compute symbolic Schneider-Hartlap bounds.

    Parameters
    ----------
    order
        Correlation coefficient whose bounds are requested.
    simplify
        Final simplification strategy. The default ``"none"`` is
        recommended for higher orders.

    Returns
    -------
    r_lower
        Exact symbolic lower bound.
    r_upper
        Exact symbolic upper bound.
    """
    lower, upper, context = _bounds_field_elements(order)

    return (
        _field_element_to_expression(
            lower,
            context,
            simplify=simplify,
        ),
        _field_element_to_expression(
            upper,
            context,
            simplify=simplify,
        ),
    )


@lru_cache(maxsize=None)
def _sh_coordinate_field_element(
    order: int,
) -> tuple[Any, _RationalFunctionContext]:
    """
    Construct the Schneider-Hartlap coordinate as a field element.

    All quantities are evaluated directly in the common fraction field
    ``QQ(r1, ..., r_order)``. This avoids conversions between fraction
    fields of different orders.
    """
    _validate_positive_integer(order, "order")

    state = _symbolic_state_raw(order)
    context = state.context
    r = context.generators

    if order == 1:
        return r[0], context

    # Prediction coefficients of order n - 1, already represented in
    # QQ(r1, ..., r_n).
    phi = state.predictor_coefficients[order - 2]

    prediction = sum(
        (
            phi[j] * r[order - 2 - j]
            for j in range(order - 1)
        ),
        context.domain.zero,
    )

    # sigma_(n-1)^2 is the half-width of the admissible interval.
    half_width = state.sigma2[order - 1]

    lower = prediction - half_width
    upper = prediction + half_width

    coordinate = (
        2 * r[order - 1] - upper - lower
    ) / (
        upper - lower
    )

    return coordinate, context

def sh_coordinate_symbolic(
    order: int,
    *,
    simplify: SimplificationMode = "none",
) -> sp.Expr:
    """
    Compute a symbolic Schneider-Hartlap coordinate.

    Parameters
    ----------
    order
        Coordinate order.
    simplify
        Final simplification strategy.

    Returns
    -------
    Expr
        Exact symbolic expression for ``x_order``.
    """
    coordinate, context = _sh_coordinate_field_element(order)

    return _field_element_to_expression(
        coordinate,
        context,
        simplify=simplify,
    )


def verify_x_equals_alpha(
    order: int,
    *,
    method: SimplificationMode = "none",
) -> sp.Expr:
    """
    Verify exactly that ``x_order = alpha_order``.

    Parameters
    ----------
    order
        Recursion order to verify.
    method
        Optional final simplification strategy if a non-zero symbolic
        difference were encountered.

    Returns
    -------
    Expr
        Zero when the two exact fraction-field elements are identical.

    Notes
    -----
    Equality is tested before conversion to ordinary SymPy expressions.
    This avoids constructing and simplifying a potentially very large
    difference expression.
    """
    coordinate, context = _sh_coordinate_field_element(order)
    alpha_n = _symbolic_state_raw(order).alpha[-1]

    difference = coordinate - alpha_n

    if difference == context.domain.zero:
        return sp.Integer(0)

    return _field_element_to_expression(
        difference,
        context,
        simplify=method,
    )


def common_subexpressions(
    expression: sp.Expr,
) -> tuple[list[tuple[sp.Symbol, sp.Expr]], list[sp.Expr]]:
    """
    Perform common-subexpression elimination.

    Parameters
    ----------
    expression
        Symbolic expression to compress.

    Returns
    -------
    replacements
        Temporary symbols and their definitions.
    reduced_expressions
        Expressions written in terms of the temporary symbols.

    Notes
    -----
    Common-subexpression elimination is useful for displaying or exporting
    large high-order formulas. It does not alter the exact cached recursion.
    """
    replacements, reduced = sp.cse(
        expression,
        optimizations="basic",
    )

    return replacements, reduced


@dataclass(frozen=True, slots=True)
class CompactSymbolicSystem:
    """
    Compact symbolic representation of a Levinson recursion.

    The system stores the recursion as a directed acyclic graph of named
    intermediate quantities rather than eliminating all intermediate
    symbols into one large rational expression.

    Parameters
    ----------
    order
        Highest recursion order.
    correlations
        Correlation symbols ``(r1, ..., rN)``.
    definitions
        Ordered symbolic definitions. Each right-hand side depends only
        on correlation coefficients and symbols defined earlier.
    alpha
        PACF symbols ``(alpha_1, ..., alpha_N)``.
    sigma2
        Innovation-variance symbols
        ``(sigma2_0, ..., sigma2_N)``.
    prediction
        Prediction symbols ``(p_1, ..., p_N)``.
    lower
        Lower admissible-bound symbols.
    upper
        Upper admissible-bound symbols.
    sh_coordinate
        Schneider-Hartlap coordinate symbols.
    predictor_coefficients
        Prediction-coefficient symbols at each recursion order.
    """

    order: int
    correlations: tuple[sp.Symbol, ...]
    definitions: tuple[sp.Equality, ...]
    alpha: tuple[sp.Symbol, ...]
    sigma2: tuple[sp.Symbol, ...]
    prediction: tuple[sp.Symbol, ...]
    lower: tuple[sp.Symbol, ...]
    upper: tuple[sp.Symbol, ...]
    sh_coordinate: tuple[sp.Symbol, ...]
    predictor_coefficients: tuple[
        tuple[sp.Symbol, ...],
        ...
    ]

    def definition_map(self) -> dict[sp.Symbol, sp.Expr]:
        """
        Return the symbolic definitions as a dictionary.

        Returns
        -------
        dict
            Mapping from each intermediate symbol to its defining
            expression.
        """
        return {
            equation.lhs: equation.rhs
            for equation in self.definitions
        }

    def as_text(self) -> str:
        """
        Return the complete recursion as plain SymPy text.

        Returns
        -------
        str
            One symbolic definition per line.
        """
        return "\n".join(
            sp.sstr(equation)
            for equation in self.definitions
        )

    def as_latex(self) -> str:
        """
        Return the recursion as a LaTeX aligned environment.

        Returns
        -------
        str
            LaTeX representation of all symbolic definitions.
        """
        lines = [
            sp.latex(equation)
            for equation in self.definitions
        ]

        body = r" \\" + "\n"
        body = body.join(lines)

        return (
            "\\begin{aligned}\n"
            f"{body}\n"
            "\\end{aligned}"
        )


@lru_cache(maxsize=None)
def compact_symbolic_system(
    order: int,
) -> CompactSymbolicSystem:
    """
    Construct a compact exact symbolic Levinson recursion.

    Parameters
    ----------
    order
        Highest recursion order.

    Returns
    -------
    CompactSymbolicSystem
        Symbolic recursion graph through the requested order.

    Notes
    -----
    No recursive substitution is performed. The returned system therefore
    avoids the expression swell of fully explicit formulas.

    The number of stored prediction-coefficient updates grows as
    ``O(order**2)``.
    """
    _validate_positive_integer(order, "order")

    r = correlation_symbols(order)

    definitions: list[sp.Equality] = []

    alpha_symbols: list[sp.Symbol] = []
    sigma2_symbols: list[sp.Symbol] = [
        sp.Symbol("sigma2_0", positive=True)
    ]
    prediction_symbols: list[sp.Symbol] = []
    lower_symbols: list[sp.Symbol] = []
    upper_symbols: list[sp.Symbol] = []
    coordinate_symbols: list[sp.Symbol] = []

    predictor_rows: list[tuple[sp.Symbol, ...]] = []

    definitions.append(
        sp.Eq(
            sigma2_symbols[0],
            sp.Integer(1),
            evaluate=False,
        )
    )

    previous_phi: tuple[sp.Symbol, ...] = ()

    for n in range(1, order + 1):
        prediction_n = sp.Symbol(f"p_{n}")
        alpha_n = sp.Symbol(f"alpha_{n}")
        lower_n = sp.Symbol(f"r_lower_{n}")
        upper_n = sp.Symbol(f"r_upper_{n}")
        coordinate_n = sp.Symbol(f"x_{n}")
        sigma2_n = sp.Symbol(
            f"sigma2_{n}",
            nonnegative=True,
        )

        if n == 1:
            prediction_expression = sp.Integer(0)
        else:
            prediction_expression = sum(
                (
                    previous_phi[j] * r[n - 2 - j]
                    for j in range(n - 1)
                ),
                sp.Integer(0),
            )

        definitions.append(
            sp.Eq(
                prediction_n,
                prediction_expression,
                evaluate=False,
            )
        )

        definitions.append(
            sp.Eq(
                lower_n,
                prediction_n - sigma2_symbols[n - 1],
                evaluate=False,
            )
        )
        definitions.append(
            sp.Eq(
                upper_n,
                prediction_n + sigma2_symbols[n - 1],
                evaluate=False,
            )
        )

        definitions.append(
            sp.Eq(
                alpha_n,
                (
                    r[n - 1] - prediction_n
                ) / sigma2_symbols[n - 1],
                evaluate=False,
            )
        )

        definitions.append(
            sp.Eq(
                coordinate_n,
                (
                    2 * r[n - 1]
                    - upper_n
                    - lower_n
                ) / (
                    upper_n - lower_n
                ),
                evaluate=False,
            )
        )

        phi_row = tuple(
            sp.Symbol(f"phi_{n}_{j}")
            for j in range(1, n + 1)
        )

        if n == 1:
            definitions.append(
                sp.Eq(
                    phi_row[0],
                    alpha_n,
                    evaluate=False,
                )
            )
        else:
            for j in range(n - 1):
                definitions.append(
                    sp.Eq(
                        phi_row[j],
                        previous_phi[j]
                        - alpha_n
                        * previous_phi[n - 2 - j],
                        evaluate=False,
                    )
                )

            definitions.append(
                sp.Eq(
                    phi_row[-1],
                    alpha_n,
                    evaluate=False,
                )
            )

        definitions.append(
            sp.Eq(
                sigma2_n,
                sigma2_symbols[n - 1]
                * (1 - alpha_n**2),
                evaluate=False,
            )
        )

        prediction_symbols.append(prediction_n)
        alpha_symbols.append(alpha_n)
        lower_symbols.append(lower_n)
        upper_symbols.append(upper_n)
        coordinate_symbols.append(coordinate_n)
        sigma2_symbols.append(sigma2_n)
        predictor_rows.append(phi_row)

        previous_phi = phi_row

    return CompactSymbolicSystem(
        order=order,
        correlations=r,
        definitions=tuple(definitions),
        alpha=tuple(alpha_symbols),
        sigma2=tuple(sigma2_symbols),
        prediction=tuple(prediction_symbols),
        lower=tuple(lower_symbols),
        upper=tuple(upper_symbols),
        sh_coordinate=tuple(coordinate_symbols),
        predictor_coefficients=tuple(predictor_rows),
    )


def compact_pacf_symbolic(
    order: int,
) -> tuple[sp.Symbol, CompactSymbolicSystem]:
    """
    Return a compact symbolic representation of ``alpha_order``.

    Parameters
    ----------
    order
        Partial autocorrelation order.

    Returns
    -------
    alpha
        Symbol representing ``alpha_order``.
    system
        Ordered symbolic definitions needed to evaluate the symbol.
    """
    system = compact_symbolic_system(order)
    return system.alpha[-1], system


def compact_bounds_symbolic(
    order: int,
) -> tuple[
    sp.Symbol,
    sp.Symbol,
    CompactSymbolicSystem,
]:
    """
    Return compact symbolic bounds for ``r_order``.

    Parameters
    ----------
    order
        Correlation order.

    Returns
    -------
    r_lower
        Symbol representing the lower admissible bound.
    r_upper
        Symbol representing the upper admissible bound.
    system
        Ordered symbolic definitions needed to evaluate the bounds.
    """
    system = compact_symbolic_system(order)

    return (
        system.lower[-1],
        system.upper[-1],
        system,
    )


def verify_compact_x_equals_alpha(order: int) -> sp.Expr:
    """
    Verify ``x_order = alpha_order`` without recursive expansion.

    Parameters
    ----------
    order
        Recursion order.

    Returns
    -------
    Expr
        Zero when the identity is verified.

    Notes
    -----
    Only the local definitions of the bounds and coordinates are
    substituted. Earlier Levinson recursion steps are deliberately left
    unexpanded.
    """
    system = compact_symbolic_system(order)
    definitions = system.definition_map()

    x_n = system.sh_coordinate[-1]
    alpha_n = system.alpha[-1]
    lower_n = system.lower[-1]
    upper_n = system.upper[-1]

    x_expression = definitions[x_n].subs(
        {
            lower_n: definitions[lower_n],
            upper_n: definitions[upper_n],
        }
    )

    alpha_expression = definitions[alpha_n]

    return sp.cancel(
        x_expression - alpha_expression
    )
