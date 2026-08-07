"""Levinson--Durbin transformations between correlations and PACFs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

# Absorbs float64 roundoff around exact admissibility/boundary values (see
# DOCUMENTATION.md, "Tolerances"); unrelated to
# schurcorr.bounds._BOUNDARY_CONTINUATION_TOL, which validates
# already-computed forced continuations rather than roundoff at the
# boundary itself.
_ROUNDING_TOL = 1.0e-12


class SingularToeplitzError(Exception):
    """Raised when the PACF recursion reaches a degenerate boundary.

    The sequence is admissible, but the ``r <-> alpha`` bijection ends
    there; see ``docs/boundary_semantics.md``.
    """


def _asarray1d(x: ArrayLike, *, name: str) -> FloatArray:
    """Return input as a contiguous one-dimensional float64 array."""
    array = np.asarray(x, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    return np.ascontiguousarray(array)


def _as_sequence_1d(x: ArrayLike, *, name: str) -> list:
    """Validate one-dimensional input without converting its elements."""
    # dtype=object avoids a float64 round-trip, which would silently
    # discard precision beyond float64 (e.g. mpmath.mpf input elements).
    array = np.asarray(x, dtype=object)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    return list(array)


def _asarray_batchable(x: ArrayLike, *, name: str) -> FloatArray:
    """Return input as a contiguous one- or two-dimensional float64 array."""
    array = np.asarray(x, dtype=np.float64)

    if array.ndim not in (1, 2):
        raise ValueError(
            f"{name} must be a 1-D array (N,) or a 2-D batch "
            f"(n_samples, N); got ndim={array.ndim}."
        )

    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class _LevinsonResult:
    """Terminal data returned by a scalar Levinson recursion."""

    r: FloatArray
    alpha: FloatArray
    sigma2: FloatArray
    prediction: FloatArray
    reached_boundary: bool
    # Terminal predictor coefficients, only meaningful (non-None) once the
    # boundary is reached; see schurcorr.bounds.extend_at_boundary.
    terminal_phi: FloatArray | None = None


def _run_levinson_from_correlations(r: ArrayLike) -> _LevinsonResult:
    """Run the canonical scalar correlation-to-PACF recursion.

    Shared by :func:`pacf`, :func:`pacf_prefix`, and
    :mod:`schurcorr.bounds`. Proceeds as far as the mathematics allows and
    reports how far via ``reached_boundary`` rather than raising or
    warning on a degenerate-but-admissible input; still raises
    :class:`ValueError` for a genuinely inadmissible input (see
    ``docs/boundary_semantics.md``).
    """
    r_array = _asarray1d(r, name="r")
    n_max = r_array.size

    alpha_values: list[float] = []
    sigma2_values: list[float] = [1.0]
    predictions: list[float] = []

    phi_buffer = np.empty(n_max, dtype=np.float64)
    reached_boundary = False

    for n in range(1, n_max + 1):
        if sigma2_values[-1] <= _ROUNDING_TOL:
            reached_boundary = True
            break

        if n == 1:
            prediction = 0.0
            alpha_n = float(r_array[0])
        else:
            prediction = float(
                np.dot(
                    phi_buffer[: n - 1],
                    r_array[n - 2 :: -1],
                )
            )
            alpha_n = (
                float(r_array[n - 1]) - prediction
            ) / sigma2_values[-1]

        if abs(alpha_n) > 1.0 + _ROUNDING_TOL:
            raise ValueError(
                "Input correlations are not admissible: "
                f"computed abs(alpha_{n}) = "
                f"{abs(alpha_n):.6g} > 1."
            )

        at_boundary_hit = abs(alpha_n) >= 1.0
        if at_boundary_hit:
            alpha_n = float(np.sign(alpha_n))

        if n == 1:
            phi_buffer[0] = alpha_n
        else:
            phi_prev = phi_buffer[: n - 1].copy()
            phi_buffer[: n - 1] = phi_prev - alpha_n * phi_prev[::-1]
            phi_buffer[n - 1] = alpha_n

        if at_boundary_hit:
            alpha_values.append(alpha_n)
            predictions.append(prediction)
            sigma2_values.append(0.0)
            reached_boundary = True
            break

        sigma2_next = (
            sigma2_values[-1]
            * (1.0 - alpha_n * alpha_n)
        )

        if sigma2_next < 0.0 and sigma2_next > -_ROUNDING_TOL:
            sigma2_next = 0.0

        alpha_values.append(float(alpha_n))
        predictions.append(prediction)
        sigma2_values.append(float(sigma2_next))

    number_computed = len(alpha_values)

    return _LevinsonResult(
        r=r_array[:number_computed].copy(),
        alpha=np.asarray(alpha_values, dtype=np.float64),
        sigma2=np.asarray(sigma2_values, dtype=np.float64),
        prediction=np.asarray(predictions, dtype=np.float64),
        reached_boundary=reached_boundary,
        terminal_phi=(
            phi_buffer[:number_computed].copy()
            if reached_boundary
            else None
        ),
    )


def _run_levinson_from_pacf(alpha: ArrayLike) -> _LevinsonResult:
    """Run the canonical scalar PACF-to-correlation recursion.

    Cannot reach the singular boundary by construction (``abs(alpha_n) <
    1`` is validated up front), so ``reached_boundary`` is always
    ``False``.
    """
    alpha_array = _asarray1d(alpha, name="alpha")

    if np.any(np.abs(alpha_array) >= 1.0):
        raise ValueError(
            "All PACF coefficients must satisfy abs(alpha_n) < 1."
        )

    n_max = alpha_array.size

    # r_buffer[0] is the fixed r_0 = 1 term; r_buffer[1:n] holds
    # r_1, ..., r_(n-1) once computed, so r_buffer[:n] is exactly
    # the "r_with_zero" vector needed at step n -- avoids
    # rebuilding it via np.concatenate at every step.
    r_buffer = np.empty(n_max + 1, dtype=np.float64)
    r_buffer[0] = 1.0

    sigma2_values: list[float] = [1.0]
    predictions: list[float] = []

    phi_buffer = np.empty(n_max, dtype=np.float64)

    for n in range(1, n_max + 1):
        alpha_n = float(alpha_array[n - 1])

        if n == 1:
            phi_buffer[0] = alpha_n
            prediction = 0.0
            r_n = alpha_n
        else:
            phi_prev = phi_buffer[: n - 1].copy()

            # p_n uses the order-n predictor phi^{(n)} (phi_prev, not yet
            # updated to order n+1), matched against r_{n-1}, ..., r_1.
            prediction = float(
                np.dot(
                    phi_prev,
                    r_buffer[n - 1 : 0 : -1],
                )
            )
            r_n = prediction + alpha_n * sigma2_values[-1]

            phi_buffer[: n - 1] = phi_prev - alpha_n * phi_prev[::-1]
            phi_buffer[n - 1] = alpha_n

        r_buffer[n] = r_n
        predictions.append(prediction)

        sigma2_next = (
            sigma2_values[-1]
            * (1.0 - alpha_n * alpha_n)
        )

        if sigma2_next < 0.0 and sigma2_next > -_ROUNDING_TOL:
            warnings.warn(
                "Innovation variance became slightly negative due "
                "to roundoff and was clamped to zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            sigma2_next = 0.0

        sigma2_values.append(float(sigma2_next))

    return _LevinsonResult(
        r=r_buffer[1:].copy(),
        alpha=alpha_array.copy(),
        sigma2=np.asarray(sigma2_values, dtype=np.float64),
        prediction=np.asarray(predictions, dtype=np.float64),
        reached_boundary=False,
    )


def _pacf_2d_fast(r_array: FloatArray) -> FloatArray | None:
    """Run the PACF recursion vectorized across batch rows.

    Not guaranteed bit-identical to the per-row scalar path; see
    ``docs/development_notes.md``.
    """
    n_samples, n_max = r_array.shape

    alpha = np.empty((n_samples, n_max), dtype=np.float64)
    phi = np.empty((n_samples, n_max), dtype=np.float64)
    sigma2 = np.ones(n_samples, dtype=np.float64)

    for n in range(1, n_max + 1):
        idx = n - 1

        if np.any(sigma2 <= _ROUNDING_TOL):
            # Any row at or past the boundary: fall back to the per-row
            # scalar path, which raises SingularToeplitzError with the
            # offending sample index instead of partial/incorrect output.
            return None

        if n == 1:
            prediction = np.zeros(n_samples, dtype=np.float64)
        else:
            prediction = np.sum(
                phi[:, : n - 1] * r_array[:, n - 2 :: -1], axis=1
            )

        alpha_n = (r_array[:, idx] - prediction) / sigma2

        if np.any(np.abs(alpha_n) >= 1.0):
            return None

        alpha[:, idx] = alpha_n

        if n == 1:
            phi[:, 0] = alpha_n
        else:
            phi_prev = phi[:, : n - 1].copy()
            phi[:, : n - 1] = phi_prev - alpha_n[:, None] * phi_prev[:, ::-1]
            phi[:, n - 1] = alpha_n

        sigma2 = sigma2 * (1.0 - alpha_n * alpha_n)

    return alpha


def _from_pacf_2d(alpha_array: FloatArray) -> FloatArray:
    """Run the inverse recursion vectorized across batch rows.

    Unlike :func:`_pacf_2d_fast`, no fallback is needed: ``abs(alpha_n) <
    1`` (validated up front for every row) keeps the innovation variance
    strictly positive by construction, so the boundary is unreachable.
    """
    n_samples, n_max = alpha_array.shape

    bad_mask = np.any(np.abs(alpha_array) >= 1.0, axis=1)
    if np.any(bad_mask):
        first_bad = int(np.argmax(bad_mask))
        raise ValueError(
            f"sample {first_bad}: All PACF coefficients must satisfy "
            "abs(alpha_n) < 1."
        )

    # r_buffer[:, 0] is the fixed r_0 = 1 column; r_buffer[:, 1:n]
    # holds r_1, ..., r_(n-1) once computed (see _run_levinson_from_pacf).
    r_buffer = np.empty((n_samples, n_max + 1), dtype=np.float64)
    r_buffer[:, 0] = 1.0
    phi = np.empty((n_samples, n_max), dtype=np.float64)

    for n in range(1, n_max + 1):
        idx = n - 1
        alpha_n = alpha_array[:, idx]

        if n == 1:
            phi[:, 0] = alpha_n
            r_buffer[:, 1] = alpha_n
        else:
            phi_prev = phi[:, : n - 1].copy()
            phi[:, : n - 1] = phi_prev - alpha_n[:, None] * phi_prev[:, ::-1]
            phi[:, n - 1] = alpha_n

            r_buffer[:, n] = np.sum(
                phi[:, :n] * r_buffer[:, n - 1 :: -1], axis=1
            )

    return r_buffer[:, 1:].copy()


@dataclass(frozen=True, slots=True)
class PrefixResult:
    """
    Boundary analysis result for :func:`pacf_prefix` -- the maximal
    independent PACF prefix together with the recursion state needed to
    continue past a degenerate boundary (see docs/boundary_semantics.md,
    category 2: "mathematically degenerate boundary point").

    Attributes
    ----------
    alpha
        The independent PACF prefix ``(alpha_1, ..., alpha_order)``. If
        ``reached_boundary``, the last entry is exactly ``+1`` or ``-1``.
    order
        Number of coefficients in ``alpha``.
    reached_boundary
        Whether the recursion stopped because it hit the singular
        boundary (``sigma_order^2 = 0``) rather than exhausting ``r``.
    sigma2
        Innovation variances ``(sigma_0^2, ..., sigma_order^2)``.
    predictor
        Terminal Levinson--Durbin predictor coefficients at the boundary
        (see :func:`schurcorr.bounds.extend_at_boundary`), or ``None``
        if the boundary was not reached.
    """

    alpha: FloatArray
    order: int
    reached_boundary: bool
    sigma2: FloatArray
    predictor: FloatArray | None


def pacf_prefix(r: ArrayLike) -> PrefixResult:
    """
    Return the maximal PACF prefix of an admissible sequence.

    Use this function instead of :func:`pacf` when the sequence may reach
    a degenerate boundary: unlike :func:`pacf`, it never raises
    :class:`SingularToeplitzError` for a degenerate-but-admissible ``r``.

    Parameters
    ----------
    r
        One-dimensional correlation sequence.

    Returns
    -------
    PrefixResult
        PACF prefix and boundary information.

    Raises
    ------
    ValueError
        If ``r`` is not admissible.
    """
    r_array = _asarray1d(r, name="r")
    result = _run_levinson_from_correlations(r_array)

    return PrefixResult(
        alpha=result.alpha,
        order=result.alpha.size,
        reached_boundary=result.reached_boundary,
        sigma2=result.sigma2,
        predictor=result.terminal_phi,
    )


def _pacf_1d_strict(r_array: FloatArray) -> FloatArray:
    result = _run_levinson_from_correlations(r_array)

    if result.reached_boundary:
        n = result.alpha.size
        raise SingularToeplitzError(
            f"The Toeplitz matrix of order {n} is singular "
            f"(sigma_{n}^2 = 0); the r <-> alpha bijection breaks down at "
            "this point. Use pacf_prefix(r) for the boundary-inclusive "
            "prefix or extend_at_boundary(r, n_extra) for the recurrence-"
            "forced continuation."
        )

    return result.alpha


def pacf(r: ArrayLike) -> FloatArray:
    """
    Convert correlation coefficients to partial autocorrelations.

    Parameters
    ----------
    r
        Correlation coefficients with shape ``(N,)`` or ``(M, N)``. For a
        2-D batch, each row is treated as an independent sequence; a row
        that raises reports its own index in the error message.

    Returns
    -------
    numpy.ndarray
        Partial autocorrelations with the same shape as ``r``.

    Raises
    ------
    SingularToeplitzError
        If the recursion reaches a degenerate boundary (``sigma_n^2 =
        0``); use :func:`pacf_prefix` for the boundary-inclusive prefix,
        or :func:`schurcorr.bounds.extend_at_boundary` for the
        deterministic continuation past it.
    ValueError
        If ``r`` is not an admissible correlation sequence.

    Examples
    --------
    >>> r = from_pacf([0.5, -0.3, 0.2])
    >>> pacf(r)
    array([ 0.5, -0.3,  0.2])
    """
    r_array = _asarray_batchable(r, name="r")

    if r_array.ndim == 1:
        return _pacf_1d_strict(r_array)

    fast_result = _pacf_2d_fast(r_array)
    if fast_result is not None:
        return fast_result

    n_samples, _ = r_array.shape
    alpha_rows: list[FloatArray] = []

    for i in range(n_samples):
        try:
            alpha_rows.append(_pacf_1d_strict(r_array[i]))
        except SingularToeplitzError as error:
            raise SingularToeplitzError(f"sample {i}: {error}") from error
        except ValueError as error:
            raise ValueError(f"sample {i}: {error}") from error

    return np.stack(alpha_rows)


def from_pacf(alpha: ArrayLike) -> FloatArray:
    """
    Convert partial autocorrelations to correlation coefficients.

    Parameters
    ----------
    alpha
        Partial autocorrelations with shape ``(N,)`` or ``(M, N)``.
        Entries must lie strictly between -1 and 1. For a 2-D batch,
        each row is treated as an independent sequence.

    Returns
    -------
    numpy.ndarray
        Correlation coefficients with the same shape as ``alpha``.

    Raises
    ------
    ValueError
        If an entry lies outside ``(-1, 1)``.
    """
    alpha_array = _asarray_batchable(alpha, name="alpha")

    if alpha_array.ndim == 1:
        return _run_levinson_from_pacf(alpha_array).r

    return _from_pacf_2d(alpha_array)
