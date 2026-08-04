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
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoundaryMode = Literal["raise", "warn", "extend"]
Backend = Literal["float64", "mpmath"]

_TOL = 1.0e-12


class SingularToeplitzError(Exception):
    """
    Raised when the Levinson--Durbin recursion reaches a singular
    Toeplitz matrix, ``sigma_n^2 = 0``, while ``at_boundary='raise'``.

    At this point the angle between the two endpoint residuals that
    defines ``alpha_n`` is undefined (one residual is the zero
    vector), so the ``r <-> alpha`` bijection breaks down. This is
    distinct from a :class:`ValueError` on malformed or inadmissible
    input: it signals a boundary reached *during* an otherwise valid
    recursion.
    """


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


def _as_sequence_1d(x: ArrayLike, *, name: str) -> list:
    """
    Validate 1-D sequence input without forcing ``float64`` precision.

    Used for ``backend='mpmath'`` input that may already carry
    ``mpmath.mpf`` values (e.g. the output of a prior
    ``from_pacf(..., backend='mpmath')`` call): unlike
    :func:`_asarray1d`, this does not round-trip through a
    ``float64`` array, which would silently discard everything past
    ``float64`` precision.

    Parameters
    ----------
    x
        Input values.
    name
        Argument name used in error messages.

    Returns
    -------
    list
        The input as a plain Python list, elements unchanged.

    Raises
    ------
    ValueError
        If the input is not one-dimensional.
    """
    array = np.asarray(x, dtype=object)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    return list(array)


def _asarray_batchable(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Convert array-like input to a contiguous float array of ndim 1 or 2.

    Parameters
    ----------
    x
        Input values, either a 1-D sequence ``(N,)`` or a 2-D batch
        ``(n_samples, N)``.
    name
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        Contiguous array, unchanged in ndim (1 or 2).

    Raises
    ------
    ValueError
        If the input is neither one- nor two-dimensional.
    """
    array = np.asarray(x, dtype=np.float64)

    if array.ndim not in (1, 2):
        raise ValueError(
            f"{name} must be a 1-D array (N,) or a 2-D batch "
            f"(n_samples, N); got ndim={array.ndim}."
        )

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
    def from_correlations(
        cls,
        r: ArrayLike,
        *,
        at_boundary: BoundaryMode = "raise",
    ) -> "_LevinsonState":
        """
        Construct the recursion state from correlation coefficients.

        Parameters
        ----------
        r
            Normalized correlation coefficients ``(r_1, ..., r_N)``.
        at_boundary
            Behaviour when the recursion reaches a singular Toeplitz
            matrix (``sigma_n^2 = 0``): ``'raise'`` raises
            :class:`SingularToeplitzError`; ``'warn'`` clamps, warns,
            and returns the non-degenerate part only. ``'extend'`` is
            handled by the caller (see :func:`pacf`).
        """
        r_array = _asarray1d(r, name="r")
        n_max = r_array.size

        alpha_values: list[float] = []
        sigma2_values: list[float] = [1.0]
        predictors: list[FloatArray] = []

        # phi_buffer[:n] holds the order-n predictor coefficients;
        # reused in place across orders instead of reallocating a new
        # array at every step (the per-order snapshots callers need,
        # e.g. admissible_bounds, are still taken via .copy() below).
        phi_buffer = np.empty(n_max, dtype=np.float64)

        for n in range(1, n_max + 1):
            if sigma2_values[-1] <= _TOL:
                if at_boundary == "raise":
                    raise SingularToeplitzError(
                        f"The Toeplitz matrix of order {n - 1} is "
                        f"singular (sigma_{n - 1}^2 = 0); the r <-> "
                        "alpha bijection breaks down at this point. "
                        "Pass at_boundary='warn' to obtain the "
                        "truncated result or at_boundary='extend' for "
                        "the recurrence-forced continuation."
                    )
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
                        phi_buffer[: n - 1],
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

            at_boundary_hit = abs(alpha_n) >= 1.0
            if at_boundary_hit:
                if at_boundary == "raise":
                    raise SingularToeplitzError(
                        f"sigma_{n}^2 = 0 reached at index {n} "
                        f"(alpha_{n} = {np.sign(alpha_n):+.0f}); the "
                        "r <-> alpha bijection breaks down at this "
                        "point. Pass at_boundary='warn' to obtain the "
                        "boundary-inclusive truncated result or "
                        "at_boundary='extend' for the recurrence-"
                        "forced continuation."
                    )
                alpha_n = float(np.sign(alpha_n))

            if n == 1:
                phi_buffer[0] = alpha_n
            else:
                phi_prev = phi_buffer[: n - 1].copy()
                phi_buffer[: n - 1] = phi_prev - alpha_n * phi_prev[::-1]
                phi_buffer[n - 1] = alpha_n

            if at_boundary_hit:
                alpha_values.append(alpha_n)
                predictors.append(phi_buffer[:n].copy())
                sigma2_values.append(0.0)

                warnings.warn(
                    "Boundary point encountered with abs(alpha_n) = 1; "
                    "subsequent coefficients are uniquely determined.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

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
            predictors.append(phi_buffer[:n].copy())

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

        n_max = alpha_array.size

        # r_buffer[0] is the fixed r_0 = 1 term; r_buffer[1:n] holds
        # r_1, ..., r_(n-1) once computed, so r_buffer[:n] is exactly
        # the "r_with_zero" vector needed at step n -- avoids
        # rebuilding it via np.concatenate at every step.
        r_buffer = np.empty(n_max + 1, dtype=np.float64)
        r_buffer[0] = 1.0

        sigma2_values: list[float] = [1.0]
        predictors: list[FloatArray] = []

        phi_buffer = np.empty(n_max, dtype=np.float64)

        for n in range(1, n_max + 1):
            alpha_n = float(alpha_array[n - 1])

            if n == 1:
                phi_buffer[0] = alpha_n
                r_n = alpha_n
            else:
                phi_prev = phi_buffer[: n - 1].copy()
                phi_buffer[: n - 1] = phi_prev - alpha_n * phi_prev[::-1]
                phi_buffer[n - 1] = alpha_n

                r_n = float(
                    np.dot(
                        phi_buffer[:n],
                        r_buffer[n - 1 :: -1],
                    )
                )

            r_buffer[n] = r_n

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

            sigma2_values.append(float(sigma2_next))
            predictors.append(phi_buffer[:n].copy())

        return cls(
            r=r_buffer[1:].copy(),
            alpha=alpha_array.copy(),
            sigma2=np.asarray(
                sigma2_values,
                dtype=np.float64,
            ),
            predictor_coefficients=predictors,
        )


def pacf(
    r: ArrayLike,
    *,
    backend: Backend = "float64",
    at_boundary: BoundaryMode = "raise",
    dps: int | None = None,
) -> FloatArray | list:
    """
    Compute partial autocorrelations from correlation coefficients.

    Parameters
    ----------
    r
        Normalized correlation coefficients, either a 1-D array
        ``(r_1, ..., r_N)`` or a 2-D batch of ``n_samples`` such
        sequences with shape ``(n_samples, N)``. For a 2-D batch, the
        recursion is applied independently to each row (the same
        recursion as for the 1-D case, not a different algorithm),
        vectorized across rows for performance whenever no row reaches
        the singular boundary; a batch with a boundary-reaching row
        falls back internally to the per-row scalar path so that
        ``at_boundary`` is honoured exactly. The vectorized path is
        not guaranteed bit-identical to looping :func:`pacf` over rows
        -- the summation order of the internal prediction differs by
        up to a few ULP -- though both compute the same recursion.
        ``backend='mpmath'`` only supports 1-D input.
    backend
        ``'float64'`` (default) runs the plain ``numpy`` bijection in
        this module. ``'mpmath'`` runs the arbitrary-precision
        recursion in :mod:`schurcorr._levinson_mp` instead, a
        numerical-conditioning stress test rather than a faster or
        more "correct" version of the same computation -- see that
        module's docstring. With ``backend='mpmath'``, the return
        value is a ``list`` of ``mpmath.mpf`` (not downcast to
        ``float64``), and only ``at_boundary in ('raise', 'warn')``
        is supported.
    at_boundary
        Behaviour when the recursion reaches a singular Toeplitz
        matrix, i.e. ``sigma_n^2 = 0`` for some ``n <= N`` (see
        :class:`SingularToeplitzError`). Applied per row for a 2-D
        batch.

        - ``'raise'`` (default): raise :class:`SingularToeplitzError`.
          The PACF/bijection view and the ``r``-sequence view disagree
          at this boundary, so the default forces the caller to
          decide explicitly how to proceed.
        - ``'warn'``: clamp, emit a ``RuntimeWarning``, and return the
          non-degenerate part of ``alpha`` only. For a 1-D input, its
          length is then less than ``N``; for a 2-D batch, an affected
          row is padded with ``NaN`` past its truncation point so the
          returned array remains rectangular.
        - ``'extend'``: return a full-length ``alpha`` in which the
          coefficients past the boundary are the unique continuation
          forced by the Toeplitz structure (see
          :func:`schurcorr.sh_bounds.extend_at_boundary`). These
          boundary coefficients are exactly ``+1`` or ``-1`` (the
          sign fixed at the first boundary hit) and are *not*
          interior PACF values in ``(-1, 1)`` -- ``sigma_n^2`` stays
          exactly zero from that order on, so the bijection with an
          interior ``alpha_n`` no longer applies.
    dps
        Working precision in decimal places, only used when
        ``backend='mpmath'``. If ``None`` (default),
        ``recommended_dps(len(r))`` is used.

    Returns
    -------
    alpha
        Partial autocorrelation coefficients. For ``backend='float64'``,
        the same shape as ``r``: ``(alpha_1, ..., alpha_N)`` or
        ``(n_samples, N)``. For ``backend='mpmath'``, a ``list`` of
        ``mpmath.mpf``, length ``N`` (or less, under
        ``at_boundary='warn'``).

    Raises
    ------
    SingularToeplitzError
        If ``at_boundary='raise'`` (the default) and the recursion
        reaches a singular Toeplitz matrix (in any row, for a 2-D
        batch).
    """
    if backend not in ("float64", "mpmath"):
        raise ValueError(
            f"backend must be 'float64' or 'mpmath'; got {backend!r}."
        )

    if backend == "mpmath":
        if at_boundary not in ("raise", "warn"):
            raise ValueError(
                "at_boundary='extend' is not supported for "
                "backend='mpmath'; use backend='float64'."
            )

        from . import _levinson_mp

        r_list = _as_sequence_1d(r, name="r")
        return _levinson_mp._pacf_mp(
            r_list, dps=dps, at_boundary=at_boundary
        )

    if dps is not None:
        raise ValueError("dps is only used when backend='mpmath'.")

    if at_boundary not in ("raise", "warn", "extend"):
        raise ValueError(
            "at_boundary must be one of 'raise', 'warn', 'extend'; "
            f"got {at_boundary!r}."
        )

    r_array = _asarray_batchable(r, name="r")

    if r_array.ndim == 1:
        return _pacf_1d(r_array, at_boundary)

    fast_result = _pacf_2d_fast(r_array)
    if fast_result is not None:
        return fast_result

    # At least one row reaches the singular boundary or an
    # inadmissible value; fall back to the per-row scalar path, which
    # implements the full at_boundary semantics (raise / warn+NaN-pad
    # / extend) exactly.
    n_samples, n_max = r_array.shape
    alpha_rows: list[FloatArray] = []

    for i in range(n_samples):
        try:
            row_alpha = _pacf_1d(r_array[i], at_boundary)
        except SingularToeplitzError as error:
            raise SingularToeplitzError(f"sample {i}: {error}") from error
        except ValueError as error:
            raise ValueError(f"sample {i}: {error}") from error

        if row_alpha.size < n_max:
            padded = np.full(n_max, np.nan, dtype=np.float64)
            padded[: row_alpha.size] = row_alpha
            row_alpha = padded

        alpha_rows.append(row_alpha)

    return np.stack(alpha_rows)


def _pacf_2d_fast(r_array: FloatArray) -> FloatArray | None:
    """
    Vectorized fast path for :func:`pacf` over a 2-D batch.

    Runs the same recursion as :meth:`_LevinsonState.from_correlations`,
    vectorized across the batch (sample) dimension instead of looping
    over rows in Python. Returns ``None`` -- instead of partial or
    incorrect output -- the moment any row would reach the singular
    boundary or an inadmissible value, since the correct behaviour
    there depends on ``at_boundary`` (raise / warn+NaN-pad / extend);
    the caller then falls back to the per-row scalar path for the
    whole batch, which already implements that in full. When no row
    hits the boundary, ``at_boundary`` is irrelevant (all three modes
    agree on non-degenerate input), so this function does not need to
    know which mode was requested.

    Not guaranteed bit-identical to the per-row scalar path: the
    prediction ``sum(phi * r, axis=1)`` here uses a different
    floating-point summation order than the scalar path's ``np.dot``,
    which can differ by up to a few ULP. Both compute the same
    recursion (see the module-level docstring); this is a
    summation-order effect, not a different algorithm.
    """
    n_samples, n_max = r_array.shape

    alpha = np.empty((n_samples, n_max), dtype=np.float64)
    phi = np.empty((n_samples, n_max), dtype=np.float64)
    sigma2 = np.ones(n_samples, dtype=np.float64)

    for n in range(1, n_max + 1):
        idx = n - 1

        if np.any(sigma2 <= _TOL):
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


def _pacf_1d(r_array: FloatArray, at_boundary: BoundaryMode) -> FloatArray:
    """
    Compute partial autocorrelations for a single 1-D correlation
    sequence, implementing the ``at_boundary`` semantics of
    :func:`pacf`.
    """
    if at_boundary != "extend":
        state = _LevinsonState.from_correlations(
            r_array, at_boundary=at_boundary
        )
        return state.alpha

    state = _LevinsonState.from_correlations(r_array, at_boundary="warn")

    if state.sigma2[-1] > _TOL:
        return state.alpha

    n_max = r_array.size
    prefix_len = state.r.size
    sign = state.alpha[-1]

    if prefix_len < n_max:
        from .sh_bounds import extend_at_boundary

        extend_at_boundary(r_array, n_extra=n_max - prefix_len)

    return np.concatenate(
        [state.alpha, np.full(n_max - prefix_len, sign, dtype=np.float64)]
    )


def from_pacf(
    alpha: ArrayLike,
    *,
    backend: Backend = "float64",
    dps: int | None = None,
) -> FloatArray | list:
    """
    Reconstruct correlations from partial autocorrelations.

    Parameters
    ----------
    alpha
        Partial autocorrelation coefficients, either a 1-D array
        ``(alpha_1, ..., alpha_N)`` or a 2-D batch of ``n_samples``
        such sequences with shape ``(n_samples, N)``. For a 2-D
        batch, the recursion is applied independently to each row
        (the same recursion as for the 1-D case, not a different
        algorithm), vectorized across rows for performance -- unlike
        :func:`pacf`, no fallback is needed here, since
        ``abs(alpha_n) < 1`` (checked for every row up front) already
        guarantees every row stays clear of the singular boundary.
        The vectorized path is not guaranteed bit-identical to looping
        :func:`from_pacf` over rows (summation-order ULP differences,
        see :func:`pacf`'s equivalent note). ``backend='mpmath'`` only
        supports 1-D input. Accepted as plain Python floats (or
        anything ``mpmath.mpf`` accepts, for ``backend='mpmath'``);
        values are promoted to the working precision internally, the
        caller does not need to pre-construct ``mp.mpf`` values.
    backend
        ``'float64'`` (default) runs the plain ``numpy`` recursion in
        this module. ``'mpmath'`` runs the arbitrary-precision
        recursion in :mod:`schurcorr._levinson_mp` instead -- see
        :func:`pacf` and that module's docstring. With
        ``backend='mpmath'``, the return value is a ``list`` of
        ``mpmath.mpf`` (not downcast to ``float64``).
    dps
        Working precision in decimal places, only used when
        ``backend='mpmath'``. If ``None`` (default),
        ``recommended_dps(len(alpha))`` is used.

    Returns
    -------
    r
        Normalized correlation coefficients. For ``backend='float64'``,
        the same shape as ``alpha``: ``(r_1, ..., r_N)`` or
        ``(n_samples, N)``. For ``backend='mpmath'``, a ``list`` of
        ``mpmath.mpf``, length ``N``.

    Raises
    ------
    ValueError
        If any ``abs(alpha_n) >= 1``, regardless of backend.
    """
    if backend not in ("float64", "mpmath"):
        raise ValueError(
            f"backend must be 'float64' or 'mpmath'; got {backend!r}."
        )

    if backend == "mpmath":
        alpha_array = _asarray1d(alpha, name="alpha")

        if np.any(np.abs(alpha_array) >= 1.0):
            raise ValueError(
                "All PACF coefficients must satisfy abs(alpha_n) < 1."
            )

        from . import _levinson_mp

        return _levinson_mp._from_pacf_mp(alpha_array.tolist(), dps=dps)

    if dps is not None:
        raise ValueError("dps is only used when backend='mpmath'.")

    alpha_array = _asarray_batchable(alpha, name="alpha")

    if alpha_array.ndim == 1:
        state = _LevinsonState.from_pacf(alpha_array)
        return state.r

    return _from_pacf_2d(alpha_array)


def _from_pacf_2d(alpha_array: FloatArray) -> FloatArray:
    """
    Vectorized implementation of :func:`from_pacf` over a 2-D batch.

    Runs the same recursion as :meth:`_LevinsonState.from_pacf`,
    vectorized across the batch (sample) dimension. Unlike
    :func:`_pacf_2d_fast`, no fallback is needed: for any
    ``abs(alpha_n) < 1`` (validated up front, for every row, exactly
    as the scalar path does per row), the innovation variance
    ``sigma2 * (1 - alpha_n**2)`` stays strictly positive by
    construction, so the forward recursion can never reach the
    singular boundary -- there is no boundary case to fall back on.

    Not guaranteed bit-identical to the per-row scalar path; see the
    Notes in :func:`_pacf_2d_fast` (same summation-order caveat).
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
    # holds r_1, ..., r_(n-1) once computed (see _LevinsonState.from_pacf).
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


def fisher(alpha: ArrayLike) -> FloatArray:
    """
    Map partial autocorrelations to Fisher coordinates.

    The transformation is

    ``y_n = arctanh(alpha_n)``.

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
    Compute innovation variances from partial autocorrelations.

    The recursion is

    ``sigma_n^2 = sigma_(n-1)^2 * (1 - alpha_n^2)``

    with ``sigma_0^2 = 1``.

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
        Innovation variances ``(sigma_0^2, ..., sigma_N^2)``, or, for
        a 2-D batch, ``(n_samples, N + 1)``.
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

    n_samples, n_max = alpha_array.shape
    sigma2 = np.empty((n_samples, n_max + 1), dtype=np.float64)
    sigma2[:, 0] = 1.0

    for n in range(1, n_max + 1):
        alpha_n = alpha_array[:, n - 1]
        sigma2[:, n] = sigma2[:, n - 1] * (1.0 - alpha_n * alpha_n)

        clamp_mask = (sigma2[:, n] < 0.0) & (sigma2[:, n] > -_TOL)
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
