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
    """Terminal data returned by the correlation-to-PACF recursion for a
    single sequence."""

    r: FloatArray
    alpha: FloatArray
    sigma2: FloatArray
    prediction: FloatArray
    reached_boundary: bool
    # Terminal predictor coefficients, only meaningful (non-None) once the
    # boundary is reached; see schurcorr.bounds.extend_at_boundary.
    terminal_phi: FloatArray | None = None


@dataclass(frozen=True, slots=True)
class _LevinsonBatchResult:
    """Terminal data returned by the correlation-to-PACF recursion for a
    batch of sequences.

    Every field has a leading ``(n_samples, ...)`` axis. For row ``i``,
    only the first ``terminal_order[i]`` entries of ``alpha``/``prediction``
    and ``terminal_order[i] + 1`` entries of ``sigma2`` are meaningful --
    later entries are unwritten. ``phi`` holds the terminal predictor
    snapshot for that row, valid up to the same length.
    """

    alpha: FloatArray
    sigma2: FloatArray
    prediction: FloatArray
    phi: FloatArray
    terminal_order: NDArray[np.intp]
    reached_boundary: NDArray[np.bool_]
    invalid: NDArray[np.bool_]
    # abs(alpha) computed at the step where invalidity was detected, only
    # meaningful where invalid is True; used for the error message.
    invalid_alpha_abs: FloatArray


def _levinson_correlations_batch(r2d: FloatArray) -> _LevinsonBatchResult:
    """Run the correlation-to-PACF recursion, vectorized across batch rows.

    Shared by :func:`pacf` and, via :func:`_run_levinson_from_correlations`,
    by :func:`pacf_prefix` and :mod:`schurcorr.bounds`. Proceeds each row
    as far as the mathematics allows and reports how far via
    ``terminal_order``/``reached_boundary``/``invalid`` rather than raising;
    callers decide what to raise (see ``docs/boundary_semantics.md``).
    """
    n_samples, n_max = r2d.shape

    # Zero-initialized (not np.empty) so that np.where blending below never
    # mixes in uninitialized memory for rows that froze before writing a
    # given column.
    alpha = np.zeros((n_samples, n_max), dtype=np.float64)
    prediction = np.zeros((n_samples, n_max), dtype=np.float64)
    phi = np.zeros((n_samples, n_max), dtype=np.float64)
    sigma2 = np.zeros((n_samples, n_max + 1), dtype=np.float64)
    sigma2[:, 0] = 1.0

    sigma2_current = np.ones(n_samples, dtype=np.float64)
    active = np.ones(n_samples, dtype=bool)
    terminal_order = np.zeros(n_samples, dtype=np.intp)
    reached_boundary = np.zeros(n_samples, dtype=bool)
    invalid = np.zeros(n_samples, dtype=bool)
    invalid_alpha_abs = np.zeros(n_samples, dtype=np.float64)

    for n in range(1, n_max + 1):
        idx = n - 1
        active_before = active.copy()

        if not np.any(active_before):
            break

        # Rows whose sigma2 already rounded to the boundary on the previous
        # step (without alpha hitting +-1 exactly there) freeze here,
        # mirroring the top-of-loop check in the single-sequence recursion.
        pre_boundary = active_before & (sigma2_current <= _ROUNDING_TOL)
        reached_boundary[pre_boundary] = True
        terminal_order[pre_boundary] = idx
        active[pre_boundary] = False

        computing = active_before & ~pre_boundary

        if n == 1:
            pred = np.zeros(n_samples, dtype=np.float64)
        else:
            pred = np.sum(phi[:, :idx] * r2d[:, idx - 1 :: -1], axis=1)

        sigma2_safe = np.where(computing, sigma2_current, 1.0)
        alpha_n_raw = (r2d[:, idx] - pred) / sigma2_safe
        abs_alpha_n_raw = np.abs(alpha_n_raw)

        invalid_now = computing & (abs_alpha_n_raw > 1.0 + _ROUNDING_TOL)
        boundary_hit_now = computing & ~invalid_now & (abs_alpha_n_raw >= 1.0)
        continue_now = computing & ~(invalid_now | boundary_hit_now)

        invalid[invalid_now] = True
        terminal_order[invalid_now] = idx
        invalid_alpha_abs[invalid_now] = abs_alpha_n_raw[invalid_now]
        active[invalid_now] = False

        alpha_n = np.where(boundary_hit_now, np.sign(alpha_n_raw), alpha_n_raw)
        update_mask = continue_now | boundary_hit_now

        # Blend via np.where rather than boolean-indexed assignment: for an
        # interior batch (the common case) update_mask is all-True, and
        # blending avoids fancy-indexing gather/scatter overhead while
        # leaving frozen rows' already-written entries untouched.
        alpha[:, idx] = np.where(update_mask, alpha_n, alpha[:, idx])
        prediction[:, idx] = np.where(update_mask, pred, prediction[:, idx])

        # phi^{(n+1)}_j = phi^{(n)}_j - alpha_n * phi^{(n)}_{n-j}
        if n == 1:
            phi[:, 0] = np.where(update_mask, alpha_n, phi[:, 0])
        else:
            phi_prev = phi[:, :idx].copy()
            new_phi_head = phi_prev - alpha_n[:, None] * phi_prev[:, ::-1]
            phi[:, :idx] = np.where(update_mask[:, None], new_phi_head, phi_prev)
            phi[:, idx] = np.where(update_mask, alpha_n, phi[:, idx])

        sigma2_next = sigma2_current * (1.0 - alpha_n * alpha_n)
        clamp_mask = continue_now & (sigma2_next < 0.0) & (sigma2_next > -_ROUNDING_TOL)
        if np.any(clamp_mask):
            sigma2_next = np.where(clamp_mask, 0.0, sigma2_next)
        sigma2_next = np.where(boundary_hit_now, 0.0, sigma2_next)

        sigma2[:, n] = np.where(update_mask, sigma2_next, sigma2[:, n])

        reached_boundary[boundary_hit_now] = True
        terminal_order[boundary_hit_now] = n
        active[boundary_hit_now] = False

        terminal_order[continue_now] = n
        sigma2_current = np.where(continue_now, sigma2_next, sigma2_current)

    # Rows still active after n_max steps completed the interior sequence.
    terminal_order[active] = n_max

    return _LevinsonBatchResult(
        alpha=alpha,
        sigma2=sigma2,
        prediction=prediction,
        phi=phi,
        terminal_order=terminal_order,
        reached_boundary=reached_boundary,
        invalid=invalid,
        invalid_alpha_abs=invalid_alpha_abs,
    )


def _run_levinson_from_correlations(r: ArrayLike) -> _LevinsonResult:
    """Run the shared correlation-to-PACF recursion for a single sequence.

    Thin single-row adapter over :func:`_levinson_correlations_batch`, used
    by :func:`pacf_prefix` and :mod:`schurcorr.bounds`.
    """
    r_array = _asarray1d(r, name="r")
    batch = _levinson_correlations_batch(r_array[None, :])

    if batch.invalid[0]:
        n = int(batch.terminal_order[0]) + 1
        raise ValueError(
            "Input correlations are not admissible: "
            f"computed abs(alpha_{n}) = "
            f"{batch.invalid_alpha_abs[0]:.6g} > 1."
        )

    n = int(batch.terminal_order[0])
    return _LevinsonResult(
        r=r_array[:n].copy(),
        alpha=batch.alpha[0, :n].copy(),
        sigma2=batch.sigma2[0, : n + 1].copy(),
        prediction=batch.prediction[0, :n].copy(),
        reached_boundary=bool(batch.reached_boundary[0]),
        terminal_phi=(
            batch.phi[0, :n].copy() if batch.reached_boundary[0] else None
        ),
    )


def _run_levinson_from_pacf(alpha: ArrayLike) -> FloatArray:
    """Run the PACF-to-correlation recursion, vectorized across batch rows.

    Shared by :func:`from_pacf` for both a single sequence and a 2-D batch
    (a 1-D input is treated as a batch of one row internally). Cannot reach
    the singular boundary by construction (``abs(alpha_n) < 1`` is
    validated up front), so no boundary handling is needed.
    """
    alpha_array = _asarray_batchable(alpha, name="alpha")
    was_1d = alpha_array.ndim == 1
    alpha2d = alpha_array[None, :] if was_1d else alpha_array

    bad_mask = np.any(np.abs(alpha2d) >= 1.0, axis=1)
    if np.any(bad_mask):
        first_bad = int(np.argmax(bad_mask))
        prefix = "" if was_1d else f"sample {first_bad}: "
        raise ValueError(
            f"{prefix}All PACF coefficients must satisfy abs(alpha_n) < 1."
        )

    n_samples, n_max = alpha2d.shape

    # r_buffer[:, 0] is the fixed r_0 = 1 column; r_buffer[:, 1:n] holds
    # r_1, ..., r_(n-1) once computed, so r_buffer[:, :n] is exactly the
    # "r_with_zero" vector needed at step n.
    r_buffer = np.empty((n_samples, n_max + 1), dtype=np.float64)
    r_buffer[:, 0] = 1.0
    phi = np.empty((n_samples, n_max), dtype=np.float64)
    sigma2 = np.ones(n_samples, dtype=np.float64)

    for n in range(1, n_max + 1):
        idx = n - 1
        alpha_n = alpha2d[:, idx]

        if n == 1:
            pred = np.zeros(n_samples, dtype=np.float64)
        else:
            pred = np.sum(phi[:, :idx] * r_buffer[:, idx:0:-1], axis=1)

        r_buffer[:, n] = pred + alpha_n * sigma2

        if n == 1:
            phi[:, 0] = alpha_n
        else:
            phi_prev = phi[:, :idx].copy()
            phi[:, :idx] = phi_prev - alpha_n[:, None] * phi_prev[:, ::-1]
            phi[:, idx] = alpha_n

        sigma2_next = sigma2 * (1.0 - alpha_n * alpha_n)
        clamp_mask = (sigma2_next < 0.0) & (sigma2_next > -_ROUNDING_TOL)
        if np.any(clamp_mask):
            warnings.warn(
                "Innovation variance became slightly negative due "
                "to roundoff and was clamped to zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            sigma2_next = np.where(clamp_mask, 0.0, sigma2_next)
        sigma2 = sigma2_next

    r_out = r_buffer[:, 1:]
    return r_out[0].copy() if was_1d else r_out.copy()


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
    was_1d = r_array.ndim == 1
    r2d = r_array[None, :] if was_1d else r_array

    batch = _levinson_correlations_batch(r2d)

    # A batch with several problematic rows reports the smallest row
    # index, using that row's own error type -- matching the row-by-row
    # loop this replaces (see docs/development_notes.md).
    problem_mask = batch.invalid | batch.reached_boundary
    if np.any(problem_mask):
        row = int(np.argmax(problem_mask))
        prefix = "" if was_1d else f"sample {row}: "

        if batch.invalid[row]:
            n = int(batch.terminal_order[row]) + 1
            raise ValueError(
                f"{prefix}Input correlations are not admissible: "
                f"computed abs(alpha_{n}) = "
                f"{batch.invalid_alpha_abs[row]:.6g} > 1."
            )

        n = int(batch.terminal_order[row])
        raise SingularToeplitzError(
            f"{prefix}The Toeplitz matrix of order {n} is singular "
            f"(sigma_{n}^2 = 0); the r <-> alpha bijection breaks down at "
            "this point. Use pacf_prefix(r) for the boundary-inclusive "
            "prefix or extend_at_boundary(r, n_extra) for the recurrence-"
            "forced continuation."
        )

    alpha = batch.alpha.copy()
    return alpha[0] if was_1d else alpha


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
    return _run_levinson_from_pacf(alpha)
