"""
Arbitrary-precision (mpmath) counterpart to the Levinson--Durbin recursion.

This module is the arbitrary-precision counterpart to the plain
``numpy``/``float64`` recursion in :mod:`schurcorr.levinson`. It is kept
in its own module -- reached via the standalone public functions
:func:`pacf_mp` / :func:`from_pacf_mp`, not a ``backend=`` parameter on
:func:`schurcorr.levinson.pacf` -- so that the ``numpy``-only bijection
code stays visually and structurally separate from the ``mpmath``-
precision code: the two solve genuinely different problems.
``levinson.py`` implements the ``r <-> alpha`` bijection itself; this
module is a numerical-conditioning stress test of that same recursion at
arbitrary working precision (see :func:`recommended_dps`).

For random ``alpha`` drawn independently over many lags, the residual
variance ``sigma_n^2 = prod_(i<n) (1 - alpha_i^2)`` decays
geometrically (the same product that appears as the Jacobian
``det(dr/dalpha)``, see :func:`schurcorr.coordinates.jacobian`). Once
``sigma_n^2`` is smaller than ``float64``'s own rounding unit, the
difference ``r_n - p_n = alpha_n * sigma_n^2`` that the recursion
needs is no longer representable, and the information about
``alpha_n`` is lost -- not recoverable by any downstream algorithm.
This module sidesteps the limit by carrying the whole computation at
a working precision high enough that ``sigma_n^2`` never underflows
relative to that precision's own rounding unit.

References
----------
Schneider, P. and Hartlap, J. 2009, A&A 504, 705.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Sequence

import mpmath as mp

BoundaryModeMp = Literal["raise", "warn"]


def recommended_dps(
    n_max: int,
    target_exponent: int = -13,
    safety_margin: int = 11,
) -> int:
    """
    Working precision (decimal places) recommended for size ``n_max``.

    Parameters
    ----------
    n_max
        Length ``N`` of the ``alpha`` (or ``r``) sequence the
        recursion will be run on.
    target_exponent
        Desired roundtrip accuracy, expressed as
        ``10^target_exponent``.
    safety_margin
        Extra decimal places added on top of the theoretical minimum,
        to absorb trial-to-trial variance (the conditioning loss is a
        random quantity, not a fixed number, for a given ``N``).

    Returns
    -------
    int
        Recommended value for ``mpmath``'s working precision in
        decimal places (``dps``).

    Notes
    -----
    Empirically, the recursion loses about ``0.45`` decimal digits of
    precision per lag for ``alpha`` drawn from ``(-0.95, 0.95)`` (the
    wider of the two bounds used in the paper's roundtrip figure),
    independent of the working precision itself -- this is the
    conditioning loss described in the module docstring, not a
    property of ``float64`` specifically. The formula below is a
    conservative, deliberately simple affine fit to that empirical
    rate; it is calibrated for ``b=0.95`` and therefore also safe
    (with some margin to spare) for the narrower ``b=0.9`` case used
    alongside it in the same figure.

    References
    ----------
    Schneider, P. and Hartlap, J. 2009, A&A 504, 705.
    """
    return math.ceil(0.45 * n_max) + abs(target_exponent) + safety_margin


def pacf_mp(
    r: Sequence[float],
    *,
    dps: int | None = None,
    at_boundary: BoundaryModeMp = "raise",
) -> list[mp.mpf]:
    """
    Recovery recursion ``r -> alpha`` at arbitrary precision.

    Standalone arbitrary-precision counterpart to
    :func:`schurcorr.levinson.pacf`; only the number type changes, from
    ``float64`` to ``mpmath.mpf`` at an explicitly chosen working
    precision. Use this directly rather than a ``backend=`` parameter on
    :func:`schurcorr.levinson.pacf` (which does not have one -- see
    docs/boundary_semantics.md).

    Parameters
    ----------
    r
        Correlation coefficients ``r_1, ..., r_N``, as plain Python
        floats (or anything ``mpmath.mpf`` accepts); promoted to
        ``mpf`` at the working precision internally.
    dps
        Working precision in decimal places, scoped via
        ``mpmath.workdps`` (no side effect on the caller's global
        ``mpmath.mp.dps``). If ``None``, ``recommended_dps(len(r))``
        is used.
    at_boundary
        ``'raise'`` raises :class:`schurcorr.SingularToeplitzError`
        when a residual variance ``<= 0`` is encountered at the
        chosen working precision; ``'warn'`` emits a
        ``RuntimeWarning`` and returns the non-degenerate part only.
        ``'extend'`` is not supported for this backend (see
        :func:`schurcorr.levinson.pacf`).

    Returns
    -------
    list[mpmath.mpf]
        Partial autocorrelations ``alpha_1, ..., alpha_N`` (or fewer,
        under ``at_boundary='warn'``), as high-precision ``mpf``
        values -- deliberately not downcast to ``float64``.

    Raises
    ------
    schurcorr.SingularToeplitzError
        If ``at_boundary='raise'`` (the default) and a residual
        variance ``<= 0`` is encountered, i.e. ``dps`` was not high
        enough for the input sequence, or the boundary of the
        admissible region was genuinely reached.
    ValueError
        If ``at_boundary`` is not ``'raise'`` or ``'warn'`` (in
        particular, ``'extend'`` is not supported at arbitrary
        precision; use :func:`schurcorr.bounds.extend_at_boundary`,
        which operates on ``r`` directly and has no ``mpmath`` variant).

    Notes
    -----
    Complexity is ``O(N^2)`` recursion steps, as in the ``float64``
    version, but each arithmetic operation costs more than a
    ``float64`` operation, and that cost grows with ``dps`` (which in
    turn grows with ``N`` via :func:`recommended_dps`), so wall-clock
    time scales worse than ``O(N^2)`` in practice.

    References
    ----------
    Schneider, P. and Hartlap, J. 2009, A&A 504, 705.
    """
    if at_boundary not in ("raise", "warn"):
        raise ValueError(
            "at_boundary must be 'raise' or 'warn' for pacf_mp (no "
            f"'extend' support at arbitrary precision); got {at_boundary!r}."
        )

    from .levinson import _as_sequence_1d

    r = _as_sequence_1d(r, name="r")

    n_max = len(r)
    if dps is None:
        dps = recommended_dps(n_max)

    with mp.workdps(dps):
        r_mp = [mp.mpf(x) for x in r]
        alpha: list[mp.mpf] = []
        sigma_sq: list[mp.mpf] = [mp.mpf(1)]
        phi: list[mp.mpf] = []

        for n in range(1, n_max + 1):
            idx = n - 1

            if sigma_sq[-1] <= 0:
                if at_boundary == "raise":
                    from .levinson import SingularToeplitzError

                    raise SingularToeplitzError(
                        f"sigma_{idx}^2 <= 0 at working precision "
                        f"dps={dps} (mpmath backend); the r <-> alpha "
                        "bijection breaks down at alpha_"
                        f"{idx + 1}. Increase dps, or pass "
                        "at_boundary='warn'."
                    )

                warnings.warn(
                    "Degenerate Toeplitz matrix encountered at "
                    f"working precision dps={dps}; returning the "
                    "non-degenerate part only.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            if n == 1:
                p_n = mp.mpf(0)
            else:
                p_n = mp.fdot(phi, r_mp[n - 2 :: -1])

            a = (r_mp[idx] - p_n) / sigma_sq[-1]
            alpha.append(a)
            sigma_sq.append(sigma_sq[-1] * (1 - a * a))

            if n == 1:
                phi = [a]
            else:
                phi_prev = phi
                phi = [
                    phi_prev[j] - a * phi_prev[n - 2 - j]
                    for j in range(n - 1)
                ]
                phi.append(a)

    return alpha


def from_pacf_mp(
    alpha: Sequence[float],
    *,
    dps: int | None = None,
) -> list[mp.mpf]:
    """
    Forward recursion ``alpha -> r`` at arbitrary precision.

    Standalone arbitrary-precision counterpart to
    :func:`schurcorr.levinson.from_pacf`; only the number type changes,
    from ``float64`` to ``mpmath.mpf`` at an explicitly chosen working
    precision.

    Parameters
    ----------
    alpha
        Partial autocorrelations ``alpha_1, ..., alpha_N``, as plain
        Python floats (or anything ``mpmath.mpf`` accepts); promoted
        to ``mpf`` at the working precision internally.
    dps
        Working precision in decimal places, scoped via
        ``mpmath.workdps`` (no side effect on the caller's global
        ``mpmath.mp.dps``). If ``None``, ``recommended_dps(len(alpha))``
        is used.

    Returns
    -------
    list[mpmath.mpf]
        Correlation coefficients ``r_1, ..., r_N`` as high-precision
        ``mpf`` values -- deliberately not downcast to ``float64``;
        the caller decides when (and whether) to round.

    Raises
    ------
    ValueError
        If ``alpha`` is not one-dimensional, or if any
        ``abs(alpha_n) >= 1``.

    Notes
    -----
    Complexity is ``O(N^2)`` recursion steps; see :func:`pacf_mp`
    for the same remark on wall-clock scaling with ``dps``.

    References
    ----------
    Schneider, P. and Hartlap, J. 2009, A&A 504, 705.
    """
    import numpy as np

    from .levinson import _asarray1d

    alpha_array = _asarray1d(alpha, name="alpha")
    if np.any(np.abs(alpha_array) >= 1.0):
        raise ValueError(
            "All PACF coefficients must satisfy abs(alpha_n) < 1."
        )
    alpha = alpha_array.tolist()

    n_max = len(alpha)
    if dps is None:
        dps = recommended_dps(n_max)

    with mp.workdps(dps):
        alpha_mp = [mp.mpf(a) for a in alpha]
        r: list[mp.mpf] = []
        sigma_sq: list[mp.mpf] = [mp.mpf(1)]
        phi: list[mp.mpf] = []

        for n in range(1, n_max + 1):
            a = alpha_mp[n - 1]

            if n == 1:
                p_n = mp.mpf(0)
            else:
                p_n = mp.fdot(phi, r[n - 2 :: -1])

            r.append(p_n + a * sigma_sq[-1])
            sigma_sq.append(sigma_sq[-1] * (1 - a * a))

            if n == 1:
                phi = [a]
            else:
                phi_prev = phi
                phi = [
                    phi_prev[j] - a * phi_prev[n - 2 - j]
                    for j in range(n - 1)
                ]
                phi.append(a)

    return r
