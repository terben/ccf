"""Arbitrary-precision PACF transformations.

Standalone ``mpmath`` counterpart to :mod:`ccf.levinson`, for
sequences where ``sigma_n^2`` underflows ``float64`` (see
:func:`recommended_dps`).
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
    """Recommend a working precision for arbitrary-precision roundtrips.

    Parameters
    ----------
    n_max
        Sequence length.
    target_exponent
        Target error scale as a power of ten.
    safety_margin
        Additional decimal places.

    Returns
    -------
    int
        Recommended decimal precision.

    Raises
    ------
    ValueError
        If an argument is invalid.

    Notes
    -----
    This is an empirical recommendation for roundtrip calculations, not a
    mathematically guaranteed minimum precision.
    """
    if n_max < 1 or n_max != int(n_max):
        raise ValueError(f"n_max must be a positive integer; got {n_max!r}.")

    if safety_margin < 0:
        raise ValueError(
            f"safety_margin must be non-negative; got {safety_margin!r}."
        )

    # 0.45 dps/lag: empirical conditioning loss for alpha in (-0.95, 0.95)
    return math.ceil(0.45 * n_max) + abs(target_exponent) + safety_margin


def pacf_mp(
    r: Sequence[float],
    *,
    dps: int | None = None,
    at_boundary: BoundaryModeMp = "raise",
) -> list[mp.mpf]:
    """Convert correlations to PACFs using arbitrary precision.

    Arbitrary-precision counterpart to :func:`ccf.levinson.pacf`.

    Parameters
    ----------
    r
        One-dimensional correlation sequence, as plain Python floats
        (or anything ``mpmath.mpf`` accepts).
    dps
        Working precision in decimal places. By default,
        :func:`recommended_dps` is used.
    at_boundary
        Boundary behavior, either ``"raise"`` or ``"warn"``. ``"warn"``
        emits a ``RuntimeWarning`` and returns the non-degenerate
        prefix only. See Notes.

    Returns
    -------
    list[mpmath.mpf]
        Partial autocorrelations, as high-precision ``mpf`` values --
        not downcast to ``float64``.

    Raises
    ------
    SingularToeplitzError
        If the recursion reaches a degenerate boundary and
        ``at_boundary="raise"``.
    ValueError
        If an argument is invalid.

    Notes
    -----
    Unlike the ``float64`` path (split into :func:`ccf.levinson.pacf`
    and :func:`ccf.levinson.pacf_prefix`), ``at_boundary`` stays a
    mode parameter here.
    """
    if at_boundary not in ("raise", "warn"):
        raise ValueError(
            "at_boundary must be 'raise' or 'warn' for pacf_mp (no "
            f"'extend' support at arbitrary precision); got {at_boundary!r}."
        )

    from .levinson import _as_sequence_1d

    r = _as_sequence_1d(r, name="r")

    if not all(mp.isfinite(x) for x in r):
        raise ValueError("r must contain only finite values.")

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
    """Convert PACFs to correlations using arbitrary precision.

    Arbitrary-precision counterpart to :func:`ccf.levinson.from_pacf`.

    Parameters
    ----------
    alpha
        One-dimensional PACF sequence with entries in ``(-1, 1)``, as
        plain Python floats (or anything ``mpmath.mpf`` accepts).
    dps
        Working precision in decimal places. By default,
        :func:`recommended_dps` is used.

    Returns
    -------
    list[mpmath.mpf]
        Correlation coefficients, as high-precision ``mpf`` values --
        not downcast to ``float64``.

    Raises
    ------
    ValueError
        If the input is invalid.
    """
    from .levinson import _as_sequence_1d

    alpha = _as_sequence_1d(alpha, name="alpha")

    if not all(mp.isfinite(x) for x in alpha):
        raise ValueError("alpha must contain only finite values.")

    n_max = len(alpha)
    if dps is None:
        dps = recommended_dps(n_max)

    with mp.workdps(dps):
        # Promote directly to mpf at working precision -- alpha may
        # already carry more than float64 precision (e.g. mpf inputs from
        # a prior high-dps call), so admissibility is also checked here,
        # not via a float64 round-trip beforehand.
        alpha_mp = [mp.mpf(a) for a in alpha]

        if any(abs(a) >= 1 for a in alpha_mp):
            raise ValueError(
                "All PACF coefficients must satisfy abs(alpha_n) < 1."
            )

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
