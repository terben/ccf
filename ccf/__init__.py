"""Partial autocorrelations and natural coordinates for correlation functions.

Companion package for "Natural Coordinates for Constrained Correlation
Functions" (Erben, in preparation); see the README for usage.
"""

from .bounds import (
    admissible_bounds,
    admissible_volume,
    check_admissibility,
    extend_at_boundary,
    log_admissible_volume,
    sh_coordinates,
)
from .coordinates import (
    fisher,
    innovation_variances,
    inverse_fisher,
    jacobian,
    log_jacobian,
)
from .levinson import (
    PACFStatus,
    PrefixResult,
    SingularToeplitzError,
    from_pacf,
    pacf,
    pacf_prefix,
    pacf_status,
)

__all__ = [
    "PACFStatus",
    "PrefixResult",
    "SingularToeplitzError",
    "admissible_bounds",
    "admissible_volume",
    "check_admissibility",
    "extend_at_boundary",
    "fisher",
    "from_pacf",
    "innovation_variances",
    "inverse_fisher",
    "jacobian",
    "log_admissible_volume",
    "log_jacobian",
    "pacf",
    "pacf_prefix",
    "pacf_status",
    "sh_coordinates",
]

try:
    # Requires the "precision" extra (mpmath); the numpy-only core stays
    # importable without it.
    from .precision import from_pacf_mp, pacf_mp, recommended_dps
except ModuleNotFoundError as error:
    if error.name != "mpmath":
        raise
else:
    __all__ += ["from_pacf_mp", "pacf_mp", "recommended_dps"]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("ccf")
except PackageNotFoundError:
    __version__ = "0+unknown"
