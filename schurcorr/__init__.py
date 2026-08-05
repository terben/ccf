"""
schurcorr
=========

Companion package for the paper

    Natural Coordinates for Constrained Correlation Functions:
    Partial Autocorrelations and the Geometry of Positive Power Spectra

The package provides routines for

- Levinson--Durbin forward recursion from correlations to PACFs,
- inverse recursion from PACFs to correlations,
- Schneider--Hartlap admissible bounds and coordinates,
- Fisher coordinates,
- innovation variances,
- Jacobian determinants.
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
    PrefixResult,
    SingularToeplitzError,
    from_pacf,
    pacf,
    pacf_prefix,
)
from .precision import from_pacf_mp, pacf_mp, recommended_dps

__all__ = [
    "PrefixResult",
    "SingularToeplitzError",
    "admissible_bounds",
    "admissible_volume",
    "check_admissibility",
    "extend_at_boundary",
    "fisher",
    "from_pacf",
    "from_pacf_mp",
    "innovation_variances",
    "inverse_fisher",
    "jacobian",
    "log_admissible_volume",
    "log_jacobian",
    "pacf",
    "pacf_mp",
    "pacf_prefix",
    "recommended_dps",
    "sh_coordinates",
]

__version__ = "0.1.2"
