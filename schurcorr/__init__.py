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

from ._levinson_mp import recommended_dps
from .levinson import (
    SingularToeplitzError,
    fisher,
    from_pacf,
    innovation_variances,
    inverse_fisher,
    jacobian,
    log_jacobian,
    pacf,
)
from .sh_bounds import (
    admissible_bounds,
    admissible_volume,
    check_admissibility,
    extend_at_boundary,
    log_admissible_volume,
    sh_coordinates,
)

__all__ = [
    "pacf",
    "from_pacf",
    "fisher",
    "inverse_fisher",
    "innovation_variances",
    "jacobian",
    "log_jacobian",
    "admissible_bounds",
    "sh_coordinates",
    "check_admissibility",
    "extend_at_boundary",
    "admissible_volume",
    "log_admissible_volume",
    "recommended_dps",
    "SingularToeplitzError",
]

__version__ = "0.1.2"
