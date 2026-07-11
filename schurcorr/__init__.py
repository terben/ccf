"""
schurcorr
=========

Companion package for the paper

    Natural Coordinates for Constrained Correlation Functions:
    Partial Autocorrelations and the Geometry of Positive Power Spectra

The package provides routines for

- Levinson-Durbin forward recursion
- inverse recursion
- Schneider-Hartlap admissible bounds
- Fisher coordinates
- Jacobians
"""

from .levinson import (
    pacf,
    from_pacf,
    fisher,
    inverse_fisher,
    innovation_variances,
    jacobian,
    log_jacobian,
)

from .sh_bounds import (
    admissible_bounds,
    sh_coordinates,
    check_admissibility,
)

__all__ = [
    "forward",
    "inverse",
    "fisher",
    "inverse_fisher",
    "innovation_variances",
    "jacobian",
    "log_jacobian",
    "admissible_bounds",
    "sh_coordinates",
    "check_admissibility",
]

__version__ = "0.1.0"
