"""
Common data structures used throughout the schurcorr package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LevinsonInfo:
    """
    Additional quantities computed during a Levinson recursion.

    Parameters
    ----------
    sigma2
        Innovation variances.

    predictor_coefficients
        Linear prediction coefficients at each recursion order.
    """

    sigma2: np.ndarray
    predictor_coefficients: list[np.ndarray]
