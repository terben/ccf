"""
Minimal example: the r <-> alpha correspondence and the degenerate boundary.

Run directly: python examples/pacf_roundtrip.py
"""

import numpy as np

import schurcorr as sc

# An admissible interior sequence: build it from PACF coordinates, whose
# only constraint is abs(alpha_n) < 1, then recover the correlations.
alpha = np.array([0.5, -0.3, 0.2, 0.1])
r = sc.from_pacf(alpha)
print("alpha  ->", alpha)
print("r      ->", r)

alpha_reconstructed = sc.pacf(r)
print("roundtrip max error:", np.max(np.abs(alpha_reconstructed - alpha)))

# Fisher coordinates map (-1, 1) to the whole real line.
y = sc.fisher(alpha)
print("fisher ->", y)
print("inverse_fisher(y) matches alpha:", np.allclose(sc.inverse_fisher(y), alpha))

# At the boundary (sigma_n^2 = 0 for some n) the r <-> alpha bijection
# breaks down: pacf() raises, since alpha_n is no longer an independent
# coordinate past that point.
r_boundary = np.array([1.0, 0.5, 0.3])
try:
    sc.pacf(r_boundary)
except sc.SingularToeplitzError as error:
    print("pacf() at the boundary raises:", error)

# pacf_prefix() is the boundary-aware analysis: it never raises, and
# reports exactly how far the independent PACF coordinates extend.
result = sc.pacf_prefix(r_boundary)
print("pacf_prefix().alpha  ->", result.alpha)
print("pacf_prefix().order  ->", result.order)
print("pacf_prefix().reached_boundary ->", result.reached_boundary)
