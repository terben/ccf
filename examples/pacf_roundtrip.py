"""
Minimal example: the r <-> alpha correspondence and the degenerate boundary.

Run directly: python examples/pacf_roundtrip.py
"""

import numpy as np

import ccf as sc

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

# pacf_status() is the cheap, non-raising diagnostic: it classifies a
# sequence as interior / boundary / invalid without raising.
status = sc.pacf_status(r_boundary)
print("pacf_status() ->", status)

if status.boundary:
    result = sc.pacf_prefix(r_boundary)
    print("pacf_prefix().alpha  ->", result.alpha)
    print("pacf_prefix().order  ->", result.order)

# A genuinely inadmissible sequence: pacf_status() reports "invalid" and
# gives the order at which admissibility first failed.
r_invalid = np.array([0.9, 0.95, 0.99])
status_invalid = sc.pacf_status(r_invalid)
print("pacf_status(r_invalid) ->", status_invalid)
print("failed at order:", status_invalid.order)

# pacf_status()/pacf_prefix() only describe the recursion up to its first
# boundary; they do not validate coefficients supplied past it. r_1 = 1
# forces r_2 = 1 too, so the supplied 0.5 below is inconsistent even
# though pacf_status() still reports "boundary" at order 1.
r_inconsistent_tail = np.array([1.0, 0.5])
print("pacf_status(r_inconsistent_tail) ->", sc.pacf_status(r_inconsistent_tail))
print(
    "check_admissibility(r_inconsistent_tail) ->",
    sc.check_admissibility(r_inconsistent_tail),
)
