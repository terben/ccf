# Documentation style

Documentation describes how to use the code, not how it is implemented.

## Public functions

Use NumPy-style docstrings with:

1. a one-sentence summary;
2. `Parameters`;
3. `Returns`;
4. `Raises`, when relevant;
5. `Examples` only for central or non-obvious functions.

Keep descriptions concise. Refer to the paper for derivations and to
`docs/` for design rationale and boundary semantics.

## Private functions

Use a one-line docstring only when the purpose is not obvious from the name.
Do not document every parameter mechanically.

## Comments

Comments explain non-obvious choices, numerical safeguards, and indexing
conventions. Do not restate the code or record implementation history.

## Module docstrings

Use one or two sentences describing the module's public purpose.

## Terminology

Use the notation of the paper consistently (see `docs/notation.md` for
the exact index correspondence):

- correlation coefficients: `r`
- partial autocorrelations: `alpha`
- Fisher coordinates: `y`
- innovation variances: `sigma2`
- predictor coefficients: `phi`

## Tolerances

Numerical tolerances are named after what they guard, not where they
happen to be used:

- `_ROUNDING_TOL` (`schurcorr/levinson.py`) absorbs float64 roundoff
  around an exact admissibility or boundary value (e.g. a computed
  `sigma_n^2` that is mathematically zero but lands at `-1e-15`).
- `_BOUNDARY_CONTINUATION_TOL` (`schurcorr/bounds.py`) is looser: it
  compares a *supplied* coefficient against the Toeplitz-forced
  continuation computed from it, a chain of products that amplifies
  roundoff faster than the single-step comparisons `_ROUNDING_TOL`
  guards.

Do not change either value without a numerical justification and test
coverage; neither is a stylistic choice.
