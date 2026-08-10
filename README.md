# schurcorr

Reference code for "Natural Coordinates for Constrained Correlation
Functions: Partial Autocorrelations and the Geometry of Positive Power
Spectra" (Erben, in preparation): the bijection between admissible
correlation coefficients and partial autocorrelations (PACFs), and the
Schneider--Hartlap admissible-region geometry it makes explicit.

## Installation

```bash
pip install -e .
```

The core package depends only on NumPy. Optional functionality (figure
reproduction, symbolic verification, arbitrary precision, tests) is
installed via extras -- see [Optional dependencies](#optional-dependencies).

Alternatively, `./install.sh` sets up a conda environment
(`schurcorr.yml`) with every optional dependency and runs the test suite.

## Minimal example

```python
import numpy as np
import schurcorr as sc

alpha = np.array([0.5, -0.3, 0.2])
r = sc.from_pacf(alpha)
alpha_recovered = sc.pacf(r)

print(r)
print(alpha_recovered)
```

## Central functions

| Concept                | Function                             |
| ----------------------- | ------------------------------------- |
| `r -> alpha`            | `schurcorr.pacf`                      |
| `alpha -> r`            | `schurcorr.from_pacf`                 |
| boundary analysis       | `schurcorr.pacf_prefix`               |
| admissible intervals    | `schurcorr.admissible_bounds`         |
| Fisher coordinates      | `schurcorr.fisher` / `inverse_fisher` |
| arbitrary precision     | `schurcorr.pacf_mp` / `from_pacf_mp`  |
| didactic reference form | `schurcorr.reference`                 |

`pacf`, `from_pacf`, `pacf_prefix`, `admissible_bounds`, `fisher`, and
`inverse_fisher` are the primary API. `check_admissibility`,
`pacf_status`, `extend_at_boundary`, `innovation_variances`,
`jacobian`/`log_jacobian`, `admissible_volume`/`log_admissible_volume`,
and the arbitrary-precision functions form an extended API for boundary
handling, diagnostics, and high-order or ill-conditioned sequences.

For a supplied prefix `r_1, ..., r_N`, `admissible_bounds(r)` returns
bounds for each of them plus, as one extra trailing entry, the admissible
interval for the next coefficient `r_(N+1)`:

```python
lower, upper = sc.admissible_bounds(r)
next_lower, next_upper = lower[-1], upper[-1]
```

For the boundary semantics (degenerate but admissible sequences vs.
genuinely inadmissible ones), see `docs/boundary_semantics.md`. For the
exact index correspondence between the paper's notation and the code
(e.g. `sigma2`), see `docs/notation.md`. `pacf_status` reports where the
independent recursion terminates; use `check_admissibility` or
`admissible_bounds` to validate coefficients already supplied beyond a
degenerate boundary.

## Reference implementation

`schurcorr.reference` provides `pacf_reference` and `from_pacf_reference`:
short, single-sequence, loop-based implementations that follow the
paper's Levinson--Durbin recursion (Eqs. ld_p-ld_sigma) line by line,
for readers checking the code against the equations. They are not used
internally; `schurcorr.pacf` / `from_pacf` are the robust, batched,
boundary-aware implementation used throughout the package and its tests.

```python
from schurcorr.reference import pacf_reference, from_pacf_reference
```

## Reproducing the figures

```bash
python paper_plot_scripts/figure_geometry.py
python paper_plot_scripts/figure_roundtrip.py --quick
python paper_plot_scripts/figure_gaussianization.py --quick
```

Drop `--quick` (or pass `--paper`) to reproduce the publication-quality
figures at their full sample sizes; this is significantly slower. Output
is written to `figs/`. `figure_roundtrip.py`'s arbitrary-precision Panel B
runs its independent trials across multiple processes; use `-j`/`--jobs`
to control the worker count (e.g. `--paper -j 8`), or `-j 1` for serial
execution.

## Optional dependencies

```bash
pip install -e ".[plots]"      # matplotlib, scipy -- figure reproduction
pip install -e ".[symbolic]"   # sympy -- schurcorr.symbolic
pip install -e ".[precision]"  # mpmath -- pacf_mp / from_pacf_mp
pip install -e ".[test]"       # pytest, statsmodels, and all of the above
pip install -e ".[all]"        # plots + symbolic + precision
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

## License

BSD 3-Clause License.
