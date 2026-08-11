# ccf — Constrained Correlation Functions

`ccf` is the companion Python package for *Natural Coordinates for Constrained Correlation Functions: Partial Autocorrelations and the Geometry of Positive Power Spectra*.

It provides the numerical tools used in the paper and a small API for experimenting with constrained correlation functions, partial autocorrelations, admissibility bounds, and Fisher coordinates.

The central transformations implemented by the package are

$$
r \quad\longleftrightarrow\quad \alpha \quad\longleftrightarrow\quad y,
$$

where `r` denotes correlation coefficients, `alpha` partial autocorrelations, and `y` Fisher coordinates.

## Installation

Clone the repository and install the package in editable mode:

```bash id="g0sj0j"
git clone https://github.com/terben/ccf.git
cd ccf
pip install -e .
```

The core package depends only on NumPy.

Optional dependencies for figure reproduction, symbolic calculations, arbitrary-precision arithmetic, and testing can be installed separately; see [Optional dependencies](#optional-dependencies).

Alternatively, `./install.sh` sets up the complete conda environment from `ccf.yml` and runs the test suite.

## Quick start

### Generate an admissible correlation sequence

Any sequence of partial autocorrelations with `abs(alpha) < 1` defines an interior point of the admissible region:

```python id="z03wzv"
import numpy as np
import ccf

alpha = np.array([0.5, -0.3, 0.2])
r = ccf.from_pacf(alpha)

print(r)
```

Transforming back recovers the original coordinates:

```python id="3kvp1q"
alpha_recovered = ccf.pacf(r)
print(alpha_recovered)
```

### Inspect the admissible bounds

For a sequence containing (r_1,\ldots,r_N), `admissible_bounds` returns the admissible intervals for the supplied coefficients and, as the final entry, the interval for (r_{N+1}):

```python id="eqxux7"
lower, upper = ccf.admissible_bounds(r)

print("next admissible interval:", lower[-1], upper[-1])
```

### Transform to unconstrained coordinates

For an interior point, transform between partial autocorrelations and Fisher coordinates with

```python id="yjmgmv"
y = ccf.fisher(alpha)
alpha_recovered = ccf.inverse_fisher(y)
```

Thus the main numerical transformations are simply

```text id="0g6i6x"
r  <->  alpha  <->  y
```

## API at a glance

| Task                             | Function                     |
| -------------------------------- | ---------------------------- |
| correlation coefficients → PACFs | `ccf.pacf(r)`                |
| PACFs → correlation coefficients | `ccf.from_pacf(alpha)`       |
| admissible intervals             | `ccf.admissible_bounds(r)`   |
| check admissibility              | `ccf.check_admissibility(r)` |
| Fisher coordinates               | `ccf.fisher(alpha)`          |
| inverse Fisher transform         | `ccf.inverse_fisher(y)`      |

The extended API provides diagnostics and boundary handling (`pacf_status`, `pacf_prefix`, `extend_at_boundary`), innovation variances and Jacobians, admissible-region volumes, and arbitrary-precision transformations.

## Tutorial

For a more extensive executable walkthrough, see

```text id="t5ndvg"
examples/ccf_api_tutorial.py
```

The tutorial is a VS Code / Spyder notebook-style Python script that can be run as a normal script or explored cell by cell. It covers the main transformations as well as batch operations, boundary cases, admissible intervals, Fisher coordinates, Jacobians, arbitrary precision, and symbolic checks.

For mathematical derivations, refer to the paper. The files `docs/notation.md` and `docs/boundary_semantics.md` document the exact correspondence between paper notation and code conventions and the treatment of degenerate boundary sequences.

## Reference implementation

For readers who want to compare the implementation directly with the equations in the paper, `ccf.reference` provides

```python id="apj03f"
from ccf.reference import pacf_reference, from_pacf_reference
```

These are short, single-sequence implementations that follow the Levinson–Durbin recursion in the paper line by line. The main `ccf.pacf` and `ccf.from_pacf` functions are the robust, batched, boundary-aware implementations used by the package.

## Reproducing the paper figures

For a quick reproduction of the numerical figures:

```bash id="kmzq40"
python paper_plot_scripts/figure_geometry.py
python paper_plot_scripts/figure_roundtrip.py --quick
python paper_plot_scripts/figure_gaussianization.py --quick
```

Output is written to `figs/`. The `--quick` option uses reduced sample sizes for a fast functional check. Omit it, or use `--paper` where available, to run the publication calculations at their full sample sizes.

For `figure_roundtrip.py`, use `-j` / `--jobs` to control the number of worker processes used for the arbitrary-precision trials.

## Optional dependencies

The NumPy-based core is installed with

```bash id="z9kcyz"
pip install -e .
```

Additional functionality is available through extras:

```bash id="s4ipz6"
pip install -e ".[plots]"      # matplotlib, scipy — figure reproduction
pip install -e ".[symbolic]"   # sympy — symbolic checks
pip install -e ".[precision]"  # mpmath — arbitrary precision
pip install -e ".[test]"       # pytest, statsmodels, and all of the above
pip install -e ".[all]"        # plots + symbolic + precision
```

At high order or close to the boundary of the admissible region, the Levinson–Durbin recursion can become ill-conditioned in `float64`. For such cases, `ccf.pacf_mp` and `ccf.from_pacf_mp` provide arbitrary-precision alternatives; `ccf.recommended_dps` gives a suitable working precision.

## Tests

```bash id="q2hl7d"
pip install -e ".[test]"
pytest
```

## Citation

If you use `ccf` in scientific work, please cite the software and the companion paper:

> T. Erben, *Natural Coordinates for Constrained Correlation Functions: Partial Autocorrelations and the Geometry of Positive Power Spectra*, in preparation.

The original constrained-correlation-function formalism is described in:

> P. Schneider & J. Hartlap (2009), *Constrained correlation functions*, Astronomy & Astrophysics, **504**, 705–717.

## License

BSD 3-Clause License.

## Development note

Parts of the AI design, code, documentation, and tests were developed
with the assistance of ChatGPT (OpenAI) and Claude Code (Anthropic).

The mathematical concepts, algorithms, and overall project design were
developed and curated by the project author.
