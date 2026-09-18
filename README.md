# ccf — Constrained Correlation Functions


`ccf` is the reference implementation accompanying the paper *Natural Coordinates for Constrained Correlation Functions: Partial Autocorrelations and the Geometry of Positive Power Spectra*.

It provides the numerical tools used in the paper and a small API for experimenting with constrained correlation functions, partial autocorrelations, admissibility bounds, and Fisher coordinates.

The central transformations implemented by the package are

$$
r \quad\longleftrightarrow\quad \alpha \quad\longleftrightarrow\quad y,
$$

where $r=(r_1,\ldots,r_N)$ denotes the correlation coefficients, $\alpha=(\alpha_1,\ldots,\alpha_N)$ the corresponding partial autocorrelations, and $y=(y_1,\ldots,y_N)$ their Fisher coordinates.

The paper's central result is the identification of $\alpha_n$ with the coordinate $x_n$ introduced by Schneider & Hartlap (2009), $x_n = \alpha_n$, so the Fisher coordinate $y_n = \text{atanh}(\alpha_n)$ is exactly their real-line coordinate.

## Installation

Clone the repository and install the package in editable mode:

```bash id="g0sj0j"
git clone https://github.com/terben/ccf.git
cd ccf
pip install -e .
```

The core package depends only on NumPy.

Optional dependencies for figure reproduction, symbolic calculations, arbitrary-precision arithmetic, and testing can be installed separately; see [Optional dependencies](#optional-dependencies).

Alternatively, if Conda is available, set up the complete development environment and run the test suite with

```bash
./install.sh
```

## Quick start

### Generate an admissible correlation sequence

Any finite sequence of partial autocorrelations satisfying $|\alpha_k| < 1$ for every $k$ defines an interior point of the corresponding admissible correlation region:

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

For a sequence containing $r_1,\ldots,r_N$, `admissible_bounds` returns, for each coefficient, the admissible interval $p_n - \sigma_n^2 \le r_n \le p_n + \sigma_n^2$, where $p_n$ is the linear prediction of $r_n$ and $\sigma_n^2$ the residual variance, and, as the final entry, the interval for $r_{N+1}$:

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

Because $x_n = \alpha_n$, `ccf.fisher(alpha)` implements $y_n = \text{atanh}(\alpha_n)$, exactly the Schneider--Hartlap Fisher coordinate.

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

The extended API provides diagnostics and boundary handling (`pacf_status`, `pacf_prefix`, `extend_at_boundary`), residual variances (`innovation_variances`) and Jacobians, admissible-region volumes, and arbitrary-precision transformations.

A correlation sequence can also reach a degenerate boundary: at the first singular Toeplitz matrix $A_m$, the residual variance vanishes ($\sigma_m^2 = 0$) and the last independent PACF coefficient saturates ($|\alpha_{m-1}| = 1$), so no further independent PACF coordinates exist beyond order $m-1$. Because $A_m$ contains only $r_0,\ldots,r_{m-1}$, not $r_m$, any correlation coefficients supplied beyond that order are not free -- they may still have a uniquely forced Toeplitz continuation. `pacf_status`, `pacf_prefix`, and `extend_at_boundary` detect and handle this case rather than raising; see [`docs/boundary_semantics.md`](docs/boundary_semantics.md) for the full semantics.

## Tutorial

For a more extensive executable walkthrough, see

[`examples/ccf_api_tutorial.py`](examples/ccf_api_tutorial.py)

The tutorial is a VS Code / Spyder notebook-style Python script that can be run as a normal script or explored cell by cell. It covers the main transformations as well as batch operations, boundary cases, admissible intervals, Fisher coordinates, Jacobians, arbitrary precision, and symbolic checks.

For mathematical derivations, refer to the paper. The files [`docs/notation.md`](docs/notation.md) and [`docs/boundary_semantics.md`](docs/boundary_semantics.md) document the exact correspondence between paper notation and code conventions and the treatment of degenerate boundary sequences.

## Reference implementation

For readers who want to compare the implementation directly with the equations in the paper, `ccf.reference` provides

```python id="apj03f"
from ccf.reference import pacf_reference, from_pacf_reference
```

These are short, single-sequence implementations that closely follow the Levinson–Durbin recursion in the paper. The main `ccf.pacf` and `ccf.from_pacf` functions are the robust, batched, boundary-aware implementations used by the package.

## Reproducing the paper figures

The plotting scripts use LaTeX for publication-quality text rendering and therefore require a working LaTeX installation including the `amsmath` and `siunitx` packages.

For a quick reproduction of all three numerical figures:

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
python -m pip install -e .
```

Additional functionality is available through extras:

```bash id="s4ipz6"
python -m pip install -e ".[plots]"      # matplotlib, scipy — figure reproduction
python -m pip install -e ".[symbolic]"   # sympy — symbolic checks
python -m pip install -e ".[precision]"  # mpmath — arbitrary precision
python -m pip install -e ".[test]"       # pytest, statsmodels, and all of the above
python -m pip install -e ".[all]"        # plots + symbolic + precision
```

At high order or close to the boundary of the admissible region, the Levinson–Durbin recursion can become ill-conditioned in `float64`. For such cases, `ccf.pacf_mp` and `ccf.from_pacf_mp` provide arbitrary-precision alternatives; `ccf.recommended_dps` gives a suitable working precision.

## Tests

```bash id="q2hl7d"
pip install -e ".[test]"
pytest
```

## Citation

If you use `ccf` in scientific work, please cite the software and the companion paper:

> T. Erben (2026), *ccf: Partial autocorrelations and natural coordinates for correlation functions*, version 1.0.1. Zenodo.

and

> T. Erben, *Natural Coordinates for Constrained Correlation Functions: Partial Autocorrelations and the Geometry of Positive Power Spectra*, in preparation.

The original constrained-correlation-function formalism is described in:

> P. Schneider & J. Hartlap (2009), *Constrained correlation functions*, Astronomy & Astrophysics, **504**, 705–717.

## License

BSD 3-Clause License.

## Development note

Parts of the API design, code, documentation, and tests were developed
with the assistance of ChatGPT (OpenAI) and Claude Code (Anthropic).

The mathematical concepts, algorithms, and overall project design were
developed and curated by the project author.
