# schurcorr

`schurcorr` is a lightweight companion package accompanying the paper

> Natural Coordinates for Constrained Correlation Functions:
> Partial Autocorrelations and the Geometry of Positive Power Spectra

The package implements the correspondence between admissible correlation
functions and partial autocorrelation coefficients (PACFs), providing

- Levinson–Durbin forward recursion (`r -> alpha`) and inverse recursion
  (`alpha -> r`)
- boundary analysis (`pacf_prefix`) and deterministic continuation past the
  degenerate boundary (`extend_at_boundary`)
- Schneider–Hartlap admissible bounds and coordinates
- Fisher coordinates, innovation variances, and Jacobians
- arbitrary-precision recursion (`pacf_mp` / `from_pacf_mp`) for
  ill-conditioned, high-order sequences
- numerical demonstrations accompanying the paper (`paper_plot_scripts/`).

## Quick start

```python
import schurcorr as sc

alpha = sc.pacf(r)
r_reconstructed = sc.from_pacf(alpha)
y = sc.fisher(alpha)
```

See `examples/pacf_roundtrip.py` for a runnable script, including the
boundary-analysis functions.

## Installation

```bash
pip install -e .
```

To also install the test dependencies:

```bash
pip install -e ".[test]"
```

Alternatively, using the provided conda environment:

```bash
conda env create -f schurcorr.yml
conda activate schurcorr
```

## Requirements

* Python ≥ 3.10
* NumPy
* SciPy
* SymPy
* Matplotlib
* mpmath

## License

BSD 3-Clause License.
