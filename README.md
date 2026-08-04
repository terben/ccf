# schurcorr

`schurcorr` is a lightweight companion package accompanying the paper

> Natural Coordinates for Constrained Correlation Functions:
> Partial Autocorrelations and the Geometry of Positive Power Spectra

The package implements the correspondence between admissible correlation
functions and partial autocorrelation coefficients (PACFs), providing

- Levinson–Durbin forward recursion (`r -> alpha`)
- inverse recursion (`alpha -> r`)
- Schneider–Hartlap admissible bounds
- Fisher coordinates
- Jacobians
- numerical demonstrations accompanying the paper.

## Installation

```bash
pip install -e ".[test]"

## Requirements

* Python ≥ 3.10
* NumPy
* SciPy
* SymPy
* Matplotlib
* mpmath

## License

BSD 3-Clause License.
