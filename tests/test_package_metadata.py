"""Tests for package metadata and the public import surface."""

import importlib.metadata

import schurcorr as sc


def test_version_matches_installed_metadata():
    assert sc.__version__ == importlib.metadata.version("schurcorr")


def test_version_is_single_sourced_from_pyproject():
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    assert sc.__version__ == data["project"]["version"]


def test_all_exported_names_are_importable_attributes():
    for name in sc.__all__:
        assert hasattr(sc, name), f"{name!r} listed in __all__ but missing"


def test_mpmath_dependent_names_present_when_mpmath_installed():
    import mpmath  # noqa: F401 -- presence check for the "precision" extra

    for name in ("pacf_mp", "from_pacf_mp", "recommended_dps"):
        assert hasattr(sc, name)
        assert name in sc.__all__
