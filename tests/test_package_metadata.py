"""Tests for package metadata and the public import surface."""

import builtins
import importlib
import importlib.metadata
import sys

import pytest

import ccf


def test_version_matches_installed_metadata():
    assert ccf.__version__ == importlib.metadata.version("ccf")


def test_version_is_single_sourced_from_pyproject():
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    assert ccf.__version__ == data["project"]["version"]


def test_all_exported_names_are_importable_attributes():
    for name in ccf.__all__:
        assert hasattr(ccf, name), f"{name!r} listed in __all__ but missing"


def test_mpmath_dependent_names_present_when_mpmath_installed():
    import mpmath  # noqa: F401 -- presence check for the "precision" extra

    for name in ("pacf_mp", "from_pacf_mp", "recommended_dps"):
        assert hasattr(ccf, name)
        assert name in ccf.__all__


def test_import_error_unrelated_to_mpmath_is_not_swallowed(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mpmath":
            raise ModuleNotFoundError(
                "No module named 'not_mpmath'", name="not_mpmath"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "ccf", raising=False)
    monkeypatch.delitem(sys.modules, "ccf.precision", raising=False)

    with pytest.raises(ModuleNotFoundError, match="not_mpmath"):
        importlib.import_module("ccf")
