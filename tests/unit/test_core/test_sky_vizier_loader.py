"""Tests for VizieR loader internals through the public helper entry point."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from astropy.table import Table

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.loaders.vizier import core as vizier_core
from radiosim.core.sky.loaders.vizier import point_catalogs as vizier_point_catalogs
from radiosim.core.sky.registry import loader_registry


def test_vizier_loader_extracts_sources_from_fetched_catalog(monkeypatch):
    catalog = Table(
        {
            "RAJ2000": [180.0, 181.0, 182.0],
            "DEJ2000": [-30.0, -31.0, -32.0],
            "S1.4": [2500.0, 500.0, np.nan],
            "MajAxis": [30.0, 40.0, 50.0],
            "MinAxis": [20.0, 25.0, 30.0],
            "PA": [45.0, 50.0, 55.0],
            "source_name": ["src-a", "src-b", "src-c"],
            "source_id": ["id-a", "id-b", "id-c"],
        }
    )

    def fake_fetch_vizier_catalog(**kwargs):
        assert kwargs["catalog_key"] == "nvss"
        assert kwargs["max_rows"] == 10
        return catalog

    monkeypatch.setattr(
        vizier_core,
        "_fetch_vizier_catalog",
        fake_fetch_vizier_catalog,
    )

    sky = vizier_core._load_from_vizier_catalog(
        "nvss",
        flux_limit=1.0,
        max_rows=10,
        precision=PrecisionConfig.standard(),
    )

    assert sky.n_point_sources == 1
    assert sky.reference_frequency == 1_400_000_000.0
    assert sky.point is not None
    np.testing.assert_allclose(np.rad2deg(sky.point.ra_rad), [180.0])
    np.testing.assert_allclose(np.rad2deg(sky.point.dec_rad), [-30.0])
    np.testing.assert_allclose(sky.point.flux, [2.5])
    assert sky.point.morphology is not None
    np.testing.assert_allclose(sky.point.morphology.major_arcsec, [30.0])
    np.testing.assert_allclose(sky.point.morphology.minor_arcsec, [20.0])
    np.testing.assert_allclose(sky.point.morphology.pa_deg, [45.0])
    assert sky.point.metadata is not None
    np.testing.assert_array_equal(sky.point.metadata.source_name, np.array(["src-a"]))
    np.testing.assert_array_equal(sky.point.metadata.source_id, np.array(["id-a"]))


# =========================================================================
# SKY-001 regression: the registered VizieR wrappers must call
# ``_load_from_vizier_catalog`` in a way its normalized signature accepts.
#
# Commit 7b02bb2 made ``precision`` keyword-only on the private helper while
# every wrapper call site still passed it positionally, so all ten registered
# VizieR point-catalog loaders raised ``TypeError`` before any network access.
# =========================================================================

_SENTINEL = object()

# (registered loader name, extra call kwargs, expected catalog_key)
_VIZIER_POINT_LOADER_CASES = [
    ("gleam", {}, "gleam_egc"),
    ("gleam", {"catalog": "gleam_x_dr1"}, "gleam_x_dr1"),
    ("mals", {}, "mals_dr2"),
    ("mals", {"release": "dr1"}, "mals_dr1"),
    ("lotss", {}, "lotss_dr2"),
    ("lotss", {"release": "dr1"}, "lotss_dr1"),
    ("vlssr", {}, "vlssr"),
    ("tgss", {}, "tgss"),
    ("wenss", {}, "wenss"),
    ("sumss", {}, "sumss"),
    ("nvss", {}, "nvss"),
    ("3c", {}, "3c"),
    ("vlass", {}, "vlass"),
]


def _install_signature_checking_spy(monkeypatch):
    """Replace the private helper with a spy that enforces its real signature.

    The spy binds every call against ``inspect.signature`` of the *real*
    ``_load_from_vizier_catalog``, so a wrapper that passes a keyword-only
    parameter positionally raises ``TypeError`` exactly as the real helper
    would -- without the spy having to restate the signature by hand.
    """
    real_signature = inspect.signature(vizier_core._load_from_vizier_catalog)
    calls: list[dict] = []

    def spy(*args, **kwargs):
        bound = real_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        return _SENTINEL

    monkeypatch.setattr(
        vizier_point_catalogs,
        "_load_from_vizier_catalog",
        spy,
    )
    return calls


@pytest.mark.parametrize(
    ("loader_name", "extra_kwargs", "expected_catalog_key"),
    _VIZIER_POINT_LOADER_CASES,
    ids=[f"{name}-{key}" for name, _, key in _VIZIER_POINT_LOADER_CASES],
)
def test_vizier_point_loader_calls_helper_with_conforming_signature(
    monkeypatch, loader_name, extra_kwargs, expected_catalog_key
):
    """Every registered VizieR point loader must satisfy the helper signature."""
    calls = _install_signature_checking_spy(monkeypatch)
    precision = PrecisionConfig.standard()

    loader = loader_registry.loader(loader_name)
    result = loader(precision=precision, max_rows=5, **extra_kwargs)

    assert result is _SENTINEL
    assert len(calls) == 1
    arguments = calls[0]
    assert arguments["catalog_key"] == expected_catalog_key
    assert arguments["precision"] is precision
    assert arguments["max_rows"] == 5
    assert arguments["brightness_conversion"] == "planck"


def test_vizier_point_loader_forwards_every_pass_through_argument(monkeypatch):
    """The wrapper must forward all optional arguments, not just ``precision``."""
    calls = _install_signature_checking_spy(monkeypatch)
    precision = PrecisionConfig.precise()

    loader_registry.loader("nvss")(
        flux_limit=0.5,
        brightness_conversion="rayleigh-jeans",
        precision=precision,
        max_rows=None,
        allow_full_catalog=True,
    )

    arguments = calls[0]
    assert arguments["flux_limit"] == 0.5
    assert arguments["brightness_conversion"] == "rayleigh-jeans"
    assert arguments["precision"] is precision
    assert arguments["allow_full_catalog"] is True
    assert arguments["region"] is None


def test_load_nvss_wrapper_reaches_the_fetch_boundary(monkeypatch):
    """A simple factory-built wrapper runs end to end against a mocked fetch."""
    catalog = Table(
        {
            "RAJ2000": [10.0, 11.0],
            "DEJ2000": [-20.0, -21.0],
            "S1.4": [3000.0, 100.0],
            "MajAxis": [10.0, 11.0],
            "MinAxis": [5.0, 6.0],
            "PA": [15.0, 16.0],
        }
    )
    fetched: list[str] = []

    def fake_fetch_vizier_catalog(**kwargs):
        fetched.append(kwargs["catalog_key"])
        return catalog

    monkeypatch.setattr(
        vizier_core,
        "_fetch_vizier_catalog",
        fake_fetch_vizier_catalog,
    )

    sky = loader_registry.loader("nvss")(
        flux_limit=1.0,
        max_rows=10,
        precision=PrecisionConfig.standard(),
    )

    assert fetched == ["nvss"]
    assert sky.n_point_sources == 1
    assert sky.point is not None
    np.testing.assert_allclose(np.rad2deg(sky.point.ra_rad), [10.0])
    np.testing.assert_allclose(sky.point.flux, [3.0])
    assert sky.reference_frequency == 1_400_000_000.0


def test_load_gleam_wrapper_reaches_the_fetch_boundary(monkeypatch):
    """The explicitly written ``gleam`` wrapper runs end to end to the mock.

    ``configs/realistic_foreground_example.yaml`` reaches VizieR through this
    wrapper, so it is the representative end-to-end case for SKY-001.
    """
    catalog = Table(
        {
            "RAJ2000": [200.0, 201.0],
            "DEJ2000": [-25.0, -26.0],
            "Fpwide": [5.0, 0.1],
            "alpha": [-0.9, -0.6],
        }
    )
    fetched: list[str] = []

    def fake_fetch_vizier_catalog(**kwargs):
        fetched.append(kwargs["catalog_key"])
        return catalog

    monkeypatch.setattr(
        vizier_core,
        "_fetch_vizier_catalog",
        fake_fetch_vizier_catalog,
    )

    sky = vizier_point_catalogs.load_gleam(
        flux_limit=1.0,
        max_rows=10,
        precision=PrecisionConfig.standard(),
    )

    assert fetched == ["gleam_egc"]
    assert sky.n_point_sources == 1
    assert sky.point is not None
    np.testing.assert_allclose(np.rad2deg(sky.point.ra_rad), [200.0])
    np.testing.assert_allclose(sky.point.flux, [5.0])
    np.testing.assert_allclose(sky.point.spectral_index, [-0.9])
    assert sky.reference_frequency == 200_000_000.0
