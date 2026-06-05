"""Tests for VizieR loader internals through the public helper entry point."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.loaders.vizier import core as vizier_core


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
