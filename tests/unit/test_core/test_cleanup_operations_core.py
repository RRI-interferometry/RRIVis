"""Tests for the operations-core Phase-2 cleanup (spec items F1-F4, F6-F8, B2, D5).

Covers:
- F1: parallel-loading machinery relocated to ``operations/parallel.py``.
- F2: ``compute_linear_polarization`` relocated to
  ``diagnostics/polarization.py``.
- D5/F7: executor recommendation derived from registry loader category.
- F3: ``operations/__init__.py`` populated.
- F4: ``materialize_point_sources_model`` logs when it returns input unchanged.
- F6: ``BoxRegion`` out-of-range handling symmetric (clamp both dims).
- F8: ``create_from_freq_dict_maps`` exposes explicit params.
- B2: ``region.py`` uses the support pixel-solid-angle helper.
"""

from __future__ import annotations

import inspect
import logging

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers import HealpixData
from radiosim.core.sky.containers.model import SkyModel


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


# ---------------------------------------------------------------------------
# F1: parallel machinery relocation
# ---------------------------------------------------------------------------


class TestParallelRelocation:
    def test_parallel_module_exports_machinery(self):
        from radiosim.core.sky.operations import parallel

        assert hasattr(parallel, "SkyLoadError")
        assert hasattr(parallel, "SkyLoadAggregateError")
        assert hasattr(parallel, "recommend_executor_for_loaders")
        assert hasattr(parallel, "load_models_parallel")

    def test_factories_no_longer_hosts_parallel_symbols(self):
        from radiosim.core.sky.operations import factories

        assert not hasattr(factories, "load_models_parallel")
        assert not hasattr(factories, "recommend_executor_for_loaders")
        assert not hasattr(factories, "SkyLoadError")
        assert not hasattr(factories, "SkyLoadAggregateError")
        assert not hasattr(factories, "_GIL_BOUND_LOADERS")

    def test_aggregate_error_is_runtime_error(self):
        from radiosim.core.sky.operations.parallel import (
            SkyLoadAggregateError,
            SkyLoadError,
        )

        failure = SkyLoadError(
            loader_name="gleam",
            kwargs={},
            exception=ValueError("boom"),
            traceback_text="",
        )
        err = SkyLoadAggregateError([failure])
        assert isinstance(err, RuntimeError)
        assert "gleam" in str(err)


# ---------------------------------------------------------------------------
# F3: operations package re-exports
# ---------------------------------------------------------------------------


class TestOperationsPackageSurface:
    def test_operations_init_reexports(self):
        from radiosim.core.sky import operations as ops

        for name in (
            "create_from_arrays",
            "create_from_freq_dict_maps",
            "materialize_healpix_model",
            "materialize_point_sources_model",
            "SkyRegion",
            "BoxRegion",
            "subtract_bright_sources",
            "load_models_parallel",
            "recommend_executor_for_loaders",
        ):
            assert name in ops.__all__
            assert hasattr(ops, name)


# ---------------------------------------------------------------------------
# F2: compute_linear_polarization relocation
# ---------------------------------------------------------------------------


class TestLinearPolRelocation:
    def test_lives_in_diagnostics(self):
        from radiosim.core.sky.diagnostics import compute_linear_polarization
        from radiosim.core.sky.diagnostics.polarization import (
            compute_linear_polarization as direct,
        )

        assert compute_linear_polarization is direct

    def test_not_defined_in_operations_module(self):
        from radiosim.core.sky.operations import operations

        # The symbol moved out of operations.py entirely.
        assert "compute_linear_polarization" not in vars(operations)

    def test_still_computes_correctly(self, precision):
        from radiosim.core.sky.diagnostics import compute_linear_polarization

        nside = 8
        npix = hp.nside2npix(nside)
        sky = SkyModel(
            healpix=HealpixData(
                maps=np.full((1, npix), 4.0, dtype=np.float64),
                q_maps=np.full((1, npix), 0.0, dtype=np.float64),
                u_maps=np.full((1, npix), 2.0, dtype=np.float64),
                nside=nside,
                frequencies=np.array([150e6]),
            ),
            model_name="pol",
            precision=precision,
        )
        out = compute_linear_polarization(sky, frequency=150e6)
        np.testing.assert_allclose(out["P"], 2.0)
        np.testing.assert_allclose(out["chi_deg"], 45.0)
        np.testing.assert_allclose(out["frac_pol"], 0.5)


# ---------------------------------------------------------------------------
# D5 / F7: registry-category-derived executor recommendation
# ---------------------------------------------------------------------------


class TestExecutorRecommendation:
    def test_pure_catalog_loads_use_thread(self):
        from radiosim.core.sky.operations.parallel import (
            recommend_executor_for_loaders,
        )

        assert recommend_executor_for_loaders([("gleam", {}), ("nvss", {})]) == "thread"

    def test_diffuse_category_recommends_process(self):
        from radiosim.core.sky.operations.parallel import (
            recommend_executor_for_loaders,
        )

        # ``diffuse_sky`` carries category="diffuse"; the recommender must
        # pick a process pool purely from that category, with no hardcoded
        # loader-name list.
        assert (
            recommend_executor_for_loaders([("gleam", {}), ("diffuse_sky", {})])
            == "process"
        )

    def test_file_category_recommends_process(self):
        from radiosim.core.sky.operations.parallel import (
            recommend_executor_for_loaders,
        )

        assert (
            recommend_executor_for_loaders([("pyradiosky_file", {"filename": "x"})])
            == "process"
        )

    def test_decision_uses_registry_category_not_hardcoded_set(self):
        # No module-level frozenset of loader names should drive the
        # decision; the category set is the single source of truth.
        from radiosim.core.sky.operations import parallel

        assert not hasattr(parallel, "_GIL_BOUND_LOADERS")
        assert hasattr(parallel, "_GIL_BOUND_CATEGORIES")
        assert "diffuse" in parallel._GIL_BOUND_CATEGORIES

    def test_unknown_loader_skipped_gracefully(self):
        from radiosim.core.sky.operations.parallel import (
            recommend_executor_for_loaders,
        )

        assert (
            recommend_executor_for_loaders([("__no_such_loader__", {}), ("gleam", {})])
            == "thread"
        )


# ---------------------------------------------------------------------------
# F4: materialize_point_sources_model logs unchanged-return
# ---------------------------------------------------------------------------


class TestMaterializeNoOpLog:
    def test_logs_when_returning_input_unchanged(self, precision, caplog):
        from radiosim.core.sky.operations.factories import create_from_arrays
        from radiosim.core.sky.operations.operations import (
            materialize_point_sources_model,
        )

        sky = create_from_arrays(
            ra_rad=np.array([0.1, 0.2]),
            dec_rad=np.array([-0.5, -0.4]),
            flux=np.array([1.0, 2.0]),
            precision=precision,
        )

        with caplog.at_level(logging.INFO):
            out = materialize_point_sources_model(sky)

        assert out is sky
        assert any(
            "already has a point payload" in rec.getMessage() for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# F6: BoxRegion symmetric out-of-range handling
# ---------------------------------------------------------------------------


class TestBoxRegionSymmetricClamp:
    def test_oversize_width_clamps_to_360(self):
        from radiosim.core.sky.operations.region import SkyRegion

        box = SkyRegion.box(ra_deg=180.0, dec_deg=0.0, width_deg=720.0, height_deg=20.0)
        assert box.width.deg == pytest.approx(360.0)

    def test_oversize_height_clamps_to_180(self):
        from radiosim.core.sky.operations.region import SkyRegion

        # Behavior change: previously raised; now clamps, symmetric with width.
        box = SkyRegion.box(ra_deg=180.0, dec_deg=0.0, width_deg=20.0, height_deg=400.0)
        assert box.height.deg == pytest.approx(180.0)

    def test_both_dimensions_handled_the_same_way(self):
        from radiosim.core.sky.operations.region import SkyRegion

        # Neither oversize dimension raises; both saturate at their max.
        box = SkyRegion.box(ra_deg=0.0, dec_deg=0.0, width_deg=720.0, height_deg=400.0)
        assert box.width.deg == pytest.approx(360.0)
        assert box.height.deg == pytest.approx(180.0)

    def test_nonpositive_still_raises(self):
        from radiosim.core.sky.operations.region import SkyRegion

        with pytest.raises(ValueError):
            SkyRegion.box(ra_deg=0.0, dec_deg=0.0, width_deg=0.0, height_deg=10.0)
        with pytest.raises(ValueError):
            SkyRegion.box(ra_deg=0.0, dec_deg=0.0, width_deg=10.0, height_deg=-1.0)


# ---------------------------------------------------------------------------
# B2: region uses the support pixel_solid_angle helper
# ---------------------------------------------------------------------------


class TestRegionPixelSolidAngle:
    def test_region_imports_support_helper(self):
        from radiosim.core.sky.operations import region
        from radiosim.core.sky.support.healpix_geometry import pixel_solid_angle

        assert region.pixel_solid_angle is pixel_solid_angle

    def test_union_area_matches_pixel_count_times_solid_angle(self):
        from radiosim.core.sky.operations.region import SkyRegion
        from radiosim.core.sky.support.healpix_geometry import pixel_solid_angle

        union = SkyRegion.union(
            [
                SkyRegion.cone(30.0, 10.0, 3.0),
                SkyRegion.box(200.0, -20.0, 10.0, 10.0),
            ]
        )
        nside = 128
        mask = union.healpix_mask(nside)
        expected = float(int(mask.sum()) * pixel_solid_angle(nside))
        assert union.area_sr(nside=nside) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# F8: create_from_freq_dict_maps explicit params
# ---------------------------------------------------------------------------


class TestFreqDictMapsExplicitParams:
    def test_no_var_keyword_param(self):
        from radiosim.core.sky.operations.factories import create_from_freq_dict_maps

        sig = inspect.signature(create_from_freq_dict_maps)
        kinds = {p.kind for p in sig.parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD not in kinds
        # explicit named params present
        for name in (
            "precision",
            "coordinate_frame",
            "model_name",
            "reference_frequency",
            "brightness_conversion",
            "provenance",
        ):
            assert name in sig.parameters

    def test_builds_healpix_model_with_sorted_freqs(self, precision):
        from radiosim.core.sky.operations.factories import create_from_freq_dict_maps

        nside = 4
        npix = hp.nside2npix(nside)
        i_maps = {
            200e6: np.full(npix, 3.0),
            100e6: np.full(npix, 1.0),
        }
        sky = create_from_freq_dict_maps(
            i_maps,
            None,
            None,
            None,
            nside,
            precision=precision,
            model_name="dict_maps",
        )
        assert sky.healpix is not None
        # frequencies ascending
        np.testing.assert_array_equal(sky.healpix.frequencies, np.array([100e6, 200e6]))
        # channel order follows sorted freqs
        np.testing.assert_allclose(sky.healpix.maps[0], 1.0)
        np.testing.assert_allclose(sky.healpix.maps[1], 3.0)
        assert sky.model_name == "dict_maps"
