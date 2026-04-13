"""Tests for sky-visibility planning."""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import HealpixData, create_from_arrays
from rrivis.core.sky.model import SkyFormat, SkyModel
from rrivis.visualization.sky_visibility import SkyVisibilityPlanner


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


def _point_sky(precision):
    ras = np.deg2rad([30.0, 45.0, 60.0, 120.0])
    decs = np.deg2rad([-30.0, -28.0, -32.0, -30.0])
    fluxes = np.array([10.0, 5.0, 8.0, 1.0])
    zeros = np.zeros(4)
    return create_from_arrays(
        ra_rad=ras,
        dec_rad=decs,
        flux=fluxes,
        spectral_index=zeros,
        stokes_q=zeros,
        stokes_u=zeros,
        stokes_v=zeros,
        model_name="test_points",
        brightness_conversion="planck",
        precision=precision,
    )


def _combined_sky(precision, *, coordinate_frame: str = "icrs"):
    point_sky = _point_sky(precision)
    nside = 4
    npix = hp.nside2npix(nside)
    maps = np.linspace(10.0, 100.0, npix, dtype=np.float32)[None, :]
    return SkyModel(
        point=point_sky.point,
        healpix=HealpixData(
            maps=maps,
            nside=nside,
            frequencies=np.array([150e6], dtype=np.float64),
            coordinate_frame=coordinate_frame,
        ),
        source_format=SkyFormat.HEALPIX,
        reference_frequency=150e6,
        model_name="combined_sky",
        _precision=precision,
    )


class TestSkyVisibilityPlanner:
    def test_summary_plan_tracks_visibility_metrics(self, precision):
        planner = SkyVisibilityPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=5.0,
            frequency_mhz=150.0,
            field_radius_deg=5.0,
            sky_model=_point_sky(precision),
            footprint_step_seconds=3600.0,
            snapshot_step_seconds=7200.0,
            top_n_sources=2,
        )

        plan = planner.build()

        assert plan.title.startswith("Sky Visibility")
        assert plan.source_metrics is not None
        assert len(plan.track_labels) == 5
        assert plan.footprint_mask.any()
        assert plan.source_metrics.visible_any[0]
        assert not plan.source_metrics.visible_any[3]
        assert list(plan.source_metrics.top_visible_indices) == [0, 1]
        assert len(plan.snapshots) == 3

    def test_rectangular_approx_masks_ra_interval(self, precision):
        planner = SkyVisibilityPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=23.0,
            lst_end_hours=1.0,
            frequency_mhz=150.0,
            field_radius_deg=5.0,
            sky_model=_point_sky(precision),
            footprint_model="rectangular_approx",
        )

        plan = planner.build()
        ra_grid = plan.ra_grid_deg
        dec_grid = plan.dec_grid_deg
        ra_idx = int(np.where(np.isclose(ra_grid, -15.0))[0][0])
        dec_idx = int(np.where(np.isclose(dec_grid, -30.0))[0][0])
        assert plan.footprint_mask[dec_idx, ra_idx]

    def test_background_projection_accepts_galactic_metadata(self, precision):
        planner = SkyVisibilityPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=2.0,
            frequency_mhz=150.0,
            field_radius_deg=5.0,
            sky_model=_combined_sky(precision, coordinate_frame="galactic"),
            background_layer="diffuse",
        )

        plan = planner.build()

        assert plan.projected_background is not None
        assert plan.projected_background.ndim == 2

    def test_analytic_beam_uses_full_sky_projection(self, precision):
        planner = SkyVisibilityPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=3.0,
            frequency_mhz=150.0,
            field_radius_deg=5.0,
            sky_model=_point_sky(precision),
            beam_diameter_m=14.0,
            beam_config={"beam_mode": "analytic"},
            beam_reference="midpoint",
        )

        plan = planner.build()

        assert plan.beam_projection is not None
        assert plan.beam_reference_ra_deg is not None
        assert plan.beam_projection.ra_grid_deg[0] == pytest.approx(-180.0)
        assert plan.beam_projection.ra_grid_deg[-1] == pytest.approx(180.0)
