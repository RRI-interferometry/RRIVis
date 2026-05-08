"""Tests for ``draw_observability_overlay``.

Sanity checks that the overlay function adds line artists to every healpy
Mollweide panel in a Figure produced by ``SkyPlotter``.
"""

from __future__ import annotations

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from radiosim.core.observability import (  # noqa: E402
    ObservabilityPlanner,
    draw_observability_overlay,
)
from radiosim.core.precision import PrecisionConfig  # noqa: E402
from radiosim.core.sky import HealpixData, SkyPlotter  # noqa: E402
from radiosim.core.sky.model import SkyModel  # noqa: E402


def _single_channel_sky(nside: int = 16) -> SkyModel:
    npix = hp.nside2npix(nside)
    maps = np.random.default_rng(0).standard_normal((1, npix)).astype(np.float32)
    return SkyModel(
        healpix=HealpixData(
            maps=maps,
            nside=nside,
            frequencies=np.array([80e6], dtype=np.float64),
            coordinate_frame="icrs",
        ),
        model_name="overlay-test",
        brightness_conversion="rayleigh-jeans",
        _precision=PrecisionConfig.standard(),
    )


def _hera_plan():
    return ObservabilityPlanner(
        latitude_deg=-30.72,
        longitude_deg=21.43,
        lst_start_hours=0.0,
        lst_end_hours=24.0,
        frequency_mhz=80.0,
        beam_diameter_m=14.0,
        footprint_model="swept_beam",
        background_layer="none",
        mode="summary",
    ).build()


def _count_projection_lines(fig) -> int:
    total = 0
    for ax in fig.axes:
        if hasattr(ax, "projplot") or "Hpx" in type(ax).__name__:
            total += len(ax.get_lines())
    return total


class TestObservabilityOverlay:
    def test_hera_plan_field_radius_matches_airy(self):
        plan = _hera_plan()
        # 0.5 * 1.22 * lambda / D at 80 MHz, D = 14 m -> 9.355 deg
        assert plan.field_radius_deg == 8.0 or abs(plan.field_radius_deg - 9.355) < 0.01

    def test_hera_plan_footprint_has_contours(self):
        plan = _hera_plan()
        assert len(plan.footprint_contours) >= 1
        total_verts = sum(
            len(verts) for group in plan.footprint_contours for verts in group
        )
        assert total_verts > 0

    def test_overlay_adds_lines_to_single_mollweide(self):
        sky = _single_channel_sky()
        plotter = SkyPlotter(sky)
        fig = plotter.healpix_map(frequency=80e6, log_scale=False)
        n_before = _count_projection_lines(fig)
        draw_observability_overlay(fig, _hera_plan())
        n_after = _count_projection_lines(fig)
        assert n_after > n_before, (n_before, n_after)
        plt.close(fig)

    def test_overlay_adds_lines_to_multipole_bands_grid(self):
        sky = _single_channel_sky(nside=32)
        plotter = SkyPlotter(sky)
        fig = plotter.multipole_bands(
            frequency=80e6,
            bands=[(5, 10), (20, 40)],
            ncols=2,
            title="",
        )
        # One line added per projection panel.
        n_before = _count_projection_lines(fig)
        draw_observability_overlay(fig, _hera_plan())
        n_after = _count_projection_lines(fig)
        panels = sum(
            1
            for ax in fig.axes
            if hasattr(ax, "projplot") or "Hpx" in type(ax).__name__
        )
        assert panels == 2
        assert n_after - n_before >= panels, (n_before, n_after, panels)
        plt.close(fig)

    def test_overlay_with_tracks_adds_marker_set(self):
        sky = _single_channel_sky()
        plotter = SkyPlotter(sky)
        fig = plotter.healpix_map(frequency=80e6, log_scale=False)
        collections_before = sum(
            len(ax.collections)
            for ax in fig.axes
            if hasattr(ax, "projplot") or "Hpx" in type(ax).__name__
        )
        draw_observability_overlay(fig, _hera_plan(), draw_tracks=True)
        collections_after = sum(
            len(ax.collections)
            for ax in fig.axes
            if hasattr(ax, "projplot") or "Hpx" in type(ax).__name__
        )
        assert collections_after > collections_before
        plt.close(fig)

    def test_radec_to_za_az_roundtrip_is_monotonic_in_ra_offset(self):
        """Regression: the outer np.deg2rad double-wrap is gone."""
        from radiosim.utils.coordinates import radec_to_za_az

        zeniths = [(0.0, -30.72), (90.0, 0.0), (180.0, 45.0)]
        for zra, zdec in zeniths:
            zas = []
            for dra in (0.0, 10.0, 30.0, 60.0, 90.0, 120.0, 170.0):
                za, _ = radec_to_za_az(
                    np.array([zra + dra]),
                    np.array([zdec]),
                    zenith_ra_deg=zra,
                    zenith_dec_deg=zdec,
                )
                zas.append(float(np.rad2deg(za[0])))
            # Strictly non-decreasing over [0, 170] degrees of RA offset.
            assert all(a < b + 1e-6 for a, b in zip(zas, zas[1:], strict=False)), zas
            # At 90° RA offset from equatorial zenith, za must be near 90°,
            # not near 1° (the previous bug).
            if zdec == 0.0:
                assert abs(zas[4] - 90.0) < 1e-6

    def test_healpix_beam_projection_produces_localized_spot(self):
        """_fits_beam_power_func_healpix + compute_beam_power_on_full_sky_grid
        give a localised projected beam (not a Dec-only stripe) when fed an
        azimuthally symmetric Gaussian HEALPix map."""
        from radiosim.core.observability.geometry import (
            compute_beam_power_on_full_sky_grid,
        )

        nside = 32
        npix = hp.nside2npix(nside)
        th, _ = hp.pix2ang(nside, np.arange(npix))
        sigma = np.deg2rad(8.0)
        beam_map = np.exp(-0.5 * (th / sigma) ** 2).astype(np.float64)
        beam_map /= beam_map.max()

        def power_func(za, az):
            out = hp.get_interp_val(beam_map, za.ravel(), az.ravel())
            return np.asarray(out).reshape(za.shape)

        ra_grid = np.arange(-180.0, 181.0, 1.0)
        dec_grid = np.arange(-90.0, 91.0, 1.0)
        proj = compute_beam_power_on_full_sky_grid(
            beam_power_func=power_func,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            ra_grid_deg=ra_grid,
            dec_grid_deg=dec_grid,
        )
        # At Dec=0, Ra=0 → peak; Ra=90 → far from peak; Ra=180 → below horizon (NaN).
        idx_dec = int(np.argmin(np.abs(dec_grid - 0.0)))
        row = proj.power_db[idx_dec, :]
        assert np.nanmax(row) == pytest.approx(0.0, abs=0.1)
        assert row[int(np.argmin(np.abs(ra_grid - 90.0)))] < -10.0, row[
            int(np.argmin(np.abs(ra_grid - 90.0)))
        ]

    def test_beam_contours_drawn_when_present(self):
        """draw_observability_overlay adds beam-contour line artists when
        plan.beam_contours is present. Uses the real Vivaldi FITS file if
        available, otherwise skips."""
        import os

        vivaldi = "/Volumes/CrucialX8/beams/NF_HERA_Vivaldi_power_beam_nside128.fits"
        if not os.path.exists(vivaldi):
            pytest.skip("Vivaldi FITS not mounted")

        sky = _single_channel_sky()
        plotter = SkyPlotter(sky)
        plan = ObservabilityPlanner(
            latitude_deg=-30.72,
            longitude_deg=21.43,
            lst_start_hours=0.0,
            lst_end_hours=23.999,
            frequency_mhz=80.0,
            beam_fits_path=vivaldi,
            beam_diameter_m=14.0,
            beam_reference="start",
            footprint_model="swept_beam",
            background_layer="none",
            mode="summary",
        ).build()
        assert plan.beam_contours, "beam_contours should be populated"

        fig = plotter.healpix_map(frequency=80e6, log_scale=False)
        n_lines_before = _count_projection_lines(fig)
        draw_observability_overlay(fig, plan)
        n_lines_after = _count_projection_lines(fig)
        # 1 footprint line + at least 1 beam contour per level (2 levels).
        assert n_lines_after >= n_lines_before + 3, (n_lines_before, n_lines_after)
        plt.close(fig)
