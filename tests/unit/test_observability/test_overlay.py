"""Tests for ``draw_observability_overlay``.

Sanity checks that the overlay function adds line artists to every healpy
Mollweide panel in a Figure produced by ``SkyPlotter``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from radiosim.api.simulator import Simulator  # noqa: E402
from radiosim.core.observability import (  # noqa: E402
    draw_observability_overlay,
)
from radiosim.core.precision import PrecisionConfig  # noqa: E402
from radiosim.core.sky import (  # noqa: E402
    HealpixData,
)
from radiosim.core.sky.containers.model import SkyModel  # noqa: E402
from radiosim.visualization import plot_healpix_map, plot_multipole_bands  # noqa: E402


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
        precision=PrecisionConfig.standard(),
    )


def _hera_plan(tmp_path: Path, *, beam_fits_path: str | None = None):
    antenna_path = tmp_path / "overlay-antennas.txt"
    antenna_path.write_text(
        "Name Number BeamID E N U Diameter\n"
        "ANT0 0 0 0.0 0.0 0.0 14.0\n"
        "ANT1 1 0 14.0 0.0 0.0 14.0\n",
        encoding="utf-8",
    )
    beams = (
        {
            "mode": "analytic",
            "model": {
                "kind": "circular_aperture",
                "taper": {"kind": "uniform"},
            },
        }
        if beam_fits_path is None
        else {
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": beam_fits_path},
        }
    )
    simulator = Simulator.from_mapping(
        {
            "instrument": {
                "source": {
                    "kind": "layout_file",
                    "path": str(antenna_path),
                    "format": "radiosim",
                    "telescope_name": "Overlay Array",
                },
                "location": {
                    "longitude_deg": 21.43,
                    "latitude_deg": -30.72,
                    "height_m": 1073.0,
                },
            },
            "baseline_selection": {"correlations": "cross"},
            "beams": beams,
            "obs_time": {
                "start_time": "2025-01-01T00:00:00",
                "duration_seconds": 60.0,
                "time_step_seconds": 1.0,
            },
            "obs_frequency": {
                "mode": "explicit",
                "channel_frequencies_hz": [80_000_000.0],
            },
            "sky_model": {
                "sources": [{"kind": "test_sources", "num_sources": 1, "seed": 1}]
            },
            "execution": {"backend": "numpy", "offline": True},
        },
        base_dir=tmp_path,
    )
    return simulator.plan_observability(
        footprint_step_seconds=60.0,
        grid_resolution_deg=5.0,
    )


def _count_projection_lines(fig) -> int:
    total = 0
    for ax in fig.axes:
        if hasattr(ax, "projplot") or "Hpx" in type(ax).__name__:
            total += len(ax.get_lines())
    return total


class TestObservabilityOverlay:
    def test_hera_plan_uses_reference_beam_half_power(self, tmp_path):
        plan = _hera_plan(tmp_path)
        assert plan.field_radius_deg is None
        assert plan.footprint_provenance == "reference_beam_half_power"

    def test_hera_plan_footprint_has_contours(self, tmp_path):
        plan = _hera_plan(tmp_path)
        assert len(plan.footprint_contours) >= 1
        total_verts = sum(
            len(verts) for group in plan.footprint_contours for verts in group
        )
        assert total_verts > 0

    def test_overlay_adds_lines_to_single_mollweide(self, tmp_path):
        sky = _single_channel_sky()
        fig = plot_healpix_map(sky, frequency=80e6, log_scale=False)
        n_before = _count_projection_lines(fig)
        draw_observability_overlay(fig, _hera_plan(tmp_path))
        n_after = _count_projection_lines(fig)
        assert n_after > n_before, (n_before, n_after)
        plt.close(fig)

    def test_overlay_adds_lines_to_multipole_bands_grid(self, tmp_path):
        sky = _single_channel_sky(nside=32)
        fig = plot_multipole_bands(
            sky,
            frequency=80e6,
            bands=[(5, 10), (20, 40)],
            ncols=2,
            title="",
        )
        # One line added per projection panel.
        n_before = _count_projection_lines(fig)
        draw_observability_overlay(fig, _hera_plan(tmp_path))
        n_after = _count_projection_lines(fig)
        panels = sum(
            1
            for ax in fig.axes
            if hasattr(ax, "projplot") or "Hpx" in type(ax).__name__
        )
        assert panels == 2
        assert n_after - n_before >= panels, (n_before, n_after, panels)
        plt.close(fig)

    def test_overlay_with_tracks_adds_marker_set(self, tmp_path):
        sky = _single_channel_sky()
        fig = plot_healpix_map(sky, frequency=80e6, log_scale=False)
        collections_before = sum(
            len(ax.collections)
            for ax in fig.axes
            if hasattr(ax, "projplot") or "Hpx" in type(ax).__name__
        )
        draw_observability_overlay(fig, _hera_plan(tmp_path), draw_tracks=True)
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


class TestTier3GOverlaySurface:
    def test_visualization_wrapper_has_explicit_exact_signature(self):
        from radiosim.visualization.sky import overlay_observability

        assert tuple(inspect.signature(overlay_observability).parameters) == (
            "fig",
            "plan",
            "color",
            "linestyle",
            "linewidth",
            "alpha",
            "draw_footprint",
            "draw_beam",
            "beam_color",
            "beam_linestyle",
            "beam_linewidths",
            "beam_alpha",
            "draw_tracks",
            "track_color",
            "track_marker_size",
        )

    def test_visualization_package_does_not_duplicate_core_model_exports(self):
        import radiosim.visualization as visualization

        for core_name in (
            "ObservabilityPlanner",
            "ObservabilityPlan",
            "ObservabilitySnapshot",
            "ObservabilitySourceMetrics",
            "za_ring_points",
            "draw_za_rings_on_figure",
        ):
            assert core_name not in visualization.__all__
            assert not hasattr(visualization, core_name)
