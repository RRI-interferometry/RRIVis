"""Tests for rrivis.utils.diagnostics.planner — observable strip computation."""

import numpy as np
import pytest

from rrivis.core.sky.model import SkyModel
from rrivis.utils.diagnostics.planner import DiagnosticsPlanner

# ---------------------------------------------------------------------------
# Strip geometry — LST/RA conversion
# ---------------------------------------------------------------------------


class TestLSTtoRA:
    def test_lst_to_ra_simple(self):
        """LST in hours * 15 = RA in degrees."""
        planner = DiagnosticsPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=2.0,
            lst_end_hours=6.0,
            frequency_mhz=150.0,
            fov_radius_deg=5.0,
            background_mode="none",
        )
        strip = planner.compute()
        assert strip.ra_start_deg == pytest.approx(30.0)
        assert strip.ra_end_deg == pytest.approx(90.0)

    def test_lst_zero(self):
        """LST=0 → RA=0."""
        planner = DiagnosticsPlanner(
            latitude_deg=0.0,
            longitude_deg=0.0,
            lst_start_hours=0.0,
            lst_end_hours=1.0,
            frequency_mhz=100.0,
            fov_radius_deg=10.0,
            background_mode="none",
        )
        strip = planner.compute()
        assert strip.ra_start_deg == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dec bounds
# ---------------------------------------------------------------------------


class TestDecBounds:
    def test_dec_bounds(self):
        """Dec bounds = latitude ± fov_radius."""
        planner = DiagnosticsPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=5.0,
            frequency_mhz=100.0,
            fov_radius_deg=8.0,
            background_mode="none",
        )
        strip = planner.compute()
        assert strip.dec_lower_deg == pytest.approx(-38.0)
        assert strip.dec_upper_deg == pytest.approx(-22.0)

    def test_fov_from_diameter(self):
        """FOV radius should be estimated from beam diameter when not given."""
        planner = DiagnosticsPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=5.0,
            frequency_mhz=150.0,
            beam_diameter_m=14.0,
            background_mode="none",
        )
        strip = planner.compute()
        # 1.22 * (c/150e6) / 14 ≈ 0.1742 rad ≈ 9.98° HPBW → ~5° radius
        assert 3.0 < strip.fov_radius_deg < 7.0


# ---------------------------------------------------------------------------
# ObservableStrip dataclass
# ---------------------------------------------------------------------------


class TestObservableStrip:
    def test_frozen(self):
        """ObservableStrip should be immutable."""
        planner = DiagnosticsPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=3.0,
            frequency_mhz=100.0,
            fov_radius_deg=5.0,
            background_mode="none",
        )
        strip = planner.compute()
        with pytest.raises(AttributeError):
            strip.ra_start_deg = 999.0

    def test_no_sky_model(self):
        """Without a sky model, source arrays should be None."""
        planner = DiagnosticsPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=3.0,
            frequency_mhz=100.0,
            fov_radius_deg=5.0,
            background_mode="none",
        )
        strip = planner.compute()
        assert strip.source_ra_deg is None
        assert strip.projected_map is None
        assert strip.beam_projection is None

    def test_metadata_lst(self):
        """LST range should be recorded in metadata."""
        planner = DiagnosticsPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=2.0,
            lst_end_hours=6.0,
            frequency_mhz=100.0,
            fov_radius_deg=5.0,
            background_mode="none",
        )
        strip = planner.compute()
        assert strip.lst_start_hours == pytest.approx(2.0)
        assert strip.lst_end_hours == pytest.approx(6.0)
        assert strip.obstime_start_iso is None  # LST mode, no UTC time


# ---------------------------------------------------------------------------
# Source filtering
# ---------------------------------------------------------------------------


class TestSourceFiltering:
    def _make_sky_model(self):
        """Build a minimal SkyModel with 5 known sources."""
        ras = np.deg2rad([30.0, 45.0, 60.0, 120.0, 300.0])
        decs = np.deg2rad([-32.0, -28.0, -35.0, -30.0, -30.0])
        fluxes = np.array([10.0, 5.0, 8.0, 2.0, 1.0])
        alphas = np.zeros(5)
        zeros = np.zeros(5)

        return SkyModel.from_arrays(
            ra_rad=ras,
            dec_rad=decs,
            flux=fluxes,
            spectral_index=alphas,
            stokes_q=zeros,
            stokes_u=zeros,
            stokes_v=zeros,
            model_name="test",
            brightness_conversion="planck",
        )

    def test_in_strip_mask(self):
        """Sources inside the strip should be flagged."""
        sky = self._make_sky_model()
        # Strip from RA 15° to 75° (LST 1h to 5h), Dec -35 to -25
        planner = DiagnosticsPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=5.0,
            frequency_mhz=100.0,
            fov_radius_deg=5.0,
            sky_model=sky,
            background_mode="none",
        )
        strip = planner.compute()

        assert strip.source_ra_deg is not None
        assert strip.in_strip_mask is not None

        # RA 30° dec -32° should be in strip (RA 15-75, Dec -35 to -25)
        # RA 45° dec -28° should be in strip
        # RA 60° dec -35° should be in strip (edge, dec = -35)
        # RA 120° dec -30° should NOT be (RA out of range)
        # RA 300° dec -30° should NOT be (RA out of range)
        in_count = int(strip.in_strip_mask.sum())
        assert in_count >= 2  # At least the first two should be in

    def test_top_n(self):
        """Top-N indices should point to brightest in-strip sources."""
        sky = self._make_sky_model()
        planner = DiagnosticsPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=5.0,
            frequency_mhz=100.0,
            fov_radius_deg=5.0,
            sky_model=sky,
            top_n_sources=2,
            background_mode="none",
        )
        strip = planner.compute()

        assert strip.top_n_indices is not None
        assert len(strip.top_n_indices) <= 2

        # The brightest in-strip source should be first
        if len(strip.top_n_indices) >= 2:
            f0 = strip.source_flux_jy[strip.top_n_indices[0]]
            f1 = strip.source_flux_jy[strip.top_n_indices[1]]
            assert f0 >= f1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_no_time_spec_raises(self):
        """Should raise if neither LST range nor UTC time is provided."""
        planner = DiagnosticsPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            frequency_mhz=100.0,
            fov_radius_deg=5.0,
            background_mode="none",
        )
        with pytest.raises(ValueError, match="lst_start_hours"):
            planner.compute()
