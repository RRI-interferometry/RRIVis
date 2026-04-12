"""Tests for rrivis.core.jones.beam.projection — beam sky projection."""

import numpy as np
import pytest

from rrivis.core.jones.beam.projection import (
    compute_beam_power_on_radec_grid,
    create_rgba_overlay,
    extract_contours,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_beam(za_rad, az_rad, hpbw_rad=np.deg2rad(10.0)):
    """Simple azimuthally-symmetric Gaussian beam for testing."""
    sigma = hpbw_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-(za_rad**2) / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# compute_beam_power_on_radec_grid
# ---------------------------------------------------------------------------


class TestComputeBeamPower:
    def test_zenith_maps_to_peak(self):
        """The zenith point (RA_z, Dec_z) should have power ≈ 0 dB."""
        proj = compute_beam_power_on_radec_grid(
            _gaussian_beam,
            zenith_ra_deg=90.0,
            zenith_dec_deg=-30.0,
            max_za_deg=30.0,
            ra_resolution_deg=1.0,
            dec_resolution_deg=1.0,
        )
        # Find the grid cell closest to zenith
        ra_idx = np.argmin(np.abs(proj.ra_grid_deg - 90.0))
        dec_idx = np.argmin(np.abs(proj.dec_grid_deg - (-30.0)))
        peak_db = proj.power_db[dec_idx, ra_idx]
        assert peak_db == pytest.approx(0.0, abs=0.5)

    def test_output_shape(self):
        """Output grid has expected shape (n_dec, n_ra)."""
        proj = compute_beam_power_on_radec_grid(
            _gaussian_beam,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            max_za_deg=20.0,
            ra_resolution_deg=2.0,
            dec_resolution_deg=2.0,
        )
        assert proj.power_db.shape == (
            len(proj.dec_grid_deg),
            len(proj.ra_grid_deg),
        )

    def test_beyond_max_za_is_nan(self):
        """Points beyond max_za_deg should be NaN."""
        proj = compute_beam_power_on_radec_grid(
            _gaussian_beam,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            max_za_deg=15.0,
            ra_resolution_deg=2.0,
            dec_resolution_deg=2.0,
        )
        # Corners of the grid should be well beyond 15 deg from zenith
        assert np.any(np.isnan(proj.power_db))

    def test_dataclass_fields(self):
        """BeamSkyProjection has all expected fields."""
        proj = compute_beam_power_on_radec_grid(
            _gaussian_beam,
            zenith_ra_deg=45.0,
            zenith_dec_deg=-10.0,
            max_za_deg=20.0,
        )
        assert proj.zenith_ra_deg == 45.0
        assert proj.zenith_dec_deg == -10.0
        assert proj.max_za_deg == 20.0
        assert len(proj.ra_grid_deg) > 0
        assert len(proj.dec_grid_deg) > 0

    def test_symmetric_beam(self):
        """A symmetric beam should produce roughly symmetric power around zenith."""
        proj = compute_beam_power_on_radec_grid(
            _gaussian_beam,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            max_za_deg=20.0,
            ra_resolution_deg=1.0,
            dec_resolution_deg=1.0,
        )
        # Power should be similar at equal angular offsets N vs S of zenith
        ra_idx_0 = np.argmin(np.abs(proj.ra_grid_deg - 0.0))

        # 5 deg north and south
        dec_n = np.argmin(np.abs(proj.dec_grid_deg - 5.0))
        dec_s = np.argmin(np.abs(proj.dec_grid_deg - (-5.0)))
        pn = proj.power_db[dec_n, ra_idx_0]
        ps = proj.power_db[dec_s, ra_idx_0]
        if np.isfinite(pn) and np.isfinite(ps):
            assert pn == pytest.approx(ps, abs=1.0)  # within 1 dB


# ---------------------------------------------------------------------------
# create_rgba_overlay
# ---------------------------------------------------------------------------


class TestRGBAOverlay:
    def test_shape_uint32(self):
        """Overlay image should be uint32 with shape (n_dec, n_ra)."""
        proj = compute_beam_power_on_radec_grid(
            _gaussian_beam,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            max_za_deg=15.0,
            ra_resolution_deg=2.0,
            dec_resolution_deg=2.0,
        )
        result = create_rgba_overlay(proj)
        img = result["image"]
        assert img.dtype == np.uint32
        assert img.shape == proj.power_db.shape

    def test_keys_present(self):
        """Result dict should have x, y, dw, dh, ra_center, dec_center."""
        proj = compute_beam_power_on_radec_grid(
            _gaussian_beam,
            0.0,
            0.0,
            max_za_deg=10.0,
            ra_resolution_deg=2.0,
            dec_resolution_deg=2.0,
        )
        result = create_rgba_overlay(proj)
        for key in ("image", "x", "y", "dw", "dh", "ra_center", "dec_center"):
            assert key in result


# ---------------------------------------------------------------------------
# extract_contours
# ---------------------------------------------------------------------------


class TestExtractContours:
    def test_returns_correct_levels(self):
        """One entry per requested dB level."""
        proj = compute_beam_power_on_radec_grid(
            _gaussian_beam,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            max_za_deg=20.0,
            ra_resolution_deg=0.5,
            dec_resolution_deg=0.5,
        )
        levels = [-3.0, -10.0]
        result = extract_contours(proj, levels_db=levels)
        assert len(result) == 2
        # Each entry is (segments, level)
        for segments, level in result:
            assert level in levels
            assert isinstance(segments, list)

    def test_contour_has_vertices(self):
        """At least one contour should have vertices for -3 dB."""
        proj = compute_beam_power_on_radec_grid(
            _gaussian_beam,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            max_za_deg=20.0,
            ra_resolution_deg=0.5,
            dec_resolution_deg=0.5,
        )
        result = extract_contours(proj, levels_db=[-3.0])
        segments, _ = result[0]
        assert len(segments) > 0
        assert segments[0].shape[1] == 2  # (N, 2)
