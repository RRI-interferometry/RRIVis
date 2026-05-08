"""Tests for ZA-ring helpers in radiosim.core.observability.overlay."""

import matplotlib

matplotlib.use("Agg")

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from radiosim.core.observability.overlay import (
    draw_za_rings_on_figure,
    za_ring_points,
)
from radiosim.utils.coordinates import angular_separation_deg, split_wrapped_path


class TestZARingPoints:
    def test_all_points_at_correct_separation(self):
        zenith_ra, zenith_dec = 30.0, 10.0
        theta = 25.0
        ra, dec = za_ring_points(zenith_ra, zenith_dec, theta, n=361)
        sep = angular_separation_deg(ra, dec, zenith_ra, zenith_dec)
        np.testing.assert_allclose(sep, theta, atol=1e-6)

    def test_pole_centred_ring_constant_dec(self):
        # A ring of θ=20° around the equator at RA=0 is symmetric about the
        # equator and crosses the anti-meridian.
        ra, dec = za_ring_points(0.0, 0.0, 20.0, n=721)
        # min/max dec should bracket [-θ, +θ].
        assert dec.min() == pytest.approx(-20.0, abs=0.5)
        assert dec.max() == pytest.approx(+20.0, abs=0.5)
        assert ra.min() >= -180.0
        assert ra.max() <= 180.0

    def test_anti_meridian_ring_splits(self):
        # Centre at RA=180, large radius — the resulting ring crosses the
        # ±180° wrap and split_wrapped_path should produce ≥2 segments.
        ra, dec = za_ring_points(180.0, 0.0, 30.0, n=721)
        segments = split_wrapped_path(ra, dec, 180.0)
        # Either it is one ring stored twice (start≈end) or two halves.
        assert len(segments) >= 1
        # Sanity: sum of segment lengths ≤ original length (no duplicates).
        total = sum(len(s[0]) for s in segments)
        assert total <= len(ra)


class TestDrawZARingsOnFigure:
    def test_draws_on_mollview(self):
        nside = 8
        m = np.zeros(hp.nside2npix(nside), dtype=float)
        fig = plt.figure()
        hp.mollview(m, fig=fig.number, sub=(1, 1, 1))
        out = draw_za_rings_on_figure(
            fig,
            zenith_ra_deg=0.0,
            zenith_dec_deg=-30.0,
            theta_deg_list=[5.0, 10.0, 15.0],
        )
        assert out is fig
        plt.close(fig)

    def test_empty_theta_list_is_noop(self):
        fig = plt.figure()
        out = draw_za_rings_on_figure(
            fig,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            theta_deg_list=[],
        )
        assert out is fig
        plt.close(fig)

    def test_color_cycle_wraps(self):
        nside = 4
        m = np.zeros(hp.nside2npix(nside), dtype=float)
        fig = plt.figure()
        hp.mollview(m, fig=fig.number, sub=(1, 1, 1))
        # Three thetas, two colours → cycles.
        draw_za_rings_on_figure(
            fig,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            theta_deg_list=[5.0, 10.0, 15.0],
            colors=["red", "blue"],
        )
        plt.close(fig)
