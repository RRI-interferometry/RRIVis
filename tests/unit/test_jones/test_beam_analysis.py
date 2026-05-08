"""Tests for radiosim.core.jones.beam.analysis — radial profile + features."""

import healpy as hp
import numpy as np
import pytest
from scipy.special import j1

from radiosim.core.jones.beam.analysis import (
    BeamFeatures,
    BeamRadialProfile,
    azimuthal_radial_profile,
    detect_beam_features,
)
from radiosim.core.observability.geometry import compute_beam_map_on_healpix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_power(za_rad, az_rad, hpbw_deg):
    sigma = np.deg2rad(hpbw_deg) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-(za_rad**2) / (2.0 * sigma**2))


def _airy_power(za_rad, az_rad, ka):
    """Idealised Airy disc ``[2 J_1(x) / x]^2`` with ``x = ka·sin(za)``."""
    x = ka * np.sin(za_rad)
    out = np.ones_like(x)
    nz = x != 0.0
    out[nz] = (2.0 * j1(x[nz]) / x[nz]) ** 2
    return out


# ---------------------------------------------------------------------------
# compute_beam_map_on_healpix
# ---------------------------------------------------------------------------


class TestBeamMapOnHealpix:
    def test_peak_normalises_to_one(self):
        nside = 32
        beam_map = compute_beam_map_on_healpix(
            lambda za, az: _gaussian_power(za, az, hpbw_deg=10.0),
            nside=nside,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            peak_normalize=True,
        )
        assert beam_map.shape == (hp.nside2npix(nside),)
        assert beam_map.max() == pytest.approx(1.0, rel=1e-6)
        assert beam_map.min() >= 0.0

    def test_horizon_mask_zeros_below_horizon(self):
        nside = 32
        beam_map = compute_beam_map_on_healpix(
            lambda za, az: np.ones_like(za),
            nside=nside,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            max_za_deg=90.0,
            peak_normalize=False,
        )
        # With a uniform beam, peak normalisation off, anything within
        # za<=90° is 1, beyond is 0.  Half the sphere should be zeroed.
        n_zero = np.sum(beam_map == 0.0)
        n_total = hp.nside2npix(nside)
        assert 0.45 * n_total < n_zero < 0.55 * n_total


# ---------------------------------------------------------------------------
# azimuthal_radial_profile
# ---------------------------------------------------------------------------


class TestRadialProfile:
    def test_recovers_gaussian_hpbw(self):
        nside = 64
        hpbw_in = 10.0
        beam_map = compute_beam_map_on_healpix(
            lambda za, az: _gaussian_power(za, az, hpbw_deg=hpbw_in),
            nside=nside,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            peak_normalize=True,
        )
        profile = azimuthal_radial_profile(
            beam_map,
            nside=nside,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            bin_step_deg=0.25,
        )
        assert isinstance(profile, BeamRadialProfile)

        finite = np.isfinite(profile.power_db)
        idx_minus3 = int(np.argmin(np.abs(profile.power_db[finite] + 3.0)))
        za_minus3 = profile.za_deg[finite][idx_minus3]
        # Half-width at half max is HPBW/2.
        assert za_minus3 == pytest.approx(hpbw_in / 2.0, abs=0.5)

    def test_shape_matches_bin_step(self):
        nside = 16
        beam_map = np.ones(hp.nside2npix(nside), dtype=float)
        profile = azimuthal_radial_profile(
            beam_map,
            nside=nside,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            bin_step_deg=1.0,
            max_za_deg=90.0,
        )
        # 90 / 1 = 90 bins (open interval; 0..90 with step 1 → 90 centres)
        assert len(profile.za_deg) == 90

    def test_wrong_shape_raises(self):
        nside = 8
        with pytest.raises(ValueError, match="expected"):
            azimuthal_radial_profile(
                np.zeros(10),
                nside=nside,
                zenith_ra_deg=0.0,
                zenith_dec_deg=0.0,
            )


# ---------------------------------------------------------------------------
# detect_beam_features (Airy reference)
# ---------------------------------------------------------------------------


class TestDetectFeatures:
    def test_airy_first_null_and_sidelobe(self):
        # Choose ka so first null lies at za ≈ 5° (analytic root: x≈3.8317).
        first_null_deg = 5.0
        ka = 3.8317059702 / np.sin(np.deg2rad(first_null_deg))

        nside = 128
        beam_map = compute_beam_map_on_healpix(
            lambda za, az: _airy_power(za, az, ka=ka),
            nside=nside,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            max_za_deg=45.0,
            peak_normalize=True,
        )
        profile = azimuthal_radial_profile(
            beam_map,
            nside=nside,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            bin_step_deg=0.1,
            max_za_deg=45.0,
        )
        features = detect_beam_features(
            profile, prominence_db=3.0, exclude_inner_deg=2.0
        )
        assert isinstance(features, BeamFeatures)

        assert features.nulls_za_deg.size >= 1
        assert features.sidelobes_za_deg.size >= 1

        # First null near the analytic value within 0.5°.
        assert features.nulls_za_deg[0] == pytest.approx(first_null_deg, abs=0.5)

        # First sidelobe peak: x ≈ 5.1356 → za_sl = arcsin(5.1356 / ka).
        first_sl_deg = np.rad2deg(np.arcsin(5.1356223018 / ka))
        assert features.sidelobes_za_deg[0] == pytest.approx(first_sl_deg, abs=0.6)

        # Sidelobe peak level should be near the analytic ~-17.6 dB.
        assert features.sidelobes_power_db[0] == pytest.approx(-17.6, abs=2.0)

    def test_hpbw_field_finite_for_gaussian(self):
        nside = 64
        beam_map = compute_beam_map_on_healpix(
            lambda za, az: _gaussian_power(za, az, hpbw_deg=10.0),
            nside=nside,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            peak_normalize=True,
        )
        profile = azimuthal_radial_profile(
            beam_map,
            nside=nside,
            zenith_ra_deg=0.0,
            zenith_dec_deg=0.0,
            bin_step_deg=0.25,
        )
        features = detect_beam_features(profile, prominence_db=3.0)
        assert np.isfinite(features.hpbw_deg)
        assert features.hpbw_deg == pytest.approx(5.0, abs=0.5)
