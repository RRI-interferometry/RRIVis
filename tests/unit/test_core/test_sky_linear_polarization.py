"""Tests for compute_linear_polarization on healpix and point payloads."""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import (
    HealpixData,
    SkyModel,
    compute_linear_polarization,
    create_from_arrays,
)


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _polarised_healpix(
    precision: PrecisionConfig,
    *,
    q_value: float,
    u_value: float,
    i_value: float = 1.0,
    nside: int = 8,
) -> SkyModel:
    npix = hp.nside2npix(nside)
    return SkyModel(
        healpix=HealpixData(
            maps=np.full((1, npix), i_value, dtype=np.float64),
            q_maps=np.full((1, npix), q_value, dtype=np.float64),
            u_maps=np.full((1, npix), u_value, dtype=np.float64),
            nside=nside,
            frequencies=np.array([150e6]),
        ),
        model_name="test_pol",
        _precision=precision,
    )


class TestLinearPolHealpix:
    def test_pure_q_gives_chi_zero(self, precision):
        sky = _polarised_healpix(precision, q_value=2.0, u_value=0.0, i_value=4.0)
        out = compute_linear_polarization(sky, frequency=150e6)
        np.testing.assert_allclose(out["P"], 2.0)
        np.testing.assert_allclose(out["chi_deg"], 0.0)
        np.testing.assert_allclose(out["frac_pol"], 0.5)

    def test_pure_u_gives_chi_45(self, precision):
        sky = _polarised_healpix(precision, q_value=0.0, u_value=2.0, i_value=4.0)
        out = compute_linear_polarization(sky, frequency=150e6)
        np.testing.assert_allclose(out["P"], 2.0)
        np.testing.assert_allclose(out["chi_deg"], 45.0)

    def test_negative_q_gives_chi_90(self, precision):
        sky = _polarised_healpix(precision, q_value=-2.0, u_value=0.0, i_value=4.0)
        out = compute_linear_polarization(sky, frequency=150e6)
        np.testing.assert_allclose(out["chi_deg"], 90.0)

    def test_no_frequency_returns_full_cube(self, precision):
        sky = _polarised_healpix(precision, q_value=1.0, u_value=1.0)
        out = compute_linear_polarization(sky)
        assert out["P"].shape == sky.healpix.maps.shape
        np.testing.assert_allclose(out["P"], np.sqrt(2.0))

    def test_missing_quv_raises(self, precision):
        nside = 8
        npix = hp.nside2npix(nside)
        sky = SkyModel(
            healpix=HealpixData(
                maps=np.zeros((1, npix), dtype=np.float64),
                nside=nside,
                frequencies=np.array([150e6]),
            ),
            model_name="i_only",
            _precision=precision,
        )
        with pytest.raises(ValueError, match="Stokes Q and U"):
            compute_linear_polarization(sky)


class TestLinearPolPoint:
    def test_point_pure_q(self, precision):
        sky = create_from_arrays(
            ra_rad=np.array([0.0, 1.0]),
            dec_rad=np.array([0.0, 0.5]),
            flux=np.array([1.0, 2.0]),
            spectral_index=np.array([-0.7, -0.7]),
            stokes_q=np.array([0.3, 0.6]),
            stokes_u=np.array([0.0, 0.0]),
            reference_frequency=150e6,
            precision=precision,
        )
        out = compute_linear_polarization(sky)
        np.testing.assert_allclose(out["P"], [0.3, 0.6])
        np.testing.assert_allclose(out["chi_deg"], [0.0, 0.0])

    def test_point_zero_qu_returns_zero_pol(self, precision):
        # When Q/U arrays are present but all zero (the typical
        # create_from_arrays output without explicit stokes), P should be
        # zero and frac_pol zero — no error.
        sky = create_from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([1.0]),
            spectral_index=np.array([-0.7]),
            reference_frequency=150e6,
            precision=precision,
        )
        out = compute_linear_polarization(sky)
        np.testing.assert_allclose(out["P"], 0.0)
        np.testing.assert_allclose(out["frac_pol"], 0.0)
