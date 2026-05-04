"""Tests for compute_drift_scan_lightcurve / fractional_horizon_excess.

These tests use a synthetic SkyModel and a hand-built planner-free
lightcurve to exercise the bookkeeping (LST sweep, area normalisation,
horizon mask) without requiring a real beam FITS file on disk.
"""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.observability.lightcurves import (
    DriftScanLightcurve,
    compute_drift_scan_lightcurve,
    fractional_horizon_excess,
)
from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import HealpixData, SkyModel


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _uniform_sky(
    precision: PrecisionConfig, *, nside: int = 8, value_k: float = 5.0
) -> SkyModel:
    npix = hp.nside2npix(nside)
    return SkyModel(
        healpix=HealpixData(
            maps=np.full((1, npix), value_k, dtype=np.float64),
            nside=nside,
            frequencies=np.array([150e6]),
        ),
        model_name="uniform_sky",
        _precision=precision,
    )


def _isotropic_lightcurve(
    sky: SkyModel,
    lst_hours: np.ndarray,
    *,
    mask_horizon: bool,
    area_normalize: bool,
) -> DriftScanLightcurve:
    """Bypass FITS I/O by constructing the lightcurve directly with an
    isotropic beam_power_func.  Validates the integration step rather than
    the FITS-loading codepath, which the notebook smoke test covers.
    """
    from rrivis.core.observability.geometry import compute_beam_map_on_healpix

    nside = int(sky.healpix.nside)
    sky_map = sky.healpix.maps[0]
    integrated = np.empty(lst_hours.shape, dtype=float)
    mean_b = np.empty(lst_hours.shape, dtype=float) if area_normalize else None
    max_za = 90.0 if mask_horizon else 180.0

    for i, lst in enumerate(lst_hours):
        zenith_ra = float(((lst * 15.0) + 180.0) % 360.0 - 180.0)
        beam_map = compute_beam_map_on_healpix(
            lambda za, az: np.ones_like(za),
            nside=nside,
            zenith_ra_deg=zenith_ra,
            zenith_dec_deg=-30.0,
            max_za_deg=max_za,
            peak_normalize=True,
        )
        integrated[i] = float(np.sum(sky_map * beam_map))
        if mean_b is not None:
            denom = float(np.sum(beam_map))
            mean_b[i] = (
                float(np.sum(sky_map * beam_map) / denom)
                if denom > 0.0
                else float("nan")
            )

    return DriftScanLightcurve(
        lst_hours=lst_hours,
        integrated_flux=integrated,
        mean_brightness=mean_b,
        mask_horizon=mask_horizon,
        frequency_hz=float(sky.healpix.frequencies[0]),
        nside=nside,
    )


class TestDriftScanIntegration:
    def test_uniform_sky_isotropic_beam_constant_flux(self, precision):
        sky = _uniform_sky(precision)
        lst = np.linspace(0.0, 24.0, 8)
        lc = _isotropic_lightcurve(sky, lst, mask_horizon=True, area_normalize=False)
        np.testing.assert_allclose(lc.integrated_flux, lc.integrated_flux[0])

    def test_mask_horizon_reduces_flux(self, precision):
        sky = _uniform_sky(precision)
        lst = np.array([0.0, 6.0, 12.0])
        masked = _isotropic_lightcurve(
            sky, lst, mask_horizon=True, area_normalize=False
        )
        unmasked = _isotropic_lightcurve(
            sky, lst, mask_horizon=False, area_normalize=False
        )
        assert np.all(masked.integrated_flux <= unmasked.integrated_flux)

    def test_area_normalised_mean_equals_input(self, precision):
        sky = _uniform_sky(precision, value_k=12.5)
        lst = np.array([0.0, 6.0, 12.0, 18.0])
        lc = _isotropic_lightcurve(sky, lst, mask_horizon=True, area_normalize=True)
        # ⟨I⟩_beam over a uniform sky = the constant input.
        np.testing.assert_allclose(lc.mean_brightness, 12.5)


class TestFractionalHorizonExcess:
    def test_excess_non_negative(self, precision):
        sky = _uniform_sky(precision)
        lst = np.linspace(0.0, 24.0, 6)
        masked = _isotropic_lightcurve(
            sky, lst, mask_horizon=True, area_normalize=False
        )
        unmasked = _isotropic_lightcurve(
            sky, lst, mask_horizon=False, area_normalize=False
        )
        excess = fractional_horizon_excess(masked, unmasked)
        finite = np.isfinite(excess)
        assert np.all(excess[finite] >= -1e-12)

    def test_mismatched_grids_raise(self, precision):
        sky = _uniform_sky(precision)
        a = _isotropic_lightcurve(
            sky, np.array([0.0, 1.0]), mask_horizon=True, area_normalize=False
        )
        b = _isotropic_lightcurve(
            sky, np.array([0.0, 1.0, 2.0]), mask_horizon=False, area_normalize=False
        )
        with pytest.raises(ValueError, match="LST grid"):
            fractional_horizon_excess(a, b)


class TestComputeDriftScanLightcurveValidation:
    def test_no_healpix_payload_raises(self, precision):
        from rrivis.core.sky import create_from_arrays

        sky = create_from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([1.0]),
            spectral_index=np.array([-0.7]),
            reference_frequency=150e6,
            precision=precision,
        )
        with pytest.raises(ValueError, match="HEALPix payload"):
            compute_drift_scan_lightcurve(
                sky,
                latitude_deg=-30.0,
                longitude_deg=21.0,
                height_m=1000.0,
                beam_fits_path="/does/not/exist.fits",
                beam_diameter_m=14.0,
                frequency_hz=150e6,
                lst_hours=np.array([0.0]),
            )
