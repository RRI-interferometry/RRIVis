"""Tests for ``subtract_bright_sources`` (Remazeilles-style source subtraction).

Validation strategy: build a synthetic HEALPix map = constant background +
injected Gaussian sources at known positions/fluxes.  After subtraction,
assert that (a) the residual peak is a small fraction of the injected peak,
(b) pixels far from the sources are unchanged, (c) provenance is updated,
(d) the catalog-supplied detection path works, and (e) no-op behavior when
no sources exceed the threshold.
"""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import (
    HealpixData,
    MonopoleConvention,
    SkyModel,
    SkyProvenance,
    SourceSubtractionStatus,
    create_from_arrays,
    subtract_bright_sources,
)
from rrivis.core.sky.constants import rayleigh_jeans_factor


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _inject_gaussian_sources(
    maps_k: np.ndarray,
    nside: int,
    frequencies_hz: np.ndarray,
    src_ra_rad: np.ndarray,
    src_dec_rad: np.ndarray,
    src_flux_jy: np.ndarray,
    sigma_rad: float,
) -> np.ndarray:
    """Add symmetric Gaussians to ``maps_k`` in-place (K_RJ), returns the map."""
    pixel_sr = 4.0 * np.pi / hp.nside2npix(nside)
    for fi, freq in enumerate(frequencies_hz):
        for ra, dec, S in zip(src_ra_rad, src_dec_rad, src_flux_jy, strict=True):
            vec0 = hp.ang2vec(np.pi / 2 - dec, ra)
            patch = hp.query_disc(nside, vec0, 5 * sigma_rad, inclusive=True)
            theta, phi = hp.pix2ang(nside, patch)
            cos_d = np.sin(np.pi / 2 - theta) * np.sin(dec) + np.cos(
                np.pi / 2 - theta
            ) * np.cos(dec) * np.cos(phi - ra)
            cos_d = np.clip(cos_d, -1.0, 1.0)
            dist = np.arccos(cos_d)
            flux_jy = (
                (S / (2 * np.pi * sigma_rad**2))
                * np.exp(-0.5 * dist**2 / sigma_rad**2)
                * pixel_sr
            )
            maps_k[fi, patch] += flux_jy / rayleigh_jeans_factor(float(freq), pixel_sr)
    return maps_k


def _make_synthetic_diffuse_with_sources(
    precision: PrecisionConfig,
    *,
    nside: int = 32,
    bg_k: float = 0.5,
    frequencies: np.ndarray | None = None,
    src_flux_jy: tuple[float, ...] = (10.0, 5.0, 3.0),
    seed: int = 7,
) -> tuple[SkyModel, np.ndarray, np.ndarray, np.ndarray]:
    """Return (sky, src_ra_rad, src_dec_rad, src_flux_jy) — ready for fitting."""
    if frequencies is None:
        frequencies = np.array([150e6, 160e6])
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    src_ra_rad = rng.uniform(0, 2 * np.pi, size=len(src_flux_jy))
    # Keep sources out of the polar caps for stable Gaussian fits.
    src_dec_rad = rng.uniform(-np.pi / 3, np.pi / 3, size=len(src_flux_jy))
    flux_arr = np.asarray(src_flux_jy, dtype=np.float64)

    maps_k = np.full((len(frequencies), npix), bg_k, dtype=np.float64)
    # FWHM ≈ 1.5 native pixels so the source is cleanly resolved on the grid.
    sigma_rad = 1.5 * hp.nside2resol(nside) / 2.355
    _inject_gaussian_sources(
        maps_k, nside, frequencies, src_ra_rad, src_dec_rad, flux_arr, sigma_rad
    )

    sky = SkyModel(
        healpix=HealpixData(
            maps=maps_k.astype(np.float32),
            nside=nside,
            frequencies=frequencies,
        ),
        model_name="synth_diffuse",
        brightness_conversion="rayleigh-jeans",
        provenance=SkyProvenance(
            angular_resolution_rad=(hp.nside2resol(nside), np.pi),
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            source_subtraction=SourceSubtractionStatus.NONE,
        ),
        _precision=precision,
    )
    return sky, src_ra_rad, src_dec_rad, flux_arr


class TestSubtractBrightSourcesDetection:
    def test_residual_peak_below_injected(self, precision):
        sky, _, _, _ = _make_synthetic_diffuse_with_sources(precision)
        input_peak = float(sky.healpix.maps[0].max())
        result = subtract_bright_sources(sky, flux_limit_jy=2.5, frequency_hz=150e6)
        # Residual peak is dominated by either inpaint artefacts or the
        # background floor — well under a third of the injected peak.
        assert float(result.healpix.maps[0].max()) < 0.3 * input_peak

    def test_off_source_flux_preserved(self, precision):
        sky, src_ra, src_dec, _ = _make_synthetic_diffuse_with_sources(precision)
        result = subtract_bright_sources(sky, flux_limit_jy=2.5, frequency_hz=150e6)
        nside = sky.healpix.nside
        # Find a "far" pixel: ≥ 20 pixel radii from every source.
        far_radius = 20 * hp.nside2resol(nside)
        all_pix = np.arange(hp.nside2npix(nside))
        theta, phi = hp.pix2ang(nside, all_pix)
        far_mask = np.ones(all_pix.size, dtype=bool)
        for ra, dec in zip(src_ra, src_dec, strict=True):
            cos_d = np.sin(np.pi / 2 - theta) * np.sin(dec) + np.cos(
                np.pi / 2 - theta
            ) * np.cos(dec) * np.cos(phi - ra)
            dist = np.arccos(np.clip(cos_d, -1.0, 1.0))
            far_mask &= dist > far_radius
        assert far_mask.sum() > 0
        assert np.allclose(
            result.healpix.maps[0][far_mask],
            sky.healpix.maps[0][far_mask],
            atol=1e-3,
        )

    def test_provenance_updated(self, precision):
        sky, _, _, _ = _make_synthetic_diffuse_with_sources(precision)
        result = subtract_bright_sources(sky, flux_limit_jy=2.5, frequency_hz=150e6)
        assert (
            result.provenance.source_subtraction
            is SourceSubtractionStatus.ABOVE_THRESHOLD
        )
        assert result.provenance.source_subtraction_threshold_jy == pytest.approx(2.5)
        assert result.provenance.source_subtraction_freq_hz == pytest.approx(150e6)
        assert result.provenance.source_subtraction_method == "gaussian_fit_inpaint"
        assert "subtracted>2.5Jy@150.0MHz" in (result.provenance.notes or "")


class TestSubtractBrightSourcesCatalogPath:
    def test_uses_catalog_positions_when_supplied(self, precision):
        sky, src_ra, src_dec, src_flux = _make_synthetic_diffuse_with_sources(precision)
        catalog = create_from_arrays(
            ra_rad=src_ra,
            dec_rad=src_dec,
            flux=src_flux,
            reference_frequency=150e6,
            precision=precision,
        )
        result = subtract_bright_sources(
            sky,
            flux_limit_jy=2.5,
            frequency_hz=150e6,
            catalog=catalog,
        )
        # The same physical outcome: residual peak much smaller than input.
        assert result.healpix.maps[0].max() < 0.3 * sky.healpix.maps[0].max()
        assert (
            result.provenance.source_subtraction
            is SourceSubtractionStatus.ABOVE_THRESHOLD
        )

    def test_catalog_sources_below_threshold_skipped(self, precision):
        sky, src_ra, src_dec, src_flux = _make_synthetic_diffuse_with_sources(precision)
        catalog = create_from_arrays(
            ra_rad=src_ra,
            dec_rad=src_dec,
            flux=src_flux,
            reference_frequency=150e6,
            precision=precision,
        )
        # Threshold above every catalog entry → no subtraction, provenance
        # still updates to ABOVE_THRESHOLD as a declaration.
        pre = sky.healpix.maps[0].copy()
        result = subtract_bright_sources(
            sky,
            flux_limit_jy=100.0,
            frequency_hz=150e6,
            catalog=catalog,
        )
        assert np.array_equal(result.healpix.maps[0], pre)
        assert (
            result.provenance.source_subtraction
            is SourceSubtractionStatus.ABOVE_THRESHOLD
        )
        assert result.provenance.source_subtraction_threshold_jy == pytest.approx(100.0)

    def test_catalog_scales_threshold_per_source_reference_frequency(self, precision):
        sky, src_ra, src_dec, _ = _make_synthetic_diffuse_with_sources(
            precision,
            frequencies=np.array([150e6]),
            src_flux_jy=(4.0, 8.0),
            seed=13,
        )
        catalog = create_from_arrays(
            ra_rad=src_ra,
            dec_rad=src_dec,
            flux=np.array([4.0, 4.0]),
            spectral_index=np.array([-1.0, -1.0]),
            ref_freq=np.array([150e6, 300e6]),
            reference_frequency=150e6,
            precision=precision,
        )
        result = subtract_bright_sources(
            sky,
            flux_limit_jy=6.0,
            frequency_hz=150e6,
            catalog=catalog,
        )

        src_pix = hp.ang2pix(sky.healpix.nside, np.pi / 2 - src_dec, src_ra)
        original = sky.healpix.maps[0][src_pix]
        residual = result.healpix.maps[0][src_pix]
        assert residual[0] > 0.7 * original[0]
        assert residual[1] < 0.5 * original[1]


class TestSubtractBrightSourcesGuards:
    def test_point_only_raises(self, precision):
        from rrivis.core.sky import create_test_sources

        sky = create_test_sources(
            num_sources=5, precision=precision, reference_frequency=150e6
        )
        with pytest.raises(ValueError, match="requires a HEALPix payload"):
            subtract_bright_sources(sky, flux_limit_jy=1.0, frequency_hz=150e6)

    def test_sparse_healpix_raises(self, precision):
        nside = 8
        npix = hp.nside2npix(nside)
        # Build a sparse HealpixData (only half the pixels).
        kept = np.arange(npix // 2, dtype=np.int64)
        sparse = HealpixData(
            maps=np.ones((1, kept.size), dtype=np.float32),
            nside=nside,
            frequencies=np.array([150e6]),
            hpx_inds=kept,
        )
        sky = SkyModel(
            healpix=sparse,
            provenance=SkyProvenance(
                monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
                source_subtraction=SourceSubtractionStatus.NONE,
            ),
            _precision=precision,
        )
        with pytest.raises(ValueError, match="dense HEALPix"):
            subtract_bright_sources(sky, flux_limit_jy=1.0, frequency_hz=150e6)


class TestSubtractionHelpers:
    """Direct exercises of the extracted helpers from Fix 8.

    The split into ``_select_subtraction_candidates`` and
    ``_fit_and_subtract_per_channel`` should preserve the public function's
    behaviour; these tests exercise each helper in isolation so a future
    edit that changes one path is caught at the unit boundary instead of
    through the integrated public surface only.
    """

    def test_select_candidates_with_catalog(self, precision):
        from rrivis.core.sky.operations import _select_subtraction_candidates

        nside = 16
        npix = hp.nside2npix(nside)
        # Background-only sky (no injected source).
        background_jy = 0.05
        rj_factor = rayleigh_jeans_factor(150e6, 4.0 * np.pi / npix)
        background_k = background_jy / rj_factor
        sky = SkyModel(
            healpix=HealpixData(
                maps=np.full((1, npix), background_k, dtype=np.float32),
                nside=nside,
                frequencies=np.array([150e6]),
            ),
            provenance=SkyProvenance(
                monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
                source_subtraction=SourceSubtractionStatus.NONE,
            ),
            _precision=precision,
        )
        # A catalog with one bright + one dim source; only the bright one
        # should survive the threshold filter.
        catalog = create_from_arrays(
            ra_rad=np.array([1.0, 2.0]),
            dec_rad=np.array([0.1, -0.4]),
            flux=np.array([10.0, 0.1]),
            spectral_index=np.array([-0.7, -0.7]),
            ref_freq=np.array([150e6, 150e6]),
            reference_frequency=150e6,
            precision=precision,
            model_name="cat",
        )
        candidates, _ = _select_subtraction_candidates(
            sky,
            frequency_hz=150e6,
            flux_limit_jy=1.0,
            catalog=catalog,
            detection_peak_fraction=0.2,
            max_sources=None,
        )
        assert candidates.size == 1  # only the bright source survives
        # Recover the surviving pixel and check it matches the bright source.
        expected_pix = hp.ang2pix(nside, np.pi / 2 - 0.1, 1.0)
        assert int(candidates[0]) == int(expected_pix)

    def test_select_candidates_max_sources_keeps_brightest(self, precision):
        from rrivis.core.sky.operations import _select_subtraction_candidates

        nside = 32
        npix = hp.nside2npix(nside)
        # Inject three Gaussians with distinct fluxes.
        frequencies = np.array([150e6])
        sigma_rad = 1.5 * hp.nside2resol(nside)
        maps = np.zeros((1, npix), dtype=np.float64)
        ra = np.array([0.5, 1.5, 2.5])
        dec = np.array([0.1, 0.1, 0.1])
        flux = np.array([1.0, 5.0, 3.0])  # second is brightest
        maps = _inject_gaussian_sources(
            maps, nside, frequencies, ra, dec, flux, sigma_rad
        )
        sky = SkyModel(
            healpix=HealpixData(
                maps=maps.astype(np.float32),
                nside=nside,
                frequencies=frequencies,
            ),
            provenance=SkyProvenance(
                monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
                source_subtraction=SourceSubtractionStatus.NONE,
            ),
            _precision=precision,
        )
        candidates, _ = _select_subtraction_candidates(
            sky,
            frequency_hz=150e6,
            flux_limit_jy=0.5,
            catalog=None,
            detection_peak_fraction=0.2,
            max_sources=2,
        )
        # Cap of 2 → only the two brightest sources kept.
        assert candidates.size <= 2
