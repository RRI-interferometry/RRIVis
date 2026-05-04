"""Tests for ``with_monopole`` and ``with_monopole_subtracted`` operations."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import (
    HealpixData,
    MonopoleConvention,
    SkyCoverage,
    SkyModel,
    SkyProvenance,
    SkyRegion,
    SourceSubtractionStatus,
    create_test_sources,
    with_monopole,
    with_monopole_subtracted,
)


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _healpix_sky(
    precision: PrecisionConfig,
    *,
    value_k: float = 100.0,
    nside: int = 8,
    convention: MonopoleConvention = MonopoleConvention.ABSOLUTE_NO_CMB,
    monopole_k: float | None = None,
) -> SkyModel:
    npix = hp.nside2npix(nside)
    return SkyModel(
        healpix=HealpixData(
            maps=np.full((2, npix), value_k, dtype=np.float32),
            nside=nside,
            frequencies=np.array([150e6, 160e6]),
        ),
        source_format="healpix_map",
        model_name="test_diffuse",
        provenance=SkyProvenance(
            angular_resolution_rad=(hp.nside2resol(nside), np.pi),
            sky_coverage=SkyCoverage.FULL_SKY,
            coverage_fraction=1.0,
            monopole_convention=convention,
            monopole_k=monopole_k if monopole_k is not None else float(value_k),
            source_subtraction=SourceSubtractionStatus.NONE,
        ),
        _precision=precision,
    )


class TestWithMonopole:
    def test_adds_value_to_healpix_stokes_i(self, precision):
        sky = _healpix_sky(precision, value_k=100.0, monopole_k=100.0)
        shifted = with_monopole(sky, 2.725)
        assert shifted.healpix.maps[0].mean() == pytest.approx(102.725, rel=1e-5)
        assert shifted.provenance.monopole_k == pytest.approx(102.725)

    def test_preserves_q_u_v_unchanged(self, precision):
        npix = hp.nside2npix(8)
        sky = SkyModel(
            healpix=HealpixData(
                maps=np.ones((1, npix), dtype=np.float32),
                nside=8,
                frequencies=np.array([150e6]),
                q_maps=np.full((1, npix), 0.1, dtype=np.float32),
                u_maps=np.full((1, npix), 0.2, dtype=np.float32),
            ),
            source_format="healpix_map",
            provenance=SkyProvenance(
                sky_coverage=SkyCoverage.FULL_SKY,
                coverage_fraction=1.0,
                monopole_k=1.0,
            ),
            _precision=precision,
        )
        shifted = with_monopole(sky, 5.0)
        assert shifted.healpix.maps[0].mean() == pytest.approx(6.0)
        assert np.all(shifted.healpix.q_maps == 0.1)
        assert np.all(shifted.healpix.u_maps == 0.2)

    def test_convention_is_set(self, precision):
        sky = _healpix_sky(precision, convention=MonopoleConvention.ABSOLUTE_NO_CMB)
        shifted = with_monopole(
            sky,
            2.725,
            convention=MonopoleConvention.ABSOLUTE_WITH_CMB,
        )
        assert (
            shifted.provenance.monopole_convention
            is MonopoleConvention.ABSOLUTE_WITH_CMB
        )

    def test_string_convention_accepted(self, precision):
        sky = _healpix_sky(precision)
        shifted = with_monopole(sky, 1.0, convention="absolute_with_cmb")
        assert (
            shifted.provenance.monopole_convention
            is MonopoleConvention.ABSOLUTE_WITH_CMB
        )

    def test_point_only_leaves_arrays_untouched(self, precision):
        sky = create_test_sources(
            num_sources=5, precision=precision, reference_frequency=150e6
        )
        shifted = with_monopole(sky, 1.5)
        assert np.array_equal(sky.point.flux, shifted.point.flux)
        # Provenance still updates.
        assert shifted.provenance.monopole_k == pytest.approx(1.5)

    def test_does_not_mutate_input(self, precision):
        sky = _healpix_sky(precision, value_k=100.0, monopole_k=100.0)
        _ = with_monopole(sky, 10.0)
        assert sky.healpix.maps[0].mean() == pytest.approx(100.0)
        assert sky.provenance.monopole_k == pytest.approx(100.0)

    def test_rejects_partial_sky_models(self, precision):
        sky = _healpix_sky(precision, value_k=100.0, monopole_k=100.0)
        partial = sky.filter_region(SkyRegion.cone(180.0, 0.0, 20.0))
        with pytest.raises(ValueError, match="full-sky"):
            with_monopole(partial, 2.725)


class TestWithMonopoleSubtracted:
    def test_zeroes_healpix_mean(self, precision):
        sky = _healpix_sky(precision, value_k=100.0, monopole_k=100.0)
        zeroed = with_monopole_subtracted(sky)
        assert zeroed.healpix.maps[0].mean() == pytest.approx(0.0, abs=1e-4)
        assert (
            zeroed.provenance.monopole_convention is MonopoleConvention.MEAN_SUBTRACTED
        )
        assert zeroed.provenance.monopole_k == pytest.approx(0.0)

    def test_raises_on_already_subtracted(self, precision):
        sky = _healpix_sky(
            precision,
            convention=MonopoleConvention.MEAN_SUBTRACTED,
            monopole_k=0.0,
        )
        with pytest.raises(ValueError, match="already mean-subtracted"):
            with_monopole_subtracted(sky)

    def test_preserves_angular_structure(self, precision):
        nside = 8
        npix = hp.nside2npix(nside)
        # Create a non-uniform map so mean subtraction leaves structure intact.
        values = np.linspace(0.0, 200.0, npix, dtype=np.float32)
        sky = SkyModel(
            healpix=HealpixData(
                maps=values[None, :],
                nside=nside,
                frequencies=np.array([150e6]),
            ),
            source_format="healpix_map",
            provenance=SkyProvenance(
                sky_coverage=SkyCoverage.FULL_SKY,
                coverage_fraction=1.0,
                monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
                monopole_k=float(values.mean()),
                source_subtraction=SourceSubtractionStatus.NONE,
            ),
            _precision=precision,
        )
        zeroed = with_monopole_subtracted(sky)
        # The shape (up to a DC offset) is the same.
        orig_centered = values - values.mean()
        assert np.allclose(
            zeroed.healpix.maps[0],
            orig_centered.astype(zeroed.healpix.maps.dtype),
            atol=1e-4,
        )

    def test_point_only_updates_provenance_only(self, precision):
        sky = create_test_sources(
            num_sources=5, precision=precision, reference_frequency=150e6
        )
        zeroed = with_monopole_subtracted(sky)
        assert np.array_equal(sky.point.flux, zeroed.point.flux)
        assert (
            zeroed.provenance.monopole_convention is MonopoleConvention.MEAN_SUBTRACTED
        )

    def test_rejects_partial_sky_models(self, precision):
        sky = _healpix_sky(precision, value_k=100.0, monopole_k=100.0)
        partial = sky.filter_region(SkyRegion.cone(180.0, 0.0, 20.0))
        with pytest.raises(ValueError, match="full-sky"):
            with_monopole_subtracted(partial)
