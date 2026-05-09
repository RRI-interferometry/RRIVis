"""Tests for provenance coverage union during sky-model combination."""

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    HealpixData,
    MonopoleConvention,
    SkyCoverage,
    SkyFootprint,
    SkyProvenance,
    SourceSubtractionStatus,
    create_from_arrays,
)
from radiosim.core.sky.combine import _combine_models
from radiosim.core.sky.constants import BrightnessConversion
from radiosim.core.sky.model import SkyModel


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _make_partial_point_model(
    precision: PrecisionConfig,
    footprint: SkyFootprint | None,
    *,
    name: str,
) -> SkyModel:
    provenance = SkyProvenance(
        sky_coverage=SkyCoverage.PARTIAL_SKY,
        coverage_footprint=footprint,
        monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        source_subtraction=SourceSubtractionStatus.NONE,
        notes=name,
    )
    return create_from_arrays(
        ra_rad=np.array([0.1]),
        dec_rad=np.array([0.2]),
        flux=np.array([1.0]),
        spectral_index=np.array([-0.7]),
        ref_freq=np.array([150e6]),
        reference_frequency=150e6,
        model_name=name,
        brightness_conversion=BrightnessConversion.PLANCK,
        precision=precision,
        provenance=provenance,
    )


def _make_full_diffuse_model(precision: PrecisionConfig) -> SkyModel:
    nside = 8
    npix = hp.nside2npix(nside)
    return SkyModel(
        healpix=HealpixData(
            maps=np.full((1, npix), 30.0, dtype=np.float32),
            nside=nside,
            frequencies=np.array([150e6]),
            coordinate_frame="icrs",
        ),
        reference_frequency=150e6,
        model_name="diffuse",
        brightness_conversion=BrightnessConversion.PLANCK,
        provenance=SkyProvenance(
            sky_coverage=SkyCoverage.FULL_SKY,
            coverage_fraction=1.0,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=30.0,
            source_subtraction=SourceSubtractionStatus.ALL,
            notes="diffuse",
        ),
        precision=precision,
    )


class TestCoverageUnion:
    def test_full_diffuse_plus_partial_catalog_is_full_sky(self, precision):
        diffuse = _make_full_diffuse_model(precision)
        footprint = SkyFootprint(nside=8, hpx_inds=np.array([1, 2, 3]))
        catalog = _make_partial_point_model(precision, footprint, name="catalog")

        combined = _combine_models([diffuse, catalog], precision=precision)

        assert combined.provenance.sky_coverage is SkyCoverage.FULL_SKY
        assert combined.provenance.coverage_fraction == pytest.approx(1.0)

    def test_same_partial_footprint_preserves_fraction(self, precision):
        footprint = SkyFootprint(nside=8, hpx_inds=np.array([1, 2, 3]))
        model_a = _make_partial_point_model(precision, footprint, name="a")
        model_b = _make_partial_point_model(precision, footprint, name="b")

        combined = _combine_models([model_a, model_b], precision=precision)

        assert combined.provenance.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert combined.provenance.coverage_fraction == pytest.approx(
            footprint.coverage_fraction
        )
        assert combined.provenance.coverage_footprint == footprint

    def test_disjoint_partial_footprints_union_to_larger_fraction(self, precision):
        footprint_a = SkyFootprint(nside=8, hpx_inds=np.array([1, 2]))
        footprint_b = SkyFootprint(nside=8, hpx_inds=np.array([5, 6, 7]))
        model_a = _make_partial_point_model(precision, footprint_a, name="a")
        model_b = _make_partial_point_model(precision, footprint_b, name="b")

        combined = _combine_models([model_a, model_b], precision=precision)

        assert combined.provenance.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert combined.provenance.coverage_footprint is not None
        assert combined.provenance.coverage_fraction == pytest.approx(
            5 / hp.nside2npix(8)
        )

    def test_known_partial_plus_unknown_partial_downgrades_to_unknown(self, precision):
        footprint = SkyFootprint(nside=8, hpx_inds=np.array([1, 2, 3]))
        model_known = _make_partial_point_model(precision, footprint, name="known")
        model_unknown = _make_partial_point_model(precision, None, name="unknown")

        combined = _combine_models([model_known, model_unknown], precision=precision)

        assert combined.provenance.sky_coverage is SkyCoverage.UNKNOWN
        assert combined.provenance.coverage_fraction is None
        assert combined.provenance.coverage_footprint is None
