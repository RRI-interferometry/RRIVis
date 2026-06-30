"""HEALPix ordering field tests for HealpixData and end-to-end NEST threading."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    HealpixData,
    MonopoleConvention,
    SkyModel,
    SkyProvenance,
    create_test_sources,
    materialize_healpix_model,
    prepare_sky_model,
    subtract_bright_sources,
)
from radiosim.core.sky.combine.regrid import _resolve_common_healpix_ordering
from radiosim.core.sky.containers.constants import rayleigh_jeans_factor
from radiosim.core.sky.containers.model import SkyFormat


def _basic_kwargs(nside: int = 8) -> dict:
    npix = hp.nside2npix(nside)
    return {
        "maps": np.zeros((1, npix), dtype=np.float32),
        "nside": nside,
        "frequencies": np.asarray([100e6], dtype=np.float64),
    }


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.fast()


class TestHealpixDataOrdering:
    def test_default_is_ring(self) -> None:
        data = HealpixData(**_basic_kwargs())
        assert data.ordering == "ring"

    def test_nest_is_accepted(self) -> None:
        data = HealpixData(ordering="nest", **_basic_kwargs())
        assert data.ordering == "nest"

    def test_invalid_ordering_raises(self) -> None:
        with pytest.raises(ValueError, match="ordering must be"):
            HealpixData(ordering="weird", **_basic_kwargs())

    def test_ordering_is_lowercased(self) -> None:
        data = HealpixData(ordering="NEST", **_basic_kwargs())
        assert data.ordering == "nest"

    def test_ordering_differs_breaks_equality(self) -> None:
        precision = PrecisionConfig.standard()
        a = SkyModel(
            healpix=HealpixData(ordering="ring", **_basic_kwargs()),
            precision=precision,
        )
        b = SkyModel(
            healpix=HealpixData(ordering="nest", **_basic_kwargs()),
            precision=precision,
        )
        assert a != b


class TestNestEndToEnd:
    def test_materialize_healpix_model_emits_nest_ordering(
        self, precision: PrecisionConfig
    ) -> None:
        sky = create_test_sources(
            num_sources=5, precision=precision, reference_frequency=150e6
        )
        freqs = np.asarray([150e6], dtype=np.float64)
        nest_sky = materialize_healpix_model(
            sky,
            nside=32,
            frequencies=freqs,
            ref_frequency=150e6,
            ordering="nest",
        )
        assert nest_sky.healpix is not None
        assert nest_sky.healpix.ordering == "nest"
        assert nest_sky.healpix.is_nested

    def test_materialize_nest_matches_ring_twin_after_reorder(
        self, precision: PrecisionConfig
    ) -> None:
        sky = create_test_sources(
            num_sources=8, precision=precision, reference_frequency=150e6
        )
        freqs = np.asarray([150e6], dtype=np.float64)
        nside = 32
        ring_sky = materialize_healpix_model(
            sky,
            nside=nside,
            frequencies=freqs,
            ref_frequency=150e6,
            ordering="ring",
        )
        nest_sky = materialize_healpix_model(
            sky,
            nside=nside,
            frequencies=freqs,
            ref_frequency=150e6,
            ordering="nest",
        )
        assert ring_sky.healpix is not None
        assert nest_sky.healpix is not None
        ring_from_nest = nest_sky.healpix.reordered("ring").maps
        np.testing.assert_allclose(ring_from_nest, ring_sky.healpix.maps, rtol=1e-5)

    def test_prepare_sky_model_point_binning_preserves_nest_ref(
        self, precision: PrecisionConfig
    ) -> None:
        nside = 16
        npix = hp.nside2npix(nside)
        freqs = np.asarray([150e6], dtype=np.float64)
        diffuse = SkyModel(
            healpix=HealpixData(
                maps=np.full((1, npix), 0.1, dtype=np.float32),
                nside=nside,
                frequencies=freqs,
                ordering="nest",
            ),
            reference_frequency=float(freqs[0]),
            provenance=SkyProvenance(
                monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            ),
            precision=precision,
        )
        points = create_test_sources(
            num_sources=3,
            precision=precision,
            reference_frequency=float(freqs[0]),
        )
        combined = prepare_sky_model(
            [diffuse, points],
            representation=SkyFormat.HEALPIX,
            mixed_model_policy="allow",
            precision=precision,
        )
        assert combined.healpix is not None
        assert combined.healpix.ordering == "nest"

    def test_mixed_ordering_combine_raises(self, precision: PrecisionConfig) -> None:
        nside = 8
        npix = hp.nside2npix(nside)
        freqs = np.asarray([150e6], dtype=np.float64)
        ring = SkyModel(
            healpix=HealpixData(
                maps=np.zeros((1, npix), dtype=np.float32),
                nside=nside,
                frequencies=freqs,
                ordering="ring",
            ),
            precision=precision,
        )
        nest = SkyModel(
            healpix=HealpixData(
                maps=np.zeros((1, npix), dtype=np.float32),
                nside=nside,
                frequencies=freqs,
                ordering="nest",
            ),
            precision=precision,
        )
        with pytest.raises(ValueError, match="different ordering"):
            _resolve_common_healpix_ordering([ring, nest])

    def test_subtract_bright_sources_restores_nest_ordering(
        self, precision: PrecisionConfig
    ) -> None:
        nside = 32
        npix = hp.nside2npix(nside)
        freqs = np.asarray([150e6, 160e6], dtype=np.float64)
        maps = np.full((len(freqs), npix), 0.5, dtype=np.float64)
        src_ra = np.array([1.1])
        src_dec = np.array([0.2])
        src_flux = np.array([10.0])
        sigma_rad = 1.5 * hp.nside2resol(nside) / 2.355
        pixel_sr = 4.0 * np.pi / npix
        for fi, freq in enumerate(freqs):
            vec0 = hp.ang2vec(np.pi / 2 - src_dec[0], src_ra[0])
            patch = hp.query_disc(nside, vec0, 5 * sigma_rad, inclusive=True)
            theta, phi = hp.pix2ang(nside, patch)
            cos_d = np.sin(np.pi / 2 - theta) * np.sin(src_dec[0]) + np.cos(
                np.pi / 2 - theta
            ) * np.cos(src_dec[0]) * np.cos(phi - src_ra[0])
            cos_d = np.clip(cos_d, -1.0, 1.0)
            dist = np.arccos(cos_d)
            flux_jy = (
                (src_flux[0] / (2 * np.pi * sigma_rad**2))
                * np.exp(-0.5 * dist**2 / sigma_rad**2)
                * pixel_sr
            )
            maps[fi, patch] += flux_jy / rayleigh_jeans_factor(float(freq), pixel_sr)

        nest_maps = np.stack(
            [hp.reorder(maps[fi], r2n=True) for fi in range(len(freqs))], axis=0
        )
        sky = SkyModel(
            healpix=HealpixData(
                maps=nest_maps.astype(np.float32),
                nside=nside,
                frequencies=freqs,
                ordering="nest",
            ),
            brightness_conversion="rayleigh-jeans",
            provenance=SkyProvenance(
                monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            ),
            precision=precision,
        )
        result = subtract_bright_sources(
            sky, flux_limit_jy=2.5, frequency_hz=150e6, max_sources=5
        )
        assert result.healpix is not None
        assert result.healpix.ordering == "nest"
