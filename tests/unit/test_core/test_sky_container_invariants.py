"""Regression tests for container invariants hardened in the 2026-06 audit.

Covers: array immutability (the copy-on-write contract), boolean-mask
validation on ``masked()`` entry points, ``nside`` legality, ``hpx_inds``
uniqueness, and point-array ndim enforcement.
"""

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    HealpixData,
    PointSourceData,
    SkyRegion,
    create_from_arrays,
)


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


def _point(n=4, precision=None):
    return create_from_arrays(
        ra_rad=np.linspace(0.0, 1.0, n),
        dec_rad=np.linspace(-0.5, 0.5, n),
        flux=np.linspace(1.0, 4.0, n),
        spectral_index=np.full(n, -0.7),
        stokes_q=np.zeros(n),
        stokes_u=np.zeros(n),
        stokes_v=np.zeros(n),
        reference_frequency=100e6,
        precision=precision,
    )


def _healpix(nside=8, n_freq=2, precision=None):
    npix = hp.nside2npix(nside)
    return HealpixData(
        maps=np.ones((n_freq, npix), dtype=np.float64),
        nside=nside,
        frequencies=np.array([100e6, 101e6][:n_freq], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Immutability (copy-on-write contract)
# ---------------------------------------------------------------------------


class TestArraysAreReadOnly:
    def test_point_arrays_are_frozen(self, precision):
        sky = _point(precision=precision)
        assert sky.point.flux.flags.writeable is False
        assert sky.point.ra_rad.flags.writeable is False
        with pytest.raises(ValueError):
            sky.point.flux[0] = 999.0

    def test_healpix_maps_are_frozen(self):
        hpx = _healpix()
        assert hpx.maps.flags.writeable is False
        with pytest.raises(ValueError):
            hpx.maps[0, 0] = 5.0

    def test_caller_array_cannot_corrupt_model_after_construction(self, precision):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        sky = create_from_arrays(
            ra_rad=arr,
            dec_rad=np.zeros(4),
            flux=np.ones(4),
            spectral_index=np.full(4, -0.7),
            stokes_q=np.zeros(4),
            stokes_u=np.zeros(4),
            stokes_v=np.zeros(4),
            reference_frequency=100e6,
            precision=precision,
        )
        # The stored array is read-only; attempting to mutate the shared
        # buffer raises rather than silently corrupting the "frozen" model.
        with pytest.raises(ValueError):
            sky.point.ra_rad[0] = -1.0


# ---------------------------------------------------------------------------
# Mask validation on masked() entry points
# ---------------------------------------------------------------------------


class TestMaskValidation:
    def test_boolean_mask_works(self, precision):
        sky = _point(n=4, precision=precision)
        out = sky.point.masked(np.array([True, False, True, False]))
        assert out.n_sources == 2

    def test_integer_mask_rejected(self, precision):
        sky = _point(n=4, precision=precision)
        with pytest.raises(ValueError, match="boolean"):
            sky.point.masked(np.array([0, 2]))

    def test_wrong_length_mask_rejected(self, precision):
        sky = _point(n=4, precision=precision)
        with pytest.raises(ValueError, match="shape"):
            sky.point.masked(np.array([True, False]))


# ---------------------------------------------------------------------------
# HealpixData validators
# ---------------------------------------------------------------------------


class TestHealpixValidators:
    def test_invalid_nside_rejected(self):
        with pytest.raises(ValueError, match="NSIDE"):
            HealpixData(
                maps=np.ones((1, 100), dtype=np.float64),
                nside=100,  # not a power of two
                frequencies=np.array([100e6]),
            )

    def test_duplicate_hpx_inds_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            HealpixData(
                maps=np.ones((1, 3), dtype=np.float64),
                nside=8,
                frequencies=np.array([100e6]),
                hpx_inds=np.array([0, 0, 1]),
            )

    def test_non_finite_frequencies_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            HealpixData(
                maps=np.ones((1, hp.nside2npix(8)), dtype=np.float64),
                nside=8,
                frequencies=np.array([np.inf]),
            )

    def test_nest_ordering_round_trips_pixel_coords(self):
        # A NEST-ordered map must report pixel coords using NEST geometry.
        nside = 8
        npix = hp.nside2npix(nside)
        hpx = HealpixData(
            maps=np.ones((1, npix), dtype=np.float64),
            nside=nside,
            frequencies=np.array([100e6]),
            ordering="nest",
        )
        coords = hpx.pixel_coords
        # Expected RING-vs-NEST distinction: pix 1 has different coordinates.
        theta_nest, phi_nest = hp.pix2ang(nside, np.arange(npix), nest=True)
        np.testing.assert_allclose(coords.ra.rad, phi_nest, atol=1e-12)

    def test_ring_and_nest_twins_materialize_to_same_sky(self):
        # Build a RING map, reorder a copy to NEST; materializing point sources
        # from each must yield the same set of (ra, dec, flux) — i.e. NEST is
        # honored end-to-end, not silently read as RING.
        from radiosim.core.sky import materialize_point_sources_model
        from radiosim.core.sky.containers.model import SkyModel

        nside = 8
        npix = hp.nside2npix(nside)
        rng = np.random.default_rng(0)
        ring_maps = np.zeros((1, npix), dtype=np.float64)
        hot = rng.choice(npix, size=12, replace=False)
        ring_maps[0, hot] = 50.0  # Kelvin
        prec = PrecisionConfig.standard()

        def _materialize(hpx):
            sky = SkyModel(healpix=hpx, reference_frequency=100e6, precision=prec)
            pt = materialize_point_sources_model(sky, frequency=100e6, lossy=True)
            order = np.argsort(pt.point.ra_rad)
            return (
                pt.point.ra_rad[order],
                pt.point.dec_rad[order],
                pt.point.flux[order],
            )

        ring = HealpixData(maps=ring_maps, nside=nside, frequencies=np.array([100e6]))
        nest = ring.reordered("nest")
        assert nest.ordering == "nest"

        ra_r, dec_r, flux_r = _materialize(ring)
        ra_n, dec_n, flux_n = _materialize(nest)
        np.testing.assert_allclose(ra_r, ra_n, atol=1e-9)
        np.testing.assert_allclose(dec_r, dec_n, atol=1e-9)
        np.testing.assert_allclose(flux_r, flux_n, rtol=1e-6)


# ---------------------------------------------------------------------------
# Point-array ndim enforcement
# ---------------------------------------------------------------------------


class TestReferenceFrequencyReanchor:
    def test_reanchor_rescales_flux_power_law(self, precision):
        sky = create_from_arrays(
            ra_rad=np.zeros(3),
            dec_rad=np.zeros(3),
            flux=np.array([10.0, 20.0, 5.0]),
            spectral_index=np.array([-1.0, -0.5, -2.0]),
            stokes_q=np.zeros(3),
            stokes_u=np.zeros(3),
            stokes_v=np.zeros(3),
            ref_freq=np.full(3, 100e6),
            reference_frequency=100e6,
            precision=precision,
        )
        out = sky.with_reference_frequency(200e6)
        scale = (200e6 / 100e6) ** np.array([-1.0, -0.5, -2.0])
        np.testing.assert_allclose(
            out.point.flux, np.array([10.0, 20.0, 5.0]) * scale, rtol=1e-5
        )
        np.testing.assert_allclose(out.point.ref_freq, np.full(3, 200e6), rtol=1e-6)
        assert out.reference_frequency == 200e6

    def test_reanchor_rejects_spectral_coeffs(self, precision):
        sky = create_from_arrays(
            ra_rad=np.zeros(2),
            dec_rad=np.zeros(2),
            flux=np.array([10.0, 20.0]),
            spectral_index=np.array([-0.7, -0.7]),
            stokes_q=np.zeros(2),
            stokes_u=np.zeros(2),
            stokes_v=np.zeros(2),
            ref_freq=np.full(2, 100e6),
            spectral_coeffs=np.array([[-0.7, 0.1], [-0.7, 0.1]]),
            reference_frequency=100e6,
            precision=precision,
        )
        with pytest.raises(NotImplementedError, match="spectral_coeffs"):
            sky.with_reference_frequency(200e6)


class TestHealpixUnitsLoadBearing:
    def test_jy_per_sr_input_converted_to_kelvin(self):
        nside = 4
        npix = hp.nside2npix(nside)
        freq = 150e6
        sr_value = 1.0  # Jy/sr
        hpx = HealpixData(
            maps=np.full((1, npix), sr_value, dtype=np.float64),
            nside=nside,
            frequencies=np.array([freq]),
            i_unit="Jy/sr",
        )
        # Stored map is now Kelvin and the unit reflects that.
        assert hpx.i_unit == "K"
        from radiosim.core.sky.containers.constants import (
            flux_density_to_brightness_temp,
        )

        expected = flux_density_to_brightness_temp(
            np.array([sr_value]), freq, 1.0, method="rayleigh-jeans"
        )[0]
        np.testing.assert_allclose(hpx.maps[0], expected, rtol=1e-9)

    def test_unknown_unit_rejected(self):
        with pytest.raises(ValueError, match="unit must be one of"):
            HealpixData(
                maps=np.ones((1, hp.nside2npix(4)), dtype=np.float64),
                nside=4,
                frequencies=np.array([100e6]),
                i_unit="Jy/foo",
            )


class TestRngSeedProvenance:
    def test_poisson_confusion_records_seed(self):
        from radiosim.core.sky.loaders.synthetic import load_poisson_confusion

        sky = load_poisson_confusion(
            flux_range_jy=(0.01, 1.0),
            region=SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=3.0),
            reference_frequency=154e6,
            seed=12345,
            precision=PrecisionConfig.standard(),
        )
        assert sky.provenance.rng_seed == 12345

    def test_seed_none_resolved_and_recorded(self):
        from radiosim.core.sky.loaders.synthetic import load_poisson_confusion

        sky = load_poisson_confusion(
            flux_range_jy=(0.01, 1.0),
            region=SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=3.0),
            reference_frequency=154e6,
            seed=None,
            precision=PrecisionConfig.standard(),
        )
        assert isinstance(sky.provenance.rng_seed, int)


class TestSkyModelCheck:
    def test_check_passes_on_valid_model(self, precision):
        sky = _point(precision=precision)
        sky.check()  # should not raise

    def test_check_collects_acceptability_problems(self, precision):
        sky = _point(precision=precision)
        bad = PointSourceData(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([-5.0]),  # negative
            spectral_index=np.array([np.nan]),  # non-finite
            stokes_q=np.zeros(1),
            stokes_u=np.zeros(1),
            stokes_v=np.zeros(1),
            ref_freq=np.array([100e6]),
        )
        bad_sky = sky.replace(point=bad)
        with pytest.raises(ValueError, match=r"check\(\) found"):
            bad_sky.check()


class TestSpectralType:
    def test_power_law_default(self, precision):
        from radiosim.core.sky import SpectralType

        sky = _point(precision=precision)
        assert sky.point.spectral_type is SpectralType.POWER_LAW

    def test_per_channel_when_spectrum_present(self, precision):
        from radiosim.core.sky import SpectralType
        from radiosim.core.sky.containers.point import PointSpectrum

        sky = _point(n=2, precision=precision)
        spec = PointSpectrum(flux=np.ones((2, 2)), frequencies=np.array([100e6, 200e6]))
        p = sky.point
        new_point = PointSourceData(
            ra_rad=p.ra_rad,
            dec_rad=p.dec_rad,
            flux=p.flux,
            spectral_index=p.spectral_index,
            stokes_q=p.stokes_q,
            stokes_u=p.stokes_u,
            stokes_v=p.stokes_v,
            ref_freq=p.ref_freq,
            spectrum=spec,
        )
        assert new_point.spectral_type is SpectralType.PER_CHANNEL


class TestPointNdim:
    def test_2d_core_array_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            PointSourceData(
                ra_rad=np.zeros((2, 2)),
                dec_rad=np.zeros(2),
                flux=np.ones(2),
                spectral_index=np.zeros(2),
                stokes_q=np.zeros(2),
                stokes_u=np.zeros(2),
                stokes_v=np.zeros(2),
                ref_freq=np.full(2, 100e6),
            )


class TestSpectralRepresentationLayering:
    def _dual(self, precision):
        # A source carrying BOTH a higher-order log-polynomial (spectral_coeffs
        # with >1 term) AND a per-channel PointSpectrum.
        from radiosim.core.sky.containers.point import PointSpectrum

        sky = _point(n=2, precision=precision)
        p = sky.point
        spec = PointSpectrum(flux=np.ones((2, 2)), frequencies=np.array([100e6, 200e6]))
        return PointSourceData(
            ra_rad=p.ra_rad,
            dec_rad=p.dec_rad,
            flux=p.flux,
            spectral_index=p.spectral_index,
            stokes_q=p.stokes_q,
            stokes_u=p.stokes_u,
            stokes_v=p.stokes_v,
            ref_freq=p.ref_freq,
            spectral_coeffs=np.array([[-0.7, 0.1, 0.01], [-0.7, 0.1, 0.01]]),
            spectrum=spec,
        )

    def test_populated_spectral_fields_lists_all_present(self, precision):
        from radiosim.core.sky import SpectralType

        dual = self._dual(precision)
        assert dual.populated_spectral_fields == frozenset(
            {
                SpectralType.POWER_LAW,
                SpectralType.LOG_POLYNOMIAL,
                SpectralType.PER_CHANNEL,
            }
        )

    def test_assert_single_representation_raises_on_overlap(self, precision):
        dual = self._dual(precision)
        with pytest.raises(ValueError):
            dual.assert_single_spectral_representation()

    def test_assert_single_representation_passes_power_law_only(self, precision):
        sky = _point(precision=precision)
        from radiosim.core.sky import SpectralType

        assert sky.point.populated_spectral_fields == frozenset(
            {SpectralType.POWER_LAW}
        )
        # No raise for the plain power-law-only source.
        sky.point.assert_single_spectral_representation()


class TestPointSourceDataDtypes:
    def _valid_kwargs(self):
        return {
            "ra_rad": np.array([0.0, 1.0], dtype=np.float64),
            "dec_rad": np.array([0.0, 0.5], dtype=np.float64),
            "flux": np.array([1.0, 2.0], dtype=np.float64),
            "spectral_index": np.array([-0.7, -0.8], dtype=np.float64),
            "stokes_q": np.zeros(2, dtype=np.float64),
            "stokes_u": np.zeros(2, dtype=np.float64),
            "stokes_v": np.zeros(2, dtype=np.float64),
            "ref_freq": np.full(2, 100e6, dtype=np.float64),
        }

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_core_float_dtypes_pass(self, dtype):
        kwargs = {name: arr.astype(dtype) for name, arr in self._valid_kwargs().items()}
        point = PointSourceData(**kwargs)
        assert point.flux.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.int64, np.complex128, object])
    def test_core_non_float_dtypes_rejected(self, dtype):
        kwargs = self._valid_kwargs()
        kwargs["flux"] = np.array([1, 2], dtype=dtype)
        with pytest.raises(ValueError, match="floating dtype"):
            PointSourceData(**kwargs)

    @pytest.mark.parametrize("dtype", [np.int64, np.complex128, object])
    def test_spectral_coeffs_non_float_dtypes_rejected(self, dtype):
        kwargs = self._valid_kwargs()
        kwargs["spectral_coeffs"] = np.array([[1, 2], [3, 4]], dtype=dtype)
        with pytest.raises(ValueError, match="spectral_coeffs.*floating dtype"):
            PointSourceData(**kwargs)
