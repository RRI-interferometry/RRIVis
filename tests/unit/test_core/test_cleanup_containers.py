"""Characterization + behavior tests for the Phase-2 ``containers`` cleanup.

Covers spec items E1-E6 (Task 2.1):

* E1 -- flat-kwargs construction of ``PointSourceData`` is the live path and
  produces the same object as the nested-block construction.
* E2 -- ``n_sky_elements_for`` agrees with the canonical scalar accessors.
* E3 -- ``filter_region`` preserves the fields whose redundant ``replace()``
  forwards were dropped.
* E4 -- ``_coerce_inputs`` does not mutate the caller's kwargs mapping.
* E5 -- the frequency axis is always float64 (HEALPix and per-channel point)
  even when the flux precision is float32.
* E6 -- ``HealpixData.to_dense(fill=np.nan)`` distinguishes un-observed pixels
  from measured zeros.

These are invariant / known-case assertions (no golden snapshots): an
incorrect refactor breaks an invariant rather than a magic number.
"""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    BrightnessConversion,
    HealpixData,
    PointSourceData,
    PointSpectrum,
    SkyFormat,
    SkyModel,
    SkyRegion,
    create_from_arrays,
)
from radiosim.core.sky.containers import (
    PointMetadata,
    PointMorphology,
    PointPolarization,
)


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


@pytest.fixture
def fast_precision() -> PrecisionConfig:
    return PrecisionConfig.fast()


# ---------------------------------------------------------------------------
# E1 -- flat-kwargs construction routes through point_source_data_from_mapping
# and is equivalent to nested construction
# ---------------------------------------------------------------------------


class TestFlatKwargsConstruction:
    def _core(self, n: int = 3) -> dict[str, np.ndarray]:
        return {
            "ra_rad": np.linspace(0.0, 1.0, n),
            "dec_rad": np.linspace(-0.5, 0.5, n),
            "flux": np.linspace(1.0, 3.0, n),
            "spectral_index": np.full(n, -0.7),
            "stokes_q": np.zeros(n),
            "stokes_u": np.zeros(n),
            "stokes_v": np.zeros(n),
            "ref_freq": np.full(n, 100e6),
        }

    def test_flat_and_nested_are_equal(self, precision) -> None:
        # The flat column-dict path now lives in point_source_data_from_mapping:
        # flat morphology/polarization/metadata keys get packed into the nested
        # sub-blocks there, producing the same object as direct nested
        # construction of PointSourceData.
        from radiosim.core.sky.support.point_builder import (
            point_source_data_from_mapping,
        )

        n = 3
        core = self._core(n)
        major = np.full(n, 10.0)
        minor = np.full(n, 5.0)
        pa = np.full(n, 30.0)
        rm = np.full(n, 1.5)
        names = np.array(["a", "b", "c"])

        flat = point_source_data_from_mapping(
            {
                **core,
                "major_arcsec": major,
                "minor_arcsec": minor,
                "pa_deg": pa,
                "rotation_measure": rm,
                "source_name": names,
            },
            precision=precision,
        )
        nested = point_source_data_from_mapping(
            {
                **core,
                "morphology": PointMorphology(
                    major_arcsec=major, minor_arcsec=minor, pa_deg=pa
                ),
                "polarization": PointPolarization(rotation_measure=rm),
                "metadata": PointMetadata(source_name=names),
            },
            precision=precision,
        )
        assert flat == nested
        # The flat path actually populated the nested sub-blocks.
        assert flat.morphology is not None
        assert flat.polarization is not None
        assert flat.metadata is not None
        np.testing.assert_array_equal(flat.morphology.major_arcsec, major)
        np.testing.assert_array_equal(flat.polarization.rotation_measure, rm)
        np.testing.assert_array_equal(flat.metadata.source_name, names)

    def test_create_from_arrays_uses_flat_path(self, precision) -> None:
        # create_from_arrays forwards flat morphology/polarization/metadata
        # kwargs through point_source_data_from_mapping, which packs them into
        # the nested sub-blocks.
        n = 4
        sky = create_from_arrays(
            ra_rad=np.linspace(0.0, 1.0, n),
            dec_rad=np.linspace(-0.5, 0.5, n),
            flux=np.linspace(1.0, 4.0, n),
            rotation_measure=np.full(n, 2.0),
            major_arcsec=np.full(n, 12.0),
            minor_arcsec=np.full(n, 6.0),
            pa_deg=np.full(n, 45.0),
            reference_frequency=100e6,
            precision=precision,
        )
        assert sky.point.morphology is not None
        assert sky.point.polarization is not None

    def test_constructor_is_nested_only(self, precision) -> None:
        # The raw PointSourceData constructor no longer accepts flat per-source
        # kwargs: packing was centralized in point_source_data_from_mapping, so
        # the constructor forbids extras and rejects a flat morphology kwarg
        # (pydantic raises ValidationError, a ValueError subclass) instead of
        # silently dropping the column.
        n = 2
        core = self._core(n)
        with pytest.raises(ValueError, match="[Uu]nexpected keyword"):
            PointSourceData(
                **core,
                major_arcsec=np.full(n, 1.0),
                minor_arcsec=np.full(n, 1.0),
                pa_deg=np.full(n, 1.0),
            )

    def test_morphology_all_or_none_enforced_in_builder(self, precision) -> None:
        # The morphology all-or-none rule survives the centralization: passing a
        # partial flat morphology (only major_arcsec) through the builder raises.
        from radiosim.core.sky.support.point_builder import (
            point_source_data_from_mapping,
        )

        n = 2
        core = self._core(n)
        with pytest.raises(ValueError, match="all set or all None"):
            point_source_data_from_mapping(
                {**core, "major_arcsec": np.full(n, 1.0)},
                precision=precision,
            )


# ---------------------------------------------------------------------------
# E2 -- canonical count accessors
# ---------------------------------------------------------------------------


def _make_healpix_sky(precision, *, nside: int = 8) -> SkyModel:
    npix = hp.nside2npix(nside)
    return SkyModel(
        healpix=HealpixData(
            maps=np.ones((1, npix), dtype=np.float64),
            nside=nside,
            frequencies=np.array([100e6], dtype=np.float64),
        ),
        reference_frequency=100e6,
        precision=precision,
    )


class TestCanonicalCounts:
    def test_n_sky_elements_matches_scalar_accessors(self, precision) -> None:
        nside = 8
        sky = _make_healpix_sky(precision, nside=nside)
        assert sky.n_sky_elements_for(SkyFormat.HEALPIX) == sky.n_healpix_pixels
        assert sky.n_healpix_pixels == hp.nside2npix(nside)
        # No point payload -> both report 0.
        assert sky.n_sky_elements_for(SkyFormat.POINT_SOURCES) == sky.n_point_sources
        assert sky.n_point_sources == 0

    def test_n_sky_elements_string_and_enum_agree(self, precision) -> None:
        sky = _make_healpix_sky(precision)
        assert sky.n_sky_elements_for("healpix_map") == sky.n_sky_elements_for(
            SkyFormat.HEALPIX
        )


# ---------------------------------------------------------------------------
# E3 -- filter_region preserves the fields whose forwards were dropped
# ---------------------------------------------------------------------------


class TestFilterRegionPreservesFields:
    def test_metadata_fields_survive_filter(self, precision) -> None:
        n = 6
        sky = create_from_arrays(
            ra_rad=np.deg2rad(np.linspace(80.0, 90.0, n)),
            dec_rad=np.deg2rad(np.linspace(18.0, 26.0, n)),
            flux=np.ones(n),
            reference_frequency=100e6,
            model_name="orig-name",
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=precision,
        )
        region = SkyRegion.cone(ra_deg=85.0, dec_deg=22.0, radius_deg=10.0)
        filtered = sky.filter_region(region)

        assert filtered.model_name == sky.model_name == "orig-name"
        assert filtered.reference_frequency == sky.reference_frequency
        assert filtered.brightness_conversion == sky.brightness_conversion
        assert filtered.precision is sky.precision


# ---------------------------------------------------------------------------
# E4 -- _coerce_inputs does not mutate the caller's kwargs mapping
# ---------------------------------------------------------------------------


class TestCoerceInputsNoMutation:
    def test_caller_dict_unchanged_via_typeadapter(self, precision) -> None:
        # The dict branch of _coerce_inputs is reached when pydantic validates
        # a raw mapping (TypeAdapter / model_validate path). Before the E4 fix
        # this mutated the caller's dict in place; it must not now.
        from pydantic import TypeAdapter

        point = PointSourceData.empty()
        kwargs = {
            "point": point,
            "brightness_conversion": "rayleigh-jeans",
            "provenance": {"sky_coverage": "full_sky"},
            "reference_frequency": 100e6,
            "precision": precision,
        }
        snapshot = dict(kwargs)
        sky = TypeAdapter(SkyModel).validate_python(kwargs)
        assert sky is not None
        # The original mapping must be unchanged: the string enum was NOT
        # coerced in place, the dict provenance was NOT replaced, the payload
        # was NOT swapped for a precision-cast copy.
        assert kwargs == snapshot
        assert kwargs["brightness_conversion"] == "rayleigh-jeans"
        assert isinstance(kwargs["provenance"], dict)
        assert kwargs["point"] is point

    def test_caller_kwargs_unchanged_via_constructor(self, precision) -> None:
        # The ArgsKwargs branch (SkyModel(**kwargs)) must also leave the
        # passed values intact and still build a valid model.
        point = PointSourceData.empty()
        kwargs = {
            "point": point,
            "brightness_conversion": "rayleigh-jeans",
            "provenance": {"sky_coverage": "full_sky"},
            "reference_frequency": 100e6,
            "precision": precision,
        }
        snapshot = dict(kwargs)
        sky = SkyModel(**kwargs)
        assert sky.brightness_conversion == BrightnessConversion.RAYLEIGH_JEANS
        assert kwargs == snapshot
        assert kwargs["point"] is point

    def test_dict_reusable_for_second_model(self, precision) -> None:
        kwargs = {
            "healpix": HealpixData(
                maps=np.ones((1, hp.nside2npix(4)), dtype=np.float64),
                nside=4,
                frequencies=np.array([100e6], dtype=np.float64),
            ),
            "brightness_conversion": "planck",
            "precision": precision,
        }
        first = SkyModel(**kwargs)
        second = SkyModel(**kwargs)
        assert first.brightness_conversion == second.brightness_conversion
        assert first == second


# ---------------------------------------------------------------------------
# E5 -- frequency axis is always float64, independent of flux precision
# ---------------------------------------------------------------------------


class TestFrequencyAxisDtypePolicy:
    def test_healpix_frequencies_float64_under_fast(self, fast_precision) -> None:
        npix = hp.nside2npix(4)
        sky = SkyModel(
            healpix=HealpixData(
                maps=np.ones((2, npix), dtype=np.float64),
                nside=4,
                frequencies=np.array([100e6, 150e6], dtype=np.float32),
            ),
            reference_frequency=100e6,
            precision=fast_precision,
        )
        # Flux/storage precision is float32 under fast() ...
        assert sky.healpix.maps.dtype == np.float32
        # ... but the frequency axis is pinned to float64.
        assert sky.healpix.frequencies.dtype == np.float64

    def test_point_spectrum_frequencies_float64_under_fast(
        self, fast_precision
    ) -> None:
        n = 3
        spectrum = PointSpectrum(
            flux=np.ones((2, n), dtype=np.float32),
            frequencies=np.array([100e6, 200e6], dtype=np.float32),
        )
        point = PointSourceData(
            ra_rad=np.linspace(0.0, 1.0, n),
            dec_rad=np.linspace(-0.5, 0.5, n),
            flux=np.ones(n),
            spectral_index=np.full(n, -0.7),
            stokes_q=np.zeros(n),
            stokes_u=np.zeros(n),
            stokes_v=np.zeros(n),
            ref_freq=np.full(n, 100e6),
            spectrum=spectrum,
        )
        sky = SkyModel(
            point=point,
            reference_frequency=100e6,
            precision=fast_precision,
        )
        # Flux follows the fast() precision (float32) ...
        assert sky.point.spectrum.flux.dtype == np.float32
        # ... while the frequency axis stays float64 (same policy as HEALPix).
        assert sky.point.spectrum.frequencies.dtype == np.float64

    def test_both_axes_share_policy(self, fast_precision) -> None:
        sky_hp = SkyModel(
            healpix=HealpixData(
                maps=np.ones((1, hp.nside2npix(4)), dtype=np.float64),
                nside=4,
                frequencies=np.array([120e6], dtype=np.float64),
            ),
            reference_frequency=120e6,
            precision=fast_precision,
        )
        spectrum = PointSpectrum(
            flux=np.ones((1, 2), dtype=np.float64),
            frequencies=np.array([120e6], dtype=np.float64),
        )
        point = PointSourceData(
            ra_rad=np.zeros(2),
            dec_rad=np.zeros(2),
            flux=np.ones(2),
            spectral_index=np.full(2, -0.7),
            stokes_q=np.zeros(2),
            stokes_u=np.zeros(2),
            stokes_v=np.zeros(2),
            ref_freq=np.full(2, 120e6),
            spectrum=spectrum,
        )
        sky_pt = SkyModel(
            point=point, reference_frequency=120e6, precision=fast_precision
        )
        assert (
            sky_hp.healpix.frequencies.dtype
            == sky_pt.point.spectrum.frequencies.dtype
            == np.float64
        )


# ---------------------------------------------------------------------------
# E6 -- to_dense(fill=...) partial-sky fill semantics
# ---------------------------------------------------------------------------


def _make_sparse_healpix(*, nside: int = 8) -> tuple[HealpixData, np.ndarray]:
    hpx_inds = np.array([2, 17, 123], dtype=np.int64)
    # Deliberately include a measured zero so we can distinguish it from fill.
    maps = np.array([[0.0, 2.0, 3.0]], dtype=np.float64)
    data = HealpixData(
        maps=maps,
        nside=nside,
        frequencies=np.array([100e6], dtype=np.float64),
        hpx_inds=hpx_inds,
    )
    return data, hpx_inds


class TestToDenseFill:
    def test_default_fill_zero_preserves_observed(self) -> None:
        data, hpx_inds = _make_sparse_healpix()
        dense = data.to_dense()
        assert not dense.is_sparse
        assert dense.maps.shape[1] == hp.nside2npix(data.nside)
        # Observed pixels preserved exactly (including the measured zero).
        np.testing.assert_array_equal(dense.maps[0, hpx_inds], data.maps[0])
        # Un-observed pixels are 0.0 -- indistinguishable from the measured 0.
        unobserved = np.setdiff1d(np.arange(dense.maps.shape[1]), hpx_inds)
        assert np.all(dense.maps[0, unobserved] == 0.0)

    def test_nan_fill_distinguishes_unobserved_from_measured_zero(self) -> None:
        data, hpx_inds = _make_sparse_healpix()
        dense = data.to_dense(fill=np.nan)

        # The measured-zero pixel (hpx_inds[0]) is a real 0.0, NOT NaN.
        assert dense.maps[0, hpx_inds[0]] == 0.0
        assert not np.isnan(dense.maps[0, hpx_inds[0]])
        np.testing.assert_array_equal(dense.maps[0, hpx_inds], data.maps[0])

        # Every un-observed pixel is NaN -- distinguishable from a measured 0.
        unobserved = np.setdiff1d(np.arange(dense.maps.shape[1]), hpx_inds)
        assert np.all(np.isnan(dense.maps[0, unobserved]))

        # Exactly the un-observed pixels are NaN.
        assert int(np.sum(np.isnan(dense.maps[0]))) == unobserved.size

    def test_nan_fill_applies_to_polarization_maps(self) -> None:
        hpx_inds = np.array([1, 5], dtype=np.int64)
        data = HealpixData(
            maps=np.array([[1.0, 2.0]], dtype=np.float64),
            nside=8,
            frequencies=np.array([100e6], dtype=np.float64),
            hpx_inds=hpx_inds,
            q_maps=np.array([[0.0, 0.5]], dtype=np.float64),
        )
        dense = data.to_dense(fill=np.nan)
        unobserved = np.setdiff1d(np.arange(dense.maps.shape[1]), hpx_inds)
        assert np.all(np.isnan(dense.q_maps[0, unobserved]))
        # The measured Q=0 at hpx_inds[0] survives as a real zero.
        assert dense.q_maps[0, hpx_inds[0]] == 0.0

    def test_dense_input_returns_self_regardless_of_fill(self) -> None:
        npix = hp.nside2npix(4)
        data = HealpixData(
            maps=np.ones((1, npix), dtype=np.float64),
            nside=4,
            frequencies=np.array([100e6], dtype=np.float64),
        )
        assert data.to_dense(fill=np.nan) is data
        assert data.to_dense() is data
