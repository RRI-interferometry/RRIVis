"""First-class hybrid model tests (point + healpix in one SkyModel).

Covers:
- Hybrid construction (both payloads populated).
- ``SkyModel.formats`` returns both members.
- ``replace`` round-trips a hybrid.
- ``materialize_*`` produces a hybrid by default and a single-format model
  with ``clear_other=True``.
- ``_combine_models`` preserves both payloads when inputs span types.
- ``to_pyradiosky`` rejects a hybrid without an explicit representation.
"""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    HealpixData,
    PointSourceData,
    SkyFormat,
    SkyModel,
    create_from_arrays,
    materialize_healpix_model,
    materialize_point_sources_model,
)
from radiosim.core.sky.combine.engine import _combine_models
from radiosim.core.sky.io.serialization import to_pyradiosky


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


@pytest.fixture
def hybrid_sky(precision: PrecisionConfig) -> SkyModel:
    """A directly-constructed hybrid model with non-empty point + healpix."""
    nside = 8
    npix = hp.nside2npix(nside)
    freqs = np.asarray([100e6, 150e6], dtype=np.float64)
    point = PointSourceData(
        ra_rad=np.asarray([0.1, 0.2, 0.3]),
        dec_rad=np.asarray([-0.1, 0.0, 0.1]),
        flux=np.asarray([1.0, 2.0, 3.0]),
        spectral_index=np.full(3, -0.7),
        stokes_q=np.zeros(3),
        stokes_u=np.zeros(3),
        stokes_v=np.zeros(3),
        ref_freq=np.full(3, 150e6),
    )
    healpix = HealpixData(
        maps=np.full((len(freqs), npix), 5.0, dtype=np.float32),
        nside=nside,
        frequencies=freqs,
        coordinate_frame="icrs",
    )
    return SkyModel(
        point=point,
        healpix=healpix,
        model_name="hybrid",
        reference_frequency=150e6,
        precision=precision,
    )


class TestHybridConstruction:
    def test_formats_contains_both(self, hybrid_sky: SkyModel) -> None:
        assert hybrid_sky.formats == {SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX}

    def test_counts_per_payload(self, hybrid_sky: SkyModel) -> None:
        assert hybrid_sky.n_point_sources == 3
        assert hybrid_sky.n_healpix_pixels == hp.nside2npix(8)

    def test_replace_preserves_hybrid(self, hybrid_sky: SkyModel) -> None:
        replaced = hybrid_sky.replace(model_name="renamed")
        assert replaced.formats == {SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX}
        assert replaced.model_name == "renamed"
        assert replaced.point is not None
        assert replaced.healpix is not None


class TestMaterializeClearOther:
    def test_materialize_healpix_default_drops_point(
        self, precision: PrecisionConfig
    ) -> None:
        sky = create_from_arrays(
            ra_rad=np.asarray([0.1, 0.2]),
            dec_rad=np.asarray([0.0, 0.0]),
            flux=np.asarray([1.0, 2.0]),
            reference_frequency=150e6,
            precision=precision,
        )
        out = materialize_healpix_model(sky, nside=8, frequencies=np.asarray([150e6]))
        assert out.formats == {SkyFormat.HEALPIX}
        assert out.point is None

    def test_materialize_healpix_keep_other_keeps_point(
        self, precision: PrecisionConfig
    ) -> None:
        sky = create_from_arrays(
            ra_rad=np.asarray([0.1, 0.2]),
            dec_rad=np.asarray([0.0, 0.0]),
            flux=np.asarray([1.0, 2.0]),
            reference_frequency=150e6,
            precision=precision,
        )
        out = materialize_healpix_model(
            sky, nside=8, frequencies=np.asarray([150e6]), clear_other=False
        )
        assert out.formats == {SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX}

    def test_materialize_point_default_drops_healpix(
        self, hybrid_sky: SkyModel
    ) -> None:
        healpix_only = hybrid_sky.replace(point=None)
        out = materialize_point_sources_model(healpix_only, frequency=100e6, lossy=True)
        assert out.formats == {SkyFormat.POINT_SOURCES}
        assert out.healpix is None

    def test_materialize_point_keep_other_keeps_healpix(
        self, hybrid_sky: SkyModel
    ) -> None:
        healpix_only = hybrid_sky.replace(point=None)
        out = materialize_point_sources_model(
            healpix_only, frequency=100e6, lossy=True, clear_other=False
        )
        assert out.formats == {SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX}


class TestCombineModelsHybridAuto:
    def test_combine_point_plus_healpix_yields_hybrid(
        self, precision: PrecisionConfig
    ) -> None:
        nside = 8
        npix = hp.nside2npix(nside)
        freqs = np.asarray([100e6, 150e6], dtype=np.float64)

        point_only = create_from_arrays(
            ra_rad=np.asarray([0.1, 0.2]),
            dec_rad=np.asarray([0.0, 0.0]),
            flux=np.asarray([1.0, 2.0]),
            reference_frequency=150e6,
            precision=precision,
        )
        healpix_only = SkyModel(
            healpix=HealpixData(
                maps=np.full((len(freqs), npix), 7.0, dtype=np.float32),
                nside=nside,
                frequencies=freqs,
                coordinate_frame="icrs",
            ),
            precision=precision,
        )

        combined = _combine_models(
            [point_only, healpix_only],
            precision=precision,
            mixed_model_policy="allow",
        )
        assert combined.formats == {SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX}
        assert combined.n_point_sources == 2
        assert combined.healpix is not None


class TestSerializationRejectsHybrid:
    def test_to_pyradiosky_requires_explicit_format(self, hybrid_sky: SkyModel) -> None:
        with pytest.raises(ValueError, match="both point and HEALPix"):
            to_pyradiosky(hybrid_sky)

    def test_to_pyradiosky_explicit_format_works(self, hybrid_sky: SkyModel) -> None:
        # Smoke-test that explicit representation doesn't raise.
        ps = to_pyradiosky(hybrid_sky, representation="point_sources")
        assert ps is not None
