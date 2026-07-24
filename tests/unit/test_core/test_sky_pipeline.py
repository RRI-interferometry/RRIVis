"""Tests for sky-model orchestration helpers."""

import healpy as hp
import numpy as np
import pytest
from pydantic import ValidationError

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import HealpixData, PointSourceData
from radiosim.core.sky.combine.options import PrepareSkyOptions
from radiosim.core.sky.combine.pipeline import prepare_sky_model
from radiosim.core.sky.containers.model import SkyFormat, SkyModel


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


def make_healpix_model(
    *,
    nside: int = 8,
    freqs: np.ndarray | None = None,
    precision: PrecisionConfig,
) -> SkyModel:
    if freqs is None:
        freqs = np.array([100e6, 101e6], dtype=np.float64)
    npix = hp.nside2npix(nside)
    return SkyModel(
        healpix=HealpixData(
            maps=np.ones((len(freqs), npix), dtype=np.float32),
            nside=nside,
            frequencies=freqs,
        ),
        reference_frequency=float(freqs[0]),
        model_name="diffuse",
        precision=precision,
    )


class TestPrepareSkyModel:
    def test_existing_healpix_explicit_frequency_grid_is_respected(self, precision):
        sky = make_healpix_model(
            freqs=np.array([100e6, 101e6], dtype=np.float64),
            precision=precision,
        )
        with pytest.raises(ValueError, match="frequency grid does not match"):
            prepare_sky_model(
                [sky],
                representation=SkyFormat.HEALPIX,
                nside=None,
                frequencies=np.array([100e6, 101e6, 102e6]),
            )

    def test_representation_none_returns_single_model_unchanged(self, precision):
        sky = make_healpix_model(precision=precision)
        out = prepare_sky_model([sky])
        assert out is sky

    def test_representation_none_preserves_hybrid_inputs(self, precision):
        # One healpix model + one point model → hybrid output preserved.
        healpix_sky = make_healpix_model(precision=precision)
        point_sky = SkyModel(
            point=PointSourceData(
                ra_rad=np.array([0.5]),
                dec_rad=np.array([0.1]),
                flux=np.array([1.0]),
                spectral_index=np.array([-0.7]),
                stokes_q=np.array([0.0]),
                stokes_u=np.array([0.0]),
                stokes_v=np.array([0.0]),
                ref_freq=np.array([100e6]),
            ),
            reference_frequency=100e6,
            model_name="src",
            precision=precision,
        )
        out = prepare_sky_model(
            [healpix_sky, point_sky],
            representation=None,
            mixed_model_policy="allow",
            precision=precision,
        )
        # Hybrid: both formats populated.
        assert SkyFormat.POINT_SOURCES in out.formats
        assert SkyFormat.HEALPIX in out.formats

    @pytest.mark.parametrize("field", ("beam_fwhm_rad", "nside_safety_factor"))
    def test_removed_beam_advisor_fields_are_strictly_rejected(
        self,
        field,
        precision,
    ):
        sky = make_healpix_model(precision=precision)

        with pytest.raises(ValidationError, match=field):
            PrepareSkyOptions(**{field: 1.0})
        with pytest.raises(TypeError, match=field):
            prepare_sky_model(
                [sky],
                representation=SkyFormat.HEALPIX,
                nside=8,
                **{field: 1.0},
            )
