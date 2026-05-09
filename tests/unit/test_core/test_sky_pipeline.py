"""Tests for sky-model orchestration helpers."""

import logging

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import HealpixData, PointSourceData
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
    def test_existing_healpix_frequency_config_is_respected(self, precision):
        sky = make_healpix_model(
            freqs=np.array([100e6, 101e6], dtype=np.float64),
            precision=precision,
        )
        obs_frequency_config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 2.0,
            "frequency_unit": "MHz",
        }
        with pytest.raises(ValueError, match="frequency grid does not match"):
            prepare_sky_model(
                [sky],
                representation=SkyFormat.HEALPIX,
                nside=None,
                frequencies=None,
                obs_frequency_config=obs_frequency_config,
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

    def test_beam_advisor_fires_for_single_model(self, caplog, precision):
        """The beam-aware nside advisor must fire even when only one model is
        passed (the old single-model fast path returned before the check)."""
        sky = make_healpix_model(precision=precision, nside=4)  # very coarse
        with caplog.at_level(
            logging.WARNING, logger="radiosim.core.sky.combine.pipeline"
        ):
            prepare_sky_model(
                [sky],
                representation=SkyFormat.HEALPIX,
                nside=4,
                beam_fwhm_rad=np.deg2rad(0.5),  # 0.5 deg beam << pixel size
            )
        assert any("nside=4" in rec.message for rec in caplog.records)
