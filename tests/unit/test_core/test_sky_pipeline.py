"""Tests for sky-model orchestration helpers."""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import HealpixData
from rrivis.core.sky.model import SkyFormat, SkyModel
from rrivis.core.sky.pipeline import prepare_sky_model


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
        source_format=SkyFormat.HEALPIX,
        reference_frequency=float(freqs[0]),
        model_name="diffuse",
        _precision=precision,
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
