"""Tests for ``radiosim.utils.healpix`` (NSIDE advisor)."""

from __future__ import annotations

import logging

import healpy as hp
import numpy as np
import pytest

from radiosim.utils.healpix import pixel_too_coarse, recommend_nside_for_beam


class TestRecommendNside:
    def test_one_degree_beam_safety_five(self):
        nside = recommend_nside_for_beam(np.deg2rad(1.0))
        # 1 deg / 5 ≈ 0.2 deg ≈ 3.5e-3 rad.  hp.nside2resol(64) ≈ 0.016 rad
        # (too coarse); nside 512 gives ~2e-3 rad.
        assert nside >= 128
        assert hp.nside2resol(nside) <= np.deg2rad(1.0) / 5.0

    def test_one_arcmin_beam(self):
        nside = recommend_nside_for_beam(np.deg2rad(1.0 / 60.0))
        assert nside >= 8192
        assert hp.nside2resol(nside) <= np.deg2rad(1.0 / 60.0) / 5.0

    def test_returns_power_of_two(self):
        for beam_deg in (5.0, 1.0, 0.1, 0.01):
            nside = recommend_nside_for_beam(np.deg2rad(beam_deg))
            assert nside & (nside - 1) == 0, f"{nside} not power of two"

    def test_safety_factor_override(self):
        """Smaller safety factor relaxes the rule and yields a smaller nside."""
        strict = recommend_nside_for_beam(np.deg2rad(1.0), safety_factor=10.0)
        loose = recommend_nside_for_beam(np.deg2rad(1.0), safety_factor=2.0)
        assert strict >= loose

    def test_rejects_non_positive_beam(self):
        with pytest.raises(ValueError, match="positive finite"):
            recommend_nside_for_beam(0.0)
        with pytest.raises(ValueError, match="positive finite"):
            recommend_nside_for_beam(-1.0)
        with pytest.raises(ValueError, match="positive finite"):
            recommend_nside_for_beam(float("nan"))

    def test_rejects_non_positive_safety_factor(self):
        with pytest.raises(ValueError, match="positive finite"):
            recommend_nside_for_beam(np.deg2rad(1.0), safety_factor=0.0)


class TestPixelTooCoarse:
    def test_true_for_oversized_pixels(self):
        assert pixel_too_coarse(64, np.deg2rad(1.0)) is True
        # At nside 1024 pixel size is well below 1°/5.
        assert pixel_too_coarse(1024, np.deg2rad(1.0)) is False

    def test_disabled_for_non_positive_beam(self):
        assert pixel_too_coarse(64, None) is False
        assert pixel_too_coarse(64, 0.0) is False
        assert pixel_too_coarse(64, -1.0) is False


class TestPipelineAdvisory:
    """Verify prepare_sky_model emits the advisory warning via logger.warning."""

    def test_warns_on_too_coarse_nside(self, caplog):
        from radiosim.core.precision import PrecisionConfig
        from radiosim.core.sky import create_test_sources
        from radiosim.core.sky.model import SkyFormat
        from radiosim.core.sky.pipeline import prepare_sky_model

        precision = PrecisionConfig.standard()
        sky = create_test_sources(
            num_sources=5,
            precision=precision,
            reference_frequency=150e6,
        )

        with caplog.at_level(logging.WARNING, logger="radiosim.core.sky.pipeline"):
            prepare_sky_model(
                [sky],
                representation=SkyFormat.HEALPIX,
                nside=32,  # deliberately coarse
                frequencies=np.asarray([150e6]),
                precision=precision,
                beam_fwhm_rad=np.deg2rad(1.0),  # 1° beam
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("exceeds beam_fwhm" in r.getMessage() for r in warnings), (
            "Expected the nside-advisor warning in pipeline logs."
        )

    def test_silent_when_nside_adequate(self, caplog):
        from radiosim.core.precision import PrecisionConfig
        from radiosim.core.sky import create_test_sources
        from radiosim.core.sky.model import SkyFormat
        from radiosim.core.sky.pipeline import prepare_sky_model

        precision = PrecisionConfig.standard()
        sky = create_test_sources(
            num_sources=5,
            precision=precision,
            reference_frequency=150e6,
        )

        with caplog.at_level(logging.WARNING, logger="radiosim.core.sky.pipeline"):
            prepare_sky_model(
                [sky],
                representation=SkyFormat.HEALPIX,
                nside=1024,  # fine enough for a 1° beam
                frequencies=np.asarray([150e6]),
                precision=precision,
                beam_fwhm_rad=np.deg2rad(1.0),
            )
        # No advisor warning should fire.
        assert not any("exceeds beam_fwhm" in r.getMessage() for r in caplog.records)

    def test_silent_when_beam_unknown(self, caplog):
        """Without a declared beam FWHM the advisor stays quiet even at low nside."""
        from radiosim.core.precision import PrecisionConfig
        from radiosim.core.sky import create_test_sources
        from radiosim.core.sky.model import SkyFormat
        from radiosim.core.sky.pipeline import prepare_sky_model

        precision = PrecisionConfig.standard()
        sky = create_test_sources(
            num_sources=5,
            precision=precision,
            reference_frequency=150e6,
        )
        with caplog.at_level(logging.WARNING, logger="radiosim.core.sky.pipeline"):
            prepare_sky_model(
                [sky],
                representation=SkyFormat.HEALPIX,
                nside=8,
                frequencies=np.asarray([150e6]),
                precision=precision,
            )
        assert not any("exceeds beam_fwhm" in r.getMessage() for r in caplog.records)
