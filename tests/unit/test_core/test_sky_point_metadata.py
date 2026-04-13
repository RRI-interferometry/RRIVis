"""Tests for point-source metadata preservation across sky-model APIs."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import create_from_arrays
from rrivis.core.sky._loaders_pyradiosky import (
    LossyConversionWarning,
    load_pyradiosky_file,
)
from rrivis.core.sky._serialization import to_pyradiosky


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


class FakePyRadioSkyPointModel:
    """Minimal pyradiosky-like point container for loader tests."""

    def __init__(self) -> None:
        coords = SkyCoord(
            ra=np.array([0.1, 0.2]) * u.rad,
            dec=np.array([0.3, 0.4]) * u.rad,
            frame="icrs",
        )
        self.component_type = "point"
        self.freq_array = np.array([100e6, 200e6]) * u.Hz
        self.reference_frequency = None
        self.spectral_type = "full"
        self.Ncomponents = 2
        self.ra = coords.ra
        self.dec = coords.dec
        self.name = np.array(["src-a", "src-b"])
        self.stokes = (
            np.array(
                [
                    [[1.0, 2.0], [2.0, 4.0]],
                    [[0.1, 0.2], [0.2, 0.4]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=np.float64,
            )
            * u.Jy
        )
        self.extra_columns = np.rec.fromarrays(
            [
                np.array(["A", "B"]),
                np.array(["gleam", "nvss"]),
            ],
            names=["source_id", "catalog"],
        )

    def read(self, filename: str, filetype: str | None = None) -> None:
        del filename, filetype


def test_to_pyradiosky_preserves_point_metadata(precision):
    sky = create_from_arrays(
        ra_rad=np.array([0.1, 0.2]),
        dec_rad=np.array([0.3, 0.4]),
        flux=np.array([1.0, 2.0]),
        spectral_index=np.array([-0.7, -0.5]),
        ref_freq=np.array([150e6, 150e6]),
        source_name=np.array(["src-a", "src-b"]),
        source_id=np.array(["A", "B"]),
        extra_columns={"catalog": np.array(["gleam", "nvss"])},
        precision=precision,
    )

    psky = to_pyradiosky(sky)

    np.testing.assert_array_equal(psky.name, np.array(["src-a", "src-b"]))
    assert psky.extra_columns is not None
    assert set(psky.extra_columns.dtype.names) == {"catalog", "source_id"}
    np.testing.assert_array_equal(psky.extra_columns["catalog"], ["gleam", "nvss"])
    np.testing.assert_array_equal(psky.extra_columns["source_id"], ["A", "B"])


def test_load_pyradiosky_warns_and_preserves_point_metadata(
    precision,
    monkeypatch,
    tmp_path,
):
    import rrivis.core.sky._loaders_pyradiosky as module

    monkeypatch.setattr(module, "PyRadioSkyModel", FakePyRadioSkyPointModel)
    filename = tmp_path / "fake.skyh5"
    filename.write_text("placeholder")

    with pytest.warns(LossyConversionWarning, match="collapses the per-channel"):
        sky = load_pyradiosky_file(str(filename), precision=precision)

    assert sky.point is not None
    np.testing.assert_array_equal(sky.point.source_name, np.array(["src-a", "src-b"]))
    np.testing.assert_array_equal(sky.point.source_id, np.array(["A", "B"]))
    np.testing.assert_array_equal(
        sky.point.extra_columns["catalog"],
        np.array(["gleam", "nvss"]),
    )


def test_load_pyradiosky_can_reject_lossy_point_spectrum(
    precision,
    monkeypatch,
    tmp_path,
):
    import rrivis.core.sky._loaders_pyradiosky as module

    monkeypatch.setattr(module, "PyRadioSkyModel", FakePyRadioSkyPointModel)
    filename = tmp_path / "fake.skyh5"
    filename.write_text("placeholder")

    with pytest.raises(ValueError, match="collapses the per-channel spectrum"):
        load_pyradiosky_file(
            str(filename),
            spectral_loss_policy="error",
            precision=precision,
        )
