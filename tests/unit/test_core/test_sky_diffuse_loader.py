"""Tests for diffuse HEALPix loader behavior."""

from __future__ import annotations

import sys
import types

import healpy as hp
import numpy as np

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import SkyCoverage, SkyRegion
from radiosim.core.sky.loaders.diffuse import load_pysm3


def _install_fake_pysm3(monkeypatch, *, map_value: float = 42.0) -> None:
    fake_units = types.ModuleType("pysm3.units")
    fake_units.Hz = 1.0
    fake_units.K_RJ = object()
    fake_units.cmb_equivalencies = lambda _freq: []

    class FakeEmission:
        def __init__(self, data: np.ndarray) -> None:
            self._data = data
            self.shape = data.shape

        def to(
            self, _unit: object, equivalencies: object | None = None
        ) -> FakeEmission:
            return self

        def __getitem__(self, index: int) -> np.ndarray:
            return self._data[index]

    class FakeSky:
        def __init__(self, nside: int, preset_strings: list[str]) -> None:
            self.nside = nside
            self.preset_strings = preset_strings

        def get_emission(self, _freq: float) -> FakeEmission:
            npix = hp.nside2npix(self.nside)
            data = np.full((1, npix), map_value, dtype=np.float64)
            return FakeEmission(data)

    fake_pysm3 = types.ModuleType("pysm3")
    fake_pysm3.Sky = FakeSky
    fake_pysm3.units = fake_units

    monkeypatch.setitem(sys.modules, "pysm3", fake_pysm3)
    monkeypatch.setitem(sys.modules, "pysm3.units", fake_units)


def test_pysm3_region_returns_sparse_partial_sky(monkeypatch):
    _install_fake_pysm3(monkeypatch)
    precision = PrecisionConfig.standard()

    sky = load_pysm3(
        components="s1",
        nside=8,
        frequencies=np.array([150e6]),
        region=SkyRegion.cone(180.0, -30.0, 10.0),
        precision=precision,
    )

    assert sky.healpix is not None
    assert sky.healpix.is_sparse
    assert sky.healpix.n_pixels < sky.healpix.full_n_pixels
    assert sky.provenance.sky_coverage is SkyCoverage.PARTIAL_SKY
    assert sky.provenance.monopole_k is None


def test_pysm3_full_sky_monopole_uses_full_map(monkeypatch):
    _install_fake_pysm3(monkeypatch, map_value=37.0)
    precision = PrecisionConfig.standard()

    sky = load_pysm3(
        components="s1",
        nside=4,
        frequencies=np.array([150e6]),
        precision=precision,
    )

    assert sky.healpix is not None
    assert not sky.healpix.is_sparse
    assert sky.provenance.sky_coverage is SkyCoverage.FULL_SKY
    assert sky.provenance.monopole_k == 37.0
