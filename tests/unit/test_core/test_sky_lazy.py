"""Tests for memmap-backed sky-model paths."""

from __future__ import annotations

import os

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import HealpixData
from rrivis.core.sky._allocation import allocate_cube, ensure_scratch_dir, finalize_cube
from rrivis.core.sky.combine import combine_models
from rrivis.core.sky.convert import point_sources_to_healpix_maps
from rrivis.core.sky.model import SkyFormat, SkyModel


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


def make_point_model(n: int, precision: PrecisionConfig, seed: int = 0) -> SkyModel:
    rng = np.random.default_rng(seed)
    return SkyModel.from_arrays(
        ra_rad=rng.uniform(0, 2 * np.pi, n),
        dec_rad=rng.uniform(-np.pi / 2, np.pi / 2, n),
        flux=rng.uniform(0.1, 10.0, n),
        spectral_index=np.full(n, -0.7),
        ref_freq=np.full(n, 100e6),
        reference_frequency=100e6,
        model_name="ps",
        precision=precision,
    )


def make_healpix_model(
    nside: int,
    n_freq: int,
    *,
    precision: PrecisionConfig,
    with_pol: bool = False,
) -> SkyModel:
    npix = hp.nside2npix(nside)
    freqs = np.linspace(100e6, 200e6, n_freq)
    i_maps = np.ones((n_freq, npix), dtype=np.float32) * 2.0
    q_maps = np.ones((n_freq, npix), dtype=np.float32) * 0.1 if with_pol else None
    u_maps = np.ones((n_freq, npix), dtype=np.float32) * 0.05 if with_pol else None
    return SkyModel(
        healpix=HealpixData(
            maps=i_maps,
            nside=nside,
            frequencies=freqs,
            q_maps=q_maps,
            u_maps=u_maps,
        ),
        native_representation=SkyFormat.HEALPIX,
        active_representation=SkyFormat.HEALPIX,
        model_name="hp",
        _precision=precision,
    )


class TestAllocationHelpers:
    def test_allocate_ram(self):
        arr = allocate_cube((3, 12), np.float32, None, "i_maps")
        assert not isinstance(arr, np.memmap)
        assert arr.shape == (3, 12)
        assert np.all(arr == 0)

    def test_allocate_memmap(self, tmp_path):
        scratch = ensure_scratch_dir(str(tmp_path))
        arr = allocate_cube((2, 24), np.float32, scratch, "i_maps")
        assert isinstance(arr, np.memmap)
        assert (tmp_path / "i_maps.dat").exists()

    def test_finalize_read_only(self, tmp_path):
        scratch = ensure_scratch_dir(str(tmp_path))
        arr = allocate_cube((2, 24), np.float32, scratch, "i_maps")
        arr[0] = 1.0
        ro = finalize_cube(arr, scratch, "i_maps")
        assert isinstance(ro, np.memmap)
        with pytest.raises(ValueError):
            ro[0] = 2.0

    def test_ensure_scratch_dir_creates_temp(self):
        path = ensure_scratch_dir(None)
        assert os.path.isdir(path)
        assert "rrivis_sky_" in os.path.basename(path)


class TestPointSourcesToHealpixMemmap:
    def test_parity_ram_vs_memmap(self, precision, tmp_path):
        sky = make_point_model(50, precision, seed=42)
        kwargs = {
            "ra_rad": sky.ra_rad,
            "dec_rad": sky.dec_rad,
            "flux": sky.flux,
            "spectral_index": sky.spectral_index,
            "spectral_coeffs": None,
            "stokes_q": sky.stokes_q,
            "stokes_u": sky.stokes_u,
            "stokes_v": sky.stokes_v,
            "rotation_measure": None,
            "nside": 8,
            "frequencies": np.linspace(100e6, 200e6, 4),
            "ref_frequency": 100e6,
            "brightness_conversion": "planck",
        }
        i_ram, q_ram, u_ram, v_ram = point_sources_to_healpix_maps(**kwargs)
        i_mm, q_mm, u_mm, v_mm = point_sources_to_healpix_maps(
            **kwargs,
            memmap_path=str(tmp_path),
        )
        np.testing.assert_array_equal(i_ram, i_mm)
        assert q_ram is None and q_mm is None
        assert u_ram is None and u_mm is None
        assert v_ram is None and v_mm is None
        assert isinstance(i_mm, np.memmap)


class TestSkyModelMaterializationMemmap:
    def test_materialize_healpix_parity(self, precision, tmp_path):
        sky = make_point_model(20, precision)
        freqs = np.linspace(100e6, 200e6, 4)
        ram = sky.materialize_healpix(nside=8, frequencies=freqs)
        mm = sky.materialize_healpix(
            nside=8,
            frequencies=freqs,
            memmap_path=str(tmp_path),
        )
        np.testing.assert_array_equal(ram.healpix_maps, mm.healpix_maps)
        assert not isinstance(ram.healpix_maps, np.memmap)
        assert isinstance(mm.healpix_maps, np.memmap)


class TestCombineMemmap:
    def test_combine_healpix_parity(self, precision, tmp_path):
        sky1 = make_healpix_model(nside=8, n_freq=4, precision=precision)
        sky2 = make_healpix_model(nside=8, n_freq=4, precision=precision)
        ram = combine_models([sky1, sky2], representation=SkyFormat.HEALPIX)
        mm = combine_models(
            [sky1, sky2],
            representation=SkyFormat.HEALPIX,
            memmap_path=str(tmp_path),
        )
        np.testing.assert_array_equal(ram.healpix_maps, mm.healpix_maps)
        assert isinstance(mm.healpix_maps, np.memmap)
