"""Phase 6 — tests for lazy/memmap-backed HEALPix cube construction.

Covers the ``memmap_path`` kwarg on:
- ``point_sources_to_healpix_maps`` (convert.py)
- ``combine_healpix`` / ``combine_models`` (combine.py)
- ``SkyModel.with_healpix_maps`` / ``SkyModel.with_representation`` (model.py)
- The ``_allocation`` helper primitives

The diffuse, pysm3, FITS, and pyradiosky loader paths are verified only at
the signature level here — they depend on external packages/files and are
exercised manually against real data outside the unit test suite.
"""

from __future__ import annotations

import os
import tracemalloc

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import SkyModel
from rrivis.core.sky._allocation import (
    allocate_cube,
    ensure_scratch_dir,
    finalize_cube,
)
from rrivis.core.sky.combine import combine_models
from rrivis.core.sky.convert import point_sources_to_healpix_maps
from rrivis.core.sky.model import SkyFormat

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


def _make_ps_sky(n: int, precision: PrecisionConfig, seed: int = 0) -> SkyModel:
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


def _make_hp_sky(
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
        _healpix_maps=i_maps,
        _healpix_q_maps=q_maps,
        _healpix_u_maps=u_maps,
        _healpix_nside=nside,
        _observation_frequencies=freqs,
        _native_format=SkyFormat.HEALPIX,
        _active_mode=SkyFormat.HEALPIX,
        model_name="hp",
        _precision=precision,
    )


# ---------------------------------------------------------------------------
# _allocation primitives
# ---------------------------------------------------------------------------


class TestAllocationHelpers:
    def test_allocate_ram(self):
        a = allocate_cube((3, 12), np.float32, None, "i_maps")
        assert not isinstance(a, np.memmap)
        assert a.shape == (3, 12)
        assert np.all(a == 0)

    def test_allocate_memmap(self, tmp_path):
        scratch = ensure_scratch_dir(str(tmp_path))
        a = allocate_cube((2, 24), np.float32, scratch, "i_maps")
        assert isinstance(a, np.memmap)
        assert a.shape == (2, 24)
        assert np.all(a == 0)
        assert (tmp_path / "i_maps.dat").exists()

    def test_finalize_read_only(self, tmp_path):
        scratch = ensure_scratch_dir(str(tmp_path))
        a = allocate_cube((2, 24), np.float32, scratch, "i_maps")
        a[0] = 1.0
        a[1] = 2.5
        ro = finalize_cube(a, scratch, "i_maps")
        assert isinstance(ro, np.memmap)
        assert np.all(ro[0] == 1.0)
        assert np.all(ro[1] == 2.5)
        with pytest.raises(ValueError):
            ro[0] = 99.0  # read-only

    def test_finalize_ram_passthrough(self):
        a = allocate_cube((2, 6), np.float32, None, "i_maps")
        ro = finalize_cube(a, None, "i_maps")
        assert ro is a

    def test_ensure_scratch_dir_creates_temp(self):
        d = ensure_scratch_dir(None)
        assert os.path.isdir(d)
        assert "rrivis_sky_" in os.path.basename(d)


# ---------------------------------------------------------------------------
# point_sources_to_healpix_maps
# ---------------------------------------------------------------------------


class TestPointSourcesToHealpixMemmap:
    def _get_inputs(self, precision):
        sky = _make_ps_sky(50, precision, seed=42)
        return {
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

    def test_parity_ram_vs_memmap(self, precision, tmp_path):
        kw = self._get_inputs(precision)
        i_ram, q_ram, u_ram, v_ram = point_sources_to_healpix_maps(**kw)
        i_mm, q_mm, u_mm, v_mm = point_sources_to_healpix_maps(
            **kw, memmap_path=str(tmp_path)
        )
        np.testing.assert_array_equal(i_ram, i_mm)
        # Q/U/V are None in both paths when no polarization
        assert q_ram is None and q_mm is None
        assert u_ram is None and u_mm is None
        assert v_ram is None and v_mm is None

    def test_memmap_type(self, precision, tmp_path):
        kw = self._get_inputs(precision)
        i_ram, _, _, _ = point_sources_to_healpix_maps(**kw)
        i_mm, _, _, _ = point_sources_to_healpix_maps(**kw, memmap_path=str(tmp_path))
        assert not isinstance(i_ram, np.memmap)
        assert isinstance(i_mm, np.memmap)

    def test_memmap_file_exists(self, precision, tmp_path):
        kw = self._get_inputs(precision)
        point_sources_to_healpix_maps(**kw, memmap_path=str(tmp_path))
        assert (tmp_path / "i_maps.dat").exists()


# ---------------------------------------------------------------------------
# SkyModel.with_healpix_maps and with_representation
# ---------------------------------------------------------------------------


class TestWithHealpixMapsMemmap:
    def test_with_healpix_maps_parity(self, precision, tmp_path):
        sky = _make_ps_sky(20, precision)
        freqs = np.linspace(100e6, 200e6, 4)
        ram = sky.with_healpix_maps(nside=8, frequencies=freqs)
        mm = sky.with_healpix_maps(
            nside=8, frequencies=freqs, memmap_path=str(tmp_path)
        )
        np.testing.assert_array_equal(ram.healpix_maps, mm.healpix_maps)
        assert not isinstance(ram.healpix_maps, np.memmap)
        assert isinstance(mm.healpix_maps, np.memmap)
        assert (tmp_path / "i_maps.dat").exists()

    def test_with_representation_routes_memmap_path(self, precision, tmp_path):
        sky = _make_ps_sky(20, precision)
        freqs = np.linspace(100e6, 200e6, 4)
        converted = sky.with_representation(
            SkyFormat.HEALPIX,
            nside=8,
            frequencies=freqs,
            memmap_path=str(tmp_path),
        )
        assert converted.mode == SkyFormat.HEALPIX
        assert isinstance(converted.healpix_maps, np.memmap)


# ---------------------------------------------------------------------------
# combine_models (HEALPix merge)
# ---------------------------------------------------------------------------


class TestCombineMemmap:
    def test_combine_healpix_parity(self, precision, tmp_path):
        sky1 = _make_hp_sky(nside=8, n_freq=4, precision=precision)
        sky2 = _make_hp_sky(nside=8, n_freq=4, precision=precision)
        ram = combine_models([sky1, sky2], representation=SkyFormat.HEALPIX)
        mm = combine_models(
            [sky1, sky2],
            representation=SkyFormat.HEALPIX,
            memmap_path=str(tmp_path),
        )
        np.testing.assert_array_equal(ram.healpix_maps, mm.healpix_maps)
        assert not isinstance(ram.healpix_maps, np.memmap)
        assert isinstance(mm.healpix_maps, np.memmap)
        assert (tmp_path / "i_maps.dat").exists()

    def test_combine_healpix_polarized_parity(self, precision, tmp_path):
        sky1 = _make_hp_sky(nside=8, n_freq=3, precision=precision, with_pol=True)
        sky2 = _make_hp_sky(nside=8, n_freq=3, precision=precision, with_pol=True)
        ram = combine_models([sky1, sky2], representation=SkyFormat.HEALPIX)
        mm = combine_models(
            [sky1, sky2],
            representation=SkyFormat.HEALPIX,
            memmap_path=str(tmp_path),
        )
        np.testing.assert_array_equal(ram.healpix_maps, mm.healpix_maps)
        np.testing.assert_array_equal(ram.healpix_q_maps, mm.healpix_q_maps)
        np.testing.assert_array_equal(ram.healpix_u_maps, mm.healpix_u_maps)
        assert isinstance(mm.healpix_maps, np.memmap)
        assert isinstance(mm.healpix_q_maps, np.memmap)
        assert isinstance(mm.healpix_u_maps, np.memmap)


# ---------------------------------------------------------------------------
# Peak allocation bound (integration test)
# ---------------------------------------------------------------------------


class TestPeakAllocationBound:
    """Memmap path should have dramatically lower tracemalloc peak.

    This asserts the *purpose* of Phase 6: when memmap_path is given, we
    don't build the full cube in RAM.  We use tracemalloc (not RSS)
    because it ignores OS page cache — exactly what we want.
    """

    def test_peak_ram_less_than_full_cube(self, precision, tmp_path):
        # nside=32 × 20 freqs × float32 × 3 Stokes ≈ ~740 KB per Stokes
        # cube. At this size, the RAM path allocates all three stokes
        # cubes; the memmap path allocates ~1 row per stokes map.
        nside = 32
        n_freq = 20
        npix = hp.nside2npix(nside)
        single_slice_bytes = npix * 4
        full_cube_bytes = n_freq * single_slice_bytes

        # Build a point-source model to convert.  Do this OUTSIDE the
        # tracemalloc window so the PS arrays don't count toward peak.
        sky = _make_ps_sky(200, precision, seed=1)
        freqs = np.linspace(100e6, 200e6, n_freq)

        # Memmap path peak
        tracemalloc.start()
        mm = sky.with_healpix_maps(
            nside=nside, frequencies=freqs, memmap_path=str(tmp_path)
        )
        _, peak_mm = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # RAM path peak
        tmp2 = tmp_path / "_unused"
        tmp2.mkdir(exist_ok=True)
        tracemalloc.start()
        ram = sky.with_healpix_maps(nside=nside, frequencies=freqs)
        _, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Correctness parity
        np.testing.assert_array_equal(mm.healpix_maps, ram.healpix_maps)

        # The memmap peak should be substantially lower than the RAM
        # peak.  We don't assert a tight per-slice bound because the
        # convert path has its own O(npix) scratch buffers; the real
        # signal is "memmap is noticeably cheaper than RAM".
        assert peak_mm < peak_ram, (
            f"memmap peak {peak_mm} should be less than RAM peak {peak_ram}"
        )
        # The RAM peak should be at least the full cube size (since
        # that's the minimum working set for the non-memmap path).
        assert peak_ram >= full_cube_bytes * 0.5, (
            f"RAM peak {peak_ram} should be on the order of the full cube "
            f"{full_cube_bytes}"
        )


# ---------------------------------------------------------------------------
# Loader signature checks (signatures only — loaders touch external data)
# ---------------------------------------------------------------------------


class TestLoaderSignatures:
    def test_load_diffuse_sky_accepts_memmap_path(self):
        import inspect

        from rrivis.core.sky._loaders_diffuse import load_diffuse_sky

        params = inspect.signature(load_diffuse_sky).parameters
        assert "memmap_path" in params

    def test_load_pysm3_accepts_memmap_path(self):
        import inspect

        from rrivis.core.sky._loaders_diffuse import load_pysm3

        params = inspect.signature(load_pysm3).parameters
        assert "memmap_path" in params

    def test_load_fits_image_accepts_memmap_path(self):
        import inspect

        from rrivis.core.sky._loaders_fits import load_fits_image

        params = inspect.signature(load_fits_image).parameters
        assert "memmap_path" in params

    def test_load_pyradiosky_file_accepts_memmap_path(self):
        import inspect

        from rrivis.core.sky._loaders_pyradiosky import load_pyradiosky_file

        params = inspect.signature(load_pyradiosky_file).parameters
        assert "memmap_path" in params
