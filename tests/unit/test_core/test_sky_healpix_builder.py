"""Tests for shared HEALPix loader assembly helpers."""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.core.sky.loaders._healpix_builder import (
    build_healpix_from_stokes_cube,
    extract_stokes_component,
)


class MaskRegion:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = np.asarray(mask, dtype=bool)

    def healpix_mask(self, nside: int, coordinate_frame: str = "icrs") -> np.ndarray:
        assert nside == 1
        assert coordinate_frame in {"icrs", "galactic"}
        return self._mask


def test_extract_stokes_component_returns_none_when_missing():
    stokes = np.arange(2 * 3).reshape(2, 3)

    assert np.array_equal(extract_stokes_component(stokes, "I"), stokes[0])
    assert np.array_equal(extract_stokes_component(stokes, 2), stokes[1])
    assert extract_stokes_component(stokes, "U") is None
    assert extract_stokes_component(stokes, "V", n_available=2) is None


def test_build_dense_full_sky_healpix_data():
    nside = 1
    npix = 12
    frequencies = np.array([100e6, 110e6])
    rows = [
        (np.arange(npix, dtype=float),),
        (np.arange(npix, dtype=float) + 100.0,),
    ]

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=rows,
        nside=nside,
        frequencies=frequencies,
        coordinate_frame="icrs",
    )

    assert not healpix.is_sparse
    assert healpix.hpx_inds is None
    assert healpix.maps.shape == (2, npix)
    assert np.array_equal(healpix.frequencies, frequencies)
    assert np.allclose(healpix.maps[1], np.arange(npix) + 100.0)


def test_build_sparse_healpix_data_preserves_indices():
    hpx_inds = np.array([2, 5, 7])
    frequencies = np.array([100e6])
    rows = [(np.array([1.0, 2.0, 3.0]),)]

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=rows,
        nside=1,
        frequencies=frequencies,
        coordinate_frame="galactic",
        hpx_inds=hpx_inds,
    )

    assert healpix.is_sparse
    assert healpix.coordinate_frame == "galactic"
    assert np.array_equal(healpix.hpx_inds, hpx_inds)
    assert np.array_equal(healpix.maps, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))


def test_region_crop_dense_input_builds_sparse_output():
    nside = 1
    npix = 12
    mask = np.zeros(npix, dtype=bool)
    mask[[1, 4, 9]] = True
    rows = [(np.arange(npix, dtype=float),)]

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=rows,
        nside=nside,
        frequencies=np.array([100e6]),
        coordinate_frame="icrs",
        region=MaskRegion(mask),
    )

    assert healpix.is_sparse
    assert np.array_equal(healpix.hpx_inds, np.array([1, 4, 9]))
    assert np.array_equal(healpix.maps, np.array([[1.0, 4.0, 9.0]], dtype=np.float32))


def test_region_crop_sparse_input_builds_sparse_output():
    hpx_inds = np.array([0, 3, 5, 8])
    mask = np.zeros(12, dtype=bool)
    mask[[3, 8]] = True
    rows = [(np.array([10.0, 30.0, 50.0, 80.0]),)]

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=rows,
        nside=1,
        frequencies=np.array([100e6]),
        coordinate_frame="icrs",
        hpx_inds=hpx_inds,
        region=MaskRegion(mask),
    )

    assert healpix.is_sparse
    assert np.array_equal(healpix.hpx_inds, np.array([3, 8]))
    assert np.array_equal(healpix.maps, np.array([[30.0, 80.0]], dtype=np.float32))


def test_build_iquv_allocates_present_polarization_maps():
    npix = 12
    row = (
        np.full(npix, 1.0),
        np.full(npix, 2.0),
        np.full(npix, 3.0),
        np.full(npix, 4.0),
    )

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=[row],
        nside=1,
        frequencies=np.array([100e6]),
        coordinate_frame="icrs",
    )

    assert np.array_equal(healpix.maps, np.full((1, npix), 1.0, dtype=np.float32))
    assert np.array_equal(healpix.q_maps, np.full((1, npix), 2.0, dtype=np.float32))
    assert np.array_equal(healpix.u_maps, np.full((1, npix), 3.0, dtype=np.float32))
    assert np.array_equal(healpix.v_maps, np.full((1, npix), 4.0, dtype=np.float32))


def test_i_only_leaves_polarization_maps_absent():
    healpix = build_healpix_from_stokes_cube(
        stokes_rows=[(np.ones(12), None, None, None)],
        nside=1,
        frequencies=np.array([100e6]),
        coordinate_frame="icrs",
    )

    assert healpix.q_maps is None
    assert healpix.u_maps is None
    assert healpix.v_maps is None


def test_memmap_output_is_finalized_read_only(tmp_path):
    healpix = build_healpix_from_stokes_cube(
        stokes_rows=[(np.ones(12),)],
        nside=1,
        frequencies=np.array([100e6]),
        coordinate_frame="icrs",
        memmap_dir=str(tmp_path),
    )

    assert isinstance(healpix.maps, np.memmap)
    assert healpix.maps.mode == "r"
    with pytest.raises(ValueError, match="read-only"):
        healpix.maps[0, 0] = 2.0
