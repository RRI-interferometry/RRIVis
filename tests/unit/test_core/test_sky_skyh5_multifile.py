"""Tests for the ``skyh5_multifile`` loader.

The loader reads a set of single-frequency skyh5 files and stacks them along
the frequency axis.  These tests exercise both the HEALPix and point-source
branches with minimal fixtures written through pyradiosky.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import Latitude, Longitude, SkyCoord
from pyradiosky import SkyModel as PyRadioSkyModel

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky._registry import ensure_default_loaders_registered
from rrivis.core.sky.model import SkyFormat
from rrivis.core.sky.region import ConeRegion

ensure_default_loaders_registered()
from rrivis.core.sky.registry import loader_registry  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _write_healpix_skyh5(
    path: Path,
    freq_hz: float,
    *,
    nside: int = 8,
    include_pol: bool = True,
    hpx_order: str = "ring",
    hpx_inds: np.ndarray | None = None,
    coord_frame: str = "icrs",
    fill: float = 1.0,
) -> None:
    """Write a minimal single-channel HEALPix skyh5 file."""
    npix = hp.nside2npix(nside)
    inds = hpx_inds if hpx_inds is not None else np.arange(npix)
    n_stored = len(inds)
    i_map = np.full(n_stored, fill, dtype=np.float64)
    # Offset each pixel so we can distinguish channel contents in tests.
    i_map = i_map + np.arange(n_stored) * 1e-3 + freq_hz * 1e-12
    stokes = np.zeros((4, 1, n_stored))
    stokes[0, 0, :] = i_map
    if include_pol:
        stokes[1, 0, :] = i_map * 0.1
        stokes[2, 0, :] = i_map * 0.2
        stokes[3, 0, :] = i_map * 0.05
    sky = PyRadioSkyModel(
        component_type="healpix",
        nside=nside,
        hpx_order=hpx_order,
        hpx_inds=inds,
        spectral_type="full",
        freq_array=np.array([freq_hz]) * u.Hz,
        stokes=stokes * (u.Jy / u.sr),
        frame=coord_frame,
    )
    sky.write_skyh5(str(path), clobber=True)


def _write_point_skyh5(
    path: Path,
    freq_hz: float,
    *,
    ra_deg: Iterable[float] | None = None,
    dec_deg: Iterable[float] | None = None,
    flux_jy: Iterable[float] | None = None,
    include_pol: bool = True,
) -> None:
    """Write a minimal single-channel point-source skyh5 file."""
    ra = np.asarray(ra_deg) if ra_deg is not None else np.array([10.0, 120.0, 250.0])
    dec = np.asarray(dec_deg) if dec_deg is not None else np.array([-20.0, 0.0, 30.0])
    flux = (
        np.asarray(flux_jy)
        if flux_jy is not None
        else np.array([1.0, 2.0, 3.0]) * (freq_hz / 100e6) ** -0.7
    )
    n = len(ra)
    skycoord = SkyCoord(
        ra=Longitude(ra, unit="deg"),
        dec=Latitude(dec, unit="deg"),
        frame="icrs",
    )
    stokes = np.zeros((4, 1, n)) * u.Jy
    stokes[0, 0, :] = flux * u.Jy
    if include_pol:
        stokes[1, 0, :] = flux * 0.1 * u.Jy
        stokes[2, 0, :] = flux * 0.2 * u.Jy
        stokes[3, 0, :] = flux * 0.05 * u.Jy
    sky = PyRadioSkyModel(
        name=np.array([f"src{i}" for i in range(n)]),
        skycoord=skycoord,
        stokes=stokes,
        spectral_type="full",
        freq_array=np.array([freq_hz]) * u.Hz,
    )
    sky.write_skyh5(str(path), clobber=True)


# --------------------------------------------------------------------------- #
# Shared invariants
# --------------------------------------------------------------------------- #


class TestInputValidation:
    def test_neither_glob_nor_list_rejected(self, precision: PrecisionConfig) -> None:
        loader = loader_registry.loader("skyh5_multifile")
        with pytest.raises(ValueError, match="exactly one"):
            loader(precision=precision)

    def test_both_glob_and_list_rejected(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        f = tmp_path / "only.skyh5"
        _write_healpix_skyh5(f, 100e6)
        loader = loader_registry.loader("skyh5_multifile")
        with pytest.raises(ValueError, match="exactly one"):
            loader(
                file_glob=str(tmp_path / "*.skyh5"),
                filenames=[str(f)],
                precision=precision,
            )

    def test_empty_glob_rejected(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        loader = loader_registry.loader("skyh5_multifile")
        with pytest.raises(ValueError, match="matched no files"):
            loader(
                file_glob=str(tmp_path / "nope*.skyh5"),
                precision=precision,
            )

    def test_missing_explicit_file_rejected(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        loader = loader_registry.loader("skyh5_multifile")
        with pytest.raises(FileNotFoundError):
            loader(
                filenames=[str(tmp_path / "does_not_exist.skyh5")],
                precision=precision,
            )

    def test_duplicate_frequencies_rejected(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        _write_healpix_skyh5(tmp_path / "a.skyh5", 100e6)
        _write_healpix_skyh5(tmp_path / "b.skyh5", 100e6)
        loader = loader_registry.loader("skyh5_multifile")
        with pytest.raises(ValueError, match="duplicate"):
            loader(
                filenames=[str(tmp_path / "a.skyh5"), str(tmp_path / "b.skyh5")],
                precision=precision,
            )


# --------------------------------------------------------------------------- #
# HEALPix branch
# --------------------------------------------------------------------------- #


class TestHealpixBranch:
    def test_stacks_sorted_frequencies(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        freqs = [200e6, 100e6, 150e6]  # scrambled write order
        for f in freqs:
            _write_healpix_skyh5(tmp_path / f"c_{int(f):09d}.skyh5", f)
        loader = loader_registry.loader("skyh5_multifile")
        sky = loader(
            file_glob=str(tmp_path / "c_*.skyh5"),
            precision=precision,
        )
        assert SkyFormat.HEALPIX in sky.formats
        assert sky.healpix is not None
        assert sky.healpix.maps.shape == (3, hp.nside2npix(8))
        np.testing.assert_array_equal(
            sky.healpix.frequencies, np.array([100e6, 150e6, 200e6])
        )
        assert sky.healpix.has_polarization
        assert sky.healpix.coordinate_frame == "icrs"

    def test_rejects_mismatched_nside(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        _write_healpix_skyh5(tmp_path / "a.skyh5", 100e6, nside=8)
        _write_healpix_skyh5(tmp_path / "b.skyh5", 200e6, nside=16)
        loader = loader_registry.loader("skyh5_multifile")
        with pytest.raises(ValueError, match="mismatched nside"):
            loader(
                filenames=[str(tmp_path / "a.skyh5"), str(tmp_path / "b.skyh5")],
                precision=precision,
            )

    def test_rejects_different_hpx_inds(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        # Same nside, different sparse index arrays.
        _write_healpix_skyh5(tmp_path / "a.skyh5", 100e6, hpx_inds=np.arange(50))
        _write_healpix_skyh5(tmp_path / "b.skyh5", 200e6, hpx_inds=np.arange(10, 60))
        loader = loader_registry.loader("skyh5_multifile")
        with pytest.raises(ValueError, match="hpx_inds differ"):
            loader(
                filenames=[str(tmp_path / "a.skyh5"), str(tmp_path / "b.skyh5")],
                precision=precision,
            )

    def test_sparse_identical_indices_supported(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        inds = np.array([5, 7, 13, 22, 37])
        _write_healpix_skyh5(tmp_path / "a.skyh5", 100e6, hpx_inds=inds)
        _write_healpix_skyh5(tmp_path / "b.skyh5", 200e6, hpx_inds=inds)
        loader = loader_registry.loader("skyh5_multifile")
        sky = loader(
            filenames=[str(tmp_path / "a.skyh5"), str(tmp_path / "b.skyh5")],
            precision=precision,
        )
        assert sky.healpix is not None
        assert sky.healpix.maps.shape == (2, 5)
        assert sky.healpix.hpx_inds is not None
        np.testing.assert_array_equal(sky.healpix.hpx_inds, inds)

    def test_rejects_multi_channel_files(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        # Build a file with Nfreqs > 1 and verify rejection.
        nside = 8
        npix = hp.nside2npix(nside)
        stokes = np.random.rand(4, 2, npix) * (u.Jy / u.sr)
        sky = PyRadioSkyModel(
            component_type="healpix",
            nside=nside,
            hpx_order="ring",
            hpx_inds=np.arange(npix),
            spectral_type="full",
            freq_array=np.array([100e6, 110e6]) * u.Hz,
            stokes=stokes,
            frame="icrs",
        )
        sky.write_skyh5(str(tmp_path / "multi.skyh5"), clobber=True)
        loader = loader_registry.loader("skyh5_multifile")
        with pytest.raises(ValueError, match="exactly one"):
            loader(
                filenames=[str(tmp_path / "multi.skyh5")],
                precision=precision,
            )

    def test_dtype_respects_precision(self, tmp_path: Path) -> None:
        _write_healpix_skyh5(tmp_path / "a.skyh5", 100e6)
        _write_healpix_skyh5(tmp_path / "b.skyh5", 200e6)
        loader = loader_registry.loader("skyh5_multifile")
        sky = loader(
            filenames=[str(tmp_path / "a.skyh5"), str(tmp_path / "b.skyh5")],
            precision=PrecisionConfig.standard(),
        )
        assert sky.healpix is not None
        assert sky.healpix.maps.dtype == np.float32


# --------------------------------------------------------------------------- #
# Point branch
# --------------------------------------------------------------------------- #


class TestPointBranch:
    def test_stacks_into_per_channel_flux(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        freqs = [150e6, 100e6, 200e6]  # scrambled
        for f in freqs:
            _write_point_skyh5(tmp_path / f"p_{int(f):09d}.skyh5", f)
        loader = loader_registry.loader("skyh5_multifile")
        sky = loader(
            file_glob=str(tmp_path / "p_*.skyh5"),
            precision=precision,
        )
        assert SkyFormat.POINT_SOURCES in sky.formats
        assert sky.point is not None
        assert sky.point.n_sources == 3
        assert sky.point.spectrum is not None
        assert sky.point.spectrum.flux.shape == (3, 3)
        np.testing.assert_array_equal(
            sky.point.spectrum.frequencies, np.array([100e6, 150e6, 200e6])
        )

    def test_reference_channel_slice_contract(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        for f in [100e6, 150e6, 200e6]:
            _write_point_skyh5(tmp_path / f"p_{int(f):09d}.skyh5", f)
        loader = loader_registry.loader("skyh5_multifile")
        sky = loader(
            file_glob=str(tmp_path / "p_*.skyh5"),
            reference_frequency_hz=150e6,
            precision=precision,
        )
        assert sky.point is not None
        assert sky.point.spectrum is not None
        # flux == spectrum.flux[ref_idx, :] where ref channel is 150 MHz (idx 1).
        np.testing.assert_allclose(sky.point.flux, sky.point.spectrum.flux[1, :])
        assert sky.point.ref_freq[0] == pytest.approx(150e6)

    def test_rejects_mismatched_source_list(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        _write_point_skyh5(
            tmp_path / "a.skyh5",
            100e6,
            ra_deg=[10.0, 20.0],
            dec_deg=[0.0, 0.0],
            flux_jy=[1.0, 2.0],
        )
        _write_point_skyh5(
            tmp_path / "b.skyh5",
            200e6,
            ra_deg=[30.0, 40.0],
            dec_deg=[0.0, 0.0],
            flux_jy=[1.0, 2.0],
        )
        loader = loader_registry.loader("skyh5_multifile")
        with pytest.raises(ValueError, match="RA/Dec differ"):
            loader(
                filenames=[str(tmp_path / "a.skyh5"), str(tmp_path / "b.skyh5")],
                precision=precision,
            )

    def test_region_filter_on_point_branch(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        _write_point_skyh5(
            tmp_path / "a.skyh5",
            100e6,
            ra_deg=[10.0, 120.0, 250.0],
            dec_deg=[0.0, 0.0, 0.0],
            flux_jy=[1.0, 2.0, 3.0],
        )
        _write_point_skyh5(
            tmp_path / "b.skyh5",
            200e6,
            ra_deg=[10.0, 120.0, 250.0],
            dec_deg=[0.0, 0.0, 0.0],
            flux_jy=[1.0, 2.0, 3.0],
        )
        # Cone around RA=120, Dec=0 with generous radius; keep only middle source.
        region = ConeRegion(ra_deg=120.0, dec_deg=0.0, radius_deg=1.0)
        loader = loader_registry.loader("skyh5_multifile")
        sky = loader(
            filenames=[str(tmp_path / "a.skyh5"), str(tmp_path / "b.skyh5")],
            precision=precision,
            region=region,
        )
        assert sky.point is not None
        assert sky.point.n_sources == 1
        assert sky.point.spectrum is not None
        assert sky.point.spectrum.flux.shape == (2, 1)


# --------------------------------------------------------------------------- #
# Mixed branch rejection
# --------------------------------------------------------------------------- #


class TestMixed:
    def test_mixed_component_types_rejected(
        self, tmp_path: Path, precision: PrecisionConfig
    ) -> None:
        _write_healpix_skyh5(tmp_path / "a.skyh5", 100e6)
        _write_point_skyh5(tmp_path / "b.skyh5", 200e6)
        loader = loader_registry.loader("skyh5_multifile")
        with pytest.raises(ValueError, match="mixed component_type"):
            loader(
                filenames=[str(tmp_path / "a.skyh5"), str(tmp_path / "b.skyh5")],
                precision=precision,
            )
