"""Tests for FITS sky loader edge cases."""

import numpy as np
import pytest
from astropy.io import fits

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import SkyRegion
from radiosim.core.sky.loaders.fits import load_fits_image


def test_fits_loader_missing_file_has_actionable_error(tmp_path):
    missing = tmp_path / "missing.fits"

    with pytest.raises(OSError, match="Could not open FITS image file"):
        load_fits_image(
            str(missing),
            nside=1,
            precision=PrecisionConfig.standard(),
        )


def test_fits_loader_stokes_i_only_avoids_polarization_allocations(
    monkeypatch, tmp_path
):
    import reproject

    from radiosim.core.sky.support import allocation as _allocation

    def fake_reproject_to_healpix(image_and_wcs, frame, nside, **kwargs):
        image_2d, _wcs = image_and_wcs
        npix = 12 * nside**2
        return np.full(npix, image_2d[0, 0], dtype=np.float64), np.ones(npix)

    allocations: list[str] = []
    real_allocate_cube = _allocation.allocate_cube

    def spy_allocate_cube(shape, dtype, scratch, name):
        allocations.append(name)
        return real_allocate_cube(shape, dtype, scratch, name)

    monkeypatch.setattr(reproject, "reproject_to_healpix", fake_reproject_to_healpix)
    monkeypatch.setattr(_allocation, "allocate_cube", spy_allocate_cube)

    data = np.ones((2, 2), dtype=np.float64)
    hdu = fits.PrimaryHDU(data)
    header = hdu.header
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["RESTFRQ"] = 100e6
    header["BUNIT"] = "Jy/pixel"

    fits_path = tmp_path / "stokes_i_only.fits"
    hdu.writeto(fits_path)

    sky = load_fits_image(
        str(fits_path),
        nside=1,
        brightness_conversion="planck",
        precision=PrecisionConfig.standard(),
    )

    assert allocations == ["i_maps"]
    assert sky.healpix is not None
    assert sky.healpix.q_maps is None
    assert sky.healpix.u_maps is None
    assert sky.healpix.v_maps is None


def test_fits_loader_region_filter_returns_sparse_healpix(monkeypatch, tmp_path):
    import reproject

    def fake_reproject_to_healpix(image_and_wcs, frame, nside, **kwargs):
        assert kwargs["nested"] is False
        npix = 12 * nside**2
        return np.arange(npix, dtype=np.float64), np.ones(npix)

    monkeypatch.setattr(reproject, "reproject_to_healpix", fake_reproject_to_healpix)

    data = np.ones((2, 2), dtype=np.float64)
    hdu = fits.PrimaryHDU(data)
    header = hdu.header
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["RESTFRQ"] = 100e6
    header["BUNIT"] = "K"

    fits_path = tmp_path / "region_filtered.fits"
    hdu.writeto(fits_path)

    sky = load_fits_image(
        str(fits_path),
        nside=4,
        region=SkyRegion.cone(180.0, -30.0, 20.0),
        precision=PrecisionConfig.standard(),
    )

    assert sky.healpix is not None
    assert sky.healpix.is_sparse
    assert sky.healpix.n_pixels < sky.healpix.full_n_pixels


def test_fits_loader_can_return_nested_healpix_order(monkeypatch, tmp_path):
    import healpy as hp
    import reproject

    calls: list[bool] = []

    def fake_reproject_to_healpix(image_and_wcs, frame, nside, **kwargs):
        calls.append(kwargs["nested"])
        npix = hp.nside2npix(nside)
        return np.arange(npix, dtype=np.float64), np.ones(npix)

    monkeypatch.setattr(reproject, "reproject_to_healpix", fake_reproject_to_healpix)

    data = np.ones((2, 2), dtype=np.float64)
    hdu = fits.PrimaryHDU(data)
    header = hdu.header
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["RESTFRQ"] = 100e6
    header["BUNIT"] = "K"

    fits_path = tmp_path / "nested.fits"
    hdu.writeto(fits_path)

    sky = load_fits_image(
        str(fits_path),
        nside=4,
        nested=True,
        precision=PrecisionConfig.standard(),
    )

    assert calls == [True]
    assert sky.healpix is not None
    assert sky.healpix.is_nested
    np.testing.assert_array_equal(sky.healpix.maps[0], np.arange(hp.nside2npix(4)))


def test_fits_loader_preserves_signed_polarized_stokes(monkeypatch, tmp_path):
    import reproject

    def fake_reproject_to_healpix(image_and_wcs, frame, nside, **kwargs):
        image_2d, _wcs = image_and_wcs
        npix = 12 * nside**2
        return np.full(npix, image_2d[0, 0], dtype=np.float64), np.ones(npix)

    monkeypatch.setattr(reproject, "reproject_to_healpix", fake_reproject_to_healpix)

    data = np.zeros((4, 1, 2, 2), dtype=np.float64)
    data[0, 0] = 1.0
    data[1, 0] = -0.5
    data[2, 0] = 0.25
    data[3, 0] = -0.125

    hdu = fits.PrimaryHDU(data)
    header = hdu.header
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CTYPE3"] = "FREQ"
    header["CTYPE4"] = "STOKES"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRVAL3"] = 100e6
    header["CRVAL4"] = 1.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRPIX3"] = 1.0
    header["CRPIX4"] = 1.0
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    header["CDELT3"] = 1.0
    header["CDELT4"] = 1.0
    header["CUNIT3"] = "Hz"
    header["BUNIT"] = "Jy/pixel"

    fits_path = tmp_path / "signed_stokes.fits"
    hdu.writeto(fits_path)

    sky = load_fits_image(
        str(fits_path),
        nside=1,
        brightness_conversion="planck",
        precision=PrecisionConfig.standard(),
    )

    assert sky.healpix.q_maps is not None
    assert sky.healpix.u_maps is not None
    assert sky.healpix.v_maps is not None
    assert np.all(sky.healpix.q_maps[0] < 0)
    assert np.all(sky.healpix.u_maps[0] > 0)
    assert np.all(sky.healpix.v_maps[0] < 0)


def _uniform_fits(tmp_path, *, bunit, value, cdelt_deg=2.0):
    """Write a uniform 16x16 TAN image and return its path."""
    data = np.full((16, 16), value, dtype=np.float64)
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRVAL1"] = 0.0
    h["CRVAL2"] = 0.0
    h["CRPIX1"] = 8.0
    h["CRPIX2"] = 8.0
    h["CDELT1"] = -cdelt_deg
    h["CDELT2"] = cdelt_deg
    h["RESTFRQ"] = 100e6
    h["BUNIT"] = bunit
    path = tmp_path / f"uniform_{bunit.replace('/', '_')}.fits"
    hdu.writeto(path)
    return str(path)


def test_fits_jy_pixel_and_jy_sr_are_physically_consistent(tmp_path):
    """A Jy/pixel image and the physically-identical Jy/sr image must produce
    the same HEALPix map — i.e. Jy/pixel is normalized by the input pixel
    solid angle before reprojection (flux conservation), not treated as flux.
    """
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_area

    cdelt_deg = 2.0
    # Reconstruct the input FITS pixel solid angle exactly as the loader does.
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.crpix = [8.0, 8.0]
    wcs.wcs.cdelt = [-cdelt_deg, cdelt_deg]
    pixel_area_sr = float(proj_plane_pixel_area(wcs) * (np.pi / 180.0) ** 2)

    brightness = 7.0  # Jy/sr
    jy_sr = load_fits_image(
        _uniform_fits(tmp_path, bunit="Jy/sr", value=brightness, cdelt_deg=cdelt_deg),
        nside=16,
        brightness_conversion="rayleigh-jeans",
        precision=PrecisionConfig.standard(),
    )
    jy_pixel = load_fits_image(
        _uniform_fits(
            tmp_path,
            bunit="Jy/pixel",
            value=brightness * pixel_area_sr,  # same physical sky
            cdelt_deg=cdelt_deg,
        ),
        nside=16,
        brightness_conversion="rayleigh-jeans",
        precision=PrecisionConfig.standard(),
    )

    m_sr = jy_sr.healpix.maps[0]
    m_pix = jy_pixel.healpix.maps[0]
    covered = (m_sr != 0) & (m_pix != 0)
    assert covered.sum() > 10
    np.testing.assert_allclose(m_pix[covered], m_sr[covered], rtol=1e-6)
