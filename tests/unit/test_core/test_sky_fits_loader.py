"""Tests for FITS sky loader edge cases."""

import numpy as np
from astropy.io import fits

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky._loaders_fits import load_fits_image


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
