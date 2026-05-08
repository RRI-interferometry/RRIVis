"""Observability-specific geometry helpers tied to ``BeamSkyProjection``.

General coordinate-system helpers (``radec_to_za_az``, ``wrap_ra_deg``,
``angular_separation_deg``, ``split_wrapped_path``, etc.) live in
:mod:`radiosim.utils.coordinates`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from radiosim.core.jones.beam.projection import BeamSkyProjection
from radiosim.utils.coordinates import radec_to_za_az


def compute_beam_power_on_full_sky_grid(
    beam_power_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    zenith_ra_deg: float,
    zenith_dec_deg: float,
    ra_grid_deg: np.ndarray,
    dec_grid_deg: np.ndarray,
    max_za_deg: float = 90.0,
) -> BeamSkyProjection:
    """Evaluate a beam model on a full-sky RA/Dec grid."""
    ra_mesh, dec_mesh = np.meshgrid(ra_grid_deg, dec_grid_deg)
    za_rad, az_rad = radec_to_za_az(
        ra_mesh,
        dec_mesh,
        zenith_ra_deg=zenith_ra_deg,
        zenith_dec_deg=zenith_dec_deg,
    )
    power = beam_power_func(za_rad, az_rad)
    power = np.asarray(power, dtype=float)
    power[za_rad > np.deg2rad(max_za_deg)] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        power_db = 10.0 * np.log10(np.where(np.isnan(power), np.nan, power + 1e-30))
    power_db[np.isnan(power)] = np.nan
    return BeamSkyProjection(
        ra_grid_deg=ra_grid_deg,
        dec_grid_deg=dec_grid_deg,
        power_db=power_db,
        zenith_ra_deg=zenith_ra_deg,
        zenith_dec_deg=zenith_dec_deg,
        max_za_deg=max_za_deg,
    )


def compute_beam_map_on_healpix(
    beam_power_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    nside: int,
    zenith_ra_deg: float,
    zenith_dec_deg: float,
    max_za_deg: float = 90.0,
    peak_normalize: bool = True,
) -> np.ndarray:
    """Evaluate a beam power model on every HEALPix pixel.

    Pixels above the horizon (``zenith_angle > max_za_deg``) are set to
    zero.  When ``peak_normalize=True`` the returned map is divided by its
    maximum so the beam peak is exactly ``1.0``; an all-zero beam is
    returned unchanged.

    Parameters
    ----------
    beam_power_func
        Callable that takes ``(za_rad, az_rad)`` arrays of matching shape
        and returns the corresponding beam power.  This is the type
        produced by :class:`radiosim.core.jones.beam.fits.BeamFITSHandler`
        and the analytic beam helpers.
    nside
        HEALPix nside of the output map.  The function returns a dense
        ``ndarray`` of shape ``(12 * nside ** 2,)`` in RING ordering.
    zenith_ra_deg, zenith_dec_deg
        Pointing centre (instantaneous zenith) in degrees.
    max_za_deg
        Zenith angle beyond which pixels are zeroed.  Defaults to
        ``90`` (everything above the horizon).
    peak_normalize
        Divide the map by its maximum so the beam peak is ``1.0``.

    Returns
    -------
    np.ndarray
        Dense float64 HEALPix map, shape ``(12 * nside ** 2,)``.
    """
    import healpy as hp

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    pix_ra_deg = np.degrees(phi)
    pix_dec_deg = 90.0 - np.degrees(theta)

    za_rad, az_rad = radec_to_za_az(
        pix_ra_deg,
        pix_dec_deg,
        zenith_ra_deg=zenith_ra_deg,
        zenith_dec_deg=zenith_dec_deg,
    )
    beam_map = np.asarray(beam_power_func(za_rad, az_rad), dtype=float)
    beam_map[za_rad > np.deg2rad(max_za_deg)] = 0.0

    if peak_normalize:
        peak = float(np.nanmax(beam_map))
        if peak > 0:
            beam_map = beam_map / peak
    return beam_map
