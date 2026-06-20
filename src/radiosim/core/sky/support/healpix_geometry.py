"""Shared HEALPix geometry helpers.

Consolidates three pieces of geometry that were duplicated across the sky
package:

* :func:`pixel_solid_angle` — the ``4π / npix`` pixel-area expression
  inlined in ``subtraction.py``, ``region.py``, and ``convert.py``
  (spec item B2).
* :func:`gnomonic_rotate` — the tangent-plane (gnomonic) projection
  written inline in ``operations/subtraction.py`` (spec item B4). The
  convention is preserved bit-for-bit from
  ``subtraction._gnomonic_patch_coords``.
* :func:`ring_ordered_row` — the dense RING-ordered scatter of sparse
  values used by the pyradiosky / skyh5 HEALPix loaders (spec item B8).
"""

from __future__ import annotations

import numpy as np

#: Clamp floor for the angular-distance cosine in the gnomonic projection.
#: Preserved verbatim from ``subtraction._gnomonic_patch_coords`` so the
#: extracted helper reproduces the original convention exactly.
_GNOMONIC_COS_C_FLOOR: float = 1e-12


def pixel_solid_angle(nside: int) -> float:
    """Return the HEALPix pixel solid angle in steradians.

    Equals ``4π / npix`` with ``npix = 12 * nside**2`` — the single shared
    definition replacing the inline ``4 * np.pi / npix`` expression that
    appeared at several call sites.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE resolution.

    Returns
    -------
    float
        Solid angle subtended by one pixel, in steradians.
    """
    return float(4.0 * np.pi / (12.0 * int(nside) ** 2))


def gnomonic_rotate(
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    ra0_rad: float,
    dec0_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic (tangent-plane) ``(x, y)`` projection in radians.

    Projects sky coordinates ``(ra_rad, dec_rad)`` onto the tangent plane
    at the point ``(ra0_rad, dec0_rad)``. The convention is identical to
    the one previously inlined as ``subtraction._gnomonic_patch_coords``
    (which projected via HEALPix pixel longitudes/latitudes): here
    ``ra``/``dec`` play the roles of the pixel longitude (``phi``) and
    latitude (``π/2 − θ``) respectively.

    Parameters
    ----------
    ra_rad, dec_rad : np.ndarray
        Right ascension and declination of each point (radians).
    ra0_rad, dec0_rad : float
        Right ascension and declination of the tangent point (radians).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tangent-plane ``(x, y)`` coordinates in radians.
    """
    lat0 = dec0_rad
    lat = dec_rad
    dlon = ra_rad - ra0_rad

    cos_c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dlon)
    cos_c = np.where(cos_c <= _GNOMONIC_COS_C_FLOOR, _GNOMONIC_COS_C_FLOOR, cos_c)
    x = np.cos(lat) * np.sin(dlon) / cos_c
    y = (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)) / cos_c
    return x, y


def ring_ordered_row(
    values: np.ndarray,
    hpx_inds: np.ndarray,
    npix: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Scatter sparse ``values`` into a dense RING-ordered length-``npix`` row.

    Reproduces the dense-scatter branch of the loaders' ``_ring_ordered_row``
    (``full = np.zeros(npix); full[pix] = row``), generalised with an
    explicit ``fill`` value for the unobserved pixels.

    Parameters
    ----------
    values : np.ndarray
        Stored values, one per sparse pixel index (file order).
    hpx_inds : np.ndarray
        RING-ordered HEALPix pixel index for each entry of ``values``.
    npix : int
        Length of the dense output row (``12 * nside**2``).
    fill : float, default 0.0
        Value written to pixels absent from ``hpx_inds``.

    Returns
    -------
    np.ndarray
        Dense length-``npix`` row (float64) with ``values`` scattered at
        ``hpx_inds`` and ``fill`` elsewhere.
    """
    row = np.asarray(values, dtype=np.float64)
    full = np.full(int(npix), fill, dtype=np.float64)
    full[np.asarray(hpx_inds)] = row
    return full
