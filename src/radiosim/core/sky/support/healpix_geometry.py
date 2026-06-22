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

import gc

import healpy as hp
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


def close_memmap(arr: np.ndarray) -> None:
    """Flush and release a NumPy memmap without directly closing ``arr._mmap``.

    NumPy does not expose a public close method on ``memmap``.  The least
    private cleanup path is to flush the memmap and drop references to the
    mapping-owning object so the mmap finalizer can run.  This is intended for
    exception cleanup where the array will not be returned to callers.
    """
    if not isinstance(arr, np.memmap):
        return
    arr.flush()
    root: np.ndarray = arr
    while isinstance(getattr(root, "base", None), np.ndarray):
        root = root.base
    if isinstance(root, np.memmap):
        root.flush()
    gc.collect()


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


def ordered_row(
    values: np.ndarray,
    *,
    builder_handles_scatter: bool,
    pix: np.ndarray | None,
    npix: int,
    is_nested: bool,
) -> np.ndarray:
    """Map a stored Stokes row into the RING-ordered dense layout the builder expects.

    Single shared replacement for the two ``_ring_ordered_row`` closures that
    were duplicated in the pyradiosky and skyh5 HEALPix loaders (spec item B8).
    The branch precedence is, in order:

    1. **builder** — when ``builder_handles_scatter`` is true the downstream
       ``build_healpix_from_stokes_cube`` performs the sparse scatter itself
       (it is handed ``hpx_inds``), so the stored row is returned unchanged.
    2. **pix** — otherwise, when an explicit RING-ordered pixel-index array
       ``pix`` is given, the row is densely scattered via
       :func:`ring_ordered_row` (``full[pix] = row``).
    3. **nest** — otherwise, when the input is NEST-ordered, it is reordered to
       RING with ``healpy.reorder(..., n2r=True)``.
    4. **passthrough** — otherwise the row is already dense and RING-ordered and
       is returned as-is.

    Parameters
    ----------
    values : np.ndarray
        Stored Stokes values for one channel (file order).
    builder_handles_scatter : bool
        True when the downstream cube builder owns the sparse scatter
        (``builder_hpx_inds is not None`` at the call site).
    pix : np.ndarray, optional
        RING-ordered HEALPix pixel index for each stored value, or ``None``.
        Used only when ``builder_handles_scatter`` is false.
    npix : int
        Length of the dense output row (``12 * nside**2``).
    is_nested : bool
        Whether a dense input row is NEST-ordered (reordered to RING when so).

    Returns
    -------
    np.ndarray
        The row in the RING-ordered layout the cube builder consumes (float64).
    """
    row = np.asarray(values, dtype=np.float64)
    if builder_handles_scatter:
        return row
    if pix is not None:
        return ring_ordered_row(row, pix, npix)
    if is_nested:
        return hp.reorder(row, n2r=True)
    return row
