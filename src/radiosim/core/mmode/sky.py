r"""Analytic point, HEALPix and hybrid scalar sky coefficients.

``docs/development/sci004_mmode_design.md`` Section 7.1 fixes three rules this
module implements and nothing else.

Point components are **not silently rasterized**.  A delta-function point sky
uses analytic scalar harmonics evaluated at the exact transported source
direction, so ``a_lm = sum_s S_s conj(Y_lm(theta_s, phi_s))`` exactly rather
than through a pixel grid.  The first production scope rejects Gaussian
morphology because its baseline-dependent envelope is not one common sky field;
adding analytic extended-source harmonics requires a design successor.

HEALPix maps are integrated with the pixel solid angle.  RING and NEST inputs
must yield identical coefficients after canonical ordering, which is what the
explicit reordering below guarantees: a NEST payload is permuted into canonical
RING order and then summed by exactly the same expression, so the two results
are bit-identical rather than merely close.

A hybrid model adds point and map coefficients in the fixed
``("point", "healpix")`` order **before** any ``B_lm a_lm`` product.  It does not
run two independent m-mode solvers and add rounded outputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from radiosim.core.mmode.harmonics import (
    packed_conjugate_harmonics,
    scalar_packed_block_table,
)
from radiosim.core.mmode.types import ScalarHarmonicCoefficients, ScalarPackedTable

__all__ = [
    "healpix_scalar_coefficients",
    "hybrid_scalar_coefficients",
    "point_scalar_coefficients",
    "ring_directions",
]


def point_scalar_coefficients(
    *,
    ra_rad: Sequence[float] | np.ndarray,
    dec_rad: Sequence[float] | np.ndarray,
    flux: Sequence[float] | np.ndarray,
    lmax: int,
    mmax: int,
    table: ScalarPackedTable | None = None,
) -> ScalarHarmonicCoefficients:
    """Return the analytic delta-function coefficients of a point component.

    Parameters
    ----------
    ra_rad, dec_rad : sequence of float
        The transported source directions, as right ascension and declination
        in radians.  Colatitude is ``pi/2 - dec``.
    flux : sequence of float
        The per-source Stokes ``I`` value already resolved at the frequency the
        caller is transforming.
    lmax, mmax : int
        The retained truncation dimensions.
    table : ScalarPackedTable, optional
        Reuse an already built block table instead of rebuilding it.
    """
    right_ascension = np.atleast_1d(np.asarray(ra_rad, dtype=np.float64))
    declination = np.atleast_1d(np.asarray(dec_rad, dtype=np.float64))
    amplitude = np.atleast_1d(np.asarray(flux, dtype=np.float64))
    if right_ascension.shape != declination.shape or amplitude.shape != (
        right_ascension.shape[0],
    ):
        raise ValueError("point coordinates and fluxes must have one shape")
    resolved = (
        table if table is not None else scalar_packed_block_table(lmax=lmax, mmax=mmax)
    )
    colatitude = 0.5 * math.pi - declination
    harmonics = packed_conjugate_harmonics(resolved, colatitude, right_ascension)
    return ScalarHarmonicCoefficients(
        table=resolved, values=amplitude.astype(np.complex128) @ harmonics
    )


def ring_directions(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical RING colatitude and longitude arrays for one nside."""
    from radiosim.core.sky.support.healpy import lazy_healpy

    module = lazy_healpy
    npix = 12 * int(nside) * int(nside)
    x, y, z = module.pix2vec(int(nside), np.arange(npix), nest=False)
    theta = np.arccos(np.clip(np.asarray(z, dtype=np.float64), -1.0, 1.0))
    phi = np.mod(
        np.arctan2(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)),
        2.0 * math.pi,
    )
    return (theta, phi)


def healpix_scalar_coefficients(
    pixel_values: Sequence[float] | np.ndarray,
    *,
    nside: int,
    order: str,
    lmax: int,
    mmax: int,
    table: ScalarPackedTable | None = None,
) -> ScalarHarmonicCoefficients:
    """Return the Section 7.1 **pixel-measure** coefficients of a HEALPix map.

    Section 7.1 (as corrected) rules the map's coefficients to be exactly

    .. math::

        a_{lm}=\\sum_{\rm pix} s_{\rm pix}\\,\\Omega_{\rm pix}\\,
        \\overline{Y_{lm}(\\hat n_{\rm pix})}

    over canonical-RING pixel centres with the equal pixel solid angle
    ``Omega_pix = 4*pi/npix`` -- **the same measure the private direct oracle
    sums** -- so harmonic-versus-direct agreement tests truncation and nothing
    else, and a constant map's ``l > 0`` coefficients carry the pixel-quadrature
    residue rather than being zero.

    A continuous band-limited reinterpretation of the map, a ring-weighted
    quadrature, or any iterated transform is a *different sky object* and is
    rejected.  The displayed sum is evaluated here directly.  ``healpy``'s
    ``map2alm(..., iter=0)`` with no quadrature weights is numerically the same
    functional and agrees to ``~1e-16``, but it is an FFT/recursion route rather
    than this expression, so the explicit projection is what runs.

    ``order`` is ``"ring"`` or ``"nest"``.  A NEST payload is permuted into
    canonical RING order first, so the two orderings produce bit-identical
    coefficients rather than merely equal ones.
    """
    from radiosim.core.sky.support.healpy import lazy_healpy

    module = lazy_healpy
    resolution = int(nside)
    npix = 12 * resolution * resolution
    values = np.asarray(pixel_values, dtype=np.float64)
    if values.shape != (npix,):
        raise ValueError("the HEALPix payload must be a complete full-sky map")
    normalized = str(order).lower()
    if normalized == "nest":
        values = values[module.ring2nest(resolution, np.arange(npix))]
    elif normalized != "ring":
        raise ValueError("order must be 'ring' or 'nest'")
    resolved = (
        table if table is not None else scalar_packed_block_table(lmax=lmax, mmax=mmax)
    )
    theta, phi = ring_directions(resolution)
    harmonics = packed_conjugate_harmonics(resolved, theta, phi)
    solid_angle = 4.0 * math.pi / npix
    packed = (values.astype(np.complex128) * solid_angle) @ harmonics
    return ScalarHarmonicCoefficients(table=resolved, values=packed)


def hybrid_scalar_coefficients(
    *,
    point: ScalarHarmonicCoefficients,
    healpix: ScalarHarmonicCoefficients,
    component_order: Sequence[str] = ("point", "healpix"),
) -> ScalarHarmonicCoefficients:
    """Add point and map coefficients in the fixed Section 7.1 component order."""
    order = tuple(str(name) for name in component_order)
    if order != ("point", "healpix"):
        raise ValueError("the hybrid component order is fixed at ('point', 'healpix')")
    if point.table.block_table_sha256 != healpix.table.block_table_sha256:
        raise ValueError("hybrid components must share one packed block table")
    contributions = {"point": point.values, "healpix": healpix.values}
    total = np.zeros_like(point.values)
    for name in order:
        total = total + contributions[name]
    return ScalarHarmonicCoefficients(table=point.table, values=total)
