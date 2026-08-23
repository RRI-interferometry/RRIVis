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
    field_columns,
    packed_conjugate_harmonics,
    packed_polarized_conjugate_harmonics,
    polarized_packed_block_table,
    scalar_packed_block_table,
)
from radiosim.core.mmode.types import (
    PolarizedHarmonicCoefficients,
    PolarizedPackedTable,
    ScalarHarmonicCoefficients,
    ScalarPackedTable,
)

__all__ = [
    "healpix_polarized_coefficients",
    "healpix_scalar_coefficients",
    "hybrid_polarized_coefficients",
    "hybrid_scalar_coefficients",
    "point_polarized_coefficients",
    "point_polarized_coefficients_per_frequency",
    "point_scalar_coefficients",
    "resolve_stokes_fields",
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


# ---------------------------------------------------------------------------
# Section 7.1 full-Stokes coefficient constructions
# ---------------------------------------------------------------------------


def resolve_stokes_fields(
    stokes_i: np.ndarray,
    stokes_q: np.ndarray,
    stokes_u: np.ndarray,
    stokes_v: np.ndarray,
) -> dict[str, np.ndarray]:
    r"""Return the four Section 5.3 field values from a RadioSim Stokes payload.

    Section 5.2's bridge sends ``U -> U_H = -U`` and leaves ``I``, ``Q`` and
    ``V`` alone, and Section 5.3 expands

    .. math::

        Q_H+iU_H=\sum a^{(+2)}_{lm}\,{}_{+2}Y_{lm},\qquad
        Q_H-iU_H=\sum a^{(-2)}_{lm}\,{}_{-2}Y_{lm},

    so the ``+2`` field value is ``Q_H + i U_H = Q - i U`` and the ``-2`` field
    value is its complex conjugate.  ``I`` and ``V`` are scalar (spin-0)
    expansions and pass through unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> fields = resolve_stokes_fields(
    ...     np.array([1.0]), np.array([0.4]), np.array([-0.3]), np.array([0.2])
    ... )
    >>> complex(fields["+2"][0]), complex(fields["-2"][0])
    ((0.4+0.3j), (0.4-0.3j))
    """
    from radiosim.core.polarization import stokes_to_shaw_fields

    intensity, linear_q, linear_u, circular = stokes_to_shaw_fields(
        np.asarray(stokes_i, dtype=np.float64),
        np.asarray(stokes_q, dtype=np.float64),
        np.asarray(stokes_u, dtype=np.float64),
        np.asarray(stokes_v, dtype=np.float64),
    )
    plus = linear_q.astype(np.complex128) + 1j * linear_u.astype(np.complex128)
    return {
        "I": intensity.astype(np.complex128),
        "+2": plus,
        "-2": np.conjugate(plus),
        "V": circular.astype(np.complex128),
    }


def _contract_fields(
    table: PolarizedPackedTable,
    harmonics: np.ndarray,
    weighted: dict[str, np.ndarray],
) -> np.ndarray:
    """Contract each field's own value vector against its own packed columns."""
    packed = np.zeros(table.packed_value_count, dtype=np.complex128)
    for field in table.field_order:
        columns = field_columns(table, field)
        if columns.size == 0:
            continue
        packed[columns] = weighted[field] @ harmonics[:, columns]
    return packed


def point_polarized_coefficients(
    *,
    ra_rad: Sequence[float] | np.ndarray,
    dec_rad: Sequence[float] | np.ndarray,
    stokes: Sequence[Sequence[float]] | np.ndarray,
    lmax: int,
    mmax: int,
    tangent_frame: object = None,
    table: PolarizedPackedTable | None = None,
) -> PolarizedHarmonicCoefficients:
    r"""Return the analytic full-Stokes coefficients of a point component.

    Section 7.1: "Point components are not silently rasterized. A delta-function
    point sky uses analytic scalar and spin harmonics evaluated at the exact
    transported source direction."  For a source of Shaw-basis brightness
    ``(I, Q_H, U_H, V)`` at ``n_s`` that gives exactly

    ``a^I_lm = I conj(Y_lm(n_s))``,
    ``a^(+2)_lm = (Q_H + i U_H) conj(_{+2}Y_lm(n_s))``,
    ``a^(-2)_lm = (Q_H - i U_H) conj(_{-2}Y_lm(n_s))`` and
    ``a^V_lm = V conj(Y_lm(n_s))``.

    Parameters
    ----------
    ra_rad, dec_rad : sequence of float
        The transported source directions in radians; colatitude is
        ``pi/2 - dec``.
    stokes : array-like, shape ``(n_sources, 4)``
        Resolved ``(I, Q, U, V)`` per source at the frequency being transformed.
    lmax, mmax : int
        The retained truncation dimensions.
    tangent_frame : mapping or TangentPolarizationFrame, optional
        The declared source convention.  Section 5.1 rejects a polarized payload
        that does not declare one; an ``I``/``V``-only payload may omit it.
    table : PolarizedPackedTable, optional
        Reuse an already built block table instead of rebuilding it.
    """
    from radiosim.core.sky.containers import TangentPolarizationFrame

    right_ascension = np.atleast_1d(np.asarray(ra_rad, dtype=np.float64))
    declination = np.atleast_1d(np.asarray(dec_rad, dtype=np.float64))
    payload = np.atleast_2d(np.asarray(stokes, dtype=np.float64))
    if right_ascension.shape != declination.shape:
        raise ValueError("point coordinates must have one shape")
    if payload.shape != (right_ascension.shape[0], 4):
        raise ValueError("the point Stokes payload must be (n_sources, 4)")

    TangentPolarizationFrame.require_for(
        stokes_q=payload[:, 1],
        stokes_u=payload[:, 2],
        stokes_v=payload[:, 3],
        frame=tangent_frame,
    )

    resolved = (
        table
        if table is not None
        else polarized_packed_block_table(lmax=lmax, mmax=mmax)
    )
    colatitude = 0.5 * math.pi - declination
    harmonics = packed_polarized_conjugate_harmonics(
        resolved, colatitude, right_ascension
    )
    fields = resolve_stokes_fields(
        payload[:, 0], payload[:, 1], payload[:, 2], payload[:, 3]
    )
    return PolarizedHarmonicCoefficients(
        table=resolved,
        values=_contract_fields(resolved, harmonics, fields),
        component_order=("point",),
    )


def point_polarized_coefficients_per_frequency(
    *,
    table: PolarizedPackedTable,
    colatitude: Sequence[float] | np.ndarray,
    longitude: Sequence[float] | np.ndarray,
    stokes: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    tangent_frame: object = None,
) -> np.ndarray:
    r"""Return ``(frequency, packed_value)`` analytic point coefficients.

    This is :func:`point_polarized_coefficients` over a run's whole frequency
    axis, taking the **already transported** spherical angles of each source
    rather than catalogue coordinates: Section 7.1 evaluates the analytic
    harmonics "at the exact transported source direction", and the production
    solver holds that direction as a frozen-frame unit vector.  Re-deriving a
    declination from it and a colatitude from that declination would move the
    evaluation point by a rounding, so the angles are passed through once.

    The harmonics are evaluated once and reused across the frequency axis; only
    the resolved Stokes payload varies with frequency.

    Parameters
    ----------
    table : PolarizedPackedTable
        The retained four-field block table.
    colatitude, longitude : sequence of float
        The transported source angles in radians, one entry per source.
    stokes : array-like, shape ``(n_sources, n_frequencies, 4)``
        Resolved ``(I, Q, U, V)`` per source and frequency.
    tangent_frame : mapping or TangentPolarizationFrame, optional
        The declared source convention; Section 5.1 rejects a polarized payload
        that does not declare one and lets an ``I``/``V``-only payload omit it.
    """
    from radiosim.core.sky.containers import TangentPolarizationFrame

    theta = np.atleast_1d(np.asarray(colatitude, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(longitude, dtype=np.float64))
    payload = np.asarray(stokes, dtype=np.float64)
    if theta.shape != phi.shape:
        raise ValueError("point coordinates must have one shape")
    if payload.ndim != 3 or payload.shape[0] != theta.shape[0] or payload.shape[2] != 4:
        raise ValueError(
            "the point Stokes payload must be (n_sources, n_frequencies, 4)"
        )

    TangentPolarizationFrame.require_for(
        stokes_q=payload[:, :, 1],
        stokes_u=payload[:, :, 2],
        stokes_v=payload[:, :, 3],
        frame=tangent_frame,
    )

    harmonics = packed_polarized_conjugate_harmonics(table, theta, phi)
    values = np.zeros(
        (payload.shape[1], int(table.packed_value_count)), dtype=np.complex128
    )
    for index in range(payload.shape[1]):
        channel = payload[:, index, :]
        fields = resolve_stokes_fields(
            channel[:, 0], channel[:, 1], channel[:, 2], channel[:, 3]
        )
        values[index] = _contract_fields(table, harmonics, fields)
    return values


def _resolve_maps(
    maps: object, *, npix: int, order: str, module: object
) -> dict[str, np.ndarray]:
    """Return canonical-RING ``I/Q/U/V`` pixel arrays from a declared payload."""
    if isinstance(maps, dict):
        missing = {"I", "Q", "U", "V"} - set(maps)
        if missing:
            raise ValueError(f"the HEALPix payload is missing {sorted(missing)}")
        columns = {name: np.asarray(maps[name], dtype=np.float64) for name in "IQUV"}
    else:
        array = np.asarray(maps, dtype=np.float64)
        if array.shape != (4, npix):
            raise ValueError("an array HEALPix payload must be (4, npix)")
        columns = {name: array[index] for index, name in enumerate("IQUV")}
    for name, values in columns.items():
        if values.shape != (npix,):
            raise ValueError(f"the {name} map must be a complete full-sky map")
    normalized = str(order).lower()
    if normalized == "nest":
        permutation = module.ring2nest(int(round((npix / 12) ** 0.5)), np.arange(npix))
        columns = {name: values[permutation] for name, values in columns.items()}
    elif normalized != "ring":
        raise ValueError("order must be 'ring' or 'nest'")
    return columns


def healpix_polarized_coefficients(
    maps: object,
    *,
    nside: int,
    order: str,
    lmax: int,
    mmax: int,
    tangent_frame: object = None,
    table: PolarizedPackedTable | None = None,
) -> PolarizedHarmonicCoefficients:
    r"""Return the Section 7.1 **pixel-measure** coefficients of a polarized map.

    Section 7.1 (as corrected) rules the map's coefficients to be exactly

    .. math::

        a_{lm}=\sum_{\rm pix} s_{\rm pix}\,\Omega_{\rm pix}\,
        \overline{{}_{s}Y_{lm}(\hat n_{\rm pix})}

    over canonical-RING pixel centres with ``Omega_pix = 4*pi/npix`` -- the same
    measure the private direct oracle sums -- so a harmonic-versus-direct
    comparison tests truncation and nothing else.  A continuous band-limited
    reinterpretation, a ring-weighted quadrature, or any iterated transform is a
    *different sky object* and is rejected; the displayed sum is evaluated here
    directly.  A NEST payload is permuted into canonical RING order first, so
    the two orderings produce **bit-identical** coefficients.
    """
    from radiosim.core.sky.containers import TangentPolarizationFrame
    from radiosim.core.sky.support.healpy import lazy_healpy

    resolution = int(nside)
    npix = 12 * resolution * resolution
    columns = _resolve_maps(maps, npix=npix, order=order, module=lazy_healpy)

    TangentPolarizationFrame.require_for(
        stokes_q=columns["Q"],
        stokes_u=columns["U"],
        stokes_v=columns["V"],
        frame=tangent_frame,
    )

    resolved = (
        table
        if table is not None
        else polarized_packed_block_table(lmax=lmax, mmax=mmax)
    )
    theta, phi = ring_directions(resolution)
    harmonics = packed_polarized_conjugate_harmonics(resolved, theta, phi)
    solid_angle = 4.0 * math.pi / npix
    fields = resolve_stokes_fields(
        columns["I"], columns["Q"], columns["U"], columns["V"]
    )
    weighted = {name: values * solid_angle for name, values in fields.items()}
    return PolarizedHarmonicCoefficients(
        table=resolved,
        values=_contract_fields(resolved, harmonics, weighted),
        component_order=("healpix",),
    )


def hybrid_polarized_coefficients(
    *,
    point: PolarizedHarmonicCoefficients,
    healpix: PolarizedHarmonicCoefficients,
    component_order: Sequence[str] = ("point", "healpix"),
) -> PolarizedHarmonicCoefficients:
    """Add point and map coefficients in the fixed Section 7.1 component order.

    Section 7.1: "A hybrid model adds point and map coefficients in the fixed
    ``("point", "healpix")`` order before any ``B_lm a_lm`` product; it does not
    run two independent m-mode solvers and add rounded outputs."
    """
    resolved = tuple(str(name) for name in component_order)
    if resolved != ("point", "healpix"):
        raise ValueError("the hybrid component order is fixed at ('point', 'healpix')")
    if point.table.block_table_sha256 != healpix.table.block_table_sha256:
        raise ValueError("hybrid components must share one packed block table")
    contributions = {"point": point.values, "healpix": healpix.values}
    total = np.zeros_like(point.values)
    for name in resolved:
        total = total + contributions[name]
    return PolarizedHarmonicCoefficients(
        table=point.table, values=total, component_order=resolved
    )
