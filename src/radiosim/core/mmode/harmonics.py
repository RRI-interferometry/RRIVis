r"""Orthonormal Condon-Shortley scalar harmonics and the packed block table.

``docs/development/sci004_mmode_design.md`` Section 5.3 fixes the literal
``radiosim.shaw-polarized-harmonics.v1``: right-handed spherical coordinates
``(theta, phi)`` with colatitude ``theta in [0, pi]`` and ``phi`` increasing
eastward in ``[0, 2*pi)``, **orthonormal complex Condon-Shortley** harmonics
satisfying

.. math::

    \int {}_sY_{lm}\,\overline{{}_sY_{l'm'}}\,d\Omega
    = \delta_{ll'}\delta_{mm'},

scalar expansions for ``I`` and ``V``, and the scalar reality relation
``a[l,-m] = (-1)**m conj(a[l,m])``.

Phase M1 is scalar only, so this module binds the spin-zero half of that
contract plus the packed representation Section 5.3 makes inseparable from it.
The values are computed from the explicit normalized associated-Legendre
recurrence written out below rather than from a library default: Section 5.3
states that "library default signs, iteration counts, or packed-``alm`` order are
never inferred", so the convention is spelled in the code that produces it and
is pinned by analytic tests against the published closed form.

The packed scalar block table is signed-``m``-major over the inclusive ascending
range ``-mmax..mmax``, ascending in ``l`` inside each block, with
``l_start = abs(m)``, ``l_stop = lmax + 1``, and each row starting at the
preceding row's ``value_stop`` from zero.  Invalid ``(l, m)`` cells do not exist
and are not represented by padding whose value could enter a digest.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from types import MappingProxyType

import numpy as np

from radiosim.core.mmode.types import (
    SCALAR_BLOCK_TABLE_DOMAIN,
    ScalarHarmonicCoefficients,
    ScalarPackedTable,
    canonical_json,
    domain_digest,
)

__all__ = [
    "packed_conjugate_harmonics",
    "scalar_coefficient",
    "scalar_packed_block_table",
    "scalar_transform_reference",
    "scalar_ylm",
]


def _normalized_legendre_column(
    order: int, lmax: int, cos_theta: np.ndarray, sin_theta: np.ndarray
) -> np.ndarray:
    r"""Return ``N_lm P_l^m(cos theta)`` for one non-negative ``m``.

    The recurrence is the standard stable normalized form, carrying the
    Condon-Shortley phase explicitly in the seed:

    .. math::

        \tilde P_0^0=\sqrt{\tfrac1{4\pi}},\qquad
        \tilde P_m^m=-\sqrt{\tfrac{2m+1}{2m}}\,\sin\theta\,
            \tilde P_{m-1}^{m-1},

    .. math::

        \tilde P_{m+1}^m=\sqrt{2m+3}\,\cos\theta\,\tilde P_m^m,\qquad
        \tilde P_l^m=a_l^m\left(\cos\theta\,\tilde P_{l-1}^m
            -b_l^m\,\tilde P_{l-2}^m\right)

    with ``a_l^m = sqrt((4 l**2 - 1)/(l**2 - m**2))`` and
    ``b_l^m = sqrt(((l-1)**2 - m**2)/(4 (l-1)**2 - 1))``.
    """
    if order < 0:
        raise ValueError("the normalized column recurrence takes m >= 0")
    seed = np.full(cos_theta.shape, math.sqrt(1.0 / (4.0 * math.pi)), dtype=np.float64)
    for level in range(1, order + 1):
        seed = -math.sqrt((2.0 * level + 1.0) / (2.0 * level)) * sin_theta * seed
    column = np.zeros((lmax + 1, *cos_theta.shape), dtype=np.float64)
    if order > lmax:
        return column
    column[order] = seed
    if order + 1 <= lmax:
        column[order + 1] = math.sqrt(2.0 * order + 3.0) * cos_theta * seed
    for degree in range(order + 2, lmax + 1):
        first = math.sqrt(
            (4.0 * degree * degree - 1.0) / (degree * degree - order * order)
        )
        second = math.sqrt(
            ((degree - 1.0) ** 2 - order * order) / (4.0 * (degree - 1.0) ** 2 - 1.0)
        )
        column[degree] = first * (
            cos_theta * column[degree - 1] - second * column[degree - 2]
        )
    return column


def scalar_ylm(degree: int, order: int, colatitude: float, longitude: float) -> complex:
    """Return one orthonormal complex Condon-Shortley ``Y_lm`` value.

    Parameters
    ----------
    degree, order : int
        ``l`` and the signed ``m``, with ``abs(m) <= l``.
    colatitude : float
        ``theta`` in ``[0, pi]``.
    longitude : float
        ``phi``, increasing eastward.

    Returns
    -------
    complex
        ``Y_lm(theta, phi)`` in the orthonormal Condon-Shortley convention.
    """
    if degree < 0 or abs(order) > degree:
        raise ValueError(f"cell (l={degree}, m={order}) is not a harmonic")
    absolute = abs(int(order))
    cos_theta = np.asarray([math.cos(float(colatitude))], dtype=np.float64)
    sin_theta = np.asarray([math.sin(float(colatitude))], dtype=np.float64)
    column = _normalized_legendre_column(absolute, int(degree), cos_theta, sin_theta)
    magnitude = float(column[int(degree)][0])
    value = magnitude * complex(
        math.cos(absolute * float(longitude)), math.sin(absolute * float(longitude))
    )
    if order < 0:
        value = ((-1.0) ** absolute) * value.conjugate()
    return value


def scalar_packed_block_table(*, lmax: int, mmax: int) -> ScalarPackedTable:
    """Build Section 5.3's scalar packed block table for ``(lmax, mmax)``."""
    if lmax < 0 or mmax < 0 or mmax > lmax:
        raise ValueError("the packed table requires 0 <= mmax <= lmax")
    rows: list[dict[str, int]] = []
    cursor = 0
    for order in range(-mmax, mmax + 1):
        l_start = abs(order)
        l_stop = lmax + 1
        count = l_stop - l_start
        rows.append(
            {
                "m": order,
                "l_start": l_start,
                "l_stop": l_stop,
                "value_start": cursor,
                "value_stop": cursor + count,
            }
        )
        cursor += count
    digest = domain_digest(SCALAR_BLOCK_TABLE_DOMAIN, canonical_json(rows))
    return ScalarPackedTable(
        lmax=int(lmax),
        mmax=int(mmax),
        block_rows=tuple(MappingProxyType(dict(row)) for row in rows),
        packed_value_count=cursor,
        block_table_sha256=digest,
    )


def packed_conjugate_harmonics(
    table: ScalarPackedTable,
    colatitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Return ``conj(Y_lm)`` packed as ``(n_direction, packed_value)``.

    The conjugate placement matches Section 6's expansions: a sky coefficient is
    ``a_lm = integral(f * conj(Y_lm) dOmega)``, so a real field satisfies the
    Section 5.3 reality relation without a second sign convention appearing
    anywhere.
    """
    theta = np.atleast_1d(np.asarray(colatitude, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(longitude, dtype=np.float64))
    if theta.shape != phi.shape:
        raise ValueError("colatitude and longitude must have the same shape")
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    values = np.zeros((theta.shape[0], table.packed_value_count), dtype=np.complex128)
    columns = {
        order: _normalized_legendre_column(order, table.lmax, cos_theta, sin_theta)
        for order in range(table.mmax + 1)
    }
    for row in table.block_rows:
        order = int(row["m"])
        absolute = abs(order)
        column = columns[absolute]
        azimuth = np.exp(1j * absolute * phi)
        start = int(row["value_start"])
        for degree in range(int(row["l_start"]), int(row["l_stop"])):
            harmonic = column[degree] * azimuth
            if order < 0:
                harmonic = ((-1.0) ** absolute) * np.conjugate(harmonic)
            values[:, start + degree - int(row["l_start"])] = np.conjugate(harmonic)
    return values


def scalar_transform_reference(
    field: Callable[[float, float], float],
    *,
    lmax: int,
    mmax: int,
    theta_nodes: int | None = None,
    phi_nodes: int | None = None,
) -> ScalarHarmonicCoefficients:
    """Transform a scalar field with the slow explicit quadrature reference.

    Section 5.3 keeps this reference beside the production HEALPix-RING
    implementation precisely so analytic tests can compare the two.  The nodes
    are Gauss-Legendre in ``cos(theta)`` and uniform in ``phi``; both counts
    default well above the bandwidth the retained ``(lmax, mmax)`` can carry.
    """
    table = scalar_packed_block_table(lmax=lmax, mmax=mmax)
    latitude_count = theta_nodes if theta_nodes is not None else 2 * lmax + 16
    azimuth_count = phi_nodes if phi_nodes is not None else 4 * max(mmax, 1) + 16
    nodes, weights = np.polynomial.legendre.leggauss(latitude_count)
    theta = np.arccos(nodes)
    phi = 2.0 * math.pi * np.arange(azimuth_count, dtype=np.float64) / azimuth_count
    grid_theta, grid_phi = np.meshgrid(theta, phi, indexing="ij")
    samples = np.asarray(
        [[float(field(float(t), float(p))) for p in phi] for t in theta],
        dtype=np.float64,
    )
    harmonics = packed_conjugate_harmonics(
        table, grid_theta.reshape(-1), grid_phi.reshape(-1)
    )
    area = weights[:, None] * np.full((1, azimuth_count), 2.0 * math.pi / azimuth_count)
    packed = (samples * area).reshape(-1) @ harmonics
    return ScalarHarmonicCoefficients(table=table, values=packed)


def scalar_coefficient(
    coefficients: ScalarHarmonicCoefficients, degree: int, order: int
) -> complex:
    """Return the packed ``a_lm`` value of one ``(l, m)`` cell.

    The block table travels with the buffer, so this never infers a layout from
    a buffer length; an unrepresented cell raises rather than reading as zero.
    """
    if not isinstance(coefficients, ScalarHarmonicCoefficients):
        raise TypeError(
            "scalar_coefficient requires a ScalarHarmonicCoefficients buffer "
            "carrying its Section 5.3 block table"
        )
    index = coefficients.table.index(int(degree), int(order))
    return complex(coefficients.values[index])


def packed_values(
    table: ScalarPackedTable, values: Sequence[complex] | np.ndarray
) -> ScalarHarmonicCoefficients:
    """Bind a raw packed buffer to the table that describes it."""
    return ScalarHarmonicCoefficients(
        table=table, values=np.asarray(values, dtype=np.complex128)
    )
