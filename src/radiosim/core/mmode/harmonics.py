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

This module binds that whole contract -- the spin-zero harmonics for ``I`` and
``V``, the spin-weighted ``+-2`` harmonics for the linear pair, and both packed
representations Section 5.3 makes inseparable from their value buffers.
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
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from radiosim.core.mmode.types import (
    FIELD_ORDER,
    POLARIZED_BLOCK_FIELDS,
    SCALAR_BLOCK_TABLE_DOMAIN,
    SPIN_ORDER,
    PolarizedPackedTable,
    ScalarHarmonicCoefficients,
    ScalarPackedTable,
    canonical_json,
    domain_digest,
)

__all__ = [
    "SpinHarmonicCoefficients",
    "field_columns",
    "packed_conjugate_harmonics",
    "packed_polarized_conjugate_harmonics",
    "packed_spin_harmonics",
    "polarized_packed_block_table",
    "scalar_coefficient",
    "scalar_packed_block_table",
    "scalar_transform_reference",
    "scalar_ylm",
    "spin_transform_reference",
    "spin_ylm",
    "wigner_small_d_column",
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


# ---------------------------------------------------------------------------
# Section 5.3 spin-weighted harmonics
# ---------------------------------------------------------------------------


def _wigner_seed(first: int, second: int, colatitude: np.ndarray) -> np.ndarray:
    r"""Return ``d^{l0}_{a,b}(theta)`` at ``l0 = max(abs(a), abs(b))``.

    Wigner's finite sum

    .. math::

        d^l_{ab}(\beta)=\sqrt{(l+a)!(l-a)!(l+b)!(l-b)!}
        \sum_k\frac{(-1)^k\cos^{2l+b-a-2k}(\beta/2)\sin^{a-b+2k}(\beta/2)}
        {(l+b-k)!(l-a-k)!k!(a-b+k)!}

    runs over ``max(0, b-a) <= k <= min(l+b, l-a)``.  At ``l = l0`` those two
    bounds coincide, so exactly one term survives and the seed needs no
    cancellation-prone summation -- which is what makes it a safe start for the
    upward recursion below.  Factorials are taken through ``lgamma`` so the seed
    stays finite for the whole Section 7.3 ``lmax`` range.
    """
    order = max(abs(int(first)), abs(int(second)))
    index = max(0, int(second) - int(first))
    numerator = 0.5 * (
        math.lgamma(order + first + 1)
        + math.lgamma(order - first + 1)
        + math.lgamma(order + second + 1)
        + math.lgamma(order - second + 1)
    )
    denominator = (
        math.lgamma(order + second - index + 1)
        + math.lgamma(order - first - index + 1)
        + math.lgamma(index + 1)
        + math.lgamma(first - second + index + 1)
    )
    half_cos = np.cos(0.5 * colatitude)
    half_sin = np.sin(0.5 * colatitude)
    return (
        ((-1.0) ** index)
        * math.exp(numerator - denominator)
        * half_cos ** (2 * order + second - first - 2 * index)
        * half_sin ** (first - second + 2 * index)
    )


def wigner_small_d_column(
    first: int, second: int, lmax: int, colatitude: np.ndarray
) -> np.ndarray:
    r"""Return ``d^l_{a,b}(theta)`` for ``l = 0..lmax`` as ``(lmax+1, n_dir)``.

    The three-term recursion in ``l`` at fixed ``(a, b)``,

    .. math::

        d^{l+1}_{ab}=\frac{(2l+1)\bigl(l(l+1)\cos\beta-ab\bigr)d^{l}_{ab}
        -(l+1)\sqrt{(l^2-a^2)(l^2-b^2)}\,d^{l-1}_{ab}}
        {l\sqrt{((l+1)^2-a^2)((l+1)^2-b^2)}},

    is **self-starting**: at ``l = l0 = max(abs(a), abs(b))`` the coefficient
    ``sqrt((l^2-a^2)(l^2-b^2))`` is exactly zero, so the absent ``l0 - 1`` term
    contributes nothing and the single-term seed is the only input the recursion
    needs.  Cells below ``l0`` do not exist and are returned as exact zero rather
    than as a padded value.
    """
    a = int(first)
    b = int(second)
    theta = np.atleast_1d(np.asarray(colatitude, dtype=np.float64))
    start = max(abs(a), abs(b))
    column = np.zeros((int(lmax) + 1, theta.shape[0]), dtype=np.float64)
    if start > int(lmax):
        return column
    column[start] = _wigner_seed(a, b, theta)
    cosine = np.cos(theta)
    previous = np.zeros_like(column[start])
    for degree in range(start, int(lmax)):
        if degree == 0:
            column[1] = cosine * column[0]
        else:
            lead = (2 * degree + 1) * (degree * (degree + 1) * cosine - a * b)
            trail = (degree + 1) * math.sqrt(
                (degree * degree - a * a) * (degree * degree - b * b)
            )
            divisor = degree * math.sqrt(
                ((degree + 1) ** 2 - a * a) * ((degree + 1) ** 2 - b * b)
            )
            column[degree + 1] = (lead * column[degree] - trail * previous) / divisor
        previous = column[degree]
    return column


def spin_ylm(
    spin: int, degree: int, order: int, colatitude: float, longitude: float
) -> complex:
    r"""Return one orthonormal spin-weighted harmonic ``_{s}Y_{lm}``.

    Section 5.3 fixes orthonormal complex Condon-Shortley conventions with
    ``integral(_sY_lm conj(_sY_l'm')) = delta_ll' delta_mm'``.  The value is

    .. math:: {}_{s}Y_{lm}(\theta,\phi)
        =\sqrt{\frac{2l+1}{4\pi}}\,d^{l}_{-m,-s}(\theta)\,e^{im\phi},

    which reduces to the ordinary ``Y_lm`` at ``s = 0`` and reproduces the
    published Goldberg et al. (1967) ``l = 2`` table for ``s = +-2``.

    Parameters
    ----------
    spin : int
        The spin weight ``s``.  Section 5.3 uses ``0`` for ``I``/``V`` and
        ``+-2`` for the linear pair.
    degree, order : int
        ``l`` and the signed ``m``, with ``abs(m) <= l`` and ``abs(s) <= l``.
    colatitude : float
        ``theta`` in ``[0, pi]``.
    longitude : float
        ``phi``, increasing eastward.

    Returns
    -------
    complex
        The harmonic value.

    Raises
    ------
    ValueError
        If ``(l, m, s)`` is not a harmonic cell.  Section 5.3's invalid cells
        "do not exist"; they are never returned as a padded zero.

    Examples
    --------
    ``_{2}Y_{20} = sqrt(15/(32 pi)) sin(theta)**2`` is real and independent of
    ``phi``:

    >>> import math
    >>> value = spin_ylm(2, 2, 0, 0.9, 2.3)
    >>> abs(value - math.sqrt(15.0 / (32.0 * math.pi)) * math.sin(0.9) ** 2) < 1e-14
    True

    At ``s = 0`` it is the ordinary scalar harmonic:

    >>> abs(spin_ylm(0, 3, -2, 0.9, 2.3) - scalar_ylm(3, -2, 0.9, 2.3)) < 1e-14
    True
    """
    weight = int(spin)
    if degree < 0 or abs(int(order)) > degree or abs(weight) > degree:
        raise ValueError(
            f"cell (l={degree}, m={order}, s={spin}) is not a spin-{spin} harmonic"
        )
    column = wigner_small_d_column(
        -int(order), -weight, int(degree), np.asarray([float(colatitude)])
    )
    magnitude = float(column[int(degree)][0]) * math.sqrt(
        (2 * int(degree) + 1) / (4.0 * math.pi)
    )
    return magnitude * complex(
        math.cos(int(order) * float(longitude)),
        math.sin(int(order) * float(longitude)),
    )


def polarized_packed_block_table(*, lmax: int, mmax: int) -> PolarizedPackedTable:
    """Build Section 5.3's polarized packed block table for ``(lmax, mmax)``.

    Rows are signed-``m``-major over ``-mmax..mmax`` and field-minor over the
    fixed ``("I", "+2", "-2", "V")`` order, with ``l_start = max(abs(m),
    abs(spin))``.  The digest is taken under the same Section 14 block-table
    domain the scalar table uses.

    Examples
    --------
    A spin row starts at the spin floor while a scalar row starts at ``abs(m)``:

    >>> table = polarized_packed_block_table(lmax=4, mmax=2)
    >>> table.row("I", 0)["l_start"], table.row("+2", 0)["l_start"]
    (0, 2)
    >>> table.row("+2", -3)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    IndexError: signed m=-3 is outside the retained band
    """
    if lmax < 0 or mmax < 0 or mmax > lmax:
        raise ValueError("the packed table requires 0 <= mmax <= lmax")
    rows: list[dict[str, Any]] = []
    cursor = 0
    for order in range(-int(mmax), int(mmax) + 1):
        for index, (field, spin) in enumerate(
            zip(FIELD_ORDER, SPIN_ORDER, strict=True)
        ):
            l_start = max(abs(order), abs(int(spin)))
            l_stop = int(lmax) + 1
            width = max(0, l_stop - l_start)
            rows.append(
                {
                    "m": order,
                    "field_index": index,
                    "field_name": field,
                    "spin": int(spin),
                    "l_start": l_start,
                    "l_stop": l_stop,
                    "value_start": cursor,
                    "value_stop": cursor + width,
                }
            )
            cursor += width
    for row in rows:
        if tuple(row) != POLARIZED_BLOCK_FIELDS:  # pragma: no cover - defensive
            raise ValueError("a polarized block row is not in Section 5.3's order")
    digest = domain_digest(SCALAR_BLOCK_TABLE_DOMAIN, canonical_json(rows))
    return PolarizedPackedTable(
        lmax=int(lmax),
        mmax=int(mmax),
        block_rows=tuple(MappingProxyType(dict(row)) for row in rows),
        packed_value_count=cursor,
        block_table_sha256=digest,
    )


def field_columns(table: PolarizedPackedTable, field: str) -> np.ndarray:
    """Return the packed column indices belonging to one science field."""
    columns: list[int] = []
    for row in table.block_rows:
        if str(row["field_name"]) != str(field):
            continue
        columns.extend(range(int(row["value_start"]), int(row["value_stop"])))
    return np.asarray(columns, dtype=np.int64)


def packed_spin_harmonics(
    *,
    spin: int,
    lmax: int,
    mmax: int,
    colatitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Return ``conj(_{s}Y_lm)`` for one spin as ``(n_direction, l, signed m)``.

    The conjugate placement matches Section 6's sky expansions: a coefficient is
    ``a_lm = integral(f conj(_{s}Y_lm) dOmega)``, so the reality relations hold
    without a second sign convention appearing anywhere.
    """
    theta = np.atleast_1d(np.asarray(colatitude, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(longitude, dtype=np.float64))
    if theta.shape != phi.shape:
        raise ValueError("colatitude and longitude must have the same shape")
    weight = int(spin)
    values = np.zeros(
        (theta.shape[0], int(lmax) + 1, 2 * int(mmax) + 1), dtype=np.complex128
    )
    normalization = np.sqrt(
        (2.0 * np.arange(int(lmax) + 1, dtype=np.float64) + 1.0) / (4.0 * math.pi)
    )
    for order in range(-int(mmax), int(mmax) + 1):
        column = wigner_small_d_column(-order, -weight, int(lmax), theta)
        azimuth = np.exp(1j * order * phi)
        block = (column * normalization[:, None]) * azimuth[None, :]
        values[:, :, order + int(mmax)] = np.conjugate(block).T
    return values


def packed_polarized_conjugate_harmonics(
    table: PolarizedPackedTable,
    colatitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Return ``conj(_{s}Y_lm)`` packed as ``(n_direction, packed_value)``.

    Each column carries the spin of the field whose block it belongs to, so a
    caller multiplies by that field's own sky or kernel value and contracts once
    -- never by consulting a library ``alm`` order.
    """
    theta = np.atleast_1d(np.asarray(colatitude, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(longitude, dtype=np.float64))
    if theta.shape != phi.shape:
        raise ValueError("colatitude and longitude must have the same shape")
    values = np.zeros((theta.shape[0], table.packed_value_count), dtype=np.complex128)
    by_spin = {
        spin: packed_spin_harmonics(
            spin=spin,
            lmax=table.lmax,
            mmax=table.mmax,
            colatitude=theta,
            longitude=phi,
        )
        for spin in sorted(set(SPIN_ORDER))
    }
    for row in table.block_rows:
        start, stop = int(row["value_start"]), int(row["value_stop"])
        if stop <= start:
            continue
        cube = by_spin[int(row["spin"])]
        column = int(row["m"]) + table.mmax
        values[:, start:stop] = cube[
            :, int(row["l_start"]) : int(row["l_stop"]), column
        ]
    return values


@dataclass(frozen=True, slots=True)
class SpinHarmonicCoefficients:
    """One spin field's packed coefficients, bound to its polarized table."""

    field: str
    table: PolarizedPackedTable
    values: np.ndarray

    def coefficient(self, degree: int, order: int) -> complex:
        """Return the ``(l, m)`` coefficient of this spin field."""
        return complex(self.values[self.table.index(self.field, degree, order)])

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        array = np.array(self.values, copy=True)
        return array if dtype is None else array.astype(dtype)


def _spin_field_name(spin: int) -> str:
    """Return the Section 5.3 field name carrying one spin weight."""
    if int(spin) == 2:
        return "+2"
    if int(spin) == -2:
        return "-2"
    if int(spin) == 0:
        return "I"
    raise ValueError(f"spin {spin} is not one of Section 5.3's spin labels")


def spin_transform_reference(
    field: Callable[[float, float], complex],
    *,
    spin: int,
    lmax: int,
    mmax: int,
    theta_nodes: int | None = None,
    phi_nodes: int | None = None,
) -> SpinHarmonicCoefficients:
    """Transform one complex spin field with the slow quadrature reference.

    Section 5.3 keeps this reference beside the production constructions
    precisely so analytic tests can compare the two.  Nodes are Gauss-Legendre
    in ``cos(theta)`` and uniform in ``phi``, and both counts default well above
    the bandwidth the retained ``(lmax, mmax)`` can carry.  Both spins use the
    same node rule, which is what makes the Section 5.3 paired reality relation
    ``a^(-2)[l,m] = (-1)**m conj(a^(+2)[l,-m])`` exact rather than approximate
    on the retained grid.
    """
    table = polarized_packed_block_table(lmax=lmax, mmax=mmax)
    name = _spin_field_name(spin)
    latitude_count = theta_nodes if theta_nodes is not None else 2 * lmax + 16
    azimuth_count = phi_nodes if phi_nodes is not None else 4 * max(mmax, 1) + 16
    nodes, weights = np.polynomial.legendre.leggauss(latitude_count)
    theta = np.arccos(nodes)
    phi = 2.0 * math.pi * np.arange(azimuth_count, dtype=np.float64) / azimuth_count
    grid_theta, grid_phi = np.meshgrid(theta, phi, indexing="ij")
    samples = np.asarray(
        [[complex(field(float(t), float(p))) for p in phi] for t in theta],
        dtype=np.complex128,
    )
    harmonics = packed_polarized_conjugate_harmonics(
        table, grid_theta.reshape(-1), grid_phi.reshape(-1)
    )
    area = weights[:, None] * np.full((1, azimuth_count), 2.0 * math.pi / azimuth_count)
    packed = np.zeros(table.packed_value_count, dtype=np.complex128)
    columns = field_columns(table, name)
    packed[columns] = (samples * area).reshape(-1) @ harmonics[:, columns]
    return SpinHarmonicCoefficients(field=name, table=table, values=packed)
