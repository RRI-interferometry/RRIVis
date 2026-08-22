r"""SCI-004 phase-M1 red oracles for the scalar spherical-harmonic surface.

``docs/development/sci004_mmode_design.md`` Section 5.3 fixes the literal
``radiosim.shaw-polarized-harmonics.v1``: right-handed spherical coordinates
``(theta, phi)`` with colatitude ``theta in [0, pi]`` and ``phi`` increasing
eastward, **orthonormal complex Condon-Shortley** harmonics satisfying
``integral(sY_lm conj(sY_l'm')) = delta_ll' delta_mm'``, scalar expansions for
``I`` and ``V``, and the scalar reality relation
``a[l,-m] = (-1)**m conj(a[l,m])``.

Phase M1 is scalar only, so this module binds the ``I`` half of that contract
plus the packed representation Section 5.3 makes inseparable from it. The scalar
packed block table has rows ``(m, l_start, l_stop, value_start, value_stop)``,
is **signed-m-major** over the inclusive ascending range ``-mmax..mmax`` with
ascending ``l`` inside each row, sets ``l_start = max(abs(m), abs(spin))`` and
``l_stop = lmax + 1``, starts each row at the preceding row's ``value_stop``
beginning at zero, and has **no padding**: invalid ``(l, m, s)`` cells do not
exist and cannot enter a digest. The table digest is
``D("radiosim.mmode-packed-block-table.v1", J(block_rows))``.

Section 7.1's sky rules complete the family: a delta-function point sky uses
*analytic* harmonics evaluated at the exact transported source direction rather
than being silently rasterized; HEALPix maps are integrated with the pixel solid
angle; RING and NEST inputs must yield identical coefficients after canonical
ordering; and a hybrid model adds point and map coefficients in the fixed
``("point", "healpix")`` order before any ``B_lm a_lm`` product.

Section 12.2's analytic complex128 residual limit is ``5e-12``. The Section 13.3
owners are ``radiosim.core.mmode.harmonics`` and ``radiosim.core.mmode.sky``,
neither of which exists at ``G1``; imports are function-local so each node
yields its own Section 14.1 outcome.
"""

from __future__ import annotations

import cmath
import hashlib
import json
import math
import struct
from typing import Any

BLOCK_TABLE_DOMAIN = "radiosim.mmode-packed-block-table.v1"
HARMONIC_CONVENTION = "radiosim.shaw-polarized-harmonics.v1"

#: Section 5.3's science field order and its spin labels.
FIELD_ORDER: tuple[str, ...] = ("I", "+2", "-2", "V")
SPIN_ORDER: tuple[int, ...] = (0, 2, -2, 0)

#: Section 5.3's exact scalar block-row field order.
SCALAR_BLOCK_FIELDS: tuple[str, ...] = (
    "m",
    "l_start",
    "l_stop",
    "value_start",
    "value_stop",
)

#: Section 12.2's analytic complex128 residual limit.
ANALYTIC_RESIDUAL_LIMIT = 5e-12

LMAX = 6
MMAX = 4
NSIDE = 8

_MODE_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
modes:
  - [2, 1]
  - [3, -2]
  - [4, 0]
colatitude_rad: 0.7
longitude_rad: 1.9
""".encode()

_POINT_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
point_sources:
  - ra_deg: 45.0
    dec_deg: -30.0
    flux_jy: 2.5
""".encode()

_MAP_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
nside: {NSIDE}
constant_kelvin: 1.0
""".encode()

_HYBRID_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
nside: {NSIDE}
component_order: ["point", "healpix"]
point_sources:
  - ra_deg: 45.0
    dec_deg: -30.0
    flux_jy: 2.5
constant_kelvin: 1.0
""".encode()

_PACKED_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
field_order: ["I", "+2", "-2", "V"]
spin_order: [0, 2, -2, 0]
""".encode()

_YLM_ORACLE = (
    "tests/unit/test_core/test_sci004_scalar_harmonics.py::"
    "test_the_condon_shortley_oracle_closes_against_scipy_in_the_test_body"
)
_PACKED_ORACLE = (
    "tests/unit/test_core/test_sci004_scalar_harmonics.py::"
    "test_the_scalar_packed_layout_is_reconstructed_in_the_test_body"
)


def _case(
    case_id: str,
    requirement_id: str,
    function: str,
    fixture: bytes,
    *,
    excluded_by: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": (
            f"tests/unit/test_core/test_sci004_scalar_harmonics.py::{function}"
        ),
        "expected_failure_kind": "import",
        "expected_failure_pattern": (
            r"ModuleNotFoundError: No module named 'radiosim\.core\.mmode'"
        ),
        "fixture_defect_excluded_by": excluded_by,
        "fixture_bytes": fixture,
    }


SCI004_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m1.harmonics.single-mode",
        "sci004.section-5.3.orthonormal-condon-shortley-ylm",
        "test_single_orthonormal_condon_shortley_modes_match_the_closed_form",
        _MODE_FIXTURE,
        excluded_by=_YLM_ORACLE,
    ),
    _case(
        "m1.harmonics.scalar-reality",
        "sci004.section-5.3.scalar-reality-relation",
        "test_the_scalar_reality_relation_holds_for_a_real_field",
        _MODE_FIXTURE,
        excluded_by=_YLM_ORACLE,
    ),
    _case(
        "m1.harmonics.point-delta",
        "sci004.section-7.1.analytic-point-delta-coefficients",
        "test_an_analytic_point_delta_has_the_closed_form_coefficients",
        _POINT_FIXTURE,
        excluded_by=_YLM_ORACLE,
    ),
    _case(
        "m1.harmonics.constant-map",
        "sci004.section-7.1.healpix-solid-angle-integration",
        "test_a_constant_healpix_map_carries_only_the_monopole",
        _MAP_FIXTURE,
        excluded_by=_YLM_ORACLE,
    ),
    _case(
        "m1.harmonics.ring-nest-equality",
        "sci004.section-7.1.ring-nest-canonical-equality",
        "test_ring_and_nest_inputs_give_identical_coefficients",
        _MAP_FIXTURE,
        excluded_by=_YLM_ORACLE,
    ),
    _case(
        "m1.harmonics.hybrid-additivity",
        "sci004.section-7.1.point-plus-map-additivity",
        "test_point_and_map_coefficients_add_in_the_fixed_component_order",
        _HYBRID_FIXTURE,
        excluded_by=_YLM_ORACLE,
    ),
    _case(
        "m1.harmonics.packed-block-table",
        "sci004.section-5.3.signed-m-major-unpadded-packed-table",
        "test_the_scalar_packed_block_table_is_signed_m_major_and_unpadded",
        _PACKED_FIXTURE,
        excluded_by=_PACKED_ORACLE,
    ),
    _case(
        "m1.harmonics.block-table-digest",
        "sci004.section-14.0.packed-block-table-domain",
        "test_the_block_table_digest_uses_its_exact_domain",
        _PACKED_FIXTURE,
        excluded_by=_PACKED_ORACLE,
    ),
)

SCI004_RED_GREEN_CONTROLS: tuple[str, ...] = (_YLM_ORACLE, _PACKED_ORACLE)


# --- independent oracles, evaluated in the test body --------------------------


def _condon_shortley_ylm(
    degree: int, order: int, colatitude: float, longitude: float
) -> complex:
    """The orthonormal complex Condon-Shortley ``Y_lm``, summed here directly.

    Written out rather than delegated so the closed form the design names is
    visible in the test body; the green control cross-checks it against SciPy.
    """
    absolute = abs(order)
    normalization = math.sqrt(
        (2 * degree + 1)
        / (4.0 * math.pi)
        * math.factorial(degree - absolute)
        / math.factorial(degree + absolute)
    )
    legendre = _associated_legendre(degree, absolute, math.cos(colatitude))
    value = normalization * legendre * cmath.exp(1j * absolute * longitude)
    if order < 0:
        value = (-1.0) ** absolute * value.conjugate()
    return value


def _associated_legendre(degree: int, order: int, argument: float) -> float:
    """``P_l^m`` with the Condon-Shortley phase, by the standard recurrence."""
    if order > degree:
        return 0.0
    current = 1.0
    if order > 0:
        somx2 = math.sqrt(max(0.0, 1.0 - argument * argument))
        factor = 1.0
        for _ in range(order):
            current *= -factor * somx2
            factor += 2.0
    if degree == order:
        return current
    previous = current
    current = argument * (2 * order + 1) * previous
    for level in range(order + 2, degree + 1):
        following = (
            argument * (2 * level - 1) * current - (level + order - 1) * previous
        ) / (level - order)
        previous, current = current, following
    return current


def _scalar_block_rows(lmax: int, mmax: int) -> list[dict[str, int]]:
    """Section 5.3's scalar packed table: signed-m-major, ascending ``l``."""
    rows: list[dict[str, int]] = []
    cursor = 0
    for order in range(-mmax, mmax + 1):
        l_start = abs(order)
        l_stop = lmax + 1
        count = max(0, l_stop - l_start)
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
    return rows


def _canonical_json(value: Any) -> bytes:
    """Section 14's canonical JSON: sorted keys, tight separators, ASCII, no LF."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _domain_digest(domain: str, payload: bytes) -> str:
    """Section 14.0's ``D(d, p) = SHA256(d || NUL || U64(len(p)) || p)``."""
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + struct.pack(">Q", len(payload)) + payload
    ).hexdigest()


# --- green controls -----------------------------------------------------------


def test_the_condon_shortley_oracle_closes_against_scipy_in_the_test_body() -> None:
    """The closed form above is the orthonormal Condon-Shortley convention.

    SciPy's ``sph_harm_y(n, m, theta, phi)`` is an independent implementation of
    exactly that convention, so agreeing with it proves the oracle -- and hence
    every red node that cites it -- is not itself defective.
    """
    from scipy.special import sph_harm_y

    colatitude, longitude = 0.7, 1.9
    for degree, order in ((2, 1), (3, -2), (4, 0), (5, 5), (6, -6)):
        observed = _condon_shortley_ylm(degree, order, colatitude, longitude)
        expected = complex(sph_harm_y(degree, order, colatitude, longitude))
        assert abs(observed - expected) <= ANALYTIC_RESIDUAL_LIMIT, (degree, order)

    # The scalar reality relation the design states, on the oracle itself.
    for degree, order in ((2, 1), (3, 2), (6, 4)):
        positive = _condon_shortley_ylm(degree, order, colatitude, longitude)
        negative = _condon_shortley_ylm(degree, -order, colatitude, longitude)
        assert abs(negative - (-1.0) ** order * positive.conjugate()) <= (
            ANALYTIC_RESIDUAL_LIMIT
        )


def test_the_scalar_packed_layout_is_reconstructed_in_the_test_body() -> None:
    """Section 5.3's packed invariants, proved on the oracle table.

    Contiguity, ascending signed ``m``, ``l_start = abs(m)``, ``l_stop = lmax+1``
    and the complete absence of padding are what make the block table and the
    value buffer inseparable.
    """
    rows = _scalar_block_rows(LMAX, MMAX)

    assert [row["m"] for row in rows] == list(range(-MMAX, MMAX + 1))
    assert len(rows) == 2 * MMAX + 1
    assert rows[0]["value_start"] == 0
    cursor = 0
    for row in rows:
        assert tuple(row) == SCALAR_BLOCK_FIELDS
        assert row["l_start"] == abs(row["m"])
        assert row["l_stop"] == LMAX + 1
        assert row["value_start"] == cursor
        assert row["value_stop"] - row["value_start"] == row["l_stop"] - row["l_start"]
        cursor = row["value_stop"]
    # No padding: the packed count is exactly the number of valid (l, m) cells.
    expected = sum(
        1 for order in range(-MMAX, MMAX + 1) for degree in range(abs(order), LMAX + 1)
    )
    assert cursor == expected

    digest = _domain_digest(BLOCK_TABLE_DOMAIN, _canonical_json(rows))
    assert len(digest) == 64
    # A padded table is a different table, hence a different identity.
    padded = [dict(row, l_start=0) for row in rows]
    assert _domain_digest(BLOCK_TABLE_DOMAIN, _canonical_json(padded)) != digest


# --- Section 5.3 / 7.1 / 12.2 family 3 red oracles ----------------------------


def test_single_orthonormal_condon_shortley_modes_match_the_closed_form() -> None:
    """Section 5.3: individual ``Y_lm`` values, not a library default."""
    from radiosim.core.mmode.harmonics import scalar_ylm

    for degree, order in ((2, 1), (3, -2), (4, 0)):
        observed = complex(scalar_ylm(degree, order, 0.7, 1.9))
        expected = _condon_shortley_ylm(degree, order, 0.7, 1.9)
        assert abs(observed - expected) <= ANALYTIC_RESIDUAL_LIMIT, (degree, order)


def test_the_scalar_reality_relation_holds_for_a_real_field() -> None:
    """Section 5.3: ``a[l,-m] = (-1)**m conj(a[l,m])`` for a real sky."""
    from radiosim.core.mmode.harmonics import (
        scalar_coefficient,
        scalar_transform_reference,
    )

    coefficients = scalar_transform_reference(
        lambda colatitude, longitude: 1.0 + math.cos(colatitude) * math.cos(longitude),
        lmax=LMAX,
        mmax=MMAX,
    )
    for degree in range(1, LMAX + 1):
        for order in range(1, min(degree, MMAX) + 1):
            positive = scalar_coefficient(coefficients, degree, order)
            negative = scalar_coefficient(coefficients, degree, -order)
            residual = abs(negative - (-1.0) ** order * positive.conjugate())
            assert residual <= ANALYTIC_RESIDUAL_LIMIT, (degree, order)


def test_an_analytic_point_delta_has_the_closed_form_coefficients() -> None:
    """Section 7.1: a point sky is analytic, never silently rasterized."""
    from radiosim.core.mmode.harmonics import scalar_coefficient
    from radiosim.core.mmode.sky import point_scalar_coefficients

    flux = 2.5
    colatitude = math.radians(90.0 + 30.0)
    longitude = math.radians(45.0)
    coefficients = point_scalar_coefficients(
        ra_rad=[longitude],
        dec_rad=[-math.radians(30.0)],
        flux=[flux],
        lmax=LMAX,
        mmax=MMAX,
    )
    for degree, order in ((0, 0), (2, 1), (4, -3)):
        expected = (
            flux
            * _condon_shortley_ylm(degree, order, colatitude, longitude).conjugate()
        )
        observed = scalar_coefficient(coefficients, degree, order)
        assert abs(observed - expected) <= ANALYTIC_RESIDUAL_LIMIT, (degree, order)


def test_a_constant_healpix_map_carries_only_the_monopole() -> None:
    """Section 7.1: pixel solid angle integration, checked on the constant map."""
    import numpy as np

    from radiosim.core.mmode.harmonics import scalar_coefficient
    from radiosim.core.mmode.sky import healpix_scalar_coefficients

    npix = 12 * NSIDE * NSIDE
    coefficients = healpix_scalar_coefficients(
        np.ones(npix, dtype=np.float64),
        nside=NSIDE,
        order="ring",
        lmax=LMAX,
        mmax=MMAX,
    )
    monopole = scalar_coefficient(coefficients, 0, 0)

    assert abs(monopole - math.sqrt(4.0 * math.pi)) <= ANALYTIC_RESIDUAL_LIMIT
    for degree in range(1, LMAX + 1):
        for order in range(-min(degree, MMAX), min(degree, MMAX) + 1):
            assert abs(scalar_coefficient(coefficients, degree, order)) <= (
                ANALYTIC_RESIDUAL_LIMIT
            )


def test_ring_and_nest_inputs_give_identical_coefficients() -> None:
    """Section 7.1: RING and NEST must agree after canonical ordering."""
    import healpy as hp
    import numpy as np

    from radiosim.core.mmode.sky import healpix_scalar_coefficients

    npix = 12 * NSIDE * NSIDE
    rng = np.random.default_rng(20260821)
    ring_map = rng.normal(size=npix)
    nest_map = ring_map[hp.nest2ring(NSIDE, np.arange(npix))]

    ring = healpix_scalar_coefficients(
        ring_map, nside=NSIDE, order="ring", lmax=LMAX, mmax=MMAX
    )
    nest = healpix_scalar_coefficients(
        nest_map, nside=NSIDE, order="nest", lmax=LMAX, mmax=MMAX
    )

    assert float(np.max(np.abs(np.asarray(ring) - np.asarray(nest)))) <= (
        ANALYTIC_RESIDUAL_LIMIT
    )


def test_point_and_map_coefficients_add_in_the_fixed_component_order() -> None:
    """Section 7.1: one summed sky field, not two independent m-mode solvers."""
    import numpy as np

    from radiosim.core.mmode.sky import (
        healpix_scalar_coefficients,
        hybrid_scalar_coefficients,
        point_scalar_coefficients,
    )

    npix = 12 * NSIDE * NSIDE
    point = point_scalar_coefficients(
        ra_rad=[math.radians(45.0)],
        dec_rad=[-math.radians(30.0)],
        flux=[2.5],
        lmax=LMAX,
        mmax=MMAX,
    )
    healpix = healpix_scalar_coefficients(
        np.ones(npix, dtype=np.float64),
        nside=NSIDE,
        order="ring",
        lmax=LMAX,
        mmax=MMAX,
    )
    hybrid = hybrid_scalar_coefficients(
        point=point, healpix=healpix, component_order=("point", "healpix")
    )

    total = np.asarray(point) + np.asarray(healpix)
    assert float(np.max(np.abs(np.asarray(hybrid) - total))) == 0.0


def test_the_scalar_packed_block_table_is_signed_m_major_and_unpadded() -> None:
    """Section 5.3: the M1 scalar packed table, exactly as the design prints it."""
    from radiosim.core.mmode.harmonics import scalar_packed_block_table

    table = scalar_packed_block_table(lmax=LMAX, mmax=MMAX)
    oracle = _scalar_block_rows(LMAX, MMAX)

    assert [dict(row) for row in table.block_rows] == oracle
    assert table.field_order == FIELD_ORDER
    assert table.spin_order == SPIN_ORDER
    assert table.packed_value_count == oracle[-1]["value_stop"]
    assert table.invalid_cell_count == 0


def test_the_block_table_digest_uses_its_exact_domain() -> None:
    """Section 14.0: ``D("radiosim.mmode-packed-block-table.v1", J(block_rows))``."""
    from radiosim.core.mmode.harmonics import scalar_packed_block_table

    table = scalar_packed_block_table(lmax=LMAX, mmax=MMAX)
    expected = _domain_digest(
        BLOCK_TABLE_DOMAIN, _canonical_json(_scalar_block_rows(LMAX, MMAX))
    )

    assert table.block_table_sha256 == expected
    assert table.block_table_domain == BLOCK_TABLE_DOMAIN
