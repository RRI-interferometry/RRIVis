r"""SCI-004 phase-M2 red oracles for the full-Stokes harmonic sky.

``docs/development/sci004_mmode_design.md`` Section 7.1 rules how a resolved
``SkyModel`` becomes harmonic coefficients, and phase M2 is where that rule
finally has to hold for all four Section 5.3 science fields
``("I", "+2", "-2", "V")`` rather than for ``I`` alone.

**Point components are not silently rasterized.** A delta-function point sky uses
*analytic* scalar and spin harmonics evaluated at the exact transported source
direction.  For a source of Shaw-basis brightness ``(I, Q_H, U_H, V)`` at
``n_s`` the coefficients are therefore exactly

``a^I_lm      = I * conj(Y_lm(n_s))``
``a^(+2)_lm   = (Q_H + i U_H) * conj(_{+2}Y_lm(n_s))``
``a^(-2)_lm   = (Q_H - i U_H) * conj(_{-2}Y_lm(n_s))``
``a^V_lm      = V * conj(Y_lm(n_s))``

which follows from Section 5.3's expansions ``Q_H + i U_H = sum a^(+2) _{+2}Y``
and ``Q_H - i U_H = sum a^(-2) _{-2}Y`` together with orthonormality.  ``I`` and
``V`` are scalar (spin-0) expansions; only the linear pair carries spin.

**HEALPix maps enter as the pixel measure.**  Section 7.1 (as corrected) fixes

``a_lm = sum_pix( s_pix * Omega_pix * conj(sY_lm(n_pix)) )``

over canonical-RING pixel centres with ``Omega_pix = 4*pi/npix`` -- the same
measure the private direct oracle sums -- so a harmonic-versus-direct comparison
tests truncation and nothing else.  A continuous band-limited reinterpretation, a
ring-weighted quadrature, or any iterated transform is a *different sky object*
and is rejected.  RING and NEST inputs must yield identical coefficients after
canonical ordering.

**Hybrid models add, in a fixed order.**  ``("point", "healpix")``, before any
``B_lm a_lm`` product; two independent m-mode solves whose rounded outputs are
added is not the same object.

**Polarized payloads carry their frame.**  Section 5.1: every point or HEALPix
payload with non-zero ``Q`` or ``U`` must carry the canonical
``TangentPolarizationFrame``; a programmatic polarized input without a declared
source convention is rejected.

Section 12.2 families 3 and 4 are the required oracle set and the analytic
complex128 residual limit is ``5e-12``.  Every expected value below is built in
the test body from those displayed sums and from the published spin-weight
closed forms, never read back from the module under test.  The Section 13.4
owner is ``radiosim.core.mmode.sky``, whose polarized entry points do not exist
at ``A1``; imports are function-local so each node yields its own Section 14.1
outcome.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np

HARMONIC_CONVENTION = "radiosim.shaw-polarized-harmonics.v1"
TANGENT_FRAME_SCHEMA = "radiosim.sky-tangent-polarization.v1"

#: Section 5.3's science field order and its spin labels.
FIELD_ORDER: tuple[str, ...] = ("I", "+2", "-2", "V")
SPIN_ORDER: tuple[int, ...] = (0, 2, -2, 0)

#: Section 12.2's analytic complex128 residual limit.
ANALYTIC_RESIDUAL_LIMIT = 5e-12

#: Section 12.2's non-vacuity margin.
NON_VACUITY_FACTOR = 10.0

#: Section 5.1's canonical six-key tangent frame, restated here so this module
#: declares its own source convention rather than importing one from the surface
#: it is testing.  ``tangent_frame=None`` is the *undeclared* payload Section 5.1
#: rejects, which is why every polarized call below passes this explicitly.
CANONICAL_TANGENT_FRAME: dict[str, str] = {
    "schema_version": TANGENT_FRAME_SCHEMA,
    "coordinate_frame": "icrs",
    "axes": "north_east",
    "position_angle": "north_through_east",
    "linear_complex": "q_plus_i_u",
    "stokes_v": "iau_incoming_r_minus_l",
}

LMAX = 4
MMAX = 3
NSIDE = 4

#: The fixture source: a genuinely polarized point at a well-determined
#: direction, with all four Stokes parameters non-zero so no field can be
#: silently dropped.
SOURCE_RA_DEG = 45.0
SOURCE_DEC_DEG = -30.0
SOURCE_STOKES: tuple[float, float, float, float] = (2.5, 0.4, -0.3, 0.2)

_POINT_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
tangent_polarization_frame:
  schema_version: {TANGENT_FRAME_SCHEMA}
  coordinate_frame: icrs
  axes: north_east
  position_angle: north_through_east
  linear_complex: q_plus_i_u
  stokes_v: iau_incoming_r_minus_l
point_sources:
  - ra_deg: {SOURCE_RA_DEG}
    dec_deg: {SOURCE_DEC_DEG}
    stokes_i_jy: {SOURCE_STOKES[0]}
    stokes_q_jy: {SOURCE_STOKES[1]}
    stokes_u_jy: {SOURCE_STOKES[2]}
    stokes_v_jy: {SOURCE_STOKES[3]}
""".encode()

_MAP_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
nside: {NSIDE}
order: ring
tangent_polarization_frame:
  schema_version: {TANGENT_FRAME_SCHEMA}
  coordinate_frame: galactic
  axes: north_east
  position_angle: north_through_east
  linear_complex: q_plus_i_u
  stokes_v: iau_incoming_r_minus_l
""".encode()

_ORDERING_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
nside: {NSIDE}
orders: ["ring", "nest"]
""".encode()

_HYBRID_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
nside: {NSIDE}
component_order: ["point", "healpix"]
point_sources:
  - ra_deg: {SOURCE_RA_DEG}
    dec_deg: {SOURCE_DEC_DEG}
    stokes_i_jy: {SOURCE_STOKES[0]}
    stokes_q_jy: {SOURCE_STOKES[1]}
    stokes_u_jy: {SOURCE_STOKES[2]}
    stokes_v_jy: {SOURCE_STOKES[3]}
""".encode()

_UNDECLARED_FRAME_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
tangent_polarization_frame: null
point_sources:
  - ra_deg: {SOURCE_RA_DEG}
    dec_deg: {SOURCE_DEC_DEG}
    stokes_i_jy: {SOURCE_STOKES[0]}
    stokes_q_jy: {SOURCE_STOKES[1]}
    stokes_u_jy: {SOURCE_STOKES[2]}
    stokes_v_jy: 0.0
""".encode()

_SCALAR_ORACLE = (
    "tests/unit/test_core/test_sci004_sky_harmonics.py::"
    "test_the_scalar_point_and_pixel_measure_constructions_hold_today"
)
_RING_ORACLE = (
    "tests/unit/test_core/test_sci004_sky_harmonics.py::"
    "test_the_canonical_ring_geometry_and_pixel_solid_angle_hold_today"
)

_SKY_IMPORT_PATTERN = (
    r"ImportError: cannot import name '\w+' from 'radiosim\.core\.mmode\.sky'"
)


def _local(function: str) -> str:
    return f"tests/unit/test_core/test_sci004_sky_harmonics.py::{function}"


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
        "test_nodeid": _local(function),
        "expected_failure_kind": "missing-symbol",
        "expected_failure_pattern": _SKY_IMPORT_PATTERN,
        "fixture_defect_excluded_by": excluded_by,
        "fixture_bytes": fixture,
    }


SCI004_PHASE2_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m2.sky.point-full-stokes",
        "sci004.section-7.1.analytic-full-stokes-point-delta",
        "test_a_full_stokes_point_delta_has_the_closed_form_coefficients",
        _POINT_FIXTURE,
        excluded_by=_SCALAR_ORACLE,
    ),
    _case(
        "m2.sky.point-fields-are-independent",
        "sci004.section-5.3.four-fields-share-one-packed-table",
        "test_the_four_science_fields_share_one_table_and_stay_independent",
        _POINT_FIXTURE,
        excluded_by=_SCALAR_ORACLE,
    ),
    _case(
        "m2.sky.healpix-pixel-measure",
        "sci004.section-7.1.polarized-healpix-pixel-measure",
        "test_a_polarized_healpix_map_uses_the_exact_pixel_measure",
        _MAP_FIXTURE,
        excluded_by=_RING_ORACLE,
    ),
    _case(
        "m2.sky.healpix-v-is-scalar",
        "sci004.section-5.3.stokes-v-is-a-scalar-expansion",
        "test_the_stokes_v_map_is_expanded_with_the_scalar_harmonics",
        _MAP_FIXTURE,
        excluded_by=_RING_ORACLE,
    ),
    _case(
        "m2.sky.ring-nest-equality",
        "sci004.section-7.1.ring-and-nest-give-identical-coefficients",
        "test_ring_and_nest_polarized_payloads_give_identical_coefficients",
        _ORDERING_FIXTURE,
        excluded_by=_RING_ORACLE,
    ),
    _case(
        "m2.sky.hybrid-additivity",
        "sci004.section-7.1.hybrid-adds-in-the-fixed-component-order",
        "test_a_hybrid_model_adds_point_and_map_coefficients_field_by_field",
        _HYBRID_FIXTURE,
        excluded_by=_SCALAR_ORACLE,
    ),
    _case(
        "m2.sky.undeclared-tangent-frame",
        "sci004.section-5.1.polarized-sky-requires-a-declared-frame",
        "test_a_polarized_point_payload_without_a_frame_is_rejected",
        _UNDECLARED_FRAME_FIXTURE,
        excluded_by=_SCALAR_ORACLE,
    ),
)

SCI004_PHASE2_RED_GREEN_CONTROLS: tuple[str, ...] = (_SCALAR_ORACLE, _RING_ORACLE)


# --- closed forms derived here --------------------------------------------------


def _spin_two_closed_form(
    spin: int, degree: int, order: int, colatitude: float, longitude: float
) -> complex:
    """Return ``_{s}Y_{2m}`` for ``s = +-2`` from the published closed forms.

    The spin-weight ``+2``, ``l = 2`` table is Goldberg et al. (1967); the
    negative-spin values come from the standard relation
    ``_{-s}Y_lm = (-1)**(s+m) conj(_{s}Y_{l,-m})``.  Only ``l = 2`` is needed:
    the point oracle compares a single degree, and using one degree keeps the
    expected value a closed form rather than a second implementation.
    """
    if degree != 2:
        raise ValueError("only the degree-two closed forms are written out here")
    if spin == -2:
        return ((-1.0) ** (2 + order)) * np.conjugate(
            _spin_two_closed_form(2, degree, -order, colatitude, longitude)
        )
    if spin != 2:
        raise ValueError("spin must be +2 or -2")
    cosine = math.cos(colatitude)
    sine = math.sin(colatitude)
    phase = cmath.exp(1j * order * longitude)
    if order == 2:
        return math.sqrt(5.0 / (64.0 * math.pi)) * (1.0 + cosine) ** 2 * phase
    if order == 1:
        return math.sqrt(5.0 / (16.0 * math.pi)) * sine * (1.0 + cosine) * phase
    if order == 0:
        return complex(math.sqrt(15.0 / (32.0 * math.pi)) * sine * sine, 0.0)
    if order == -1:
        return math.sqrt(5.0 / (16.0 * math.pi)) * sine * (1.0 - cosine) * phase
    if order == -2:
        return math.sqrt(5.0 / (64.0 * math.pi)) * (1.0 - cosine) ** 2 * phase
    raise ValueError(f"m={order} is not a degree-two order")


def _source_direction() -> tuple[float, float]:
    """Return the fixture source's ``(colatitude, longitude)`` in radians."""
    declination = math.radians(SOURCE_DEC_DEG)
    return (0.5 * math.pi - declination, math.radians(SOURCE_RA_DEG))


def _shaw_linear(q: float, u: float) -> tuple[complex, complex]:
    """Return ``(Q_H + i U_H, Q_H - i U_H)`` under Section 5.2's ``U_H = -U``."""
    linear = complex(q, -u)
    return (linear, linear.conjugate())


def _polarized_pixel_maps(nside: int) -> dict[str, np.ndarray]:
    """A smooth, deterministic, genuinely polarized full-sky payload."""
    from radiosim.core.mmode.sky import ring_directions

    theta, phi = ring_directions(nside)
    return {
        "I": 1.0 + 0.25 * np.cos(theta),
        "Q": 0.3 * np.sin(theta) ** 2 * np.cos(2.0 * phi),
        "U": 0.2 * np.sin(theta) ** 2 * np.sin(2.0 * phi),
        "V": 0.1 * np.cos(theta),
    }


# --- green controls -------------------------------------------------------------


def test_the_scalar_point_and_pixel_measure_constructions_hold_today() -> None:
    """M1's scalar sky constructions are sound at ``A1``.

    ``a^I_lm = I conj(Y_lm(n_s))`` for a delta source is the identity the
    polarized oracles below extend field by field; it already holds, so a red
    failure there is the absence of the polarized entry points rather than a
    defective direction convention or a wrong colatitude.
    """
    from radiosim.core.mmode.harmonics import scalar_coefficient, scalar_ylm
    from radiosim.core.mmode.sky import point_scalar_coefficients

    colatitude, longitude = _source_direction()
    coefficients = point_scalar_coefficients(
        ra_rad=[longitude],
        dec_rad=[math.radians(SOURCE_DEC_DEG)],
        flux=[SOURCE_STOKES[0]],
        lmax=LMAX,
        mmax=MMAX,
    )
    for degree in (0, 2, 3):
        for order in range(-min(degree, MMAX), min(degree, MMAX) + 1):
            expected = SOURCE_STOKES[0] * np.conjugate(
                scalar_ylm(degree, order, colatitude, longitude)
            )
            observed = scalar_coefficient(coefficients, degree, order)
            assert abs(observed - expected) <= ANALYTIC_RESIDUAL_LIMIT, (degree, order)


def test_the_canonical_ring_geometry_and_pixel_solid_angle_hold_today() -> None:
    """The RING geometry and ``Omega_pix`` the polarized oracles reuse are sound."""
    from radiosim.core.mmode.harmonics import scalar_coefficient
    from radiosim.core.mmode.sky import healpix_scalar_coefficients, ring_directions

    npix = 12 * NSIDE * NSIDE
    theta, phi = ring_directions(NSIDE)
    assert theta.shape == (npix,)
    assert phi.shape == (npix,)

    solid_angle = 4.0 * math.pi / npix
    constant = np.ones(npix, dtype=np.float64)
    coefficients = healpix_scalar_coefficients(
        constant, nside=NSIDE, order="ring", lmax=LMAX, mmax=MMAX
    )
    # The pixel measure sums to the whole sphere, so the monopole is exactly
    # ``4 pi / sqrt(4 pi) = sqrt(4 pi)``.
    monopole = scalar_coefficient(coefficients, 0, 0)
    assert abs(monopole - math.sqrt(4.0 * math.pi)) <= ANALYTIC_RESIDUAL_LIMIT
    assert abs(float(np.sum(np.full(npix, solid_angle))) - 4.0 * math.pi) <= (
        ANALYTIC_RESIDUAL_LIMIT
    )


# --- Section 7.1 polarized point red oracles ------------------------------------


def test_a_full_stokes_point_delta_has_the_closed_form_coefficients() -> None:
    """Section 7.1: analytic scalar *and* spin harmonics at the exact direction."""
    from radiosim.core.mmode.harmonics import scalar_ylm
    from radiosim.core.mmode.sky import point_polarized_coefficients

    intensity, q, u, v = SOURCE_STOKES
    colatitude, longitude = _source_direction()
    plus, minus = _shaw_linear(q, u)

    coefficients = point_polarized_coefficients(
        ra_rad=[longitude],
        dec_rad=[math.radians(SOURCE_DEC_DEG)],
        stokes=[[intensity, q, u, v]],
        lmax=LMAX,
        mmax=MMAX,
        tangent_frame=CANONICAL_TANGENT_FRAME,
    )

    degree = 2
    for order in range(-min(degree, MMAX), min(degree, MMAX) + 1):
        scalar = np.conjugate(scalar_ylm(degree, order, colatitude, longitude))
        expected = {
            "I": intensity * scalar,
            "V": v * scalar,
            "+2": plus
            * np.conjugate(
                _spin_two_closed_form(2, degree, order, colatitude, longitude)
            ),
            "-2": minus
            * np.conjugate(
                _spin_two_closed_form(-2, degree, order, colatitude, longitude)
            ),
        }
        for field in FIELD_ORDER:
            observed = coefficients.coefficient(field, degree, order)
            assert abs(observed - expected[field]) <= ANALYTIC_RESIDUAL_LIMIT, (
                field,
                order,
            )

    # Non-vacuity: an omitted Section 5.2 ``U`` flip swaps the two linear fields
    # and must miss by far more than the analytic residual.
    wrong_plus = complex(q, u)
    assert abs(wrong_plus - plus) > NON_VACUITY_FACTOR * ANALYTIC_RESIDUAL_LIMIT


def test_the_four_science_fields_share_one_table_and_stay_independent() -> None:
    """Section 5.3: one packed table, four fields, in the exact fixed order."""
    from radiosim.core.mmode.sky import point_polarized_coefficients

    intensity, q, u, v = SOURCE_STOKES
    longitude = math.radians(SOURCE_RA_DEG)

    full = point_polarized_coefficients(
        ra_rad=[longitude],
        dec_rad=[math.radians(SOURCE_DEC_DEG)],
        stokes=[[intensity, q, u, v]],
        lmax=LMAX,
        mmax=MMAX,
        tangent_frame=CANONICAL_TANGENT_FRAME,
    )
    unpolarized = point_polarized_coefficients(
        ra_rad=[longitude],
        dec_rad=[math.radians(SOURCE_DEC_DEG)],
        stokes=[[intensity, 0.0, 0.0, 0.0]],
        lmax=LMAX,
        mmax=MMAX,
        tangent_frame=CANONICAL_TANGENT_FRAME,
    )

    assert full.table.field_order == FIELD_ORDER
    assert full.table.spin_order == SPIN_ORDER
    assert full.table.block_table_sha256 == unpolarized.table.block_table_sha256

    degree, order = 2, 1
    # Removing Q, U and V must leave ``I`` untouched and zero the other three.
    assert (
        abs(
            full.coefficient("I", degree, order)
            - unpolarized.coefficient("I", degree, order)
        )
        <= ANALYTIC_RESIDUAL_LIMIT
    )
    for field in ("+2", "-2", "V"):
        assert abs(unpolarized.coefficient(field, degree, order)) <= (
            ANALYTIC_RESIDUAL_LIMIT
        )
        assert abs(full.coefficient(field, degree, order)) > (
            NON_VACUITY_FACTOR * ANALYTIC_RESIDUAL_LIMIT
        ), field


# --- Section 7.1 polarized HEALPix red oracles ----------------------------------


def test_a_polarized_healpix_map_uses_the_exact_pixel_measure() -> None:
    """Section 7.1: ``a_lm = sum_pix s_pix Omega_pix conj(sY_lm(n_pix))``."""
    from radiosim.core.mmode.sky import healpix_polarized_coefficients, ring_directions

    maps = _polarized_pixel_maps(NSIDE)
    theta, phi = ring_directions(NSIDE)
    npix = 12 * NSIDE * NSIDE
    solid_angle = 4.0 * math.pi / npix

    coefficients = healpix_polarized_coefficients(
        maps,
        nside=NSIDE,
        order="ring",
        lmax=LMAX,
        mmax=MMAX,
        tangent_frame=CANONICAL_TANGENT_FRAME,
    )

    linear = maps["Q"].astype(np.complex128) - 1j * maps["U"].astype(np.complex128)
    degree = 2
    for order in range(-min(degree, MMAX), min(degree, MMAX) + 1):
        plus_basis = np.asarray(
            [
                _spin_two_closed_form(2, degree, order, float(t), float(p))
                for t, p in zip(theta, phi, strict=True)
            ],
            dtype=np.complex128,
        )
        expected_plus = complex(np.sum(linear * solid_angle * np.conjugate(plus_basis)))
        assert abs(coefficients.coefficient("+2", degree, order) - expected_plus) <= (
            ANALYTIC_RESIDUAL_LIMIT
        ), order

        minus_basis = np.asarray(
            [
                _spin_two_closed_form(-2, degree, order, float(t), float(p))
                for t, p in zip(theta, phi, strict=True)
            ],
            dtype=np.complex128,
        )
        expected_minus = complex(
            np.sum(np.conjugate(linear) * solid_angle * np.conjugate(minus_basis))
        )
        assert abs(coefficients.coefficient("-2", degree, order) - expected_minus) <= (
            ANALYTIC_RESIDUAL_LIMIT
        ), order


def test_the_stokes_v_map_is_expanded_with_the_scalar_harmonics() -> None:
    """Section 5.3: ``I`` and ``V`` are spin-0 expansions, not spin-2 ones."""
    from radiosim.core.mmode.harmonics import scalar_coefficient
    from radiosim.core.mmode.sky import (
        healpix_polarized_coefficients,
        healpix_scalar_coefficients,
    )

    maps = _polarized_pixel_maps(NSIDE)
    polarized = healpix_polarized_coefficients(
        maps,
        nside=NSIDE,
        order="ring",
        lmax=LMAX,
        mmax=MMAX,
        tangent_frame=CANONICAL_TANGENT_FRAME,
    )
    scalar_v = healpix_scalar_coefficients(
        maps["V"], nside=NSIDE, order="ring", lmax=LMAX, mmax=MMAX
    )
    scalar_i = healpix_scalar_coefficients(
        maps["I"], nside=NSIDE, order="ring", lmax=LMAX, mmax=MMAX
    )

    for degree in range(LMAX + 1):
        for order in range(-min(degree, MMAX), min(degree, MMAX) + 1):
            assert (
                abs(
                    polarized.coefficient("V", degree, order)
                    - scalar_coefficient(scalar_v, degree, order)
                )
                <= ANALYTIC_RESIDUAL_LIMIT
            ), (degree, order)
            assert (
                abs(
                    polarized.coefficient("I", degree, order)
                    - scalar_coefficient(scalar_i, degree, order)
                )
                <= ANALYTIC_RESIDUAL_LIMIT
            ), (degree, order)


def test_ring_and_nest_polarized_payloads_give_identical_coefficients() -> None:
    """Section 7.1: identical, not merely equal, after canonical ordering."""
    from radiosim.core.mmode.sky import healpix_polarized_coefficients
    from radiosim.core.sky.support.healpy import lazy_healpy

    npix = 12 * NSIDE * NSIDE
    ring_maps = _polarized_pixel_maps(NSIDE)
    permutation = lazy_healpy.nest2ring(NSIDE, np.arange(npix))
    nest_maps = {name: values[permutation] for name, values in ring_maps.items()}

    from_ring = healpix_polarized_coefficients(
        ring_maps,
        nside=NSIDE,
        order="ring",
        lmax=LMAX,
        mmax=MMAX,
        tangent_frame=CANONICAL_TANGENT_FRAME,
    )
    from_nest = healpix_polarized_coefficients(
        nest_maps,
        nside=NSIDE,
        order="nest",
        lmax=LMAX,
        mmax=MMAX,
        tangent_frame=CANONICAL_TANGENT_FRAME,
    )

    assert from_ring.table.block_table_sha256 == from_nest.table.block_table_sha256
    assert np.array_equal(np.asarray(from_ring.values), np.asarray(from_nest.values)), (
        "RING and NEST payloads must produce bit-identical packed buffers"
    )


def test_a_hybrid_model_adds_point_and_map_coefficients_field_by_field() -> None:
    """Section 7.1: one summed sky in the fixed ``("point", "healpix")`` order."""
    from radiosim.core.mmode.sky import (
        healpix_polarized_coefficients,
        hybrid_polarized_coefficients,
        point_polarized_coefficients,
    )

    intensity, q, u, v = SOURCE_STOKES
    point = point_polarized_coefficients(
        ra_rad=[math.radians(SOURCE_RA_DEG)],
        dec_rad=[math.radians(SOURCE_DEC_DEG)],
        stokes=[[intensity, q, u, v]],
        lmax=LMAX,
        mmax=MMAX,
        tangent_frame=CANONICAL_TANGENT_FRAME,
    )
    healpix = healpix_polarized_coefficients(
        _polarized_pixel_maps(NSIDE),
        nside=NSIDE,
        order="ring",
        lmax=LMAX,
        mmax=MMAX,
        tangent_frame=CANONICAL_TANGENT_FRAME,
    )
    hybrid = hybrid_polarized_coefficients(point=point, healpix=healpix)

    assert np.allclose(
        np.asarray(hybrid.values),
        np.asarray(point.values) + np.asarray(healpix.values),
        atol=ANALYTIC_RESIDUAL_LIMIT,
    )
    assert hybrid.component_order == ("point", "healpix")

    # Section 7.1 fixes the order; a reordered request is a different object.
    raised = None
    try:
        hybrid_polarized_coefficients(
            point=point, healpix=healpix, component_order=("healpix", "point")
        )
    except ValueError as error:  # pragma: no cover - the red path
        raised = error
    assert raised is not None, "the hybrid component order is fixed"


def test_a_polarized_point_payload_without_a_frame_is_rejected() -> None:
    """Section 5.1: a declared source convention is mandatory when Q or U is set."""
    from radiosim.core.mmode.sky import point_polarized_coefficients

    intensity, q, u, _ = SOURCE_STOKES
    raised = None
    try:
        point_polarized_coefficients(
            ra_rad=[math.radians(SOURCE_RA_DEG)],
            dec_rad=[math.radians(SOURCE_DEC_DEG)],
            stokes=[[intensity, q, u, 0.0]],
            lmax=LMAX,
            mmax=MMAX,
            tangent_frame=None,
        )
    except ValueError as error:  # pragma: no cover - the red path
        raised = error
    assert raised is not None, (
        "a polarized point payload without a declared tangent frame is rejected"
    )
