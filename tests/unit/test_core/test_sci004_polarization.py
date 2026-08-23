r"""SCI-004 phase-M2 red oracles for the polarized harmonic surface.

``docs/development/sci004_mmode_design.md`` Section 5 fixes three things phase M1
deliberately did not build, and this module binds all three.

**Section 5.1 -- canonical sky metadata.** A strict frozen
``TangentPolarizationFrame`` carries exactly ``schema_version``,
``coordinate_frame``, ``axes = "north_east"``,
``position_angle = "north_through_east"``, ``linear_complex = "q_plus_i_u"`` and
``stokes_v = "iau_incoming_r_minus_l"``. Every point or HEALPix payload with
non-zero ``Q`` or ``U`` must carry it; an ``I``/``V``-only payload may omit it; a
programmatic polarized input without a declared source convention is rejected.
A HEALPix/CMB ``U`` convention is converted explicitly to RadioSim IAU
North-through-East before canonical storage, and Section 5.1 requires the sign to
be pinned "with a rotated pure-Q map" -- which is what
:func:`test_a_healpix_cmb_payload_is_converted_to_iau_north_through_east` does.

**Section 5.2 -- the RadioSim-to-Shaw bridge.** RadioSim's sky electric vector is
ordered ``(North, East)``; Shaw's is the spherical ``(theta, phi)`` basis with
``theta`` pointing South and ``phi`` East. The exact bridge is

.. math::

    D=\operatorname{diag}(-1,1),\qquad e_{\theta\phi}=De_{NE},\qquad
    J_{\theta\phi}=J_{NE}D,

giving ``I_H = I``, ``Q_H = Q``, ``U_H = -U`` and ``V_H = V``. In one unchanged
ordered basis RadioSim's ``P^V_RS`` has the *opposite matrix sign* from Shaw's
``P^V``; after the bridge the physical IAU ``V`` field has the same sign, and no
additional fitted or configurable V flip is allowed. The accepted SCI-006 east-X
permutation stays **inside** ``J_NE``: it is not replaced by ``D``, and the two
matrices are not even similar -- ``D`` is diagonal, the permutation is
antidiagonal.

**Section 5.3 -- spin-weighted harmonics.** The literal
``radiosim.shaw-polarized-harmonics.v1`` fixes orthonormal complex
Condon-Shortley harmonics, scalar expansions for ``I`` and ``V``, Shaw's
spin-labelled expansions

.. math::

    Q_H+iU_H=\sum a^{(+2)}_{lm}\,{}_{+2}Y_{lm},\qquad
    Q_H-iU_H=\sum a^{(-2)}_{lm}\,{}_{-2}Y_{lm},

and the explicit paired spin-reality relation
``a^(-2)[l,m] = (-1)**m * conj(a^(+2)[l,-m])``. The packed table is
signed-``m``-major with the science field order ``("I", "+2", "-2", "V")`` and
spin order ``(0, +2, -2, 0)``, and ``l_start = max(abs(m), abs(spin))`` -- which
is what distinguishes a polarized row from the scalar row M1 already ships.

Section 12.2 family 4 is the required oracle set: individual spin ``+2/-2``
modes, the spin reality relation, HEALPix/CMB-to-IAU ``U`` conversion, pure
``Q``/``U``/``V``, and the exact ``D``/SCI-006 east-X/circular signs. The analytic
complex128 residual limit is ``5e-12``.

Every expected value below is derived in the test body from a published closed
form or from the design's own algebra, never from the module under test. The
Section 13.4 owners are ``radiosim.core.mmode.harmonics``,
``radiosim.core.polarization`` and ``radiosim.core.sky.containers``; none of the
phase-M2 symbols exists at ``A1``, so imports are function-local and each node
yields its own Section 14.1 outcome.

This module also declares the phase-M2 red case for the authoritative Tier 7
capability node, because Section 9 makes that characterization file the record of
the polarized capability flip and the characterization files themselves carry no
red-record machinery -- exactly as ``test_sci004_strategy.py`` did for M1.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np

#: Section 5.3's fixed harmonic literal.
HARMONIC_CONVENTION = "radiosim.shaw-polarized-harmonics.v1"

#: Section 5.1's exact tangent-frame literal and its six-key surface.
TANGENT_FRAME_SCHEMA = "radiosim.sky-tangent-polarization.v1"
TANGENT_FRAME_KEYS: tuple[str, ...] = (
    "schema_version",
    "coordinate_frame",
    "axes",
    "position_angle",
    "linear_complex",
    "stokes_v",
)

#: Section 5.2's Stokes-V basis-bridge literal.
STOKES_V_BASIS_BRIDGE = "radiosim.stokes-ne-theta-phi.v1"

#: Section 5.3's science field order and its spin labels.
FIELD_ORDER: tuple[str, ...] = ("I", "+2", "-2", "V")
SPIN_ORDER: tuple[int, ...] = (0, 2, -2, 0)

#: Section 12.2's analytic complex128 residual limit.
ANALYTIC_RESIDUAL_LIMIT = 5e-12

#: Section 12.2's non-vacuity margin: a wrong sign or omitted permutation must
#: miss by more than ten times its corresponding passing residual.
NON_VACUITY_FACTOR = 10.0

LMAX = 4
MMAX = 3

_SPIN_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
spins: [2, -2]
modes:
  - [2, 0]
  - [2, 1]
  - [2, 2]
  - [2, -2]
colatitude_rad: 0.9
longitude_rad: 2.3
""".encode()

_REALITY_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
field_order: ["I", "+2", "-2", "V"]
spin_order: [0, 2, -2, 0]
real_linear_field: true
""".encode()

_BRIDGE_FIXTURE = f"""\
stokes_v_basis_bridge: {STOKES_V_BASIS_BRIDGE}
stokes:
  I: 1.0
  Q: 0.2
  U: 0.3
  V: 0.4
bridge_matrix: [[-1.0, 0.0], [0.0, 1.0]]
""".encode()

_EAST_X_FIXTURE = f"""\
stokes_v_basis_bridge: {STOKES_V_BASIS_BRIDGE}
receptor_basis: linear_xy
feed_rotation_deg: 0.0
stokes:
  I: 1.0
  Q: 0.2
  U: 0.3
  V: 0.4
""".encode()

_CIRCULAR_FIXTURE = f"""\
stokes_v_basis_bridge: {STOKES_V_BASIS_BRIDGE}
receptor_basis: circular_rl
stokes:
  I: 1.0
  Q: 0.0
  U: 0.0
  V: 0.4
""".encode()

_CMB_FIXTURE = f"""\
tangent_polarization_frame:
  schema_version: {TANGENT_FRAME_SCHEMA}
  coordinate_frame: galactic
  axes: north_east
  position_angle: north_through_west
  linear_complex: q_plus_i_u
  stokes_v: iau_incoming_r_minus_l
position_angle_deg: 45.0
polarized_intensity_jy: 2.0
""".encode()

_MISSING_FRAME_FIXTURE = b"""\
stokes:
  I: 1.0
  Q: 0.5
  U: -0.25
  V: 0.0
tangent_polarization_frame: null
"""

_INTENSITY_ONLY_FIXTURE = b"""\
stokes:
  I: 1.0
  Q: 0.0
  U: 0.0
  V: 0.3
tangent_polarization_frame: null
"""

_PACKED_FIXTURE = f"""\
harmonic_convention: {HARMONIC_CONVENTION}
lmax: {LMAX}
mmax: {MMAX}
field_order: ["I", "+2", "-2", "V"]
spin_order: [0, 2, -2, 0]
""".encode()

_CAPABILITY_FIXTURE = b"""\
capability_cases:
  - {case_kind: property, simulator: mmode, property: supports_polarization, expected: true}
  - {case_kind: property, simulator: rime, property: supports_polarization, expected: true}
"""

#: Section 14.2 names this exact Tier 7 node for the M2 capability flip.
TIER7_PROPERTY_NODEID = (
    "tests/characterization/test_tier7_current_behavior.py::"
    "test_mmode_m1_capability_truth"
)

_COHERENCY_ORACLE = (
    "tests/unit/test_core/test_sci004_polarization.py::"
    "test_the_iau_coherency_and_east_x_permutation_hold_today"
)
_SCALAR_ORACLE = (
    "tests/unit/test_core/test_sci004_polarization.py::"
    "test_the_scalar_condon_shortley_harmonics_close_in_the_test_body"
)

_HARMONICS_IMPORT_PATTERN = (
    r"ImportError: cannot import name '\w+' from "
    r"'radiosim\.core\.mmode\.harmonics'"
)
_POLARIZATION_IMPORT_PATTERN = (
    r"ImportError: cannot import name '\w+' from 'radiosim\.core\.polarization'"
)
_CONTAINER_IMPORT_PATTERN = (
    r"ImportError: cannot import name 'TangentPolarizationFrame' from "
    r"'radiosim\.core\.sky\.containers'"
)


def _local(function: str) -> str:
    return f"tests/unit/test_core/test_sci004_polarization.py::{function}"


def _case(
    case_id: str,
    requirement_id: str,
    nodeid: str,
    kind: str,
    pattern: str,
    fixture: bytes,
    *,
    excluded_by: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": nodeid,
        "expected_failure_kind": kind,
        "expected_failure_pattern": pattern,
        "fixture_defect_excluded_by": excluded_by,
        "fixture_bytes": fixture,
    }


SCI004_PHASE2_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m2.polarization.spin-two-closed-forms",
        "sci004.section-5.3.spin-two-single-modes",
        _local("test_the_spin_two_harmonics_match_their_published_closed_forms"),
        "missing-symbol",
        _HARMONICS_IMPORT_PATTERN,
        _SPIN_FIXTURE,
        excluded_by=_SCALAR_ORACLE,
    ),
    _case(
        "m2.polarization.spin-orthonormality",
        "sci004.section-5.3.orthonormal-spin-harmonics",
        _local("test_the_spin_harmonics_are_orthonormal_within_one_spin_weight"),
        "missing-symbol",
        _HARMONICS_IMPORT_PATTERN,
        _SPIN_FIXTURE,
        excluded_by=_SCALAR_ORACLE,
    ),
    _case(
        "m2.polarization.spin-conjugate-relation",
        "sci004.section-5.3.negative-spin-conjugate-relation",
        _local("test_the_negative_spin_harmonic_is_the_published_conjugate_relation"),
        "missing-symbol",
        _HARMONICS_IMPORT_PATTERN,
        _SPIN_FIXTURE,
        excluded_by=_SCALAR_ORACLE,
    ),
    _case(
        "m2.polarization.spin-reality",
        "sci004.section-5.3.paired-spin-reality-relation",
        _local("test_the_paired_spin_reality_relation_holds_for_a_real_linear_field"),
        "missing-symbol",
        _HARMONICS_IMPORT_PATTERN,
        _REALITY_FIXTURE,
        excluded_by=_SCALAR_ORACLE,
    ),
    _case(
        "m2.polarization.packed-spin-table",
        "sci004.section-5.3.packed-l-start-max-abs-m-abs-spin",
        _local("test_the_polarized_packed_table_starts_each_row_at_max_abs_m_abs_spin"),
        "missing-symbol",
        _HARMONICS_IMPORT_PATTERN,
        _PACKED_FIXTURE,
        excluded_by=_SCALAR_ORACLE,
    ),
    _case(
        "m2.polarization.shaw-bridge",
        "sci004.section-5.2.d-bridge-flips-only-u",
        _local("test_the_shaw_basis_bridge_is_diag_minus_one_one_and_flips_only_u"),
        "missing-symbol",
        _POLARIZATION_IMPORT_PATTERN,
        _BRIDGE_FIXTURE,
        excluded_by=_COHERENCY_ORACLE,
    ),
    _case(
        "m2.polarization.east-x-survives-the-bridge",
        "sci004.section-5.2.east-x-permutation-stays-inside-j-ne",
        _local("test_the_sci006_east_x_permutation_survives_the_shaw_bridge"),
        "missing-symbol",
        _POLARIZATION_IMPORT_PATTERN,
        _EAST_X_FIXTURE,
        excluded_by=_COHERENCY_ORACLE,
    ),
    _case(
        "m2.polarization.circular-v-sign",
        "sci004.section-5.2.circular-parallel-hand-v-sign",
        _local("test_the_circular_parallel_hand_difference_is_the_unflipped_iau_v"),
        "missing-symbol",
        _POLARIZATION_IMPORT_PATTERN,
        _CIRCULAR_FIXTURE,
        excluded_by=_COHERENCY_ORACLE,
    ),
    _case(
        "m2.polarization.cmb-u-conversion",
        "sci004.section-5.1.healpix-cmb-u-converted-to-iau",
        _local("test_a_healpix_cmb_payload_is_converted_to_iau_north_through_east"),
        "missing-symbol",
        _CONTAINER_IMPORT_PATTERN,
        _CMB_FIXTURE,
        excluded_by=_COHERENCY_ORACLE,
    ),
    _case(
        "m2.polarization.missing-tangent-frame",
        "sci004.section-5.1.polarized-payload-requires-a-declared-frame",
        _local("test_a_polarized_payload_without_a_declared_tangent_frame_is_rejected"),
        "missing-symbol",
        _CONTAINER_IMPORT_PATTERN,
        _MISSING_FRAME_FIXTURE,
        excluded_by=_COHERENCY_ORACLE,
    ),
    _case(
        "m2.polarization.intensity-only-omits-the-frame",
        "sci004.section-5.1.intensity-only-payload-may-omit-the-frame",
        _local("test_an_intensity_and_v_only_payload_may_omit_the_tangent_frame"),
        "missing-symbol",
        _CONTAINER_IMPORT_PATTERN,
        _INTENSITY_ONLY_FIXTURE,
        excluded_by=_COHERENCY_ORACLE,
    ),
    _case(
        "m2.capability.mmode-supports-polarization",
        "sci004.section-9.mmode-supports-polarization-true-after-m2",
        TIER7_PROPERTY_NODEID,
        "assertion",
        (
            r"AssertionError: accepted phase M2 flips "
            r"MModeSimulator\.supports_polarization to True"
        ),
        _CAPABILITY_FIXTURE,
        excluded_by=_COHERENCY_ORACLE,
    ),
)

SCI004_PHASE2_RED_GREEN_CONTROLS: tuple[str, ...] = (
    _COHERENCY_ORACLE,
    _SCALAR_ORACLE,
)


# --- closed forms derived here, never read from the module under test ---------


def _spin_two_closed_form(order: int, colatitude: float, longitude: float) -> complex:
    """Return ``_{+2}Y_{2m}`` from the published Goldberg-et-al. closed forms.

    The five ``l = 2`` spin-weight ``+2`` harmonics are standard (Goldberg et
    al. 1967; the same table the CMB literature prints):

    ``_{2}Y_{22} = sqrt(5/(64 pi)) (1+cos t)**2 exp(2 i p)``
    ``_{2}Y_{21} = sqrt(5/(16 pi)) sin t (1+cos t) exp(i p)``
    ``_{2}Y_{20} = sqrt(15/(32 pi)) sin(t)**2``
    ``_{2}Y_{2-1} = sqrt(5/(16 pi)) sin t (1-cos t) exp(-i p)``
    ``_{2}Y_{2-2} = sqrt(5/(64 pi)) (1-cos t)**2 exp(-2 i p)``

    Each is unit-normalized on the sphere, which
    :func:`test_the_spin_harmonics_are_orthonormal_within_one_spin_weight`
    re-derives numerically rather than assuming.
    """
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


def _sphere_quadrature(nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre in ``cos(theta)`` crossed with uniform azimuths."""
    abscissa, weights = np.polynomial.legendre.leggauss(nodes)
    theta = np.arccos(abscissa)
    azimuths = 2 * nodes
    phi = 2.0 * math.pi * np.arange(azimuths, dtype=np.float64) / azimuths
    grid_theta, grid_phi = np.meshgrid(theta, phi, indexing="ij")
    area = np.broadcast_to(
        weights[:, None] * (2.0 * math.pi / azimuths), grid_theta.shape
    )
    return (grid_theta.reshape(-1), grid_phi.reshape(-1), np.asarray(area).reshape(-1))


def _radiosim_brightness(intensity: float, q: float, u: float, v: float) -> np.ndarray:
    """Section 5.2's ``(North, East)`` brightness matrix, written out here."""
    return 0.5 * np.array(
        [
            [intensity + q, u + 1j * v],
            [u - 1j * v, intensity - q],
        ],
        dtype=np.complex128,
    )


def _shaw_brightness(intensity: float, q: float, u: float, v: float) -> np.ndarray:
    """Shaw's ``(theta, phi)`` brightness matrix, whose ``P^V`` sign is opposite.

    Section 5.2: "In one unchanged ordered basis, RadioSim's ``P^V_RS`` has the
    opposite matrix sign from Shaw's ``P^V``; after the required ``D`` basis
    bridge the physical IAU ``V`` field has the same sign."  Writing both
    matrices out is what makes that sentence checkable instead of quotable.
    """
    return 0.5 * np.array(
        [
            [intensity + q, u - 1j * v],
            [u + 1j * v, intensity - q],
        ],
        dtype=np.complex128,
    )


#: Section 5.2's exact bridge, restated here as the oracle's own constant.
_D_BRIDGE = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

#: SCI-006's ``P``: the IAU ``(North, East)`` sky basis mapped to ``(X=east,
#: Y=north)``.  It is antidiagonal; ``D`` is diagonal.
_EAST_X_PERMUTATION = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


# --- green controls -----------------------------------------------------------


def test_the_iau_coherency_and_east_x_permutation_hold_today() -> None:
    """The polarization algebra this module builds on is already correct at ``A1``.

    ``stokes_to_coherency`` is shipped production code carrying CLAUDE-normative
    half-power IAU convention, and SCI-006's ruling that ``P`` maps the IAU
    ``(North, East)`` sky basis to ``(X = east, Y = north)`` is what makes
    ``XX - YY == -Q`` the correct ideal response for an east-oriented X feed.
    Both facts hold here, so a red failure below is the absence of the phase-M2
    bridge and spin surface, not a defective fixture.
    """
    from radiosim.core.polarization import stokes_to_coherency

    intensity, q, u, v = 1.0, 0.2, 0.3, 0.4
    coherency = np.asarray(stokes_to_coherency(intensity, q, u, v))

    assert np.allclose(
        coherency,
        _radiosim_brightness(intensity, q, u, v),
        atol=ANALYTIC_RESIDUAL_LIMIT,
    )
    # Tr(B) == I, not 2I: the half-power convention CLAUDE.md makes normative.
    assert abs(complex(np.trace(coherency)) - intensity) <= ANALYTIC_RESIDUAL_LIMIT

    east_x = _EAST_X_PERMUTATION @ coherency @ _EAST_X_PERMUTATION.T
    assert abs(complex(east_x[0, 0] - east_x[1, 1]) + q) <= ANALYTIC_RESIDUAL_LIMIT

    # The two matrices Section 5.2 refuses to conflate.
    assert not np.array_equal(_D_BRIDGE, _EAST_X_PERMUTATION)
    assert float(np.linalg.det(_D_BRIDGE)) == -1.0
    assert float(np.linalg.det(_EAST_X_PERMUTATION)) == -1.0
    assert np.array_equal(_D_BRIDGE @ _D_BRIDGE, np.eye(2))


def test_the_scalar_condon_shortley_harmonics_close_in_the_test_body() -> None:
    """M1's scalar harmonics are sound, so a spin failure is the missing surface.

    ``Y_20 = sqrt(5/(16 pi)) (3 cos(t)**2 - 1)`` is the standard orthonormal
    Condon-Shortley value; it is written out here and compared against the
    shipped ``scalar_ylm``.  The quadrature helper the spin oracles use is
    exercised at the same time.
    """
    from radiosim.core.mmode.harmonics import scalar_ylm

    colatitude, longitude = 0.9, 2.3
    expected = math.sqrt(5.0 / (16.0 * math.pi)) * (
        3.0 * math.cos(colatitude) ** 2 - 1.0
    )
    assert abs(scalar_ylm(2, 0, colatitude, longitude) - expected) <= (
        ANALYTIC_RESIDUAL_LIMIT
    )

    theta, phi, area = _sphere_quadrature(2 * LMAX + 8)
    values = np.asarray(
        [scalar_ylm(2, 1, float(t), float(p)) for t, p in zip(theta, phi, strict=True)],
        dtype=np.complex128,
    )
    assert abs(float(np.sum(area * np.abs(values) ** 2)) - 1.0) <= (
        ANALYTIC_RESIDUAL_LIMIT
    )


# --- Section 5.3 spin-harmonic red oracles ------------------------------------


def test_the_spin_two_harmonics_match_their_published_closed_forms() -> None:
    """Section 5.3: individual spin ``+2`` modes, against Goldberg's table."""
    from radiosim.core.mmode.harmonics import spin_ylm

    colatitude, longitude = 0.9, 2.3
    for order in (-2, -1, 0, 1, 2):
        observed = spin_ylm(2, 2, order, colatitude, longitude)
        expected = _spin_two_closed_form(order, colatitude, longitude)
        assert abs(observed - expected) <= ANALYTIC_RESIDUAL_LIMIT, order

    # A spin-weight ``s`` harmonic does not exist for ``l < abs(s)``: Section 5.3
    # says invalid ``(l, m, s)`` cells "do not exist and are not represented by
    # padding whose value could enter a digest".
    for degree in (0, 1):
        raised = None
        try:
            spin_ylm(2, degree, 0, colatitude, longitude)
        except ValueError as error:  # pragma: no cover - the red path
            raised = error
        assert raised is not None, f"l={degree} is not a spin-two harmonic"


def test_the_spin_harmonics_are_orthonormal_within_one_spin_weight() -> None:
    """Section 5.3: ``integral(sY_lm conj(sY_l'm')) = delta_ll' delta_mm'``."""
    from radiosim.core.mmode.harmonics import spin_ylm

    theta, phi, area = _sphere_quadrature(2 * LMAX + 12)
    cells = [(2, 0), (2, 1), (3, 1), (3, -2), (4, 2)]
    for spin in (2, -2):
        sampled = {
            cell: np.asarray(
                [
                    spin_ylm(spin, cell[0], cell[1], float(t), float(p))
                    for t, p in zip(theta, phi, strict=True)
                ],
                dtype=np.complex128,
            )
            for cell in cells
        }
        for left in cells:
            for right in cells:
                overlap = complex(
                    np.sum(area * sampled[left] * np.conjugate(sampled[right]))
                )
                expected = 1.0 if left == right else 0.0
                assert abs(overlap - expected) <= ANALYTIC_RESIDUAL_LIMIT, (
                    spin,
                    left,
                    right,
                )


def test_the_negative_spin_harmonic_is_the_published_conjugate_relation() -> None:
    """Section 5.3: ``_{-s}Y_lm = (-1)**(s+m) conj(_{s}Y_{l,-m})``.

    This is the relation that makes the two spin fields a pair rather than two
    independent expansions, and it is the relation the paired reality rule below
    is derived from.
    """
    from radiosim.core.mmode.harmonics import spin_ylm

    colatitude, longitude = 0.9, 2.3
    for degree in (2, 3, 4):
        for order in range(-degree, degree + 1):
            expected = ((-1.0) ** (2 + order)) * np.conjugate(
                spin_ylm(2, degree, -order, colatitude, longitude)
            )
            observed = spin_ylm(-2, degree, order, colatitude, longitude)
            assert abs(observed - expected) <= ANALYTIC_RESIDUAL_LIMIT, (degree, order)

    # Non-vacuity: dropping the ``(-1)**(s+m)`` factor is a different convention
    # and must miss by far more than the passing residual for at least one cell.
    wrong = np.conjugate(spin_ylm(2, 3, 1, colatitude, longitude))
    assert abs(spin_ylm(-2, 3, -1, colatitude, longitude) - wrong) > (
        NON_VACUITY_FACTOR * ANALYTIC_RESIDUAL_LIMIT
    )


def test_the_paired_spin_reality_relation_holds_for_a_real_linear_field() -> None:
    """Section 5.3: ``a^(-2)[l,m] = (-1)**m conj(a^(+2)[l,-m])``.

    The relation is a theorem, not a convention: for real ``Q_H`` and ``U_H``,
    ``Q_H - i U_H`` is the complex conjugate of ``Q_H + i U_H``, and applying
    ``conj(_{s}Y_lm) = (-1)**(s+m) _{-s}Y_{l,-m}`` to the ``+2`` expansion and
    relabelling ``m -> -m`` produces exactly the displayed pairing.  A real
    analytic field is transformed here and both coefficient sets are compared.
    """
    from radiosim.core.mmode.harmonics import spin_transform_reference

    def linear(colatitude: float, longitude: float) -> complex:
        """A real ``(Q_H, U_H)`` field written as ``Q_H + i U_H``."""
        q = math.sin(colatitude) ** 2 * math.cos(2.0 * longitude)
        u = math.sin(colatitude) ** 2 * math.sin(2.0 * longitude) * 0.5
        return complex(q, u)

    positive = spin_transform_reference(linear, spin=2, lmax=LMAX, mmax=MMAX)
    negative = spin_transform_reference(
        lambda t, p: complex(np.conjugate(linear(t, p))), spin=-2, lmax=LMAX, mmax=MMAX
    )

    for degree in range(2, LMAX + 1):
        for order in range(-min(degree, MMAX), min(degree, MMAX) + 1):
            expected = ((-1.0) ** order) * np.conjugate(
                positive.coefficient(degree, -order)
            )
            observed = negative.coefficient(degree, order)
            assert abs(observed - expected) <= ANALYTIC_RESIDUAL_LIMIT, (degree, order)


def test_the_polarized_packed_table_starts_each_row_at_max_abs_m_abs_spin() -> None:
    """Section 5.3: signed-``m``-major, field-minor, ``l_start = max(|m|, |s|)``.

    The packed layout is what makes a sky and a transfer cell joinable without
    ever consulting a library ``alm`` order, so its arithmetic is reconstructed
    here from the design text rather than read back from the table.
    """
    from radiosim.core.mmode.harmonics import polarized_packed_block_table

    table = polarized_packed_block_table(lmax=LMAX, mmax=MMAX)

    expected: list[dict[str, Any]] = []
    cursor = 0
    for order in range(-MMAX, MMAX + 1):
        for index, (field, spin) in enumerate(
            zip(FIELD_ORDER, SPIN_ORDER, strict=True)
        ):
            l_start = max(abs(order), abs(spin))
            l_stop = LMAX + 1
            expected.append(
                {
                    "m": order,
                    "field_index": index,
                    "field_name": field,
                    "spin": spin,
                    "l_start": l_start,
                    "l_stop": l_stop,
                    "value_start": cursor,
                    "value_stop": cursor + (l_stop - l_start),
                }
            )
            cursor += l_stop - l_start

    observed = [dict(row) for row in table.block_rows]
    assert observed == expected
    assert table.packed_value_count == cursor
    # Section 5.3's exact row field order, which the digest preimage depends on.
    assert tuple(observed[0]) == (
        "m",
        "field_index",
        "field_name",
        "spin",
        "l_start",
        "l_stop",
        "value_start",
        "value_stop",
    )
    # No padding: an invalid cell has zero width rather than a hashable zero.
    for row in observed:
        assert row["value_stop"] - row["value_start"] == row["l_stop"] - row["l_start"]
        assert row["l_start"] == max(abs(row["m"]), abs(row["spin"]))


# --- Section 5.2 basis-bridge red oracles -------------------------------------


def test_the_shaw_basis_bridge_is_diag_minus_one_one_and_flips_only_u() -> None:
    """Section 5.2: ``D = diag(-1, 1)`` and ``(I, Q, U, V) -> (I, Q, -U, V)``.

    The expected field map is *derived* here: transporting RadioSim's
    ``(North, East)`` brightness matrix with ``D P D`` and reading the result off
    against Shaw's own ``(theta, phi)`` matrix -- whose ``P^V`` sign is opposite
    in one unchanged basis -- yields ``U_H = -U`` and ``V_H = +V`` with no
    additional fitted flip anywhere.
    """
    from radiosim.core.polarization import shaw_basis_bridge, stokes_to_shaw_fields

    assert np.array_equal(np.asarray(shaw_basis_bridge(), dtype=np.float64), _D_BRIDGE)

    intensity, q, u, v = 1.0, 0.2, 0.3, 0.4
    transported = _D_BRIDGE @ _radiosim_brightness(intensity, q, u, v) @ _D_BRIDGE.T
    derived = _shaw_brightness(intensity, q, -u, v)
    assert np.allclose(transported, derived, atol=ANALYTIC_RESIDUAL_LIMIT)

    observed = stokes_to_shaw_fields(intensity, q, u, v)
    assert tuple(float(value) for value in observed) == (intensity, q, -u, v)

    # Non-vacuity: an omitted bridge leaves ``U_H = +U``, which the transported
    # matrix comparison must reject by far more than the analytic residual.
    unbridged = _shaw_brightness(intensity, q, u, v)
    assert float(np.max(np.abs(transported - unbridged))) > (
        NON_VACUITY_FACTOR * ANALYTIC_RESIDUAL_LIMIT
    )


def test_the_sci006_east_x_permutation_survives_the_shaw_bridge() -> None:
    """Section 5.2: the east-X permutation stays **inside** ``J_NE``.

    ``J_thetaphi = J_NE D``, so an east-X receptor evaluated through the bridged
    basis must reproduce exactly the visibility the unbridged ``(North, East)``
    route already produces -- SCI-006's ``XX - YY == -Q``.  Replacing the
    permutation by ``D``, or applying ``D`` twice, changes that answer.
    """
    from radiosim.core.polarization import shaw_basis_bridge, stokes_to_shaw_fields

    intensity, q, u, v = 1.0, 0.2, 0.3, 0.4
    bridge = np.asarray(shaw_basis_bridge(), dtype=np.float64)

    reference = (
        _EAST_X_PERMUTATION
        @ _radiosim_brightness(intensity, q, u, v)
        @ _EAST_X_PERMUTATION.conj().T
    )
    shaw_fields = tuple(
        float(value) for value in stokes_to_shaw_fields(intensity, q, u, v)
    )
    jones_theta_phi = _EAST_X_PERMUTATION @ bridge
    bridged = (
        jones_theta_phi @ _shaw_brightness(*shaw_fields) @ jones_theta_phi.conj().T
    )

    assert np.allclose(bridged, reference, atol=ANALYTIC_RESIDUAL_LIMIT)
    assert abs(complex(bridged[0, 0] - bridged[1, 1]) + q) <= ANALYTIC_RESIDUAL_LIMIT

    # Non-vacuity: substituting ``D`` for the permutation is the defect SCI-006
    # ruled on, and it must miss by more than ten times the passing residual.
    substituted = bridge @ _shaw_brightness(*shaw_fields) @ bridge.conj().T
    assert float(np.max(np.abs(substituted - reference))) > (
        NON_VACUITY_FACTOR * ANALYTIC_RESIDUAL_LIMIT
    )


def test_the_circular_parallel_hand_difference_is_the_unflipped_iau_v() -> None:
    """Section 5.2: after the bridge the physical IAU ``V`` keeps its sign.

    In the circular basis the parallel hands satisfy ``RR - LL = +V`` for the
    IAU incoming ``R - L`` convention Section 5.1 names.  The circular receptor
    matrix is written out here from that convention and the identity is required
    to survive the ``D`` bridge with no configurable second flip.
    """
    from radiosim.core.polarization import shaw_basis_bridge, stokes_to_shaw_fields

    intensity, q, u, v = 1.0, 0.0, 0.0, 0.4
    bridge = np.asarray(shaw_basis_bridge(), dtype=np.float64)
    # ``(N, E) -> (R, L)`` for the IAU incoming convention.
    circular = np.array([[1.0, 1j], [1.0, -1j]], dtype=np.complex128) / math.sqrt(2.0)

    reference = circular @ _radiosim_brightness(intensity, q, u, v) @ circular.conj().T
    assert abs(complex(reference[0, 0] - reference[1, 1]) - v) <= (
        ANALYTIC_RESIDUAL_LIMIT
    )

    shaw_fields = tuple(
        float(value) for value in stokes_to_shaw_fields(intensity, q, u, v)
    )
    jones_theta_phi = circular @ bridge
    bridged = (
        jones_theta_phi @ _shaw_brightness(*shaw_fields) @ jones_theta_phi.conj().T
    )
    assert np.allclose(bridged, reference, atol=ANALYTIC_RESIDUAL_LIMIT)
    assert abs(complex(bridged[0, 0] - bridged[1, 1]) - v) <= ANALYTIC_RESIDUAL_LIMIT


# --- Section 5.1 tangent-frame red oracles ------------------------------------


def test_a_healpix_cmb_payload_is_converted_to_iau_north_through_east() -> None:
    """Section 5.1: the sign is pinned with a rotated pure-Q map.

    A pure-``Q`` source rotated to position angle ``45`` degrees has, in the IAU
    North-through-East convention, ``Q = p cos(2 chi) = 0`` and
    ``U = p sin(2 chi) = +p``.  The HEALPix/CMB convention measures the angle
    North-through-West, so its stored ``U`` is the negative of the IAU value; the
    loader must convert explicitly before canonical storage rather than
    relabelling the payload.
    """
    from radiosim.core.sky.containers import TangentPolarizationFrame

    canonical = TangentPolarizationFrame(
        schema_version=TANGENT_FRAME_SCHEMA,
        coordinate_frame="galactic",
        axes="north_east",
        position_angle="north_through_east",
        linear_complex="q_plus_i_u",
        stokes_v="iau_incoming_r_minus_l",
    )
    assert tuple(canonical.as_mapping()) == TANGENT_FRAME_KEYS

    polarized_intensity = 2.0
    position_angle = math.radians(45.0)
    iau_q = polarized_intensity * math.cos(2.0 * position_angle)
    iau_u = polarized_intensity * math.sin(2.0 * position_angle)
    assert abs(iau_q) <= ANALYTIC_RESIDUAL_LIMIT
    assert abs(iau_u - polarized_intensity) <= ANALYTIC_RESIDUAL_LIMIT

    cmb_q, cmb_u = iau_q, -iau_u
    converted_q, converted_u = TangentPolarizationFrame.to_canonical(
        stokes_q=cmb_q,
        stokes_u=cmb_u,
        position_angle="north_through_west",
    )
    assert abs(float(converted_q) - iau_q) <= ANALYTIC_RESIDUAL_LIMIT
    assert abs(float(converted_u) - iau_u) <= ANALYTIC_RESIDUAL_LIMIT

    # Non-vacuity: copying ``U`` through unchanged is the defect Section 5.1
    # forbids, and it lands a full ``2 p`` away from the canonical value.
    assert abs(cmb_u - iau_u) > NON_VACUITY_FACTOR * ANALYTIC_RESIDUAL_LIMIT


def test_a_polarized_payload_without_a_declared_tangent_frame_is_rejected() -> None:
    """Section 5.1: a programmatic polarized input with no source convention."""
    from radiosim.core.sky.containers import TangentPolarizationFrame

    raised = None
    try:
        TangentPolarizationFrame.require_for(
            stokes_q=0.5, stokes_u=-0.25, stokes_v=0.0, frame=None
        )
    except ValueError as error:  # pragma: no cover - the red path
        raised = error
    assert raised is not None, (
        "a non-zero Q or U payload without a declared tangent frame is rejected"
    )


def test_an_intensity_and_v_only_payload_may_omit_the_tangent_frame() -> None:
    """Section 5.1: "An I/V-only payload may omit the tangent block"."""
    from radiosim.core.sky.containers import TangentPolarizationFrame

    resolved = TangentPolarizationFrame.require_for(
        stokes_q=0.0, stokes_u=0.0, stokes_v=0.3, frame=None
    )
    assert resolved is None
