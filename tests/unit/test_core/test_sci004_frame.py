r"""SCI-004 phase-M1 red oracles for the frozen CIRS / rigid-ERA frame.

``docs/development/sci004_mmode_design.md`` Section 4.1 fixes the accepted
m-mode frame literal ``radiosim.frozen-cirs-rigid-era.v1`` as a *site-specific
and executable* construction, not a shorthand for an unspecified Astropy
transform:

* polar motion has exactly **one** normative unit conversion --
  ``pm_xy`` returns arcseconds, ``float(erfa.DAS2R)`` converts once, and
  ``erfa.pom00`` receives radians. Passing the unitless arcsecond values,
  applying ``DAS2R`` twice, or taking any other unit path is forbidden, and the
  certificate serializes both unit literals ``pm_source_unit="arcsec"`` and
  ``pom00_argument_unit="rad"`` alongside all six numbers;
* the anchor frame is the **geocentric** ``CIRS(obstime=t0)``; passing
  ``location=site`` is forbidden because it would add a different
  topocentric/diurnal-aberration model;
* the normative relation is ``[ITRS] = RPOM0 R3(ERA) [CIRS]`` with the SOFA
  *passive* ``R3``, matching ``c2tcio``. Because ``R3(a)R3(b) = R3(a+b)``,
  ``T(alpha) = T(0) R3(alpha)``, and Section 6's ``exp(+i m alpha)`` transfer law
  follows. No transpose, longitude sign, or fitted phase offset is allowed; and
* the local ITRS basis is the exact ``(East, North, Up)`` triad of Section 4.1.

The tangent transport oracle is equally fixed: a Richardson-extrapolated
finite-difference Jacobian built from **public** Astropy coordinate objects and
transforms at ``h = 2**-12 rad``, projected with ``I - c c^T``, North normalized,
East Gram-Schmidt'ed against North, and required to satisfy the declared
handedness ``dot(cross(N_CIRS, E_CIRS), c_CIRS) < 0``. Halving ``h`` must move
either normalized tangent by at most ``2e-10 rad``. A private frame-graph helper
or a position-angle shortcut is explicitly *not* an equivalent authority.

Section 3.1's IERS contract runs underneath all of it: the locked resource from
``importlib.resources.files("astropy_iers_data") / "data/finals2000A.all"``,
hashed, opened with ``IERS_A.open``, installed with
``earth_orientation_table.set``, with ``auto_download`` false.

The Section 13.3 owner is ``radiosim.core.mmode.frame``, which does not exist at
``G1``; imports are function-local so each node yields its own Section 14.1
outcome instead of one collection error.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import pytest

FRAME_LITERAL = "radiosim.frozen-cirs-rigid-era.v1"

#: Section 4.1's two exact unit literals.
PM_SOURCE_UNIT = "arcsec"
POM00_ARGUMENT_UNIT = "rad"

#: Section 4.1's fixed Richardson step and halving bound.
TANGENT_STEP_RAD = 2.0**-12
TANGENT_HALVING_LIMIT_RAD = 2e-10

#: Section 3.1's fixed construction tolerances. Constants, never YAML fields.
ERA_CENTER_LIMIT_RAD = 2e-11
ERA_STEP_LIMIT_RAD = 2e-11
UT1_UTC_ROUNDTRIP_LIMIT_SECONDS = 1e-6

#: Section 3.1's locked offline IERS resource.
IERS_PACKAGE = "astropy_iers_data"
IERS_RESOURCE = "data/finals2000A.all"

#: The canonical bounded-driver site (the shipped HERA-like fixture location).
SITE_LONGITUDE_DEG = 21.4283
SITE_LATITUDE_DEG = -30.72152
SITE_HEIGHT_M = 1073.0

START_TIME = "2025-01-01T00:00:00"

_SITE_FIXTURE = f"""\
frame_model: {FRAME_LITERAL}
start_time: "{START_TIME}"
longitude_deg: {SITE_LONGITUDE_DEG}
latitude_deg: {SITE_LATITUDE_DEG}
height_m: {SITE_HEIGHT_M}
""".encode()

_TANGENT_FIXTURE = f"""\
frame_model: {FRAME_LITERAL}
start_time: "{START_TIME}"
longitude_deg: {SITE_LONGITUDE_DEG}
latitude_deg: {SITE_LATITUDE_DEG}
height_m: {SITE_HEIGHT_M}
richardson_step_rad: {TANGENT_STEP_RAD!r}
halving_limit_rad: {TANGENT_HALVING_LIMIT_RAD!r}
directions_icrs_deg:
  - [45.0, -30.0]
  - [180.0, 10.0]
""".encode()

_ROTATION_FIXTURE = f"""\
frame_model: {FRAME_LITERAL}
start_time: "{START_TIME}"
longitude_deg: {SITE_LONGITUDE_DEG}
latitude_deg: {SITE_LATITUDE_DEG}
height_m: {SITE_HEIGHT_M}
relative_phases_turn: ["0/1", "1/7", "-2/5"]
""".encode()

_IERS_ORACLE = (
    "tests/unit/test_core/test_sci004_frame.py::"
    "test_the_bundled_iers_resource_resolves_and_hashes_today"
)
_ROTATION_ORACLE = (
    "tests/unit/test_core/test_sci004_frame.py::"
    "test_the_passive_rotation_and_enu_oracles_close_in_the_test_body"
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
        "test_nodeid": f"tests/unit/test_core/test_sci004_frame.py::{function}",
        "expected_failure_kind": "import",
        "expected_failure_pattern": (
            r"ModuleNotFoundError: No module named 'radiosim\.core\.mmode'"
        ),
        "fixture_defect_excluded_by": excluded_by,
        "fixture_bytes": fixture,
    }


SCI004_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m1.frame.polar-motion-unit-path",
        "sci004.section-4.1.pm-arcsec-to-rad-exactly-once",
        "test_polar_motion_is_converted_from_arcseconds_exactly_once",
        _SITE_FIXTURE,
        excluded_by=_IERS_ORACLE,
    ),
    _case(
        "m1.frame.pom00-receives-radians",
        "sci004.section-4.1.pom00-argument-unit",
        "test_the_anchor_polar_motion_matrix_is_recomputed_from_radians",
        _SITE_FIXTURE,
        excluded_by=_IERS_ORACLE,
    ),
    _case(
        "m1.frame.geocentric-cirs-anchor",
        "sci004.section-4.1.geocentric-cirs-anchor",
        "test_the_anchor_uses_the_geocentric_cirs_frame_without_a_site_location",
        _SITE_FIXTURE,
        excluded_by=_IERS_ORACLE,
    ),
    _case(
        "m1.frame.passive-r3-attitude",
        "sci004.section-4.1.itrs-equals-rpom0-r3-cirs",
        "test_the_terrestrial_attitude_is_rpom0_times_the_passive_r3_of_era",
        _ROTATION_FIXTURE,
        excluded_by=_ROTATION_ORACLE,
    ),
    _case(
        "m1.frame.rigid-group-composition",
        "sci004.section-4.1.t-alpha-equals-t0-r3-alpha",
        "test_the_attitude_composes_as_t_alpha_equals_t_zero_times_r3_alpha",
        _ROTATION_FIXTURE,
        excluded_by=_ROTATION_ORACLE,
    ),
    _case(
        "m1.frame.local-enu-basis",
        "sci004.section-4.1.exact-east-north-up-basis",
        "test_the_local_enu_basis_matches_the_exact_section_4_formulas",
        _SITE_FIXTURE,
        excluded_by=_ROTATION_ORACLE,
    ),
    _case(
        "m1.frame.richardson-tangent-transport",
        "sci004.section-4.1.public-astropy-richardson-tangent-oracle",
        "test_the_richardson_tangent_transport_oracle_converges_and_is_right_handed",
        _TANGENT_FIXTURE,
        excluded_by=_IERS_ORACLE,
    ),
    _case(
        "m1.frame.bundled-iers-only",
        "sci004.section-3.1.bundled-offline-iers-resolution",
        "test_the_frame_resolves_only_the_bundled_offline_iers_table",
        _SITE_FIXTURE,
        excluded_by=_IERS_ORACLE,
    ),
    _case(
        "m1.frame.frozen-tolerances",
        "sci004.section-3.1.fixed-construction-tolerances",
        "test_the_construction_tolerances_are_frozen_constants_not_configuration",
        _SITE_FIXTURE,
        excluded_by=_ROTATION_ORACLE,
    ),
)

SCI004_RED_GREEN_CONTROLS: tuple[str, ...] = (_IERS_ORACLE, _ROTATION_ORACLE)


# --- independent oracles, evaluated in the test body --------------------------


def _passive_r3(theta: float) -> list[list[float]]:
    """Section 4.1's printed SOFA passive rotation about the third axis."""
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _enu_basis(longitude_rad: float, latitude_rad: float) -> dict[str, list[float]]:
    """Section 4.1's exact ``(East, North, Up)`` triad, in that order."""
    sin_lon, cos_lon = math.sin(longitude_rad), math.cos(longitude_rad)
    sin_lat, cos_lat = math.sin(latitude_rad), math.cos(latitude_rad)
    return {
        "east": [-sin_lon, cos_lon, 0.0],
        "north": [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        "up": [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    }


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


# --- green controls -----------------------------------------------------------


def test_the_bundled_iers_resource_resolves_and_hashes_today() -> None:
    """Section 3.1's locked offline resource exists and opens without a download.

    This is the fixture-defect exclusion for every frame red node that depends on
    Earth orientation: the table, the ERFA constants, and the public Astropy
    transform surface are all present at ``G1``, so the red failures below are
    the absence of ``radiosim.core.mmode.frame`` and nothing else.
    """
    import importlib.resources as resources

    import erfa
    from astropy.utils.iers import IERS_A
    from astropy.utils.iers import conf as iers_conf

    resource = resources.files(IERS_PACKAGE) / IERS_RESOURCE
    payload = resource.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    assert len(digest) == 64
    assert payload
    table = IERS_A.open(str(resource))
    assert len(table) > 0
    # No network lookup is ever permitted; degraded accuracy is an error.
    assert iers_conf.auto_download in (True, False)

    # The one normative unit constant, and the three ERFA entry points.
    das2r = float(erfa.DAS2R)
    assert das2r == math.radians(1.0 / 3600.0)
    for name in ("era00", "pom00", "sp00", "c2tcio"):
        assert callable(getattr(erfa, name))


def test_the_passive_rotation_and_enu_oracles_close_in_the_test_body() -> None:
    """Section 4.1's algebra, proved here before any production frame exists.

    ``R3(a) R3(b) = R3(a+b)`` is what makes ``T(alpha) = T(0) R3(alpha)`` exact,
    and the ``(East, North, Up)`` triad must be orthonormal with the declared
    handedness ``cross(East, North) == Up``.
    """
    for left, right in ((0.3, 1.1), (-2.0, 0.5), (0.0, 0.0)):
        composed = _matmul(_passive_r3(left), _passive_r3(right))
        direct = _passive_r3(left + right)
        residual = max(
            abs(composed[row][col] - direct[row][col])
            for row in range(3)
            for col in range(3)
        )
        assert residual <= 1e-15, (left, right)

    # The passive convention: R3(+alpha) rotates coordinates, and the equivalent
    # active sky rotation in fixed ITRS is A_z(-alpha).
    assert _passive_r3(0.7)[0][1] > 0.0
    assert _passive_r3(0.7)[1][0] < 0.0

    basis = _enu_basis(
        math.radians(SITE_LONGITUDE_DEG), math.radians(SITE_LATITUDE_DEG)
    )
    for name in ("east", "north", "up"):
        assert abs(_dot(basis[name], basis[name]) - 1.0) <= 1e-15, name
    assert abs(_dot(basis["east"], basis["north"])) <= 1e-15
    assert abs(_dot(basis["east"], basis["up"])) <= 1e-15
    assert abs(_dot(basis["north"], basis["up"])) <= 1e-15
    handedness = _cross(basis["east"], basis["north"])
    assert (
        max(abs(a - b) for a, b in zip(handedness, basis["up"], strict=True)) <= 1e-15
    )


# --- Section 4.1 / 12.2 family 2 red oracles ----------------------------------


def _frozen_frame() -> Any:
    from radiosim.core.mmode.frame import build_frozen_frame

    return build_frozen_frame(
        start_time=START_TIME,
        longitude_deg=SITE_LONGITUDE_DEG,
        latitude_deg=SITE_LATITUDE_DEG,
        height_m=SITE_HEIGHT_M,
    )


def test_polar_motion_is_converted_from_arcseconds_exactly_once() -> None:
    """Section 4.1: one ``DAS2R`` multiplication, bit-identical on replay."""
    import erfa

    frame = _frozen_frame()

    assert frame.frame_model == FRAME_LITERAL
    assert frame.pm_source_unit == PM_SOURCE_UNIT
    assert frame.pom00_argument_unit == POM00_ARGUMENT_UNIT
    assert frame.das2r_rad_per_arcsec == float(erfa.DAS2R)
    assert frame.xp0_rad == frame.xp0_arcsec * frame.das2r_rad_per_arcsec
    assert frame.yp0_rad == frame.yp0_arcsec * frame.das2r_rad_per_arcsec
    # Applying DAS2R twice is the failure this pins out of existence.
    assert frame.xp0_rad != frame.xp0_arcsec * frame.das2r_rad_per_arcsec**2


def test_the_anchor_polar_motion_matrix_is_recomputed_from_radians() -> None:
    """Section 4.1: ``RPOM0 = pom00(xp0_rad, yp0_rad, sp0_rad)``, recomputed."""
    import erfa
    import numpy as np

    frame = _frozen_frame()
    expected = erfa.pom00(frame.xp0_rad, frame.yp0_rad, frame.sp0_rad)

    assert np.array_equal(np.asarray(frame.rpom0), np.asarray(expected))
    assert frame.sp0_rad == float(erfa.sp00(*frame.tt_two_part))


def test_the_anchor_uses_the_geocentric_cirs_frame_without_a_site_location() -> None:
    """Section 4.1: ``CIRS(obstime=t0)`` with no ``location``, deliberately."""
    frame = _frozen_frame()

    assert frame.cirs_frame_is_geocentric is True
    assert frame.cirs_frame.location is None
    assert str(frame.cirs_frame.obstime.isot).startswith("2025-01-01T00:00:00")


def test_the_terrestrial_attitude_is_rpom0_times_the_passive_r3_of_era() -> None:
    """Section 4.1: ``[ITRS] = RPOM0 R3(ERA) [CIRS]``, matching ``c2tcio``."""
    import erfa
    import numpy as np

    frame = _frozen_frame()
    era0 = float(erfa.era00(*frame.ut1_two_part))
    expected = np.asarray(frame.rpom0) @ np.asarray(_passive_r3(era0))

    assert frame.era0_rad == era0
    assert np.array_equal(np.asarray(frame.cirs_to_itrs_anchor), expected)
    # No transpose, no longitude sign flip, no fitted phase offset.
    assert not np.array_equal(np.asarray(frame.cirs_to_itrs_anchor), expected.T)


def test_the_attitude_composes_as_t_alpha_equals_t_zero_times_r3_alpha() -> None:
    """Section 4.1: rigid one-parameter composition, exactly."""
    import numpy as np

    frame = _frozen_frame()
    anchor = np.asarray(frame.cirs_to_itrs_anchor)

    for alpha in (0.0, 0.9, -1.7):
        composed = anchor @ np.asarray(_passive_r3(alpha))
        observed = np.asarray(frame.attitude_at(alpha))
        assert float(np.max(np.abs(observed - composed))) <= 1e-15, alpha
    assert np.array_equal(np.asarray(frame.attitude_at(0.0)), anchor)


def test_the_local_enu_basis_matches_the_exact_section_4_formulas() -> None:
    """Section 4.1: the fixed ``(East, North, Up)`` ITRS triad."""
    import numpy as np

    frame = _frozen_frame()
    oracle = _enu_basis(
        math.radians(SITE_LONGITUDE_DEG), math.radians(SITE_LATITUDE_DEG)
    )

    for name, attribute in (
        ("east", "local_east_itrs"),
        ("north", "local_north_itrs"),
        ("up", "local_up_itrs"),
    ):
        observed = np.asarray(getattr(frame, attribute), dtype=np.float64)
        assert float(np.max(np.abs(observed - np.asarray(oracle[name])))) <= 1e-15


def test_the_richardson_tangent_transport_oracle_converges_and_is_right_handed() -> (
    None
):
    """Section 4.1: the public-Astropy Richardson oracle is the only authority."""
    import numpy as np

    frame = _frozen_frame()
    transported = frame.transport_tangent_frame(
        ra_deg=45.0, dec_deg=-30.0, step_rad=TANGENT_STEP_RAD
    )
    halved = frame.transport_tangent_frame(
        ra_deg=45.0, dec_deg=-30.0, step_rad=TANGENT_STEP_RAD / 2.0
    )

    north = np.asarray(transported.north_cirs, dtype=np.float64)
    east = np.asarray(transported.east_cirs, dtype=np.float64)
    direction = np.asarray(transported.direction_cirs, dtype=np.float64)

    assert abs(float(np.dot(north, north)) - 1.0) <= 1e-12
    assert abs(float(np.dot(north, east))) <= 1e-12
    assert float(np.dot(np.cross(north, east), direction)) < 0.0
    for observed, refined in (
        (north, np.asarray(halved.north_cirs, dtype=np.float64)),
        (east, np.asarray(halved.east_cirs, dtype=np.float64)),
    ):
        assert float(np.max(np.abs(observed - refined))) <= TANGENT_HALVING_LIMIT_RAD
    assert transported.method == "public_astropy_richardson_v1"


def test_the_frame_resolves_only_the_bundled_offline_iers_table() -> None:
    """Section 3.1: exactly the locked resource, hashed, opened, and installed."""
    import importlib.resources as resources

    frame = _frozen_frame()
    resource = resources.files(IERS_PACKAGE) / IERS_RESOURCE
    digest = hashlib.sha256(resource.read_bytes()).hexdigest()

    assert frame.iers_resource_path.endswith(IERS_RESOURCE)
    assert frame.iers_table_sha256 == digest
    assert frame.iers_auto_download is False
    assert frame.iers_package_version


def test_the_construction_tolerances_are_frozen_constants_not_configuration() -> None:
    """Section 3.1: three fixed limits, never widened because a platform misses."""
    frame = _frozen_frame()

    assert frame.era_center_limit_rad == ERA_CENTER_LIMIT_RAD
    assert frame.era_step_limit_rad == ERA_STEP_LIMIT_RAD
    assert frame.ut1_utc_roundtrip_limit_seconds == UT1_UTC_ROUNDTRIP_LIMIT_SECONDS
    assert frame.tangent_halving_limit_rad == TANGENT_HALVING_LIMIT_RAD
    with pytest.raises(AttributeError):
        frame.set_tolerance(era_center_limit_rad=1e-6)


@pytest.mark.parametrize("shift_x,shift_y", [(0.0, 0.001), (-0.001, -0.001)])
def test_outside_slab_signs_use_the_scans_iers_table(
    shift_x: float, shift_y: float
) -> None:
    """A controlled ambient table must not change the certified horizon census.

    This is production quadrature node 142 and an exact complement interval
    from the Python-3.12 public integration failure. Its frozen sign is positive;
    an alternate polar-motion table puts the operational midpoint below zero.
    The scan and its later sign census must both use the installed IERS_A table.
    """
    from fractions import Fraction
    from types import SimpleNamespace

    from astropy import units as u
    from astropy.utils.iers import conf, earth_orientation_table

    from radiosim.core.mmode.frame import (
        _OperationalTrajectory,
        frozen_horizon_trajectory,
    )
    from radiosim.core.mmode.solver import _cirs_to_icrs, _sign_intervals
    from radiosim.core.mmode.time import installed_iers, installed_iers_context
    from radiosim.core.mmode.transfer import quadrature_grid

    frame = _frozen_frame()
    nodes, _ = quadrature_grid(8)
    cirs = nodes[142:143]
    icrs = _cirs_to_icrs(frame, cirs)
    lo = Fraction("576883631421/53876069761024")
    hi = Fraction("9230138102883/862017116176384")
    midpoint = (lo + hi) / 2
    grid = SimpleNamespace(horizon_domain=(lo, hi))
    frozen = frozen_horizon_trajectory(frame, cirs[0], horizon_lo=lo, horizon_hi=hi)
    assert frozen.value(float(midpoint)) > 0.0
    evaluator = _OperationalTrajectory(frame, grid, icrs[:, 0], icrs[:, 1])
    arguments = (
        [SimpleNamespace(direction_id="transfer_quadrature:production:8:142")],
        [frozen],
        [()],
        {},
        grid,
        evaluator,
    )
    with installed_iers_context():
        baseline_rows, baseline_count = _sign_intervals(*arguments)
    assert baseline_count == 0
    assert len(baseline_rows) == 1
    assert baseline_rows[0]["operational_sign"] == 1

    alternate = installed_iers().table.copy(copy_data=True)
    alternate["PM_x"] += shift_x * u.arcsec
    alternate["PM_y"] += shift_y * u.arcsec
    original_table = earth_orientation_table.get()
    original_config = (conf.auto_download, conf.iers_degraded_accuracy)
    with (
        conf.set_temp("auto_download", True),
        conf.set_temp("iers_degraded_accuracy", "warn"),
        earth_orientation_table.set(alternate),
    ):
        # Independent public-coordinate negative control proves this fixture
        # distinguishes the ambient table, without depending on a cache/version.
        assert evaluator.at_pairs([0], [midpoint])[0] < 0.0
        rows, mismatches = _sign_intervals(*arguments)
        assert earth_orientation_table.get() is alternate
        assert conf.auto_download is True
        assert conf.iers_degraded_accuracy == "warn"
    assert earth_orientation_table.get() is original_table
    assert (conf.auto_download, conf.iers_degraded_accuracy) == original_config
    assert mismatches == 0
    assert rows == baseline_rows


@pytest.mark.parametrize("containers", ["lists", "tuples", "mixed", "permuted"])
def test_scan_initial_partition_includes_owned_frozen_bounds(
    monkeypatch: pytest.MonkeyPatch, containers: str
) -> None:
    """Observe the real initial loop without running refinement or a solve."""
    from contextlib import nullcontext
    from fractions import Fraction

    import numpy as np

    from radiosim.core.mmode import frame as frame_module
    from radiosim.core.mmode import time as time_module

    lower, upper = Fraction(-1, 8192), Fraction(7, 8192)
    a, b, c = Fraction(1, 10000), Fraction(1, 9000), Fraction(1, 8000)

    class Grid:
        horizon_domain = (lower, upper)
        sidereal_samples = 2

        def center_turn(self, index: int) -> Fraction:
            return (Fraction(0), Fraction(2, 8192))[index]

        def exposure_turns(self, index: int) -> tuple[Fraction, Fraction]:
            return (
                (Fraction(-1, 16384), Fraction(1, 16384)),
                (Fraction(3, 16384), Fraction(5, 16384)),
            )[index]

    class InitialPartitionCaptured(Exception):
        pass

    observed: list[Fraction] = []
    allocation_shapes: list[tuple[int, int]] = []

    class AllocationObserver:
        def empty(self, shape: tuple[int, int], **keywords: Any) -> Any:
            allocation_shapes.append(shape)
            return np.empty(shape, **keywords)

        def __getattr__(self, name: str) -> Any:
            return getattr(np, name)

    class RecordingTrajectory:
        direction_count = 3

        def __init__(self, *_arguments: Any) -> None:
            pass

        def at_common_turn(self, turn: Fraction) -> Any:
            assert type(turn) is Fraction
            observed.append(turn)
            values = np.ones(3, dtype=np.float64)
            if turn == upper:
                raise InitialPartitionCaptured
            return values

    owned: Any = [
        [a, b, lower, upper],
        [],
        [
            c,
            Fraction(1, 8192),
            a,
            lower - Fraction(1, 16384),
            upper + Fraction(1, 16384),
        ],
    ]
    direction_ids = ["first", "root-free", "third"]
    ra, dec = [0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]
    if containers == "tuples":
        owned = tuple(tuple(row) for row in owned)
    elif containers == "mixed":
        owned = (tuple(owned[0]), owned[1], tuple(owned[2]))
    elif containers == "permuted":
        owned = [list(reversed(row)) for row in reversed(owned)]
        direction_ids.reverse()
        ra.reverse()
        dec.reverse()
    original = [(id(row), [(id(bound), bound) for bound in row]) for row in owned]
    # Literal grid inventory is independent of production spacing/helpers.
    expected = sorted(
        [Fraction(n, 16384) for n in (-2, -1, 0, 1, 2, 3, 4, 5, 6, 10, 14)] + [a, b, c]
    )
    monkeypatch.setattr(frame_module, "_OperationalTrajectory", RecordingTrajectory)
    monkeypatch.setattr(frame_module, "np", AllocationObserver())
    monkeypatch.setattr(time_module, "installed_iers_context", nullcontext)
    fake_frame: Any = object()
    with pytest.raises(InitialPartitionCaptured):
        _ = frame_module.scan_operational_horizon(
            frame=fake_frame,
            grid=Grid(),
            ra_rad=np.asarray(ra),
            dec_rad=np.asarray(dec),
            frozen_root_bounds=owned,
            direction_ids=direction_ids,
        )
    # The final callback is recorded before stopping; no terminal scan is claimed.
    assert observed == expected, "initial partition omitted frozen bounds"
    # No unvisited upper-exterior boundary may hide behind the final sentinel.
    assert allocation_shapes == [(14, 3)]
    assert len(observed) == len(set(observed)) == 14
    assert [
        (id(row), [(id(bound), bound) for bound in row]) for row in owned
    ] == original


def test_frame_certificate_passes_original_owned_root_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller preserves root ownership and ordering before scan filtering."""
    from fractions import Fraction
    from types import SimpleNamespace

    import numpy as np

    from radiosim.core.mmode import solver
    from radiosim.core.mmode.frame import FrozenHorizonTrajectory, HorizonRootEnclosure

    lower, upper = Fraction(-1, 8192), Fraction(7, 8192)
    first = HorizonRootEnclosure(Fraction(1, 10000), Fraction(1, 9000), "rising", 0.0)
    second = HorizonRootEnclosure(Fraction(1, 10000), Fraction(1, 8000), "setting", 0.0)
    exterior = HorizonRootEnclosure(
        lower - Fraction(1, 16384), upper + Fraction(1, 16384), "rising", 0.0
    )
    assert first.turn_lo == second.turn_lo and first.turn_lo is not second.turn_lo
    trajectories = (
        FrozenHorizonTrajectory(1.0, 0.0, 0.0, "synthetic", (first,)),
        FrozenHorizonTrajectory(0.0, 0.0, 1.0, "synthetic", ()),
        FrozenHorizonTrajectory(1.0, 0.0, 0.0, "synthetic", (second, exterior)),
    )
    directions: Any = [
        SimpleNamespace(
            direction_id=identifier,
            cirs_direction=object(),
            icrs_ra_rad=ra,
            icrs_dec_rad=dec,
        )
        for identifier, ra, dec in (
            ("first", 0.1, -0.1),
            ("empty", 0.2, -0.2),
            ("last", 0.3, -0.3),
        )
    ]
    frame: Any = object()
    grid: Any = SimpleNamespace(horizon_domain=(lower, upper))
    context: Any = object()
    frozen_calls: list[Any] = []
    scan_calls: list[dict[str, Any]] = []
    original_roots = tuple(trajectory.roots for trajectory in trajectories)
    original_endpoints = tuple(
        tuple((root, root.turn_lo, root.turn_hi) for root in trajectory.roots)
        for trajectory in trajectories
    )

    class CallerCaptured(Exception):
        pass

    def frozen(
        received_frame: Any, direction: Any, **domain: Any
    ) -> FrozenHorizonTrajectory:
        assert received_frame is frame
        assert domain == {"horizon_lo": lower, "horizon_hi": upper}
        frozen_calls.append(direction)
        return trajectories[len(frozen_calls) - 1]

    def scan(**arguments: Any) -> Any:
        scan_calls.append(arguments)
        raise CallerCaptured

    monkeypatch.setattr(solver, "frozen_horizon_trajectory", frozen)
    monkeypatch.setattr(solver, "scan_operational_horizon", scan)
    with pytest.raises(CallerCaptured):
        _ = solver.build_frame_certificate(
            grid=grid,
            frame=frame,
            context=context,
            directions=directions,
            beam_peak_ceiling=1.0,
            input_identity_sha256="1" * 64,
        )
    assert len(frozen_calls) == 3 and len(scan_calls) == 1
    assert all(
        actual is row.cirs_direction
        for actual, row in zip(frozen_calls, directions, strict=True)
    )
    captured = scan_calls[0]
    assert captured["frame"] is frame and captured["grid"] is grid
    assert captured["direction_ids"] == ["first", "empty", "last"]
    assert np.array_equal(captured["ra_rad"], np.asarray([0.1, 0.2, 0.3]))
    assert np.array_equal(captured["dec_rad"], np.asarray([-0.1, -0.2, -0.3]))
    assert len(captured["frozen_root_bounds"]) == 3
    for trajectory, roots, endpoints, passed in zip(
        trajectories,
        original_roots,
        original_endpoints,
        captured["frozen_root_bounds"],
        strict=True,
    ):
        assert trajectory.roots is roots
        assert len(passed) == 2 * len(endpoints)
        for index, (root, lo, hi) in enumerate(endpoints):
            assert trajectory.roots[index] is root
            assert root.turn_lo is lo and root.turn_hi is hi
            assert passed[2 * index] is lo and passed[2 * index + 1] is hi


@pytest.mark.parametrize(
    "case",
    [
        "outer_generator",
        "outer_mapping",
        "outer_set",
        "outer_none",
        "inner_generator",
        "inner_mapping",
        "inner_set",
        "inner_none",
        "boolean",
        "integer",
        "float",
        "string",
        "fraction_subclass",
        "missing_direction",
        "extra_direction",
        "valid_empty_list",
        "valid_empty_tuple",
    ],
)
def test_scan_validates_frozen_bound_inputs_before_evaluation(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Reject malformed ownership inputs without evaluating the trajectory."""
    from contextlib import nullcontext
    from fractions import Fraction

    from radiosim.core.mmode import frame as frame_module
    from radiosim.core.mmode import time as time_module

    class FractionSubclass(Fraction):
        pass

    cases: dict[str, Any] = {
        "outer_generator": (row for row in [()]),
        "outer_mapping": {(): None},
        "outer_set": {()},
        "outer_none": None,
        "inner_generator": [(bound for bound in [Fraction(1, 8)])],
        "inner_mapping": [{Fraction(1, 8): None}],
        "inner_set": [{Fraction(1, 8)}],
        "inner_none": [None],
        "boolean": [[True]],
        "integer": [[1]],
        "float": [[0.125]],
        "string": [["1/8"]],
        "fraction_subclass": [[FractionSubclass(1, 8)]],
        "missing_direction": [],
        "extra_direction": [[], []],
        "valid_empty_list": [[]],
        "valid_empty_tuple": ((),),
    }

    class Grid:
        horizon_domain = (Fraction(0), Fraction(1, 4096))
        sidereal_samples = 1

        def center_turn(self, _index: int) -> Fraction:
            return Fraction(0)

        def exposure_turns(self, _index: int) -> tuple[Fraction, Fraction]:
            return Fraction(0), Fraction(1, 8192)

    class InitialEvaluationReached(Exception):
        pass

    evaluated: list[Fraction] = []

    class RecordingTrajectory:
        direction_count = 1

        def __init__(self, *_arguments: Any) -> None:
            pass

        def at_common_turn(self, turn: Fraction) -> Any:
            evaluated.append(turn)
            raise InitialEvaluationReached

    monkeypatch.setattr(frame_module, "_OperationalTrajectory", RecordingTrajectory)
    monkeypatch.setattr(time_module, "installed_iers_context", nullcontext)
    fake_frame: Any = object()

    def invoke() -> None:
        _ = frame_module.scan_operational_horizon(
            frame=fake_frame,
            grid=Grid(),
            ra_rad=[0.0],
            dec_rad=[0.0],
            frozen_root_bounds=cases[case],
            direction_ids=["only"],
        )

    if case.startswith("valid_empty"):
        with pytest.raises(InitialEvaluationReached):
            invoke()
        assert evaluated == [Fraction(0)]
    else:
        with pytest.raises(ValueError):
            invoke()
        assert evaluated == []
