"""Tier 7G: the ``Z`` term's mathematics, scalings, flags, and Faraday boundary.

``Tier7JonesSciencePlan.md`` Section 20.8.  The ionosphere is one factor with
two physically distinct halves sharing one electron column:

.. code-block:: text

    Z_p(s, nu) = exp( i phi_TEC ) * F( psi_F )

    phi_TEC = -2 pi k_TEC sTEC / nu          k_TEC = 40.308e16 / c
    psi_F   = RM_ion lambda^2
    sTEC(s) = VTEC / cos( arcsin( R_E cos(el) / (R_E + h) ) )

Invariants asserted here: **I2** (every declared flag is numerically true, and
every declared ``False`` has a witness), **I3** (the direction-batched shape),
**I4** (a positive excess column is a *negative* phase), and **I8** (the sky's
intrinsic rotation measure and the ionospheric one compose exactly and are not
double-counted), plus Section 20.8's own statements: the ``1/nu`` and
``1/nu^2`` laws asserted **separately**, unitarity everywhere, the
antenna-common/direction-varying behaviour of a uniform screen, and the
antenna-differential behaviour of a gradient.

Oracles are written out in the test body rather than imported from the module
under test (Section 29.1).  The slant factor gets a genuinely independent one:
a ray-sphere intersection in Cartesian coordinates, which shares no line of
trigonometry with the closed form.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.jones.directions import DirectionBatch
from radiosim.core.jones.ionosphere import (
    EARTH_RADIUS_M,
    TEC_PHASE_CONSTANT_HZ_PER_TECU,
    IonosphereJones,
    ResolvedTecModel,
    faraday_angle_rad,
    slant_factor,
)
from radiosim.core.jones.polarization_leakage import (
    LeakageCoefficient,
    PolarizationLeakageJones,
)
from radiosim.core.jones_errors import InvalidJonesConfigError
from radiosim.core.polarization import coherency_to_stokes, stokes_to_coherency
from radiosim.core.sky.containers.constants import C_LIGHT

_BACKEND = get_backend("numpy")

#: The HERA site the shipped fixture uses.
_SITE_LATITUDE_RAD = math.radians(-30.72152)

#: A two-antenna East-West layout in metres, long enough that a TEC gradient
#: separates the two pierce points measurably.
_POSITIONS = np.array([[0.0, 0.0, 0.0], [3000.0, 1000.0, 0.0]], dtype=np.float64)

_SHELL_HEIGHT_M = 350_000.0


# ---------------------------------------------------------------------------
# Oracles, transcribed from Section 20.8 or built independently
# ---------------------------------------------------------------------------


def plan_slant_factor(alt_rad: np.ndarray, shell_height_m: float) -> np.ndarray:
    """Section 20.8's mapping, transcribed with the arcsine written out."""
    return 1.0 / np.cos(
        np.arcsin(EARTH_RADIUS_M * np.cos(alt_rad) / (EARTH_RADIUS_M + shell_height_m))
    )


def ray_sphere_slant_factor(alt_rad: np.ndarray, shell_height_m: float) -> np.ndarray:
    """An independent oracle: intersect the ray with the shell in Cartesians.

    Put the antenna at ``(0, R_E)`` in the plane containing the line of sight,
    point a unit ray at elevation ``el``, solve ``|p + t d| = R_E + h`` for the
    positive root, and read the secant of the angle between the ray and the
    local radial at the pierce point.  No trigonometric identity is reused, so a
    sign or an ``arcsin``/``arccos`` slip in the closed form would show up here.
    """
    altitude = np.atleast_1d(np.asarray(alt_rad, dtype=np.float64))
    origin = np.stack(
        [np.zeros_like(altitude), np.full_like(altitude, EARTH_RADIUS_M)], axis=-1
    )
    direction = np.stack([np.cos(altitude), np.sin(altitude)], axis=-1)
    shell_radius = EARTH_RADIUS_M + shell_height_m
    b = 2.0 * np.sum(origin * direction, axis=-1)
    c = np.sum(origin * origin, axis=-1) - shell_radius**2
    distance = 0.5 * (-b + np.sqrt(b * b - 4.0 * c))
    pierce = origin + distance[..., None] * direction
    radial = pierce / np.linalg.norm(pierce, axis=-1, keepdims=True)
    return 1.0 / np.sum(radial * direction, axis=-1)


def plan_dispersive_phase(
    slant_tec_tecu: np.ndarray, frequency_hz: float
) -> np.ndarray:
    """Section 20.8's ``phi_TEC``, transcribed with the constant written out."""
    return -2.0 * math.pi * 1.3445e9 * np.asarray(slant_tec_tecu) / frequency_hz


def faraday_matrix(angle_rad: float) -> np.ndarray:
    """The field rotation ``F(psi) = R(psi)^T``, written out.

    ``R``, the receptor and parallactic rotation, rotates the *frame* and so
    lowers the observed polarization angle.  Faraday rotation rotates the
    *field* and raises it, which is the transpose; see the module docstring of
    :mod:`radiosim.core.jones.ionosphere` and invariant I8 below.
    """
    return np.array(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _directions(
    alt_deg: np.ndarray,
    az_deg: np.ndarray | None = None,
) -> DirectionBatch:
    alt = np.radians(np.atleast_1d(np.asarray(alt_deg, dtype=np.float64)))
    if az_deg is None:
        az = np.linspace(0.0, 2.0 * np.pi, alt.size, endpoint=False)
    else:
        az = np.radians(np.atleast_1d(np.asarray(az_deg, dtype=np.float64)))
    return DirectionBatch.from_horizontal(
        alt_rad=alt,
        az_rad=az,
        dir_l=np.cos(alt) * np.sin(az),
        dir_m=np.cos(alt) * np.cos(az),
        dir_n=np.sin(alt),
        latitude_rad=_SITE_LATITUDE_RAD,
        local_sidereal_time_rad=0.0,
    )


def _term(
    *,
    vertical_tec_tecu: float = 12.0,
    gradient_east_tecu_per_km: float = 0.0,
    gradient_north_tecu_per_km: float = 0.0,
    rotation_measures_rad_m2: tuple[float, ...] = (0.0, 0.0),
    minimum_elevation_deg: float = 0.0,
    shell_height_m: float = _SHELL_HEIGHT_M,
) -> IonosphereJones:
    return IonosphereJones(
        tec_model=ResolvedTecModel(
            vertical_tec_tecu=vertical_tec_tecu,
            gradient_east_tecu_per_km=gradient_east_tecu_per_km,
            gradient_north_tecu_per_km=gradient_north_tecu_per_km,
        ),
        antenna_positions_enu_m=_POSITIONS,
        shell_height_m=shell_height_m,
        rotation_measures_rad_m2=np.array(rotation_measures_rad_m2, dtype=np.float64),
        minimum_elevation_deg=minimum_elevation_deg,
    )


def _evaluate(
    term: IonosphereJones,
    directions: DirectionBatch,
    *,
    antenna_idx: int = 0,
    frequency_hz: float = 1.5e8,
) -> np.ndarray:
    return np.asarray(
        term.compute_jones_batch(
            antenna_idx=antenna_idx,
            directions=directions,
            frequency_hz=frequency_hz,
            freq_idx=0,
            time_mjd=60_676.0,
            time_idx=0,
            backend=_BACKEND,
            dtype=np.complex128,
        )
    )


# ---------------------------------------------------------------------------
# The dispersive constant and the slant mapping
# ---------------------------------------------------------------------------


def test_the_tec_phase_constant_is_the_published_one() -> None:
    """``k_TEC`` equals ``40.308e16 / c``, which is Section 20.8's ``1.3445e9``.

    Asserted as a derivation and as a number: the excess path is
    ``40.308 TEC / nu^2`` metres (TMS eq. 13.128) and one TECU is ``1e16``
    electrons m^-2, so a transcription slip in either would show here rather
    than as a quietly wrong ionosphere.
    """
    assert TEC_PHASE_CONSTANT_HZ_PER_TECU == pytest.approx(40.308e16 / C_LIGHT, rel=0.0)
    assert TEC_PHASE_CONSTANT_HZ_PER_TECU == pytest.approx(1.3445e9, rel=5e-5)


def test_the_slant_factor_matches_an_independent_ray_sphere_intersection() -> None:
    """The thin-shell mapping, against Cartesian geometry, to machine precision."""
    altitudes = np.radians(np.arange(0.5, 90.01, 0.5))

    computed = slant_factor(altitudes, shell_height_m=_SHELL_HEIGHT_M)

    np.testing.assert_allclose(
        computed,
        ray_sphere_slant_factor(altitudes, _SHELL_HEIGHT_M),
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        computed,
        plan_slant_factor(altitudes, _SHELL_HEIGHT_M),
        rtol=1e-12,
        atol=0.0,
    )


def test_the_slant_factor_is_one_at_zenith_and_bounded_at_the_horizon() -> None:
    """Section 20.8's own numbers, and the reason ``Z`` cannot diverge.

    ``T``'s ``1 / sin(el)`` grows without bound; the thin-shell factor does not,
    because the shell is a sphere of finite radius above a sphere.  About 3.14
    at the horizon for a 350 km shell.
    """
    assert float(slant_factor(0.5 * np.pi, shell_height_m=_SHELL_HEIGHT_M)) == 1.0
    horizon = float(slant_factor(0.0, shell_height_m=_SHELL_HEIGHT_M))
    assert 3.13 < horizon < 3.15
    # Monotone in elevation: a lower line of sight crosses more electrons.
    altitudes = np.radians(np.arange(0.0, 90.01, 1.0))
    factors = slant_factor(altitudes, shell_height_m=_SHELL_HEIGHT_M)
    assert np.all(np.diff(factors) < 0.0)


def test_a_higher_shell_gives_a_smaller_slant_factor() -> None:
    """The shell height is a real parameter, not a decorative one."""
    low = float(slant_factor(np.radians(10.0), shell_height_m=250_000.0))
    high = float(slant_factor(np.radians(10.0), shell_height_m=450_000.0))
    assert high < low


# ---------------------------------------------------------------------------
# The two scalings, asserted separately (Section 20.8)
# ---------------------------------------------------------------------------


def test_the_dispersive_phase_scales_exactly_as_one_over_frequency() -> None:
    """``phi_TEC nu`` is constant across the band, to 1e-12 relative."""
    term = _term(vertical_tec_tecu=18.0)
    directions = _directions(np.array([25.0, 55.0, 85.0]))
    frequencies = np.linspace(5.0e7, 3.0e8, 12)

    products = np.array(
        [
            term.dispersive_phase_rad(0, directions, float(frequency)) * frequency
            for frequency in frequencies
        ]
    )

    for column in products.T:
        assert np.std(column) / abs(np.mean(column)) < 1e-12


def test_the_faraday_angle_scales_exactly_as_one_over_frequency_squared() -> None:
    """``psi_F nu^2`` is constant across the band: the ``lambda^2`` law."""
    term = _term(rotation_measures_rad_m2=(0.7, -0.4))
    frequencies = np.linspace(5.0e7, 3.0e8, 12)

    products = np.array(
        [
            term.faraday_angle_rad(0, float(frequency)) * frequency**2
            for frequency in frequencies
        ]
    )

    assert np.std(products) / abs(np.mean(products)) < 1e-12
    # And the closed form itself, at one frequency.
    wavelength = C_LIGHT / 1.5e8
    assert term.faraday_angle_rad(0, 1.5e8) == pytest.approx(
        0.7 * wavelength**2, rel=1e-15
    )
    assert faraday_angle_rad(0.7, 1.5e8) == pytest.approx(0.7 * wavelength**2, rel=0.0)


def test_the_two_halves_are_separable_because_their_scalings_differ() -> None:
    """A ``1/nu`` phase and a ``1/nu^2`` rotation cannot be confused for one another.

    Halving the frequency doubles the dispersive phase and *quadruples* the
    rotation angle.  That is the observational statement that ``Z`` carries two
    effects and not one.
    """
    term = _term(vertical_tec_tecu=10.0, rotation_measures_rad_m2=(0.5, 0.5))
    directions = _directions(np.array([60.0]))

    high_phase = float(term.dispersive_phase_rad(0, directions, 3.0e8)[0])
    low_phase = float(term.dispersive_phase_rad(0, directions, 1.5e8)[0])
    high_angle = term.faraday_angle_rad(0, 3.0e8)
    low_angle = term.faraday_angle_rad(0, 1.5e8)

    assert low_phase / high_phase == pytest.approx(2.0, rel=1e-12)
    assert low_angle / high_angle == pytest.approx(4.0, rel=1e-12)


def test_a_positive_column_produces_a_negative_phase() -> None:
    """Invariant I4: the tier's one sign convention, on ``Z``.

    RadioSim's geometric phase is ``exp(-2 pi i b.s)``, so every propagation
    term makes a positive excess path a *negative* phase.  A positive electron
    column advances the phase velocity, and the sign that reaches the matrix is
    the same one ``Kd``, ``Rc`` and ``T`` use.
    """
    term = _term(vertical_tec_tecu=25.0)
    directions = _directions(np.array([40.0, 80.0]))

    phase = term.dispersive_phase_rad(0, directions, 1.2e8)

    assert np.all(phase < 0.0)
    np.testing.assert_allclose(
        phase,
        plan_dispersive_phase(term.slant_tec_tecu(0, directions), 1.2e8),
        rtol=1e-4,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# The matrix itself: shape, unitarity, and the declared flags (I2, I3)
# ---------------------------------------------------------------------------


def test_the_batch_has_one_matrix_per_direction() -> None:
    """Invariant I3: ``Z`` is direction-dependent, so it returns ``(n_dir, 2, 2)``."""
    term = _term()
    directions = _directions(np.linspace(15.0, 85.0, 9))

    block = _evaluate(term, directions)

    assert block.shape == (9, 2, 2)
    assert block.dtype == np.complex128
    assert term.is_direction_dependent is True


def test_the_matrix_is_the_transcribed_product_of_a_phase_and_a_rotation() -> None:
    """The whole of Section 20.8's ``Z``, element by element."""
    term = _term(vertical_tec_tecu=14.0, rotation_measures_rad_m2=(1.3, 0.0))
    directions = _directions(np.array([20.0, 50.0, 80.0]))
    frequency = 1.1e8

    block = _evaluate(term, directions, frequency_hz=frequency)

    phase = term.dispersive_phase_rad(0, directions, frequency)
    angle = term.faraday_angle_rad(0, frequency)
    for index in range(directions.n_dir):
        expected = np.exp(1j * phase[index]) * faraday_matrix(angle)
        np.testing.assert_allclose(block[index], expected, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("rotation_measure", [0.0, 0.9, -2.5])
@pytest.mark.parametrize("vertical_tec", [0.0, 5.0, 40.0])
def test_z_is_unitary_for_every_swept_parameter(
    rotation_measure: float, vertical_tec: float
) -> None:
    """Invariant I2 for the one flag that is a genuine constant on ``Z``.

    A scalar phase times a real rotation preserves power: the ionosphere delays
    and rotates the field, it does not absorb it.  This is exactly the property
    that separates ``Z`` from ``T``'s opacity.
    """
    term = _term(
        vertical_tec_tecu=vertical_tec,
        rotation_measures_rad_m2=(rotation_measure, rotation_measure),
    )
    directions = _directions(np.linspace(10.0, 88.0, 7))

    block = _evaluate(term, directions)

    assert term.is_unitary() is True
    for matrix in block:
        np.testing.assert_allclose(
            matrix @ matrix.conj().T, np.eye(2), rtol=0.0, atol=1e-13
        )


def test_the_scalar_and_diagonal_flags_are_true_exactly_without_faraday() -> None:
    """Invariant I2 in both directions: a declared ``True`` holds numerically ...

    ... and a declared ``False`` has a witness.  Without a rotation measure
    ``Z`` is ``exp(i phi) I2``, which is scalar and diagonal; with one it is
    neither, and the off-diagonal entries are the witness.
    """
    directions = _directions(np.linspace(20.0, 80.0, 5))

    scalar_term = _term(rotation_measures_rad_m2=(0.0, 0.0))
    assert scalar_term.is_scalar() is True
    assert scalar_term.is_diagonal() is True
    for matrix in _evaluate(scalar_term, directions):
        np.testing.assert_allclose(matrix, matrix[0, 0] * np.eye(2), rtol=0.0, atol=0.0)

    rotating_term = _term(rotation_measures_rad_m2=(0.8, 0.8))
    assert rotating_term.is_scalar() is False
    assert rotating_term.is_diagonal() is False
    off_diagonal = np.array(
        [abs(matrix[0, 1]) for matrix in _evaluate(rotating_term, directions)]
    )
    assert float(np.min(off_diagonal)) > 1e-3


def test_a_zero_screen_is_the_identity_and_says_so() -> None:
    """R7's condition, computed from the resolved numbers rather than declared."""
    empty = _term(vertical_tec_tecu=0.0, rotation_measures_rad_m2=(0.0, 0.0))
    assert empty.is_identity() is True
    for matrix in _evaluate(empty, _directions(np.array([30.0, 70.0]))):
        np.testing.assert_allclose(matrix, np.eye(2), rtol=0.0, atol=0.0)

    assert _term(vertical_tec_tecu=1e-9).is_identity() is False
    assert _term(rotation_measures_rad_m2=(0.0, 1e-9)).is_identity() is False
    assert (
        _term(vertical_tec_tecu=0.0, gradient_east_tecu_per_km=0.01).is_identity()
        is False
    )


def test_the_frequency_dependence_flag_tracks_the_resolved_screen() -> None:
    """A term with no electrons and no rotation is not chromatic, and says so."""
    assert _term().is_frequency_dependent is True
    assert (
        _term(
            vertical_tec_tecu=0.0, rotation_measures_rad_m2=(0.3, 0.0)
        ).is_frequency_dependent
        is True
    )
    assert (
        _term(
            vertical_tec_tecu=0.0, rotation_measures_rad_m2=(0.0, 0.0)
        ).is_frequency_dependent
        is False
    )


# ---------------------------------------------------------------------------
# What a uniform screen does, and what a gradient does
# ---------------------------------------------------------------------------


def test_a_uniform_screen_is_antenna_common_but_direction_varying() -> None:
    """Section 20.8's discriminating statement, both halves of it.

    A constant vertical column gives every antenna the same phase for a given
    direction -- so a single source at zenith changes no visibility at all --
    while the phase still varies across a wide field through the slant factor.
    """
    term = _term(vertical_tec_tecu=20.0)
    wide = _directions(np.array([15.0, 45.0, 89.0]))

    first = term.dispersive_phase_rad(0, wide, 1.5e8)
    second = term.dispersive_phase_rad(1, wide, 1.5e8)

    np.testing.assert_array_equal(first, second)
    assert float(np.ptp(first)) > 1.0


def test_a_gradient_separates_the_two_antennas() -> None:
    """The minimal model with a closure-visible effect, and why it is not constant.

    The pierce points of two antennas are separated by their baseline plus the
    obliquity of the two lines of sight, so a gradient makes the column -- and
    therefore the phase -- differ between them.  A constant screen cannot do
    that at all, which is why both models exist.
    """
    gradient = _term(vertical_tec_tecu=20.0, gradient_east_tecu_per_km=0.5)
    directions = _directions(np.array([30.0, 60.0]), np.array([90.0, 90.0]))

    first = gradient.slant_tec_tecu(0, directions)
    second = gradient.slant_tec_tecu(1, directions)

    assert np.all(np.abs(first - second) > 1e-6)
    # The antenna 3 km further East sees the larger column under an eastward
    # gradient, and the difference is the baseline projection, not noise.
    assert np.all(second > first)


def test_the_pierce_point_offset_grows_towards_the_horizon() -> None:
    """A near-zenith line of sight pierces overhead; a low one, hundreds of km away."""
    from radiosim.core.jones.ionosphere import pierce_point_offset_m

    east_high, north_high = pierce_point_offset_m(
        np.radians(85.0), np.radians(90.0), shell_height_m=_SHELL_HEIGHT_M
    )
    east_low, north_low = pierce_point_offset_m(
        np.radians(10.0), np.radians(90.0), shell_height_m=_SHELL_HEIGHT_M
    )

    assert abs(float(north_high)) < 1.0e-6 * EARTH_RADIUS_M
    assert float(east_high) < float(east_low)
    assert float(east_low) > 500_000.0
    assert abs(float(north_low)) < 1.0e-6 * EARTH_RADIUS_M


def test_a_per_antenna_rotation_measure_reaches_only_that_antenna() -> None:
    """The Faraday sub-block's per-antenna override is a real per-antenna value."""
    term = _term(rotation_measures_rad_m2=(0.0, 1.5))

    assert term.faraday_angle_rad(0, 1.5e8) == 0.0
    assert term.faraday_angle_rad(1, 1.5e8) != 0.0
    with pytest.raises(IndexError, match="antenna rows"):
        term.faraday_angle_rad(2, 1.5e8)


# ---------------------------------------------------------------------------
# I8 -- Faraday composition, and no double count (defect D18)
# ---------------------------------------------------------------------------


def _polarization_angle(coherency: np.ndarray) -> float:
    """Return ``chi = 0.5 atan2(U, Q)`` from a 2x2 coherency matrix."""
    _, stokes_q, stokes_u, _ = coherency_to_stokes(coherency)
    return 0.5 * math.atan2(float(np.real(stokes_u)), float(np.real(stokes_q)))


def test_the_sky_and_the_ionosphere_rotate_the_angle_additively() -> None:
    """Invariant I8: source RM, ionospheric RM, and both -- exactly composing.

    The two rotations live in different objects: the sky model rotates a
    source's own ``(Q, U)`` before the coherency matrix is built
    (``core/sky/containers/spectral.py``), and ``Z`` rotates the propagated
    field afterwards.  This test runs all three cases through the *production*
    code of each and asserts that the observed polarization angle is the sum,
    to 1e-12 -- which is the statement that they compose rather than
    double-count.
    """
    from radiosim.core.sky.containers.spectral import apply_faraday_rotation

    frequency = 1.2e8
    reference_frequency = 1.5e8
    wavelength_squared = (C_LIGHT / frequency) ** 2
    reference_wavelength_squared = (C_LIGHT / reference_frequency) ** 2
    source_rm = 3.0
    ionospheric_rm = 1.25

    stokes_q = np.array([0.6])
    stokes_u = np.array([-0.2])
    scale = np.array([1.0])
    base_angle = 0.5 * math.atan2(float(stokes_u[0]), float(stokes_q[0]))

    def sky_rotated(rotation_measure: float) -> tuple[float, float]:
        rotated_q, rotated_u = apply_faraday_rotation(
            stokes_q,
            stokes_u,
            np.array([rotation_measure]),
            frequency,
            reference_frequency,
            scale,
        )
        return float(rotated_q[0]), float(rotated_u[0])

    def observed_angle(
        rotation_measure_sky: float, rotation_measure_ion: float
    ) -> float:
        rotated_q, rotated_u = sky_rotated(rotation_measure_sky)
        coherency = np.asarray(
            stokes_to_coherency(1.0, rotated_q, rotated_u, 0.0), dtype=np.complex128
        )
        term = _term(
            vertical_tec_tecu=0.0,
            rotation_measures_rad_m2=(rotation_measure_ion, rotation_measure_ion),
        )
        jones = _evaluate(term, _directions(np.array([70.0])), frequency_hz=frequency)[
            0
        ]
        return _polarization_angle(jones @ coherency @ jones.conj().T)

    sky_only = observed_angle(source_rm, 0.0)
    ionosphere_only = observed_angle(0.0, ionospheric_rm)
    both = observed_angle(source_rm, ionospheric_rm)

    sky_shift = source_rm * (wavelength_squared - reference_wavelength_squared)
    ionospheric_shift = ionospheric_rm * wavelength_squared

    assert sky_only - base_angle == pytest.approx(sky_shift, abs=1e-12)
    assert ionosphere_only - base_angle == pytest.approx(ionospheric_shift, abs=1e-12)
    assert both - base_angle == pytest.approx(sky_shift + ionospheric_shift, abs=1e-12)
    # And the composition really is a sum and not a double application.
    assert both - base_angle == pytest.approx(
        (sky_only - base_angle) + (ionosphere_only - base_angle), abs=1e-12
    )


def test_the_faraday_rotation_leaves_stokes_i_and_v_alone() -> None:
    """A real rotation moves ``(Q, U)`` and nothing else -- the physics of it."""
    term = _term(vertical_tec_tecu=0.0, rotation_measures_rad_m2=(1.1, 1.1))
    jones = _evaluate(term, _directions(np.array([50.0])))[0]
    coherency = np.asarray(
        stokes_to_coherency(2.0, 0.4, -0.3, 0.15), dtype=np.complex128
    )

    rotated = jones @ coherency @ jones.conj().T

    before = coherency_to_stokes(coherency)
    after = coherency_to_stokes(rotated)
    assert float(np.real(after[0])) == pytest.approx(
        float(np.real(before[0])), abs=1e-14
    )
    assert float(np.real(after[3])) == pytest.approx(
        float(np.real(before[3])), abs=1e-14
    )
    linear_before = math.hypot(float(np.real(before[1])), float(np.real(before[2])))
    linear_after = math.hypot(float(np.real(after[1])), float(np.real(after[2])))
    assert linear_after == pytest.approx(linear_before, abs=1e-14)


# ---------------------------------------------------------------------------
# Commutation: where Z's rotation genuinely matters in the chain
# ---------------------------------------------------------------------------


def test_the_faraday_rotation_does_not_commute_with_leakage() -> None:
    """``Z D != D Z`` for a real leakage, which is why the chain order is physics.

    A scalar phase commutes with everything; a rotation does not.  If ``Z``'s
    two halves were ever collapsed into one scalar the test below would pass
    trivially, so it is asserted on the *matrices* rather than on the claim.
    """
    directions = _directions(np.array([60.0]))
    ionosphere = _evaluate(_term(rotation_measures_rad_m2=(0.9, 0.9)), directions)[0]
    leakage_term = PolarizationLeakageJones(
        d_terms=(
            (
                LeakageCoefficient(coefficients=(0.08 + 0.03j,)),
                LeakageCoefficient(coefficients=(-0.05 + 0.02j,)),
            ),
            (
                LeakageCoefficient(coefficients=(0.08 + 0.03j,)),
                LeakageCoefficient(coefficients=(-0.05 + 0.02j,)),
            ),
        )
    )
    leakage = np.asarray(
        leakage_term.compute_jones_batch(
            antenna_idx=0,
            directions=directions,
            frequency_hz=1.5e8,
            freq_idx=0,
            time_mjd=60_676.0,
            time_idx=0,
            backend=_BACKEND,
            dtype=np.complex128,
        )
    )[0]

    commutator = ionosphere @ leakage - leakage @ ionosphere

    assert float(np.max(np.abs(commutator))) > 1e-3


def test_the_dispersive_phase_alone_commutes_with_everything() -> None:
    """The other half of the same statement: a scalar really is a scalar."""
    directions = _directions(np.array([60.0]))
    scalar = _evaluate(_term(rotation_measures_rad_m2=(0.0, 0.0)), directions)[0]
    arbitrary = np.array([[0.3, -1.2j], [0.7 + 0.1j, 2.0]], dtype=np.complex128)

    np.testing.assert_allclose(
        scalar @ arbitrary, arbitrary @ scalar, rtol=0.0, atol=1e-15
    )


# ---------------------------------------------------------------------------
# R13, the low-elevation guard
# ---------------------------------------------------------------------------


def test_a_direction_below_the_minimum_elevation_is_rejected_with_r13() -> None:
    """R13's message, verbatim for ``Z`` (Section 24, adapted clause).

    The thin-shell factor is bounded at the horizon, so the sentence names the
    approximation rather than a divergence; everything else is R13 as written.
    """
    term = _term(minimum_elevation_deg=5.0)

    with pytest.raises(InvalidJonesConfigError) as caught:
        _evaluate(term, _directions(np.array([40.0, 3.0])))

    assert str(caught.value) == (
        "jones.Z.minimum_elevation_deg=5.0 excludes no direction, but the "
        "thin-shell mapping function is not valid below 5.0 deg; raise the "
        "minimum elevation or the horizon mask."
    )


def test_a_field_entirely_above_the_minimum_elevation_is_evaluated() -> None:
    """The guard is a floor, not a refusal to run."""
    term = _term(minimum_elevation_deg=5.0)

    block = _evaluate(term, _directions(np.array([5.0, 45.0, 89.0])))

    assert block.shape == (3, 2, 2)


def test_a_zero_minimum_elevation_accepts_every_visible_direction() -> None:
    """``0`` is the explicit way to say "I accept the low-elevation model"."""
    term = _term(minimum_elevation_deg=0.0)

    block = _evaluate(term, _directions(np.array([0.05, 30.0])))

    assert np.all(np.isfinite(block))


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shell_height_m": 0.0}, "shell_height_m must be positive"),
        (
            {"minimum_elevation_deg": 90.0},
            r"minimum_elevation_deg must be in \[0, 90\)",
        ),
        (
            {"minimum_elevation_deg": -1.0},
            r"minimum_elevation_deg must be in \[0, 90\)",
        ),
    ],
)
def test_the_constructor_rejects_an_impossible_screen(kwargs, match: str) -> None:
    """The constructor is reachable from library code that never sees a document."""
    with pytest.raises(ValueError, match=match):
        _term(**kwargs)


def test_the_rotation_measures_must_have_one_entry_per_antenna_row() -> None:
    """A term whose tables disagree with the instrument would index the wrong sky."""
    with pytest.raises(ValueError, match="one entry per"):
        IonosphereJones(
            tec_model=ResolvedTecModel(vertical_tec_tecu=5.0),
            antenna_positions_enu_m=_POSITIONS,
            shell_height_m=_SHELL_HEIGHT_M,
            rotation_measures_rad_m2=np.zeros(3),
            minimum_elevation_deg=0.0,
        )


def test_the_resolved_screen_is_in_the_terms_own_record() -> None:
    """``get_config`` reports what the term will actually apply."""
    term = _term(vertical_tec_tecu=9.0, rotation_measures_rad_m2=(0.5, 0.25))

    config = term.get_config()

    assert config["name"] == "Z"
    assert config["term_status"] == "implemented"
    assert config["vertical_tec_tecu"] == 9.0
    assert config["shell_height_m"] == _SHELL_HEIGHT_M
    assert config["rotation_measures_rad_m2"] == [0.5, 0.25]
