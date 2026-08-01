"""Tier 7G: the ``T`` term's delay, mapping functions, opacity, and flags.

``Tier7JonesSciencePlan.md`` Section 20.9:

.. code-block:: text

    T_p(s, nu) = a_opacity(s) exp( -2 pi i nu tau_trop(s) ) I2

    tau_trop(s)  = ( ZHD m_h(el) + ZWD m_w(el) ) / c
    a_opacity(s) = exp( -tau_0 / (2 sin el) )

Invariants asserted here: **I2** (declared flags verified numerically, with a
witness for each declared ``False``), **I3** (the batched shape), **I4** (a
positive delay is a negative phase), and **I10** (the opacity's power/voltage
factor of two, on a baseline of two identical antennas), plus Section 20.9's own
statements: ``T`` is scalar for every parameter combination, unitary exactly
when the opacity is disabled, exactly linear in frequency, and rejected below a
configurable minimum elevation (R13).

The three-way delay discrimination
----------------------------------
``T``, ``Kd`` and ``Z`` are the tier's three delay-like terms and they are
distinguished by two independent axes rather than by their names:

============  ==========================  ==============================
Term          Phase versus frequency      Varies with direction?
============  ==========================  ==============================
``Kd``        ``-2 pi nu tau``, linear    no -- one number per feed
``T``         ``-2 pi nu tau(el)``        yes -- through the elevation
``Z``         ``-2 pi k sTEC / nu``       yes -- through the slant column
============  ==========================  ==============================

``test_the_three_delay_like_terms_are_distinguishable`` asserts every cell of
that table on the terms themselves.

Oracles
-------
The Saastamoinen delay is checked against its published sea-level value; the
Niell mapping functions against the values his Figure 2 shows at 5 degrees, the
``1/sin(el)`` limit at high elevation, and the exact ``1`` at zenith; the day of
year against astropy over nineteen years of dates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy.time import Time

from radiosim.backends import get_backend
from radiosim.core.jones.directions import DirectionBatch
from radiosim.core.jones.troposphere import (
    NIELL_LATITUDES_DEG,
    SPEED_OF_LIGHT_M_PER_S,
    TroposphereJones,
    day_of_year_from_mjd,
    niell_mapping_function,
    saastamoinen_zenith_hydrostatic_delay_m,
    simple_mapping_function,
)
from radiosim.core.jones_errors import InvalidJonesConfigError
from radiosim.core.sky.containers.constants import C_LIGHT

_BACKEND = get_backend("numpy")

_SITE_LATITUDE_DEG = -30.72152
_SITE_HEIGHTS_M = np.array([1073.0, 1073.0], dtype=np.float64)
_TIME_MJD = 60_676.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _directions(alt_deg) -> DirectionBatch:
    alt = np.radians(np.atleast_1d(np.asarray(alt_deg, dtype=np.float64)))
    az = np.linspace(0.0, 2.0 * np.pi, alt.size, endpoint=False)
    return DirectionBatch.from_horizontal(
        alt_rad=alt,
        az_rad=az,
        dir_l=np.cos(alt) * np.sin(az),
        dir_m=np.cos(alt) * np.cos(az),
        dir_n=np.sin(alt),
        latitude_rad=math.radians(_SITE_LATITUDE_DEG),
        local_sidereal_time_rad=0.0,
    )


def _term(
    *,
    hydrostatic_m: float = 2.3,
    wet_m: float = 0.1,
    mapping_function: str = "niell",
    zenith_opacity: float | None = None,
    minimum_elevation_deg: float = 0.0,
) -> TroposphereJones:
    return TroposphereJones(
        zenith_hydrostatic_delay_m=np.full(2, hydrostatic_m),
        zenith_wet_delay_m=np.full(2, wet_m),
        mapping_function=mapping_function,
        latitude_deg=_SITE_LATITUDE_DEG,
        heights_m=_SITE_HEIGHTS_M,
        zenith_opacity=zenith_opacity,
        minimum_elevation_deg=minimum_elevation_deg,
    )


def _evaluate(
    term: TroposphereJones,
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
            time_mjd=_TIME_MJD,
            time_idx=0,
            backend=_BACKEND,
            dtype=np.complex128,
        )
    )


# ---------------------------------------------------------------------------
# The day of year, against astropy
# ---------------------------------------------------------------------------


def test_the_delay_uses_the_canonical_speed_of_light() -> None:
    """The Jones package's own ``c`` is the SI value, not a rounded one."""
    assert SPEED_OF_LIGHT_M_PER_S == float(C_LIGHT)


def test_the_day_of_year_matches_astropy_over_nineteen_years() -> None:
    """The calendar shortcut is checkable, which is what makes it allowed.

    ``day_of_year_from_mjd`` exists because an astropy ``Time`` construction
    inside the solver's ``(time, frequency)`` loop would cost more than the whole
    mapping function.  A shortcut that is not compared against the library it
    replaces is a different thing entirely, so it is compared here, across leap
    years, century boundaries and both ends of January.
    """
    samples = np.concatenate(
        [
            np.arange(55_000.0, 62_000.0, 13.0),
            np.array([58_849.0, 58_850.0, 59_214.0, 59_215.0, 60_675.9, 60_676.0]),
        ]
    )

    for mjd in samples:
        expected = Time(float(mjd), format="mjd").datetime.timetuple().tm_yday
        assert int(day_of_year_from_mjd(float(mjd))) == expected, mjd


def test_the_day_of_year_carries_the_fraction_of_the_day() -> None:
    """The seasonal sinusoid must not step at midnight."""
    assert day_of_year_from_mjd(60_676.0) == pytest.approx(
        day_of_year_from_mjd(60_676.5) - 0.5, abs=1e-12
    )


# ---------------------------------------------------------------------------
# Saastamoinen
# ---------------------------------------------------------------------------


def test_the_saastamoinen_delay_reproduces_its_published_sea_level_value() -> None:
    """About 2.31 m of dry zenith delay at standard pressure -- the number to know.

    At latitude 45 the gravity correction's cosine vanishes exactly, so the
    delay is ``0.0022768 * 1013.25`` metres and the formula can be checked by
    hand.  That is deliberately the case asserted first.
    """
    at_45 = saastamoinen_zenith_hydrostatic_delay_m(
        surface_pressure_hpa=1013.25, latitude_deg=45.0, height_m=0.0
    )
    assert at_45 == pytest.approx(0.0022768 * 1013.25, rel=1e-15)
    assert at_45 == pytest.approx(2.3070, abs=5e-4)

    at_equator = saastamoinen_zenith_hydrostatic_delay_m(
        surface_pressure_hpa=1013.25, latitude_deg=0.0, height_m=0.0
    )
    assert at_equator == pytest.approx(0.0022768 * 1013.25 / (1.0 - 0.00266), rel=1e-15)
    assert at_equator > at_45


def test_the_saastamoinen_delay_falls_with_pressure_and_rises_with_height() -> None:
    """Both dependences, in the direction the formula says."""
    sea_level = saastamoinen_zenith_hydrostatic_delay_m(
        surface_pressure_hpa=1013.25, latitude_deg=_SITE_LATITUDE_DEG, height_m=0.0
    )
    thin_air = saastamoinen_zenith_hydrostatic_delay_m(
        surface_pressure_hpa=900.0, latitude_deg=_SITE_LATITUDE_DEG, height_m=0.0
    )
    high = saastamoinen_zenith_hydrostatic_delay_m(
        surface_pressure_hpa=1013.25, latitude_deg=_SITE_LATITUDE_DEG, height_m=3000.0
    )

    assert thin_air < sea_level
    assert high > sea_level
    assert thin_air / sea_level == pytest.approx(900.0 / 1013.25, rel=1e-12)


# ---------------------------------------------------------------------------
# The mapping functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", ["hydrostatic", "wet"])
def test_every_mapping_function_is_exactly_one_at_zenith(component: str) -> None:
    """The normalization that makes a *zenith* delay meaningful at all.

    Including the height correction, which vanishes at zenith because
    ``1/sin(el)`` and the correction's own continued fraction are both ``1``
    there.  If it did not vanish, a configured zenith delay would not be the
    delay at zenith.
    """
    value = niell_mapping_function(
        0.5 * np.pi,
        component=component,
        latitude_deg=_SITE_LATITUDE_DEG,
        height_m=1073.0,
        day_of_year=200.0,
    )
    assert float(value) == pytest.approx(1.0, abs=1e-12)
    assert float(simple_mapping_function(0.5 * np.pi)) == pytest.approx(1.0, abs=1e-15)


def test_the_niell_functions_reproduce_their_published_five_degree_values() -> None:
    """Niell (1996) Figure 2: about 10.1 hydrostatic at 5 degrees, mid latitude.

    His Figure 2 shows the hydrostatic mapping function at 5 degrees oscillating
    between roughly 10.08 and 10.14 over the year for stations near 40-45
    degrees latitude, and the wet function is close to 10.75.  Both are asserted
    against those published values rather than against a value read back from
    this implementation, which is the point of an oracle.
    """
    days = np.arange(1.0, 366.0, 5.0)
    hydrostatic = np.array(
        [
            float(
                niell_mapping_function(
                    np.radians(5.0),
                    component="hydrostatic",
                    latitude_deg=45.0,
                    height_m=0.0,
                    day_of_year=day,
                )
            )
            for day in days
        ]
    )

    assert 10.05 < float(np.min(hydrostatic))
    assert float(np.max(hydrostatic)) < 10.20
    assert 0.005 < float(np.ptp(hydrostatic)) < 0.10

    wet = float(
        niell_mapping_function(np.radians(5.0), component="wet", latitude_deg=45.0)
    )
    assert wet == pytest.approx(10.75, abs=0.05)


def test_the_niell_functions_stay_below_the_flat_atmosphere_limit() -> None:
    """A curved atmosphere is thinner along the line of sight than a flat one.

    At sea level both Niell components are smaller than ``1/sin(el)`` at every
    elevation below zenith, and the gap closes as the line of sight rises -- the
    qualitative statement that separates a real mapping function from the
    elementary one.
    """
    altitudes = np.radians(np.arange(5.0, 90.0, 1.0))
    flat = simple_mapping_function(altitudes)

    for component in ("hydrostatic", "wet"):
        curved = niell_mapping_function(
            altitudes,
            component=component,
            latitude_deg=45.0,
            height_m=0.0,
            day_of_year=28.0,
        )
        assert np.all(curved < flat)
        ratio = curved / flat
        assert np.all(np.diff(ratio) > 0.0)
        assert float(ratio[-1]) == pytest.approx(1.0, abs=2e-4)


def test_the_seasonal_term_is_inverted_in_the_southern_hemisphere() -> None:
    """Niell's own treatment: half a year is added to the phase below the equator.

    The tables are indexed by *absolute* latitude, so a southern site at day
    ``d`` must reproduce the northern site at ``d + 365.25/2`` exactly.  That is
    the whole content of the inversion, and it is asserted rather than assumed.
    """
    northern = niell_mapping_function(
        np.radians(20.0),
        component="hydrostatic",
        latitude_deg=45.0,
        height_m=0.0,
        day_of_year=200.0,
    )
    southern = niell_mapping_function(
        np.radians(20.0),
        component="hydrostatic",
        latitude_deg=-45.0,
        height_m=0.0,
        day_of_year=200.0 + 0.5 * 365.25,
    )
    assert float(southern) == pytest.approx(float(northern), rel=1e-14)

    # And the seasonal term is not a no-op at that latitude.
    winter = niell_mapping_function(
        np.radians(20.0),
        component="hydrostatic",
        latitude_deg=45.0,
        height_m=0.0,
        day_of_year=28.0,
    )
    assert abs(float(winter) - float(northern)) > 1e-5


def test_the_latitude_interpolation_is_linear_between_the_tabular_rows() -> None:
    """Niell interpolates linearly in latitude and does not extrapolate."""

    def value(latitude_deg: float) -> float:
        return float(
            niell_mapping_function(
                np.radians(10.0), component="wet", latitude_deg=latitude_deg
            )
        )

    midpoint = value(37.5)
    assert midpoint == pytest.approx(0.5 * (value(30.0) + value(45.0)), rel=1e-6)
    # Outside the table the nearest row is used, so a polar and an equatorial
    # site are defined rather than extrapolated into nonsense.
    assert value(89.0) == pytest.approx(value(NIELL_LATITUDES_DEG[-1]), rel=1e-15)
    assert value(1.0) == pytest.approx(value(NIELL_LATITUDES_DEG[0]), rel=1e-15)


def test_the_height_correction_applies_only_to_the_hydrostatic_component() -> None:
    """Water vapour is not in hydrostatic equilibrium, so it gets no correction."""
    altitudes = np.radians(np.array([10.0, 30.0]))
    sea_level = niell_mapping_function(
        altitudes, component="hydrostatic", latitude_deg=45.0, height_m=0.0
    )
    elevated = niell_mapping_function(
        altitudes, component="hydrostatic", latitude_deg=45.0, height_m=2000.0
    )
    assert np.all(elevated > sea_level)

    with pytest.raises(TypeError):
        niell_mapping_function(  # type: ignore[call-arg]
            altitudes, component="wet", latitude_deg=45.0, unknown=1.0
        )


def test_an_unknown_component_is_rejected() -> None:
    with pytest.raises(ValueError, match="hydrostatic"):
        niell_mapping_function(
            np.radians(45.0),
            component="dry",  # type: ignore[arg-type]
            latitude_deg=45.0,
        )


# ---------------------------------------------------------------------------
# The matrix: shape, scalarity, the delay, and its sign (I2, I3, I4)
# ---------------------------------------------------------------------------


def test_the_batch_has_one_matrix_per_direction() -> None:
    """Invariant I3: ``T`` is direction-dependent through the elevation."""
    term = _term()
    block = _evaluate(term, _directions(np.linspace(15.0, 85.0, 6)))

    assert block.shape == (6, 2, 2)
    assert term.is_direction_dependent is True


@pytest.mark.parametrize("opacity", [None, 0.0, 0.3])
@pytest.mark.parametrize("mapping_function", ["simple", "niell"])
def test_t_is_scalar_for_every_swept_parameter(
    opacity: float | None, mapping_function: str
) -> None:
    """Invariant I2: Section 20.9's "both factors are scalars times the identity".

    Asserted exactly -- the off-diagonals are zero and the two diagonal entries
    are the same object -- rather than to a tolerance, because a scalar that is
    only nearly scalar would not commute and the whole simplification would be
    an approximation nobody declared.
    """
    term = _term(mapping_function=mapping_function, zenith_opacity=opacity)
    block = _evaluate(term, _directions(np.array([12.0, 47.0, 88.0])))

    assert term.is_scalar() is True
    assert term.is_diagonal() is True
    for matrix in block:
        assert matrix[0, 1] == 0.0
        assert matrix[1, 0] == 0.0
        assert matrix[0, 0] == matrix[1, 1]


def test_the_delay_is_the_transcribed_closed_form() -> None:
    """``tau = (ZHD m_h + ZWD m_w) / c``, with the mapping factors written out."""
    term = _term(hydrostatic_m=2.31, wet_m=0.12, mapping_function="simple")
    directions = _directions(np.array([20.0, 60.0]))

    delay = term.delay_s(0, directions, _TIME_MJD)

    expected = (
        2.31 * simple_mapping_function(directions.alt_rad)
        + 0.12 * simple_mapping_function(directions.alt_rad)
    ) / SPEED_OF_LIGHT_M_PER_S
    np.testing.assert_allclose(delay, expected, rtol=1e-15, atol=0.0)
    # About 7-8 nanoseconds at zenith, tens at low elevation: the scale of a real
    # troposphere rather than an arbitrary number.
    assert 7.0e-9 < float(np.min(delay)) < 3.0e-8


def test_a_positive_delay_produces_a_negative_phase() -> None:
    """Invariant I4, on ``T``: ``exp(-2 pi i nu tau)`` for a positive ``tau``."""
    term = _term(hydrostatic_m=2.3, wet_m=0.1, zenith_opacity=None)
    directions = _directions(np.array([35.0]))
    frequency = 1.5e8

    matrix = _evaluate(term, directions, frequency_hz=frequency)[0]

    delay = float(term.delay_s(0, directions, _TIME_MJD)[0])
    assert delay > 0.0
    assert np.angle(matrix[0, 0]) == pytest.approx(
        float(np.angle(np.exp(-2j * np.pi * frequency * delay))), abs=1e-12
    )
    assert matrix[0, 0] == pytest.approx(
        complex(np.exp(-2j * np.pi * frequency * delay)), abs=1e-14
    )


def test_the_delay_phase_is_exactly_linear_in_frequency() -> None:
    """Non-dispersive, which is what distinguishes ``T`` from ``Z``.

    The unwrapped phase divided by frequency is constant to 1e-12 across a 6:1
    band, which is a statement about the *model* and not about the numbers: the
    excess path of the neutral atmosphere does not depend on frequency.
    """
    term = _term(hydrostatic_m=2.3, wet_m=0.1)
    directions = _directions(np.array([40.0]))
    frequencies = np.linspace(5.0e7, 3.0e8, 10)
    delay = float(term.delay_s(0, directions, _TIME_MJD)[0])

    slopes = []
    for frequency in frequencies:
        matrix = _evaluate(term, directions, frequency_hz=float(frequency))[0]
        # Reconstruct the unwrapped phase from the known delay rather than from
        # the principal branch, which would wrap many times over this band.
        phase = -2.0 * np.pi * float(frequency) * delay
        assert matrix[0, 0] == pytest.approx(complex(np.exp(1j * phase)), abs=1e-12)
        slopes.append(phase / float(frequency))

    assert np.std(slopes) / abs(np.mean(slopes)) < 1e-12


def test_the_three_delay_like_terms_are_distinguishable() -> None:
    """``Kd``, ``T`` and ``Z``, separated on both of their axes at once.

    ``Kd`` and ``T`` are both linear in frequency and ``Z`` is not; ``T`` and
    ``Z`` both vary with direction and ``Kd`` does not.  Two axes, three terms,
    no pair sharing both cells -- which is why the three exist separately rather
    than as one configurable delay.
    """
    from radiosim.core.jones.delay import DelayJones
    from radiosim.core.jones.ionosphere import IonosphereJones, ResolvedTecModel

    directions = _directions(np.array([25.0, 75.0]))
    low, high = 1.0e8, 2.0e8

    troposphere = _term(hydrostatic_m=2.3, wet_m=0.1)
    instrumental = DelayJones(delays_s=np.full((2, 2), 5.0e-9))
    ionosphere = IonosphereJones(
        tec_model=ResolvedTecModel(vertical_tec_tecu=20.0),
        antenna_positions_enu_m=np.zeros((2, 3)),
        shell_height_m=350_000.0,
        rotation_measures_rad_m2=np.zeros(2),
        minimum_elevation_deg=0.0,
    )

    tropospheric_phase = np.array(
        [
            -2.0 * np.pi * frequency * troposphere.delay_s(0, directions, _TIME_MJD)
            for frequency in (low, high)
        ]
    )
    instrumental_phase = np.array(
        [-2.0 * np.pi * frequency * 5.0e-9 for frequency in (low, high)]
    )
    ionospheric_phase = np.array(
        [
            ionosphere.dispersive_phase_rad(0, directions, frequency)
            for frequency in (low, high)
        ]
    )

    # Axis 1 -- frequency.  Doubling the frequency doubles T's and Kd's phase
    # and halves Z's.
    assert np.allclose(tropospheric_phase[1] / tropospheric_phase[0], 2.0, rtol=1e-12)
    assert instrumental_phase[1] / instrumental_phase[0] == pytest.approx(
        2.0, rel=1e-12
    )
    assert np.allclose(ionospheric_phase[1] / ionospheric_phase[0], 0.5, rtol=1e-12)

    # Axis 2 -- direction.  T and Z differ between the two elevations; Kd cannot.
    assert abs(tropospheric_phase[0][0] - tropospheric_phase[0][1]) > 1e-3
    assert abs(ionospheric_phase[0][0] - ionospheric_phase[0][1]) > 1e-3
    assert instrumental_phase[0].shape == ()

    assert troposphere.is_direction_dependent is True
    assert ionosphere.is_direction_dependent is True
    assert instrumental.is_direction_dependent is False


# ---------------------------------------------------------------------------
# The opacity: I10, and the unitarity flag it falsifies
# ---------------------------------------------------------------------------


def test_the_opacity_is_a_voltage_factor_of_half_the_power_opacity() -> None:
    """Invariant I10, on the terms themselves.

    ``T`` is a voltage matrix and ``tau_0`` is defined on power, so each antenna
    contributes ``exp(-tau_0 / 2)`` at zenith and the visibility of a baseline of
    two identical antennas is scaled by exactly ``exp(-tau_0)``.  The factor of
    two is the single easiest sign-class error in this term, and this is the test
    that would catch it.
    """
    zenith_opacity = 0.4
    term = _term(hydrostatic_m=0.0, wet_m=0.0, zenith_opacity=zenith_opacity)
    zenith = _directions(np.array([90.0]))

    matrix = _evaluate(term, zenith)[0]

    voltage = abs(complex(matrix[0, 0]))
    assert voltage == pytest.approx(math.exp(-0.5 * zenith_opacity), rel=1e-14)
    # The baseline product -- what a visibility is actually scaled by.
    assert voltage * voltage == pytest.approx(math.exp(-zenith_opacity), rel=1e-14)


def test_the_opacity_follows_the_airmass_towards_the_horizon() -> None:
    """``exp(-tau_0 / (2 sin el))``: more atmosphere, more absorption."""
    term = _term(hydrostatic_m=0.0, wet_m=0.0, zenith_opacity=0.25)
    directions = _directions(np.array([10.0, 30.0, 90.0]))

    attenuation = term.opacity_attenuation(directions)

    expected = np.exp(-0.5 * 0.25 / np.sin(directions.alt_rad))
    np.testing.assert_allclose(attenuation, expected, rtol=1e-15, atol=0.0)
    assert attenuation[0] < attenuation[1] < attenuation[2] < 1.0


def test_t_is_unitary_exactly_when_the_opacity_is_disabled() -> None:
    """Invariant I2 in both directions for the flag that separates ``T`` from ``Z``."""
    directions = _directions(np.array([20.0, 70.0]))

    transparent = _term(zenith_opacity=None)
    assert transparent.is_unitary() is True
    for matrix in _evaluate(transparent, directions):
        np.testing.assert_allclose(
            matrix @ matrix.conj().T, np.eye(2), rtol=0.0, atol=1e-14
        )

    absorbing = _term(zenith_opacity=0.5)
    assert absorbing.is_unitary() is False
    for matrix in _evaluate(absorbing, directions):
        product = matrix @ matrix.conj().T
        assert float(np.real(product[0, 0])) < 0.999

    # A configured zero opacity really is transparent, and says so.
    assert _term(zenith_opacity=0.0).is_unitary() is True


def test_the_frequency_dependence_flag_tracks_the_resolved_delay() -> None:
    """Opacity alone is achromatic in RadioSim's model, and the flag says so."""
    assert _term(hydrostatic_m=2.3, wet_m=0.0).is_frequency_dependent is True
    assert (
        _term(hydrostatic_m=0.0, wet_m=0.0, zenith_opacity=0.2).is_frequency_dependent
        is False
    )


def test_a_transparent_atmosphere_with_no_delay_is_the_identity() -> None:
    """R7's condition, computed from the resolved numbers."""
    empty = _term(hydrostatic_m=0.0, wet_m=0.0, zenith_opacity=None)
    assert empty.is_identity() is True
    for matrix in _evaluate(empty, _directions(np.array([30.0, 70.0]))):
        np.testing.assert_allclose(matrix, np.eye(2), rtol=0.0, atol=0.0)

    assert _term(hydrostatic_m=0.0, wet_m=1e-6).is_identity() is False
    assert (
        _term(hydrostatic_m=0.0, wet_m=0.0, zenith_opacity=1e-6).is_identity() is False
    )
    assert _term(hydrostatic_m=0.0, wet_m=0.0, zenith_opacity=0.0).is_identity() is True


# ---------------------------------------------------------------------------
# R13, the low-elevation guard
# ---------------------------------------------------------------------------


def test_a_direction_below_the_minimum_elevation_is_rejected_with_r13() -> None:
    """R13's message, verbatim (Section 24).

    Both mapping functions grow without bound as ``el -> 0``.  RadioSim refuses
    rather than writing an unbounded delay into a visibility, and the message
    names the two things a user can change.
    """
    term = _term(minimum_elevation_deg=5.0)

    with pytest.raises(InvalidJonesConfigError) as caught:
        _evaluate(term, _directions(np.array([60.0, 2.0])))

    assert str(caught.value) == (
        "jones.T.minimum_elevation_deg=5.0 excludes no direction, but the mapping "
        "function diverges below 5.0 deg; raise the minimum elevation or the "
        "horizon mask."
    )


def test_a_field_entirely_above_the_minimum_elevation_is_evaluated() -> None:
    term = _term(minimum_elevation_deg=5.0)

    block = _evaluate(term, _directions(np.array([5.0, 45.0, 89.0])))

    assert np.all(np.isfinite(block))


# ---------------------------------------------------------------------------
# Constructor validation and the record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mapping_function": "vienna"}, "mapping_function must be one of"),
        ({"zenith_opacity": -0.1}, "zenith_opacity must be non-negative"),
        (
            {"minimum_elevation_deg": 91.0},
            r"minimum_elevation_deg must be in \[0, 90\)",
        ),
    ],
)
def test_the_constructor_rejects_an_impossible_atmosphere(kwargs, match: str) -> None:
    """The constructor is reachable from library code that never sees a document."""
    with pytest.raises(ValueError, match=match):
        _term(**kwargs)


def test_the_zenith_delays_must_have_one_entry_per_antenna_row() -> None:
    with pytest.raises(ValueError, match="one entry per"):
        TroposphereJones(
            zenith_hydrostatic_delay_m=np.full(2, 2.3),
            zenith_wet_delay_m=np.full(3, 0.1),
            mapping_function="niell",
            latitude_deg=_SITE_LATITUDE_DEG,
            heights_m=_SITE_HEIGHTS_M,
            zenith_opacity=None,
            minimum_elevation_deg=0.0,
        )


def test_the_resolved_atmosphere_is_in_the_terms_own_record() -> None:
    term = _term(hydrostatic_m=2.25, wet_m=0.08, zenith_opacity=0.03)

    config = term.get_config()

    assert config["name"] == "T"
    assert config["term_status"] == "implemented"
    assert config["mapping_function"] == "niell"
    assert config["zenith_opacity"] == 0.03
    assert config["zenith_hydrostatic_delay_m"] == [2.25, 2.25]


# ---------------------------------------------------------------------------
# I7 and I10 -- through the solver
# ---------------------------------------------------------------------------


def _cube(
    tmp_path, jones, *, polarized: bool = True, **section_overrides
) -> np.ndarray:
    from radiosim.core.visibility import calculate_visibility
    from tests.characterization.test_tier6_current_behavior import (
        WORKLOAD_LOCATION,
        WORKLOAD_TIME_GRID,
        _workload_point_sources,
    )
    from tests.unit.test_core.test_jones_resolution import (
        solver_components_with_jones,
    )

    instrument, beam_system, receptors, jones_terms, frequencies = (
        solver_components_with_jones(tmp_path, jones, **section_overrides)
    )
    return np.asarray(
        calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=_workload_point_sources(polarized=polarized, gaussian=False),
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=frequencies,
            backend=_BACKEND,
            receptors=receptors,
            jones_terms=jones_terms,
        )
    )


_SOLVER_DELAY_ONLY: dict[str, dict] = {
    "T": {
        "zenith_delay": {
            "kind": "saastamoinen",
            "surface_pressure_hpa": 1013.25,
            "zenith_wet_delay_m": 0.2,
        },
        "mapping_function": "niell",
        "minimum_elevation_deg": 0.0,
    }
}


def _sloped_layout(tmp_path) -> dict:
    """Return an ``instrument`` override whose two antennas differ in height.

    The shipped fixture's antennas share one ``U``, so their Saastamoinen zenith
    delays are identical and the tropospheric phase cancels exactly (see
    :func:`test_a_common_delay_cancels_on_a_flat_homogeneous_array`).  A real
    array on a slope does not have that symmetry, and this is the smallest
    instrument that breaks it.
    """
    layout = tmp_path / "sloped.txt"
    layout.write_text(
        "Name Number BeamID E N U Diameter\n"
        "ANT0 0 0 0.0 0.0 0.0 14.0\n"
        "ANT1 1 0 60.0 0.0 900.0 14.0\n"
    )
    return {"instrument": {"source": {"path": str(layout)}}}


def test_a_configured_troposphere_changes_the_visibilities(tmp_path) -> None:
    """I7, for ``T``: a real atmosphere is not the same run as none.

    Both of ``T``'s halves are exercised, because they reach a visibility by
    different routes: the opacity attenuates each antenna's voltage and survives
    on any array, while the delay is a scalar phase and survives only where the
    two antennas' delays differ -- here, because they sit 900 m apart in height.
    """
    absorbing_clean = _cube(tmp_path, None)
    absorbing = _cube(
        tmp_path,
        {
            "T": {
                "zenith_delay": {"kind": "explicit"},
                "minimum_elevation_deg": 0.0,
                "opacity": {"zenith_opacity": 0.3},
            }
        },
    )
    assert (
        float(np.max(np.abs(absorbing - absorbing_clean)))
        / float(np.max(np.abs(absorbing_clean)))
        > 1e-10
    )

    sloped = _sloped_layout(tmp_path)
    delay_clean = _cube(tmp_path, None, **sloped)
    delayed = _cube(tmp_path, _SOLVER_DELAY_ONLY, **sloped)
    assert (
        float(np.max(np.abs(delayed - delay_clean)))
        / float(np.max(np.abs(delay_clean)))
        > 1e-10
    )


def test_a_common_delay_cancels_on_a_flat_homogeneous_array(tmp_path) -> None:
    """A delay both antennas share changes no visibility at all -- exactly.

    ``T``'s delay is a scalar, so on an array whose antennas resolve to the same
    zenith delay it enters the RIME as ``e^{i phi} C_s e^{-i phi} = C_s``, per
    source and therefore per baseline.  An interferometer measures the
    *differential* delay; that is a statement about interferometry rather than
    about this implementation, and it is asserted exactly (``< 1e-14``) so that
    an implementation which accidentally made the delay antenna-dependent would
    fail here loudly rather than pass a loose bound.

    What breaks the symmetry in RadioSim's model is the antennas' own heights,
    through both the Saastamoinen formula and the Niell height correction.  A
    per-antenna *atmosphere* -- different pressures, or a turbulent screen --
    is out of scope (Section 4).
    """
    clean = _cube(tmp_path, None)
    delayed = _cube(tmp_path, _SOLVER_DELAY_ONLY)

    assert float(np.max(np.abs(delayed - clean))) / float(np.max(np.abs(clean))) < 1e-14


def test_the_two_mapping_functions_are_different_runs(tmp_path) -> None:
    """``mapping_function`` is a real choice, not a label."""
    sloped = _sloped_layout(tmp_path)
    niell = _cube(tmp_path, _SOLVER_DELAY_ONLY, **sloped)
    simple = _cube(
        tmp_path,
        {"T": {**_SOLVER_DELAY_ONLY["T"], "mapping_function": "simple"}},
        **sloped,
    )

    difference = np.max(np.abs(simple - niell)) / np.max(np.abs(niell))
    assert float(difference) > 1e-10


def test_the_opacity_scales_the_visibility_by_exp_minus_tau(tmp_path) -> None:
    """Invariant I10, end to end, on a real baseline of two identical antennas.

    With no delay configured ``T`` is a pure real attenuation, and the shipped
    fixture's two antennas are identical -- so every source's contribution is
    scaled by the product of the two antennas' voltage factors, which is the
    *power* factor ``exp(-tau_0 / sin el)``.  The sources are not exactly at
    zenith, so the assertion brackets the ratio between the zenith value
    ``exp(-tau_0)`` and the value at 30 degrees elevation, ``exp(-2 tau_0)``:
    the factor of two this term exists to get right is what puts it between
    them rather than at ``exp(-tau_0 / 2)``.
    """
    zenith_opacity = 0.3
    clean = _cube(tmp_path, None, polarized=False)
    attenuated = _cube(
        tmp_path,
        {
            "T": {
                "zenith_delay": {"kind": "explicit"},
                "mapping_function": "simple",
                "minimum_elevation_deg": 0.0,
                "opacity": {"zenith_opacity": zenith_opacity},
            }
        },
        polarized=False,
    )

    nonzero = np.abs(clean) > 1e-12 * float(np.max(np.abs(clean)))
    ratios = np.abs(attenuated[nonzero]) / np.abs(clean[nonzero])
    assert float(np.max(ratios)) <= math.exp(-zenith_opacity) + 1e-12
    assert float(np.min(ratios)) > math.exp(-2.0 * zenith_opacity)
    # Half the opacity -- the voltage convention taken as if it were the power
    # one -- would land the whole cube above this bound.
    assert float(np.max(ratios)) < math.exp(-0.5 * zenith_opacity)


# ---------------------------------------------------------------------------
# Section 29.1 Tier-1 evidence: the published coefficients, evaluated here
# ---------------------------------------------------------------------------


def test_the_niell_functions_match_the_published_coefficients_at_five_elevations() -> (
    None
):
    """Section 29.1's ``T`` row: ``m(el)`` at five elevations, from Table 3 and 4.

    The reference is built in this test body from Niell (1996) Tables 3 and 4,
    transcribed here a second time, and from his eq. (4) and eq. (6)-(7) written
    out: the seasonal sinusoid, the linear interpolation in latitude, the
    three-term continued fraction, and the height correction.  Nothing of the
    production module's arithmetic is reused, which is what makes this evidence
    rather than a restatement.
    """
    latitudes = (15.0, 30.0, 45.0, 60.0, 75.0)
    hydrostatic_average = (
        (1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3),
        (2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3),
        (62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3),
    )
    hydrostatic_amplitude = (
        (0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5),
        (0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5),
        (0.0, 9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5),
    )
    wet = (
        (5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4),
        (1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3),
        (4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2),
    )
    height_correction = (2.53e-5, 5.49e-3, 1.14e-3)

    def fraction(sine, a, b, c):
        return (1.0 + a / (1.0 + b / (1.0 + c))) / (sine + a / (sine + b / (sine + c)))

    def interpolate(table, latitude_deg):
        return float(np.interp(abs(latitude_deg), latitudes, table))

    def reference(elevation_deg, latitude_deg, day, height_m, component):
        sine = math.sin(math.radians(elevation_deg))
        if component == "wet":
            return fraction(sine, *(interpolate(row, latitude_deg) for row in wet))
        phase = 28.0 + (365.25 / 2.0 if latitude_deg < 0.0 else 0.0)
        seasonal = math.cos(2.0 * math.pi * (day - phase) / 365.25)
        coefficients = [
            interpolate(average, latitude_deg)
            + seasonal * interpolate(amplitude, latitude_deg)
            for average, amplitude in zip(
                hydrostatic_average, hydrostatic_amplitude, strict=True
            )
        ]
        value = fraction(sine, *coefficients)
        return value + (1.0 / sine - fraction(sine, *height_correction)) * (
            height_m / 1000.0
        )

    for latitude_deg, day, height_m in (
        (45.0, 28.0, 0.0),
        (-30.72152, 200.0, 1073.0),
        (12.0, 90.0, 2400.0),
    ):
        for elevation_deg in (5.0, 10.0, 20.0, 45.0, 90.0):
            for component in ("hydrostatic", "wet"):
                computed = float(
                    niell_mapping_function(
                        math.radians(elevation_deg),
                        component=component,
                        latitude_deg=latitude_deg,
                        height_m=height_m,
                        day_of_year=day,
                    )
                )
                assert computed == pytest.approx(
                    reference(elevation_deg, latitude_deg, day, height_m, component),
                    rel=1e-13,
                ), (latitude_deg, elevation_deg, component)
