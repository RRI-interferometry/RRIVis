"""Tier 7H: the ``Q`` term's two smearing envelopes and their bounds.

``Tier7JonesSciencePlan.md`` Section 20.11 (Bridle & Schwab 1999; TMS 2017
Section 6.4), with every reference value derived in the test body -- twice over,
where it matters: once from the closed form written out by hand, and once from a
numerical average of the *solver's own* phase over the channel and over the
integration.  The second oracle is what makes this file more than a restatement
of the implementation: it decorrelates the visibility the way an instrument
does, by averaging, and asks whether the analytic envelope agrees.

Invariant **I12** is asserted here in its corrected form (Section 27): ``Q <= 1``
always; ``Q > 0`` below the first sinc zero; ``Q = 1`` exactly at the phase
centre for the *bandwidth* factor, and for the *time* factor only on a baseline
with no East-West component, because RadioSim's phase centre is the fixed zenith
and the sky moves through it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.jones.baseline_errors import (
    EARTH_ROTATION_RAD_PER_S,
    SmearingFactorJones,
)
from radiosim.core.jones_errors import InvalidJonesConfigError, JonesEvaluationError
from tests.unit.test_core.test_jones_resolution import resolve_for

_BACKEND = get_backend("numpy")

#: The shipped fixture's site latitude, in radians.
_LATITUDE_RAD = math.radians(-30.72152)

#: One sample grid the tests resolve every term against.
_FREQUENCIES = np.array([1.0e8, 1.1e8], dtype=np.float64)
_WIDTHS = np.array([1.0e6, 4.0e6], dtype=np.float64)
_INTEGRATIONS = np.array([10.0, 10.0], dtype=np.float64)
_TIMES_MJD = np.array([60_676.0, 60_676.0 + 10.0 / 86_400.0], dtype=np.float64)

_SPEED_OF_LIGHT = 299_792_458.0


def _term(
    *,
    bandwidth: bool = True,
    time: bool = True,
    widths: np.ndarray = _WIDTHS,
    integrations: np.ndarray = _INTEGRATIONS,
) -> SmearingFactorJones:
    return SmearingFactorJones(
        bandwidth_smearing=bandwidth,
        time_smearing=time,
        channel_frequencies_hz=_FREQUENCIES,
        channel_widths_hz=np.asarray(widths, dtype=np.float64),
        integration_time_s=np.asarray(integrations, dtype=np.float64),
        sample_times_mjd=_TIMES_MJD,
        latitude_rad=_LATITUDE_RAD,
    )


def _directions_from_equatorial(
    hour_angle_rad: np.ndarray,
    dec_rad: np.ndarray,
    *,
    latitude_rad: float = _LATITUDE_RAD,
):
    """Return a batch built from the standard equatorial-to-ENU transform.

    Written out here rather than taken from the production helper, because the
    direction cosines are half of what the smearing envelopes are a statement
    about::

        l = -cos(dec) sin(H)
        m =  cos(lat) sin(dec) - sin(lat) cos(dec) cos(H)
        n =  sin(lat) sin(dec) + cos(lat) cos(dec) cos(H)
    """
    from radiosim.core.jones.directions import DirectionBatch

    hour_angle = np.asarray(hour_angle_rad, dtype=np.float64)
    dec = np.asarray(dec_rad, dtype=np.float64)
    sin_lat = math.sin(latitude_rad)
    cos_lat = math.cos(latitude_rad)

    dir_l = -np.cos(dec) * np.sin(hour_angle)
    dir_m = cos_lat * np.sin(dec) - sin_lat * np.cos(dec) * np.cos(hour_angle)
    dir_n = sin_lat * np.sin(dec) + cos_lat * np.cos(dec) * np.cos(hour_angle)
    altitude = np.arcsin(np.clip(dir_n, -1.0, 1.0))
    azimuth = np.arctan2(dir_l, dir_m)

    return DirectionBatch(
        alt_rad=altitude,
        az_rad=azimuth,
        dir_l=dir_l,
        dir_m=dir_m,
        dir_n=dir_n,
        ra_rad=np.mod(-hour_angle, 2.0 * np.pi),
        dec_rad=dec,
        hour_angle_rad=hour_angle,
        n_dir=int(dec.size),
    )


def _factor(
    term: SmearingFactorJones,
    *,
    uvw_wavelengths: np.ndarray,
    directions,
    freq_idx: int = 0,
    time_idx: int = 0,
    dtype: Any = np.float64,
) -> np.ndarray:
    pairs = tuple((0, index + 1) for index in range(len(uvw_wavelengths)))
    return np.asarray(
        term.compute_baseline_factor(
            baseline_pairs=pairs,
            baseline_uvw_wavelengths=np.asarray(uvw_wavelengths, dtype=np.float64),
            directions=directions,
            frequency_hz=float(_FREQUENCIES[freq_idx]),
            freq_idx=freq_idx,
            time_mjd=float(_TIMES_MJD[time_idx]),
            time_idx=time_idx,
            backend=_BACKEND,
            dtype=dtype,
        )
    )


def _residual_delay_s(uvw_wavelengths, directions, frequency_hz: float) -> np.ndarray:
    """``(u l + v m + w (n - 1)) / nu`` -- the delay the correlator did not remove."""
    uvw = np.asarray(uvw_wavelengths, dtype=np.float64)
    return (
        uvw[:, 0:1] * directions.dir_l
        + uvw[:, 1:2] * directions.dir_m
        + uvw[:, 2:3] * (directions.dir_n - 1.0)
    ) / float(frequency_hz)


def _fringe_rate_hz(uvw_wavelengths, directions) -> np.ndarray:
    """``omega_E [ u (n cos(lat) - m sin(lat)) + l (v sin(lat) - w cos(lat)) ]``."""
    uvw = np.asarray(uvw_wavelengths, dtype=np.float64)
    sin_lat = math.sin(_LATITUDE_RAD)
    cos_lat = math.cos(_LATITUDE_RAD)
    return EARTH_ROTATION_RAD_PER_S * (
        uvw[:, 0:1] * (directions.dir_n * cos_lat - directions.dir_m * sin_lat)
        + directions.dir_l * (uvw[:, 1:2] * sin_lat - uvw[:, 2:3] * cos_lat)
    )


# ---------------------------------------------------------------------------
# The closed forms, against the numerical average they approximate
# ---------------------------------------------------------------------------


def test_the_bandwidth_envelope_is_the_sinc_of_the_residual_delay() -> None:
    """``sinc(pi dnu tau_res)``, with ``tau_res`` measured from the phase centre.

    Written against the kernel's own phase, ``exp(-2 pi i (u l + v m + w (n-1)))``:
    the ``-1`` is what makes the residual delay vanish at the phase centre, and
    a ``tau = b.s/c`` written without it would make the envelope less than one
    at zenith on a baseline with any vertical component at all.
    """
    uvw = np.array([[120.0, -35.0, 4.0], [3.0, 900.0, -12.0]], dtype=np.float64)
    directions = _directions_from_equatorial(
        np.array([0.0, 0.05, -0.2, 0.4]),
        np.array([_LATITUDE_RAD, _LATITUDE_RAD + 0.1, -0.2, 0.6]),
    )
    term = _term(time=False)

    factor = _factor(term, uvw_wavelengths=uvw, directions=directions)

    delay = _residual_delay_s(uvw, directions, float(_FREQUENCIES[0]))
    expected = np.sinc(float(_WIDTHS[0]) * delay)
    np.testing.assert_allclose(factor, expected, rtol=1e-14, atol=0.0)


def test_the_bandwidth_envelope_matches_a_numerical_channel_average() -> None:
    """The independent oracle: average the phase across the channel and look.

    A top-hat channel of width ``dnu`` centred on ``nu0`` decorrelates a source
    whose residual delay is ``tau`` by exactly the mean of ``exp(-2 pi i nu tau)``
    over the channel, which is ``sinc(pi dnu tau)`` times the phase at the
    centre.  This computes that mean by quadrature over the *solver's* phase
    expression and compares its modulus to the envelope.
    """
    uvw_centre = np.array([[600.0, -220.0, 15.0]], dtype=np.float64)
    directions = _directions_from_equatorial(
        np.array([0.03, 0.12, -0.25]),
        np.array([_LATITUDE_RAD + 0.05, -0.35, 0.25]),
    )
    term = _term(time=False)
    factor = _factor(term, uvw_wavelengths=uvw_centre, directions=directions)

    centre = float(_FREQUENCIES[0])
    width = float(_WIDTHS[0])
    offsets = np.linspace(-0.5, 0.5, 20_001) * width
    channel = centre + offsets

    # (u, v, w) are in wavelengths, so they scale with frequency; the geometry
    # in metres is what stays fixed across the channel.
    uvw_metres = uvw_centre * (_SPEED_OF_LIGHT / centre)
    averaged = []
    for index in range(directions.n_dir):
        path_m = (
            uvw_metres[0, 0] * directions.dir_l[index]
            + uvw_metres[0, 1] * directions.dir_m[index]
            + uvw_metres[0, 2] * (directions.dir_n[index] - 1.0)
        )
        phases = np.exp(-2j * np.pi * channel * path_m / _SPEED_OF_LIGHT)
        mean = np.trapezoid(phases, channel) / width
        centre_phase = np.exp(-2j * np.pi * centre * path_m / _SPEED_OF_LIGHT)
        averaged.append((mean / centre_phase).real)

    np.testing.assert_allclose(factor[0], np.array(averaged), rtol=1e-8, atol=1e-12)


def test_the_time_envelope_matches_a_numerically_rotated_sky() -> None:
    """The independent oracle for the fringe rate: rotate the sky and average.

    Nothing in this test differentiates anything.  It advances the hour angle at
    the sidereal rate across the integration, rebuilds the direction cosines from
    scratch at each step, averages the kernel's phase, and compares the modulus
    of that average to the analytic ``sinc(pi dt nu_f)``.  A sign error, a
    latitude dropped, or a ``u``/``v`` transposition in the fringe rate all fail
    here, because the rotation itself carries none of those choices.

    The agreement tolerance is ``1e-6`` rather than machine precision because
    the envelope is the first-order (constant fringe-rate) result while the
    average is exact: the fringe rate itself drifts across a 10-second
    integration, and that curvature is the residual.
    """
    uvw = np.array([[400.0, -150.0, 25.0]], dtype=np.float64)
    hour_angles = np.array([0.0, 0.6, -1.1])
    decs = np.array([_LATITUDE_RAD, -0.15, 0.35])
    directions = _directions_from_equatorial(hour_angles, decs)
    term = _term(bandwidth=False)

    factor = _factor(term, uvw_wavelengths=uvw, directions=directions)

    integration = float(_INTEGRATIONS[0])
    offsets = np.linspace(-0.5, 0.5, 4_001) * integration
    averaged = []
    for index in range(directions.n_dir):
        rotated = _directions_from_equatorial(
            hour_angles[index] + EARTH_ROTATION_RAD_PER_S * offsets,
            np.full(offsets.size, decs[index]),
        )
        path = (
            uvw[0, 0] * rotated.dir_l
            + uvw[0, 1] * rotated.dir_m
            + uvw[0, 2] * (rotated.dir_n - 1.0)
        )
        phases = np.exp(-2j * np.pi * path)
        mean = np.trapezoid(phases, offsets) / integration
        centre = np.exp(
            -2j
            * np.pi
            * (
                uvw[0, 0] * directions.dir_l[index]
                + uvw[0, 1] * directions.dir_m[index]
                + uvw[0, 2] * (directions.dir_n[index] - 1.0)
            )
        )
        averaged.append(abs(mean))
        # And the average keeps the central phase: smearing is an amplitude
        # effect, not a phase one.
        assert abs(np.angle(mean / centre)) < 1e-6

    np.testing.assert_allclose(
        np.abs(factor[0]), np.array(averaged), rtol=1e-6, atol=1e-12
    )


def test_the_two_envelopes_multiply() -> None:
    """``Q = Q_bandwidth * Q_time``, and each is selectable on its own."""
    uvw = np.array([[500.0, 300.0, -20.0]], dtype=np.float64)
    directions = _directions_from_equatorial(
        np.array([0.2, -0.5]), np.array([-0.1, 0.4])
    )

    both = _factor(_term(), uvw_wavelengths=uvw, directions=directions)
    bandwidth = _factor(_term(time=False), uvw_wavelengths=uvw, directions=directions)
    time = _factor(_term(bandwidth=False), uvw_wavelengths=uvw, directions=directions)

    np.testing.assert_allclose(both, bandwidth * time, rtol=1e-15, atol=0.0)
    assert float(np.max(np.abs(bandwidth - 1.0))) > 1e-6
    assert float(np.max(np.abs(time - 1.0))) > 1e-9


# ---------------------------------------------------------------------------
# I12 -- bounds, the phase centre, and the limits
# ---------------------------------------------------------------------------


def test_the_envelope_never_exceeds_one_anywhere() -> None:
    """The half of I12 that holds without qualification."""
    rng = np.random.default_rng(7)
    uvw = rng.normal(scale=2_000.0, size=(11, 3))
    directions = _directions_from_equatorial(
        rng.uniform(-np.pi, np.pi, 64), rng.uniform(-1.4, 1.4, 64)
    )

    factor = _factor(_term(), uvw_wavelengths=uvw, directions=directions)

    assert float(np.max(factor)) <= 1.0
    assert np.all(np.isfinite(factor))


def test_the_bandwidth_envelope_is_exactly_one_at_the_phase_centre() -> None:
    """I12's phase-centre clause, for the factor it holds of exactly.

    The phase centre is the zenith: ``l = m = 0``, ``n = 1``.  The residual
    delay is exactly zero there for every baseline, including one with a large
    vertical component, and ``numpy.sinc(0)`` is exactly ``1``.
    """
    zenith = _directions_from_equatorial(np.array([0.0]), np.array([_LATITUDE_RAD]))
    uvw = np.array([[900.0, -400.0, 250.0]], dtype=np.float64)

    factor = _factor(_term(time=False), uvw_wavelengths=uvw, directions=zenith)

    assert factor.shape == (1, 1)
    assert factor[0, 0] == 1.0


def test_the_time_envelope_is_one_at_zenith_only_without_an_east_west_arm() -> None:
    """I12's corrected clause, and the physics behind the correction.

    RadioSim's phase centre is the *fixed zenith*, not a tracked source, so a
    source at the zenith still drifts through it during an integration.  The
    residual fringe rate at zenith is ``omega_E u cos(lat)``, which vanishes when
    and only when the baseline has no East-West component -- and a drift-scan
    array with an East-West arm really does decorrelate its own zenith source.
    """
    zenith = _directions_from_equatorial(np.array([0.0]), np.array([_LATITUDE_RAD]))
    north_south = np.array([[0.0, 900.0, 250.0]], dtype=np.float64)
    east_west = np.array([[900.0, 0.0, 0.0]], dtype=np.float64)
    term = _term(bandwidth=False)

    assert _factor(term, uvw_wavelengths=north_south, directions=zenith)[0, 0] == 1.0

    east_west_factor = float(
        _factor(term, uvw_wavelengths=east_west, directions=zenith)[0, 0]
    )
    predicted = float(
        np.sinc(
            float(_INTEGRATIONS[0])
            * EARTH_ROTATION_RAD_PER_S
            * 900.0
            * math.cos(_LATITUDE_RAD)
        )
    )
    assert east_west_factor < 1.0
    assert east_west_factor == pytest.approx(predicted, rel=1e-14)


def test_an_autocorrelation_is_never_smeared() -> None:
    """A zero baseline has no delay and no fringe, so both factors are one."""
    directions = _directions_from_equatorial(
        np.array([0.4, -0.9]), np.array([-0.3, 0.5])
    )
    factor = _factor(
        _term(),
        uvw_wavelengths=np.zeros((1, 3), dtype=np.float64),
        directions=directions,
    )

    np.testing.assert_array_equal(factor, np.ones((1, 2)))


def test_the_envelope_approaches_one_as_the_width_and_the_integration_shrink() -> None:
    """``Q -> 1`` as ``dnu -> 0`` and ``dt -> 0``, quadratically."""
    uvw = np.array([[800.0, -300.0, 40.0]], dtype=np.float64)
    directions = _directions_from_equatorial(np.array([0.5]), np.array([0.2]))

    losses = []
    for scale in (1.0, 0.5, 0.25):
        term = SmearingFactorJones(
            bandwidth_smearing=True,
            time_smearing=True,
            channel_frequencies_hz=_FREQUENCIES,
            channel_widths_hz=_WIDTHS * scale,
            integration_time_s=_INTEGRATIONS * scale,
            sample_times_mjd=_TIMES_MJD,
            latitude_rad=_LATITUDE_RAD,
        )
        losses.append(
            1.0 - float(_factor(term, uvw_wavelengths=uvw, directions=directions)[0, 0])
        )

    assert losses[0] > losses[1] > losses[2] > 0.0
    # Halving both halves the argument of each sinc, so the loss falls by four.
    np.testing.assert_allclose(losses[0] / losses[1], 4.0, rtol=1e-3)
    np.testing.assert_allclose(losses[1] / losses[2], 4.0, rtol=1e-3)


def test_a_longer_baseline_decorrelates_more_toward_the_field_edge() -> None:
    """Section 20.11's structural test: monotonic in baseline length.

    The three baselines here are the same direction at three lengths, so any
    ``u``/``v`` transposition or sign error breaks the ordering rather than
    merely shifting it.
    """
    directions = _directions_from_equatorial(
        np.array([0.0, 0.25, 0.5]),
        np.array([_LATITUDE_RAD, _LATITUDE_RAD + 0.25, _LATITUDE_RAD + 0.5]),
    )
    uvw = np.array(
        [[100.0, 60.0, 5.0], [400.0, 240.0, 20.0], [1_600.0, 960.0, 80.0]],
        dtype=np.float64,
    )

    factor = _factor(_term(), uvw_wavelengths=uvw, directions=directions)

    # At the phase centre (the first direction) the bandwidth factor is one on
    # every baseline; away from it the ordering is strict and in the right
    # direction.
    for direction_index in (1, 2):
        column = factor[:, direction_index]
        assert column[0] > column[1] > column[2]
    assert float(factor[2, 2]) < 0.999


# ---------------------------------------------------------------------------
# Q6 -- the channel width and the integration time come from the resolved grids
# ---------------------------------------------------------------------------


def test_each_channel_smears_by_its_own_declared_width() -> None:
    """Section 41 Q6: the width is per channel, and it is the *declared* one.

    The two channels of the fixture grid here carry widths that differ by a
    factor of four while their centres differ by ten percent, so a term that
    used the channel *spacing* -- or the first channel's width everywhere --
    gives a visibly different answer from the one asserted.
    """
    uvw = np.array([[700.0, -250.0, 30.0]], dtype=np.float64)
    directions = _directions_from_equatorial(np.array([0.3]), np.array([0.1]))
    term = _term(time=False)

    first = _factor(term, uvw_wavelengths=uvw, directions=directions, freq_idx=0)
    second = _factor(term, uvw_wavelengths=uvw, directions=directions, freq_idx=1)

    for index, factor in ((0, first), (1, second)):
        delay = _residual_delay_s(uvw, directions, float(_FREQUENCIES[index]))
        np.testing.assert_allclose(
            factor,
            np.sinc(float(_WIDTHS[index]) * delay),
            rtol=1e-14,
            atol=0.0,
        )
    assert abs(float(first[0, 0]) - float(second[0, 0])) > 1e-3


def test_each_sample_smears_by_its_own_integration_time() -> None:
    """The time axis of the same rule."""
    uvw = np.array([[700.0, -250.0, 30.0]], dtype=np.float64)
    directions = _directions_from_equatorial(np.array([0.3]), np.array([0.1]))
    term = _term(bandwidth=False, integrations=np.array([10.0, 40.0]))

    rate = _fringe_rate_hz(uvw, directions)
    for time_idx, integration in ((0, 10.0), (1, 40.0)):
        factor = _factor(
            term, uvw_wavelengths=uvw, directions=directions, time_idx=time_idx
        )
        np.testing.assert_allclose(
            factor, np.sinc(integration * rate), rtol=1e-14, atol=0.0
        )


def test_a_grid_index_the_term_was_not_resolved_against_is_refused() -> None:
    """Reading channel 2's width for channel 5 is exactly the silent defect."""
    term = _term()
    uvw = np.array([[100.0, 0.0, 0.0]], dtype=np.float64)
    directions = _directions_from_equatorial(np.array([0.1]), np.array([0.1]))

    with pytest.raises(JonesEvaluationError):
        term.compute_baseline_factor(
            baseline_pairs=((0, 1),),
            baseline_uvw_wavelengths=uvw,
            directions=directions,
            frequency_hz=float(_FREQUENCIES[0]),
            freq_idx=7,
            time_mjd=float(_TIMES_MJD[0]),
            time_idx=0,
            backend=_BACKEND,
            dtype=np.float64,
        )

    with pytest.raises(JonesEvaluationError) as caught:
        term.compute_baseline_factor(
            baseline_pairs=((0, 1),),
            baseline_uvw_wavelengths=uvw,
            directions=directions,
            frequency_hz=1.23e8,
            freq_idx=0,
            time_mjd=float(_TIMES_MJD[0]),
            time_idx=0,
            backend=_BACKEND,
            dtype=np.float64,
        )
    assert "Q" in str(caught.value)


# ---------------------------------------------------------------------------
# Declarations, shapes, and rejections
# ---------------------------------------------------------------------------


def test_the_term_declares_what_it_is() -> None:
    term = _term()

    assert term.name == "Q"
    assert term.term_status == "implemented"
    assert term.is_direction_dependent is True
    assert term.hadamard_target == "envelope"


def test_the_factor_is_returned_in_the_real_dtype_it_was_given() -> None:
    """The envelope is real and multiplies a real Gaussian envelope (I17)."""
    uvw = np.array([[300.0, 100.0, 0.0]], dtype=np.float64)
    directions = _directions_from_equatorial(np.array([0.2]), np.array([0.1]))

    for dtype in (np.float64, np.float32):
        factor = _factor(
            _term(), uvw_wavelengths=uvw, directions=directions, dtype=dtype
        )
        assert factor.dtype == dtype


def test_the_factor_is_shaped_by_baseline_and_direction() -> None:
    """``(B, n_dir)`` -- exactly the kernel's ``envelope`` argument shape."""
    uvw = np.array([[300.0, 100.0, 0.0], [50.0, -20.0, 3.0]], dtype=np.float64)
    for n_dir in (1, 5, 23):
        directions = _directions_from_equatorial(
            np.linspace(-0.5, 0.5, n_dir), np.linspace(-0.2, 0.6, n_dir)
        )
        assert _factor(_term(), uvw_wavelengths=uvw, directions=directions).shape == (
            2,
            n_dir,
        )


def test_a_term_with_both_kinds_disabled_cannot_be_constructed() -> None:
    """The constructor half of R16: an envelope of ones is not a term."""
    with pytest.raises(ValueError):
        SmearingFactorJones(
            bandwidth_smearing=False,
            time_smearing=False,
            channel_frequencies_hz=_FREQUENCIES,
            channel_widths_hz=_WIDTHS,
            integration_time_s=_INTEGRATIONS,
            sample_times_mjd=_TIMES_MJD,
            latitude_rad=_LATITUDE_RAD,
        )


def test_the_constructor_refuses_grids_it_cannot_index() -> None:
    for widths, integrations in (
        (np.array([1.0e6]), _INTEGRATIONS),
        (np.array([1.0e6, -1.0]), _INTEGRATIONS),
        (_WIDTHS, np.array([10.0, 0.0])),
    ):
        with pytest.raises(ValueError):
            SmearingFactorJones(
                bandwidth_smearing=True,
                time_smearing=True,
                channel_frequencies_hz=_FREQUENCIES,
                channel_widths_hz=np.asarray(widths, dtype=np.float64),
                integration_time_s=np.asarray(integrations, dtype=np.float64),
                sample_times_mjd=_TIMES_MJD,
                latitude_rad=_LATITUDE_RAD,
            )


def test_the_constructor_refuses_the_physics_keyword_a_stub_would_swallow() -> None:
    with pytest.raises(TypeError):
        SmearingFactorJones()  # type: ignore[call-arg]


def test_both_kinds_disabled_is_rejected_with_the_r16_message(tmp_path) -> None:
    """R16, verbatim."""
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {"Q": {"bandwidth_smearing": False, "time_smearing": False}},
        )

    assert str(caught.value) == (
        "jones.Q is enabled with both smearing kinds disabled; remove the "
        "section instead."
    )


def test_a_resolved_q_reads_the_runs_own_grids(tmp_path) -> None:
    """Section 20.11: neither ``dnu`` nor ``dt`` is a free parameter of the term.

    The shipped fixture is three 1 MHz channels on a 1-second cadence, and that
    is what the resolved term must carry -- not a default, and not anything the
    ``jones.Q`` block could have said.
    """
    resolved = resolve_for(
        tmp_path, {"Q": {"bandwidth_smearing": True, "time_smearing": True}}
    )

    (term,) = resolved.baseline_terms
    assert isinstance(term, SmearingFactorJones)
    np.testing.assert_allclose(term.channel_widths_hz, np.full(3, 1.0e6))
    np.testing.assert_allclose(term.integration_time_s, np.full(2, 1.0))
    assert resolved.chain_terms == ()
    assert resolved.baseline_letters == ("Q",)


def test_a_nonuniform_frequency_grid_keeps_its_declared_widths(tmp_path) -> None:
    """Section 41 Q6, at the resolution boundary.

    An explicit nonuniform array carries a declared width per channel, and those
    are the widths the term smears by -- not the spacing between the centres,
    which here is different for every pair.
    """
    resolved = resolve_for(
        tmp_path,
        {"Q": {"bandwidth_smearing": True, "time_smearing": False}},
        obs_frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [1.0e8, 1.05e8, 1.3e8],
            "channel_widths_hz": [2.0e6, 5.0e5, 8.0e6],
        },
    )

    (term,) = resolved.baseline_terms
    np.testing.assert_allclose(term.channel_widths_hz, np.array([2.0e6, 5.0e5, 8.0e6]))
    np.testing.assert_allclose(
        term.channel_frequencies_hz, np.array([1.0e8, 1.05e8, 1.3e8])
    )
