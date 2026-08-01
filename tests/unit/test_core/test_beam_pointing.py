"""Tier 7I beam physics: per-antenna pointing offsets and Ruze efficiency.

Owns invariant **I19** of ``Tier7JonesSciencePlan.md`` Section 27, in the form
Section 19.2's two 7I corrections fix it:

- a pointing offset of exactly zero is bit-identical to no offset, down to the
  ``assignment_fingerprint`` and ``scientific_sha256``, because an inert
  resolved value resolves to *absent*;
- an offset of ``delta`` moves the analytic beam's peak by exactly ``delta`` in
  great-circle angle, because the offset is a rotation of the beam frame and
  not an additive shift in ``(alt, az)``;
- the Ruze factor ``eta_s`` equals ``exp(-(4 pi sigma / lambda)^2)`` at three
  wavelengths, and the visibility amplitude on a baseline of two antennas
  sharing that ``sigma`` is scaled by exactly ``eta_s`` -- which is why the
  factor applied to the *voltage* beam is ``sqrt(eta_s)``.

Every reference value is written out in the test body (Section 29.1): the
closed forms are evaluated here from ``math`` and ``scipy.special``, never by
calling the implementation under test.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import j1  # pyright: ignore[reportMissingTypeStubs]

from radiosim.core.instrument import AntennaId
from radiosim.core.precision import PrecisionConfig
from tests.fixtures.configs import valid_config_mapping

_C_M_PER_S = 299_792_458.0

# The two antennas of the shipped test layout are both 14 m, so they resolve to
# one shared analytic handler.  That is deliberate here: it is the case in which
# a per-antenna offset must still produce two different responses.
_DIAMETER_M = 14.0


# =========================================================================
# Fixtures and closed-form oracles
# =========================================================================


def _uniform_circular_voltage(u: float) -> float:
    """Return ``2 J1(pi u) / (pi u)``, the uniform circular voltage pattern."""
    argument = math.pi * u
    if argument == 0.0:
        return 1.0
    return float(2.0 * j1(argument) / argument)


def _ruze_power_efficiency(sigma_m: float, wavelength_m: float) -> float:
    """Return Ruze (1966) ``eta_s = exp(-(4 pi sigma / lambda)^2)``."""
    return math.exp(-((4.0 * math.pi * sigma_m / wavelength_m) ** 2))


def _great_circle_angle_rad(
    altitude_rad: float,
    azimuth_rad: float,
    boresight_altitude_rad: float,
    boresight_azimuth_rad: float,
) -> float:
    """Spherical law of cosines, evaluated independently of the runtime."""
    cosine = math.sin(altitude_rad) * math.sin(boresight_altitude_rad) + math.cos(
        altitude_rad
    ) * math.cos(boresight_altitude_rad) * math.cos(azimuth_rad - boresight_azimuth_rad)
    return math.acos(min(1.0, max(-1.0, cosine)))


def _beam_system(tmp_path: Path, beams: dict[str, Any]):
    """Resolve one complete ``BeamSystem`` from an authored ``beams`` block."""
    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.beam.runtime import load_beam_system
    from radiosim.core.instrument_resolution import resolve_instrument
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config

    bundle = resolve_config(
        valid_config_mapping(tmp_path, beams=beams),
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )
    runtime = bundle.runtime
    instrument = resolve_instrument(runtime.instrument)
    state = resolve_beam_assignments(runtime.beams, instrument)
    system = load_beam_system(
        state,
        observation_frequencies_hz=runtime.frequency.channel_frequencies_hz,
        precision=runtime.execution.precision,
    )
    return system, state


def _uniform_beams(**extra: Any) -> dict[str, Any]:
    """Return a uniform-taper circular analytic block plus authored extras."""
    block: dict[str, Any] = {
        "mode": "analytic",
        "model": {"kind": "circular_aperture", "taper": {"kind": "uniform"}},
    }
    block.update(extra)
    return block


def _evaluate(
    system, antenna: AntennaId, altitude: Any, azimuth: Any, frequency: float
):
    return np.asarray(
        system.evaluate_jones(
            antenna,
            altitude_rad=np.asarray(altitude, dtype=np.float64),
            azimuth_rad=np.asarray(azimuth, dtype=np.float64),
            frequency_hz=frequency,
            time_mjd=60_000.0,
        )
    )


_ANT0 = AntennaId(0, "ANT0")
_ANT1 = AntennaId(1, "ANT1")


# =========================================================================
# I19, first clause: an inert offset is absent, not a stored zero
# =========================================================================


def test_zero_pointing_offset_resolves_to_absent(tmp_path: Path) -> None:
    """A ``(0, 0)`` offset leaves every resolved beam fingerprint untouched."""
    _bare_system, bare_state = _beam_system(tmp_path, _uniform_beams())
    _zero_system, zero_state = _beam_system(
        tmp_path,
        _uniform_beams(
            pointing={
                "default": {"azimuth_offset_deg": 30.0, "elevation_offset_deg": 0.0},
                "per_antenna": [
                    {
                        "antenna": {"kind": "number", "number": 0},
                        "azimuth_offset_deg": 0.0,
                        "elevation_offset_deg": 0.0,
                    }
                ],
            }
        ),
    )

    bare = {a.antenna_id: a for a in bare_state.assignments}
    zero = {a.antenna_id: a for a in zero_state.assignments}
    # Antenna 0 authored an exactly-zero override: it resolves to no offset at
    # all, so its assignment fingerprint is the untouched one.
    assert zero[_ANT0].pointing is None
    assert zero[_ANT0].assignment_fingerprint == bare[_ANT0].assignment_fingerprint
    # Antenna 1 takes the non-zero array-wide default, so it must differ.
    assert zero[_ANT1].pointing is not None
    assert zero[_ANT1].assignment_fingerprint != bare[_ANT1].assignment_fingerprint


def test_zero_pointing_offset_is_bit_identical_to_no_offset(tmp_path: Path) -> None:
    """I19: a zero offset reproduces the no-offset response bit for bit."""
    altitude = np.linspace(0.02, np.pi / 2.0, 41)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 41)

    bare_system, _ = _beam_system(tmp_path, _uniform_beams())
    zero_system, _ = _beam_system(
        tmp_path,
        _uniform_beams(
            pointing={
                "default": {"azimuth_offset_deg": 0.0, "elevation_offset_deg": 0.4},
                "per_antenna": [
                    {
                        "antenna": {"kind": "number", "number": 0},
                        "azimuth_offset_deg": 0.0,
                        "elevation_offset_deg": 0.0,
                    }
                ],
            }
        ),
    )
    bare = _evaluate(bare_system, _ANT0, altitude, azimuth, 100e6)
    zero = _evaluate(zero_system, _ANT0, altitude, azimuth, 100e6)
    assert np.array_equal(bare, zero)
    assert bare.dtype == zero.dtype


def test_zero_surface_error_resolves_to_absent_and_is_bit_identical(
    tmp_path: Path,
) -> None:
    """I19: ``sigma = 0`` is no Ruze factor, not a factor of one."""
    altitude = np.linspace(0.02, np.pi / 2.0, 17)
    azimuth = np.zeros(17)

    bare_system, bare_state = _beam_system(tmp_path, _uniform_beams())
    zero_system, zero_state = _beam_system(
        tmp_path,
        _uniform_beams(
            surface_error={
                "default": {"rms_surface_error_m": 0.002},
                "per_antenna": [
                    {
                        "antenna": {"kind": "number", "number": 0},
                        "rms_surface_error_m": 0.0,
                    }
                ],
            }
        ),
    )
    bare = {a.antenna_id: a for a in bare_state.assignments}
    zero = {a.antenna_id: a for a in zero_state.assignments}
    assert zero[_ANT0].surface_error is None
    assert zero[_ANT0].assignment_fingerprint == bare[_ANT0].assignment_fingerprint
    assert zero[_ANT1].surface_error is not None
    assert zero[_ANT1].assignment_fingerprint != bare[_ANT1].assignment_fingerprint

    assert np.array_equal(
        _evaluate(bare_system, _ANT0, altitude, azimuth, 100e6),
        _evaluate(zero_system, _ANT0, altitude, azimuth, 100e6),
    )


def test_the_beam_state_fingerprint_moves_only_when_science_is_present(
    tmp_path: Path,
) -> None:
    """Absent blocks reproduce the pre-7I digest; present ones must not.

    A block that resolves to absence *for every antenna* is unreachable from
    configuration -- an all-zero block is rejected outright -- so what is
    checkable is the pair of statements either side of that: no block leaves the
    state digest where it was, and any accepted block moves it.
    """
    _bare, bare_state = _beam_system(tmp_path, _uniform_beams())
    _again, again_state = _beam_system(tmp_path, _uniform_beams())
    assert again_state.state_fingerprint == bare_state.state_fingerprint

    _offset, offset_state = _beam_system(
        tmp_path,
        _uniform_beams(
            pointing={
                "per_antenna": [
                    {
                        "antenna": {"kind": "number", "number": 1},
                        "azimuth_offset_deg": 0.0,
                        "elevation_offset_deg": 0.5,
                    }
                ]
            }
        ),
    )
    assert offset_state.state_fingerprint != bare_state.state_fingerprint
    # ... and antenna 0, which the block never reaches, is untouched.
    bare_by_antenna = {a.antenna_id: a for a in bare_state.assignments}
    offset_by_antenna = {a.antenna_id: a for a in offset_state.assignments}
    assert (
        offset_by_antenna[_ANT0].assignment_fingerprint
        == bare_by_antenna[_ANT0].assignment_fingerprint
    )

    _rough, rough_state = _beam_system(
        tmp_path,
        _uniform_beams(surface_error={"default": {"rms_surface_error_m": 0.003}}),
    )
    assert rough_state.state_fingerprint != bare_state.state_fingerprint
    assert rough_state.state_fingerprint != offset_state.state_fingerprint


# =========================================================================
# I19, second clause: the peak moves by exactly delta
# =========================================================================


@pytest.mark.parametrize(
    ("azimuth_offset_deg", "elevation_offset_deg"),
    [(0.0, 0.5), (90.0, 1.25), (-137.0, 3.0), (180.0, 0.25)],
)
def test_offset_moves_the_analytic_beam_peak_by_exactly_delta(
    tmp_path: Path,
    azimuth_offset_deg: float,
    elevation_offset_deg: float,
) -> None:
    """I19: the peak lands at zenith angle ``delta_el``, azimuth ``delta_az``."""
    system, _ = _beam_system(
        tmp_path,
        _uniform_beams(
            pointing={
                "default": {
                    "azimuth_offset_deg": azimuth_offset_deg,
                    "elevation_offset_deg": elevation_offset_deg,
                }
            }
        ),
    )
    peak_altitude = np.pi / 2.0 - math.radians(elevation_offset_deg)
    peak_azimuth = math.radians(azimuth_offset_deg)
    response = _evaluate(
        system,
        _ANT0,
        np.array([peak_altitude]),
        np.array([peak_azimuth]),
        100e6,
    )
    # The uniform circular voltage pattern is exactly 1 on boresight.
    assert response[0, 0, 0].real == pytest.approx(1.0, abs=5e-13)
    assert response[0, 1, 1] == response[0, 0, 0]

    # The great-circle displacement of the peak from the nominal zenith
    # boresight is exactly the configured elevation offset.
    assert _great_circle_angle_rad(
        peak_altitude, peak_azimuth, np.pi / 2.0, 0.0
    ) == pytest.approx(math.radians(elevation_offset_deg), abs=1e-13)


def test_offset_response_equals_the_pattern_at_the_great_circle_angle(
    tmp_path: Path,
) -> None:
    """The rotation is a rotation: the response depends only on the angle."""
    elevation_offset_deg = 2.0
    azimuth_offset_deg = 40.0
    system, _ = _beam_system(
        tmp_path,
        _uniform_beams(
            pointing={
                "default": {
                    "azimuth_offset_deg": azimuth_offset_deg,
                    "elevation_offset_deg": elevation_offset_deg,
                }
            }
        ),
    )
    frequency = 100e6
    wavelength = _C_M_PER_S / frequency
    altitude = np.radians(np.array([89.0, 80.0, 70.0, 55.0, 30.0, 10.0]))
    azimuth = np.radians(np.array([0.0, 45.0, 120.0, 200.0, 300.0, 355.0]))

    response = _evaluate(system, _ANT0, altitude, azimuth, frequency)

    boresight_altitude = np.pi / 2.0 - math.radians(elevation_offset_deg)
    boresight_azimuth = math.radians(azimuth_offset_deg)
    for index in range(altitude.size):
        theta = _great_circle_angle_rad(
            float(altitude[index]),
            float(azimuth[index]),
            boresight_altitude,
            boresight_azimuth,
        )
        expected = _uniform_circular_voltage(_DIAMETER_M * math.sin(theta) / wavelength)
        assert response[index, 0, 0].real == pytest.approx(expected, abs=1e-12)


def test_azimuth_offset_alone_is_the_alt_az_keyhole_degeneracy(
    tmp_path: Path,
) -> None:
    """A pure azimuth offset rotates the pattern; it never moves the peak."""
    circular, _ = _beam_system(
        tmp_path,
        _uniform_beams(
            pointing={
                "default": {
                    "azimuth_offset_deg": 37.0,
                    "elevation_offset_deg": 0.0,
                }
            }
        ),
    )
    bare_circular, _ = _beam_system(tmp_path, _uniform_beams())
    altitude = np.linspace(0.05, np.pi / 2.0, 23)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 23)
    # A circularly symmetric beam cannot see a rotation about its own boresight.
    assert _evaluate(circular, _ANT0, altitude, azimuth, 100e6) == pytest.approx(
        _evaluate(bare_circular, _ANT0, altitude, azimuth, 100e6),
        abs=1e-12,
    )

    rectangular_beams: dict[str, Any] = {
        "mode": "analytic",
        "model": {
            "kind": "rectangular_aperture",
            "north_length_m": 20.0,
            "east_length_m": 6.0,
        },
    }
    rotated, _ = _beam_system(
        tmp_path,
        {
            **rectangular_beams,
            "pointing": {
                "default": {
                    "azimuth_offset_deg": 37.0,
                    "elevation_offset_deg": 0.0,
                }
            },
        },
    )
    bare_rect, _ = _beam_system(tmp_path, rectangular_beams)
    rotated_values = _evaluate(rotated, _ANT0, altitude, azimuth, 100e6)
    bare_values = _evaluate(bare_rect, _ANT0, altitude, azimuth, 100e6)
    # A rectangular aperture does see it: the pattern is not axisymmetric.
    assert not np.allclose(rotated_values, bare_values, atol=1e-9)
    # ... but the boresight response is untouched, because the peak did not move.
    assert _evaluate(
        rotated, _ANT0, np.array([np.pi / 2.0]), np.array([0.0]), 100e6
    ) == pytest.approx(
        _evaluate(bare_rect, _ANT0, np.array([np.pi / 2.0]), np.array([0.0]), 100e6),
        abs=1e-13,
    )


def test_pointing_offset_leaves_the_horizon_gate_on_the_true_altitude(
    tmp_path: Path,
) -> None:
    """A beam-frame rotation does not move the ground."""
    system, _ = _beam_system(
        tmp_path,
        _uniform_beams(
            pointing={
                "default": {
                    "azimuth_offset_deg": 0.0,
                    "elevation_offset_deg": 5.0,
                }
            }
        ),
    )
    altitude = np.array([-1e-9, -0.01, -0.5, -np.pi / 2.0])
    azimuth = np.zeros(4)
    response = _evaluate(system, _ANT0, altitude, azimuth, 100e6)
    assert np.array_equal(response, np.zeros_like(response))


def test_per_antenna_offsets_separate_antennas_sharing_one_handler(
    tmp_path: Path,
) -> None:
    """Both antennas are 14 m, so they share a handler and must still differ."""
    system, state = _beam_system(
        tmp_path,
        _uniform_beams(
            pointing={
                "per_antenna": [
                    {
                        "antenna": {"kind": "number", "number": 1},
                        "azimuth_offset_deg": 0.0,
                        "elevation_offset_deg": 2.0,
                    }
                ]
            }
        ),
    )
    # One handler, two antennas: the precondition this test exists for.
    assert len(state.assignments) == 2
    handler_ids = {
        handler_id for _antenna, handler_id in system.state.assignment_handler_ids
    }
    assert len(handler_ids) == 1

    altitude = np.array([np.pi / 2.0 - math.radians(2.0)])
    azimuth = np.array([0.0])
    antenna0 = _evaluate(system, _ANT0, altitude, azimuth, 100e6)
    antenna1 = _evaluate(system, _ANT1, altitude, azimuth, 100e6)
    assert antenna1[0, 0, 0].real == pytest.approx(1.0, abs=5e-13)
    assert antenna0[0, 0, 0].real < 0.999
    # ... and the response key that keeps a per-step cache honest separates them.
    assert system.response_key(_ANT0) != system.response_key(_ANT1)


def test_response_key_is_the_handler_id_when_nothing_is_configured(
    tmp_path: Path,
) -> None:
    """The absent case must not perturb the solver's cache key."""
    system, _ = _beam_system(tmp_path, _uniform_beams())
    by_antenna = dict(system.state.assignment_handler_ids)
    assert system.response_key(_ANT0) == by_antenna[_ANT0]
    assert system.response_key(_ANT1) == by_antenna[_ANT1]


# =========================================================================
# I19, third clause: Ruze
# =========================================================================


@pytest.mark.parametrize("frequency_hz", [50e6, 150e6, 350e6])
def test_ruze_efficiency_matches_the_closed_form_at_three_wavelengths(
    frequency_hz: float,
) -> None:
    """I19: ``eta_s`` is the published Ruze power efficiency."""
    from radiosim.core.beam.runtime import ruze_power_efficiency

    sigma_m = 0.012
    wavelength_m = _C_M_PER_S / frequency_hz
    assert ruze_power_efficiency(
        rms_surface_error_m=sigma_m,
        wavelength_m=wavelength_m,
    ) == pytest.approx(_ruze_power_efficiency(sigma_m, wavelength_m), rel=1e-15)


def test_ruze_rule_of_thumb_holds_at_the_shortest_usable_wavelength() -> None:
    """``lambda_min ~= 10 sigma`` is the point at which most gain is gone."""
    from radiosim.core.beam.runtime import ruze_power_efficiency

    sigma_m = 0.005
    efficiency = ruze_power_efficiency(
        rms_surface_error_m=sigma_m,
        wavelength_m=10.0 * sigma_m,
    )
    # exp(-(4 pi / 10)^2) = exp(-1.5791367...) = 0.2061529924...
    assert efficiency == pytest.approx(math.exp(-((0.4 * math.pi) ** 2)), rel=1e-15)
    assert efficiency == pytest.approx(0.20615299, abs=1e-8)
    # And the far-field limit is unity, so the factor is monotone in wavelength.
    assert (
        ruze_power_efficiency(
            rms_surface_error_m=sigma_m,
            wavelength_m=1000.0 * sigma_m,
        )
        > 0.999
    )


def test_ruze_scales_the_voltage_beam_by_the_square_root_of_the_efficiency(
    tmp_path: Path,
) -> None:
    """Section 19.2's corrected convention, asserted directly on ``E``."""
    sigma_m = 0.02
    frequency = 250e6
    wavelength = _C_M_PER_S / frequency
    efficiency = _ruze_power_efficiency(sigma_m, wavelength)

    bare, _ = _beam_system(tmp_path, _uniform_beams())
    rough, _ = _beam_system(
        tmp_path,
        _uniform_beams(surface_error={"default": {"rms_surface_error_m": sigma_m}}),
    )
    altitude = np.radians(np.array([90.0, 88.0, 85.0, 60.0]))
    azimuth = np.zeros(4)
    bare_values = _evaluate(bare, _ANT0, altitude, azimuth, frequency)
    rough_values = _evaluate(rough, _ANT0, altitude, azimuth, frequency)

    assert rough_values == pytest.approx(bare_values * math.sqrt(efficiency), abs=1e-13)
    # The physical statement the convention exists for: the power a baseline of
    # two like antennas measures is reduced by exactly eta_s, not eta_s squared.
    baseline_power = rough_values[:, 0, 0] * np.conjugate(rough_values[:, 0, 0])
    nominal_power = bare_values[:, 0, 0] * np.conjugate(bare_values[:, 0, 0])
    assert baseline_power.real == pytest.approx(
        nominal_power.real * efficiency, abs=1e-13
    )


def test_ruze_is_wavelength_dependent_not_a_constant_scale(tmp_path: Path) -> None:
    """The lambda dependence is the whole content of the Ruze equation."""
    sigma_m = 0.03
    rough, _ = _beam_system(
        tmp_path,
        _uniform_beams(surface_error={"default": {"rms_surface_error_m": sigma_m}}),
    )
    bare, _ = _beam_system(tmp_path, _uniform_beams())
    zenith = np.array([np.pi / 2.0])
    azimuth = np.array([0.0])
    ratios: list[float] = []
    for frequency in (100e6, 200e6):
        rough_value = _evaluate(rough, _ANT0, zenith, azimuth, frequency)[0, 0, 0]
        bare_value = _evaluate(bare, _ANT0, zenith, azimuth, frequency)[0, 0, 0]
        ratios.append(float(abs(rough_value) / abs(bare_value)))
        wavelength = _C_M_PER_S / frequency
        assert ratios[-1] == pytest.approx(
            math.sqrt(_ruze_power_efficiency(sigma_m, wavelength)), abs=1e-12
        )
    assert ratios[1] < ratios[0]


def test_pointing_and_ruze_compose(tmp_path: Path) -> None:
    """Both effects apply together, each in its own place."""
    sigma_m = 0.015
    frequency = 200e6
    wavelength = _C_M_PER_S / frequency
    system, _ = _beam_system(
        tmp_path,
        _uniform_beams(
            pointing={
                "default": {"azimuth_offset_deg": 0.0, "elevation_offset_deg": 1.5}
            },
            surface_error={"default": {"rms_surface_error_m": sigma_m}},
        ),
    )
    peak = _evaluate(
        system,
        _ANT0,
        np.array([np.pi / 2.0 - math.radians(1.5)]),
        np.array([0.0]),
        frequency,
    )
    expected = math.sqrt(_ruze_power_efficiency(sigma_m, wavelength))
    assert peak[0, 0, 0].real == pytest.approx(expected, abs=1e-12)


# =========================================================================
# Precision
# =========================================================================


def test_pointing_and_ruze_honor_the_resolved_beam_precision(tmp_path: Path) -> None:
    """Neither effect may widen the dtype the beam precision resolved to."""
    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.beam.runtime import load_beam_system
    from radiosim.core.instrument_resolution import resolve_instrument
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config

    bundle = resolve_config(
        valid_config_mapping(
            tmp_path,
            beams=_uniform_beams(
                pointing={
                    "default": {
                        "azimuth_offset_deg": 12.0,
                        "elevation_offset_deg": 0.75,
                    }
                },
                surface_error={"default": {"rms_surface_error_m": 0.01}},
            ),
        ),
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )
    runtime = bundle.runtime
    instrument = resolve_instrument(runtime.instrument)
    state = resolve_beam_assignments(runtime.beams, instrument)
    system = load_beam_system(
        state,
        observation_frequencies_hz=runtime.frequency.channel_frequencies_hz,
        precision=PrecisionConfig.fast(),
    )
    values = _evaluate(
        system,
        _ANT0,
        np.linspace(0.1, np.pi / 2.0, 9),
        np.zeros(9),
        100e6,
    )
    assert values.dtype == np.dtype(np.complex64)


# =========================================================================
# Rejections
# =========================================================================


def _reject(tmp_path: Path, beams: dict[str, Any]):
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config

    return resolve_config(
        valid_config_mapping(tmp_path, beams=beams),
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )


def test_an_all_zero_pointing_block_is_rejected(tmp_path: Path) -> None:
    """R7's shape: a block that is present and has no effect is a defect."""
    from radiosim.io.config_resolution import ConfigSchemaError

    with pytest.raises(ConfigSchemaError) as excinfo:
        _reject(
            tmp_path,
            _uniform_beams(
                pointing={
                    "default": {
                        "azimuth_offset_deg": 0.0,
                        "elevation_offset_deg": 0.0,
                    }
                }
            ),
        )
    assert "beams.pointing" in str(excinfo.value)
    assert "every authored offset is zero" in str(excinfo.value)


def test_an_empty_pointing_block_is_rejected(tmp_path: Path) -> None:
    from radiosim.io.config_resolution import ConfigSchemaError

    with pytest.raises(ConfigSchemaError) as excinfo:
        _reject(tmp_path, _uniform_beams(pointing={}))
    assert "every authored offset is zero" in str(excinfo.value)


def test_an_all_zero_surface_error_block_is_rejected(tmp_path: Path) -> None:
    from radiosim.io.config_resolution import ConfigSchemaError

    with pytest.raises(ConfigSchemaError) as excinfo:
        _reject(
            tmp_path,
            _uniform_beams(surface_error={"default": {"rms_surface_error_m": 0.0}}),
        )
    assert "beams.surface_error" in str(excinfo.value)
    assert "every authored surface error is zero" in str(excinfo.value)


def test_a_negative_surface_error_is_rejected(tmp_path: Path) -> None:
    from radiosim.io.config_resolution import ConfigSchemaError

    with pytest.raises(ConfigSchemaError):
        _reject(
            tmp_path,
            _uniform_beams(surface_error={"default": {"rms_surface_error_m": -0.001}}),
        )


def test_an_out_of_range_elevation_offset_is_rejected(tmp_path: Path) -> None:
    from radiosim.io.config_resolution import ConfigSchemaError

    with pytest.raises(ConfigSchemaError):
        _reject(
            tmp_path,
            _uniform_beams(
                pointing={
                    "default": {
                        "azimuth_offset_deg": 0.0,
                        "elevation_offset_deg": 91.0,
                    }
                }
            ),
        )


def test_an_unknown_pointing_antenna_is_rejected(tmp_path: Path) -> None:
    from radiosim.core.beam.errors import UnknownBeamAntennaError

    with pytest.raises(UnknownBeamAntennaError) as excinfo:
        _beam_system(
            tmp_path,
            _uniform_beams(
                pointing={
                    "per_antenna": [
                        {
                            "antenna": {"kind": "number", "number": 99},
                            "azimuth_offset_deg": 0.0,
                            "elevation_offset_deg": 1.0,
                        }
                    ]
                }
            ),
        )
    assert "beams.pointing.per_antenna[0].antenna=99" in str(excinfo.value)


def test_a_duplicate_pointing_antenna_is_rejected(tmp_path: Path) -> None:
    from radiosim.core.beam.errors import DuplicateBeamAssignmentError

    with pytest.raises(DuplicateBeamAssignmentError) as excinfo:
        _beam_system(
            tmp_path,
            _uniform_beams(
                pointing={
                    "per_antenna": [
                        {
                            "antenna": {"kind": "number", "number": 0},
                            "azimuth_offset_deg": 0.0,
                            "elevation_offset_deg": 1.0,
                        },
                        {
                            "antenna": {"kind": "name", "name": "ANT0"},
                            "azimuth_offset_deg": 0.0,
                            "elevation_offset_deg": 2.0,
                        },
                    ]
                }
            ),
        )
    assert "already assigned at index 0" in str(excinfo.value)


def test_an_unknown_surface_error_antenna_is_rejected(tmp_path: Path) -> None:
    from radiosim.core.beam.errors import UnknownBeamAntennaError

    with pytest.raises(UnknownBeamAntennaError) as excinfo:
        _beam_system(
            tmp_path,
            _uniform_beams(
                surface_error={
                    "per_antenna": [
                        {
                            "antenna": {"kind": "name", "name": "NOPE"},
                            "rms_surface_error_m": 0.01,
                        }
                    ]
                }
            ),
        )
    assert "beams.surface_error.per_antenna[0].antenna='NOPE'" in str(excinfo.value)


def test_an_unknown_key_inside_the_pointing_block_is_rejected(
    tmp_path: Path,
) -> None:
    from radiosim.io.config_resolution import ConfigSchemaError

    with pytest.raises(ConfigSchemaError):
        _reject(
            tmp_path,
            _uniform_beams(
                pointing={
                    "default": {
                        "azimuth_offset_deg": 0.0,
                        "elevation_offset_deg": 1.0,
                        "pointing_rms_rad": 0.001,
                    }
                }
            ),
        )
