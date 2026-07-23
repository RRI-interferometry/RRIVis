"""Scalar BeamFITS and canonical BeamSystem runtime tests."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiosim.core.beam import (
    BeamAngularDomainError,
    BeamEvaluationError,
    BeamFrequencyDomainError,
    NonFiniteBeamResponseError,
    ResolvedFITSBeamDefinition,
    UnsupportedBeamBasisError,
)
from radiosim.core.beam import models as beam_models
from radiosim.core.precision import PrecisionConfig
from tests.fixtures.beamfits import (
    BeamScienceVariant,
    build_scalar_efield_uvbeam,
    canonical_azimuth_grid,
    canonical_zenith_angle_grid,
    scalar_voltage_reference,
    write_scalar_efield_beamfits,
)
from tests.fixtures.configs import valid_config_mapping


def _definition(path: Path, *, interpolation: str = "linear"):
    normalized = path.resolve(strict=False)
    payload = {
        "path": normalized,
        "normalization": "peak",
        "angular_interpolation": "bilinear",
        "frequency_interpolation": interpolation,
    }
    return ResolvedFITSBeamDefinition(
        "fits",
        normalized,
        "peak",
        "bilinear",
        interpolation,
        "beams.beam.path",
        beam_models._definition_fingerprint("fits", payload),
    )


def _load(
    tmp_path: Path,
    *,
    precision: PrecisionConfig | None = None,
    interpolation: str = "linear",
):
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam.fits import _load_fits_handler

    return _load_fits_handler(
        _definition(written.path, interpolation=interpolation),
        observation_frequencies_hz=(100e6, 110e6, 120e6, 130e6),
        precision=precision or PrecisionConfig.standard(),
        handler_ordinal=0,
    )


def test_private_scalar_evaluator_preserves_phase_and_returns_jones(
    tmp_path: Path,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(110e6,),
        precision=PrecisionConfig.standard(),
        handler_ordinal=0,
    )
    result = loaded.evaluator.evaluate_numpy(
        np.array([np.pi / 2.0, np.pi / 4.0, -0.1]),
        np.array([0.0, np.pi / 2.0, np.pi]),
        110e6,
        60_000.0,
    )

    assert result.shape == (3, 2, 2)
    assert result.dtype == np.dtype(np.complex128)
    assert result[1, 0, 0].imag != 0.0
    np.testing.assert_array_equal(result[2], np.zeros((2, 2)))


def test_native_grid_voltage_feature_scale_is_available(
    tmp_path: Path,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(100e6, 120e6),
        precision=PrecisionConfig.standard(),
        handler_ordinal=0,
    )
    first = loaded.evaluator.voltage_feature_scale_rad(100e6)
    second = loaded.evaluator.voltage_feature_scale_rad(120e6)

    assert np.isfinite(first)
    assert first > 0.0
    assert first == second


def test_native_grid_feature_scale_matches_independent_analytical_oracle(
    tmp_path: Path,
) -> None:
    loaded = _load(tmp_path)
    za = canonical_zenith_angle_grid()
    delta_za = za[1] - za[0]
    delta_az = canonical_azimuth_grid()[1] - canonical_azimuth_grid()[0]
    positive_visible = za[(za > 0.0) & (za <= np.pi / 2.0)]
    horizontal = np.arccos(
        np.cos(positive_visible) ** 2 + np.sin(positive_visible) ** 2 * np.cos(delta_az)
    )
    expected = 2.0 * min(delta_za, float(np.min(horizontal)))

    assert loaded.state.voltage_feature_scale_by_frequency == tuple(
        (frequency, expected) for frequency in (100e6, 110e6, 120e6, 130e6)
    )
    assert loaded.evaluator.voltage_feature_scale_rad(105e6) == expected


def test_native_grid_feature_scale_excludes_degenerate_zenith_row(
    tmp_path: Path,
) -> None:
    loaded = _load(tmp_path)
    scale = loaded.evaluator.voltage_feature_scale_rad(100e6)

    assert scale > 0.0
    assert scale != 0.0


@pytest.mark.parametrize("frequency_hz", (99e6, 131e6))
def test_feature_scale_rejects_out_of_domain_frequency(
    tmp_path: Path,
    frequency_hz: float,
) -> None:
    loaded = _load(tmp_path)
    with pytest.raises(BeamFrequencyDomainError):
        loaded.evaluator.voltage_feature_scale_rad(frequency_hz)


def test_scientific_values_match_no_conjugation_oracle_at_native_nodes(
    tmp_path: Path,
) -> None:
    loaded = _load(tmp_path)
    azimuth_uv = canonical_azimuth_grid()[[0, 1, 4, 6]]
    radiosim_azimuth = (np.pi / 2.0 - azimuth_uv) % (2.0 * np.pi)
    altitude = np.pi / 2.0 - canonical_zenith_angle_grid()[[0, 1, 2, 4]]
    result = loaded.evaluator.evaluate_numpy(
        altitude,
        radiosim_azimuth,
        110e6,
        60_000.0,
    )
    expected = scalar_voltage_reference(
        azimuth_uv_rad=azimuth_uv,
        zenith_angle_rad=np.pi / 2.0 - altitude,
        frequency_index=np.ones(4),
    )

    np.testing.assert_allclose(result[:, 0, 0], expected, rtol=0.0, atol=2e-15)
    np.testing.assert_array_equal(result[:, 0, 1], 0.0)
    np.testing.assert_array_equal(result[:, 1, 0], 0.0)
    np.testing.assert_array_equal(result[:, 1, 1], result[:, 0, 0])
    assert result[1, 0, 0] != np.conjugate(expected[1])


def test_radiosim_cardinals_wrap_and_inputs_remain_unchanged(tmp_path: Path) -> None:
    loaded = _load(tmp_path)
    altitude = np.full(6, np.pi / 4.0)
    azimuth = np.array(
        [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi, -np.pi / 2.0]
    )
    original_altitude = altitude.copy()
    original_azimuth = azimuth.copy()
    result = loaded.evaluator.evaluate_numpy(altitude, azimuth, 110e6, 60_000.0)

    np.testing.assert_array_equal(altitude, original_altitude)
    np.testing.assert_array_equal(azimuth, original_azimuth)
    np.testing.assert_allclose(result[0], result[4], rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(result[3], result[5], rtol=0.0, atol=2e-15)


def test_mixed_visible_and_below_horizon_returns_exact_zero_for_hidden(
    tmp_path: Path,
) -> None:
    loaded = _load(tmp_path)
    altitude = np.array([-np.pi / 2.0, -0.1, 0.0, np.pi / 2.0])
    azimuth = np.array([0.0, 1.0, 2.0, 3.0])
    result = loaded.evaluator.evaluate_numpy(altitude, azimuth, 100e6, 60_000.0)

    np.testing.assert_array_equal(result[:2], np.zeros((2, 2, 2)))
    assert np.any(result[2:] != 0.0)


def test_all_below_horizon_batch_never_calls_pyuvdata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _load(tmp_path)
    private_beam = loaded.evaluator._beam

    def forbidden_interp(self: Any, **kwargs: Any):
        raise AssertionError((self, kwargs))

    monkeypatch.setattr(type(private_beam), "interp", forbidden_interp)
    result = loaded.evaluator.evaluate_numpy(
        np.array([-0.1, -np.pi / 2.0]),
        np.array([0.0, 1.0]),
        100e6,
        60_000.0,
    )
    np.testing.assert_array_equal(result, np.zeros((2, 2, 2)))


def test_exact_interpolation_call_contract_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _load(tmp_path, interpolation="cubic")
    private_beam = loaded.evaluator._beam
    original = type(private_beam).interp
    calls: list[dict[str, Any]] = []

    def recording_interp(self: Any, **kwargs: Any):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(type(private_beam), "interp", recording_interp)
    loaded.evaluator.evaluate_numpy(
        np.array([0.0, np.pi / 2.0]),
        np.array([0.0, np.pi / 2.0]),
        105e6,
        60_000.0,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["interpolation_function"] == "az_za_simple"
    assert call["freq_interp_kind"] == "cubic"
    assert call["freq_interp_tol"] == 1e-6
    assert call["return_basis_vector"] is False
    assert call["spline_opts"] == {"kx": 1, "ky": 1, "s": 0}
    np.testing.assert_allclose(
        call["az_array"],
        np.array([np.pi / 2.0, 0.0]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        call["za_array"],
        np.array([np.pi / 2.0, 0.0]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "altitude,azimuth,error_type",
    (
        (0.0, np.array([0.0]), BeamAngularDomainError),
        (np.array([0.0]), 0.0, BeamAngularDomainError),
        (np.array([[0.0]]), np.array([0.0]), BeamAngularDomainError),
        (np.array([0.0]), np.array([[0.0]]), BeamAngularDomainError),
        (np.array([0.0, 0.1]), np.array([0.0]), BeamAngularDomainError),
        (np.array([np.nan]), np.array([0.0]), NonFiniteBeamResponseError),
        (np.array([0.0]), np.array([np.inf]), NonFiniteBeamResponseError),
        (np.array([-np.pi / 2.0 - 1e-12]), np.array([0.0]), BeamAngularDomainError),
        (np.array([np.pi / 2.0 + 1e-12]), np.array([0.0]), BeamAngularDomainError),
    ),
)
def test_coordinate_input_contract_rejects_invalid_values(
    tmp_path: Path,
    altitude: Any,
    azimuth: Any,
    error_type: type[Exception],
) -> None:
    loaded = _load(tmp_path)
    with pytest.raises(error_type):
        loaded.evaluator.evaluate_numpy(altitude, azimuth, 100e6, 60_000.0)


@pytest.mark.parametrize(
    "frequency,time_mjd,error_type",
    (
        (99e6, 60_000.0, BeamFrequencyDomainError),
        (131e6, 60_000.0, BeamFrequencyDomainError),
        (0.0, 60_000.0, BeamFrequencyDomainError),
        (np.nan, 60_000.0, NonFiniteBeamResponseError),
        (100e6, np.inf, NonFiniteBeamResponseError),
        (np.float64(100e6), 60_000.0, NonFiniteBeamResponseError),
    ),
)
def test_frequency_and_time_input_contract_rejects_invalid_values(
    tmp_path: Path,
    frequency: Any,
    time_mjd: Any,
    error_type: type[Exception],
) -> None:
    loaded = _load(tmp_path)
    with pytest.raises(error_type):
        loaded.evaluator.evaluate_numpy(
            np.array([0.0]),
            np.array([0.0]),
            frequency,
            time_mjd,
        )


def test_results_are_read_only_independently_owned_and_precision_cast_once(
    tmp_path: Path,
) -> None:
    loaded = _load(tmp_path, precision=PrecisionConfig.fast())
    args = (np.array([0.2]), np.array([0.3]), 110e6, 60_000.0)
    first = loaded.evaluator.evaluate_numpy(*args)
    second = loaded.evaluator.evaluate_numpy(*args)

    assert first.dtype == np.dtype(np.complex64)
    assert not first.flags.writeable
    assert not second.flags.writeable
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError):
        first[0, 0, 0] = 0.0


class _ControlledInterpolationBeam:
    def __init__(
        self,
        *,
        response: object | None = None,
        tracker: dict[str, Any] | None = None,
    ) -> None:
        self.response = response
        self.tracker = tracker or {"active": 0, "maximum": 0, "lock": threading.Lock()}

    def interp(self, **kwargs: Any) -> object:
        with self.tracker["lock"]:
            self.tracker["active"] += 1
            self.tracker["maximum"] = max(
                self.tracker["maximum"], self.tracker["active"]
            )
        time.sleep(0.03)
        with self.tracker["lock"]:
            self.tracker["active"] -= 1
        if self.response is not None:
            return self.response
        count = len(kwargs["az_array"])
        data = np.zeros((2, 2, 1, count), dtype=np.complex128)
        data[0, 0] = 1.0 + 0.25j
        data[1, 1] = 1.0 + 0.25j
        return data, None


def _controlled_evaluator(beam: Any):
    from radiosim.core.beam.runtime import _UVBeamScalarEvaluator

    return _UVBeamScalarEvaluator(
        beam=beam,
        identity="beam-0000-controlled",
        frequency_interpolation="linear",
        frequencies_hz=np.array([100e6, 110e6]),
        scalar_absolute_tolerance=1e-12,
        scalar_relative_tolerance=1e-10,
        feature_scale_rad=0.1,
        result_dtype=np.dtype(np.complex128),
    )


def test_one_evaluator_serializes_dependency_interpolation() -> None:
    beam = _ControlledInterpolationBeam()
    evaluator = _controlled_evaluator(beam)

    def evaluate() -> np.ndarray:
        return evaluator.evaluate_numpy(
            np.array([0.2]),
            np.array([0.3]),
            105e6,
            60_000.0,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(lambda _: evaluate(), range(4)))

    assert beam.tracker["maximum"] == 1
    for result in results:
        assert not result.flags.writeable


def test_independent_evaluators_do_not_share_a_global_lock() -> None:
    tracker = {"active": 0, "maximum": 0, "lock": threading.Lock()}
    first = _controlled_evaluator(_ControlledInterpolationBeam(tracker=tracker))
    second = _controlled_evaluator(_ControlledInterpolationBeam(tracker=tracker))

    def evaluate(evaluator: Any) -> np.ndarray:
        return evaluator.evaluate_numpy(
            np.array([0.2]),
            np.array([0.3]),
            105e6,
            60_000.0,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(evaluate, first), pool.submit(evaluate, second))
        tuple(future.result() for future in futures)

    assert tracker["maximum"] == 2


@pytest.mark.parametrize(
    "response,error_type",
    (
        (
            (np.ones((2, 2, 1, 1), dtype=np.complex128), np.eye(2)),
            UnsupportedBeamBasisError,
        ),
        ((np.ones((2, 2, 2, 1), dtype=np.complex128), None), UnsupportedBeamBasisError),
        ((np.full((2, 2, 1, 1), np.nan + 0j), None), NonFiniteBeamResponseError),
        (
            (
                np.array([[[[1.0]], [[0.2]]], [[[0.0]], [[1.0]]]], dtype=np.complex128),
                None,
            ),
            UnsupportedBeamBasisError,
        ),
        (
            (
                np.array([[[[1.0]], [[0.0]]], [[[0.0]], [[0.8]]]], dtype=np.complex128),
                None,
            ),
            UnsupportedBeamBasisError,
        ),
    ),
)
def test_interpolated_response_is_revalidated(
    response: object,
    error_type: type[Exception],
) -> None:
    evaluator = _controlled_evaluator(_ControlledInterpolationBeam(response=response))
    with pytest.raises(error_type):
        evaluator.evaluate_numpy(
            np.array([0.2]),
            np.array([0.3]),
            105e6,
            60_000.0,
        )


@pytest.mark.parametrize("factor", (0.5, 1.0))
def test_tolerated_interpolated_scalar_noise_is_canonicalized(
    factor: float,
) -> None:
    scalar = 1.0 + 0.25j
    bound = 1e-12 + 1e-10 * abs(scalar)
    response = np.zeros((2, 2, 1, 1), dtype=np.complex128)
    response[0, 0, 0, 0] = scalar
    response[1, 1, 0, 0] = scalar + 0.5 * bound
    response[0, 1, 0, 0] = factor * bound
    response[1, 0, 0, 0] = -factor * bound
    evaluator = _controlled_evaluator(
        _ControlledInterpolationBeam(response=(response, None))
    )

    result = evaluator.evaluate_numpy(
        np.array([0.2]),
        np.array([0.3]),
        105e6,
        60_000.0,
    )

    np.testing.assert_array_equal(
        result,
        np.array([[[1.0 + 0.25j, 0.0], [0.0, 1.0 + 0.25j]]]),
    )


def test_interpolated_scalar_noise_above_tolerance_is_rejected() -> None:
    bound = 1e-12 + 1e-10
    response = np.zeros((2, 2, 1, 1), dtype=np.complex128)
    response[0, 0, 0, 0] = 1.0
    response[1, 1, 0, 0] = 1.0
    response[0, 1, 0, 0] = 1.01 * bound
    evaluator = _controlled_evaluator(
        _ControlledInterpolationBeam(response=(response, None))
    )

    with pytest.raises(UnsupportedBeamBasisError):
        evaluator.evaluate_numpy(
            np.array([0.2]),
            np.array([0.3]),
            105e6,
            60_000.0,
        )


@pytest.mark.parametrize(
    "error",
    (
        RuntimeError("hostile dependency runtime failure"),
        IndexError("hostile dependency index failure"),
        OverflowError("hostile dependency overflow failure"),
    ),
)
def test_unexpected_interpolation_failures_are_typed_and_chained(
    error: Exception,
) -> None:
    evaluator = _controlled_evaluator(_ControlledInterpolationBeam())

    def fail(**kwargs: Any) -> object:
        raise error

    evaluator._beam.interp = fail
    with pytest.raises(BeamEvaluationError) as caught:
        evaluator.evaluate_numpy(
            np.array([0.2]),
            np.array([0.3]),
            105e6,
            60_000.0,
        )
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "altitude,azimuth",
    (
        (np.array(["bad"]), np.array([0.0])),
        (np.array([0.0]), np.array([object()])),
        (np.array([(0.0,)], dtype=[("value", "f8")]), np.array([0.0])),
    ),
)
def test_hostile_coordinate_arrays_fail_with_typed_angular_error(
    altitude: np.ndarray,
    azimuth: np.ndarray,
) -> None:
    evaluator = _controlled_evaluator(_ControlledInterpolationBeam())
    with pytest.raises(BeamAngularDomainError) as caught:
        evaluator.evaluate_numpy(altitude, azimuth, 105e6, 60_000.0)
    assert caught.value.__cause__ is not None


def test_empty_directions_and_failure_recovery_are_deterministic() -> None:
    beam = _ControlledInterpolationBeam()
    evaluator = _controlled_evaluator(beam)
    empty = evaluator.evaluate_numpy(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        105e6,
        60_000.0,
    )

    assert empty.shape == (0, 2, 2)
    assert not empty.flags.writeable
    assert beam.tracker["maximum"] == 0

    beam.response = (
        np.ones((2, 2, 2, 1), dtype=np.complex128),
        None,
    )
    with pytest.raises(UnsupportedBeamBasisError):
        evaluator.evaluate_numpy(
            np.array([0.2]),
            np.array([0.3]),
            105e6,
            60_000.0,
        )
    beam.response = None
    recovered = evaluator.evaluate_numpy(
        np.array([0.2]),
        np.array([0.3]),
        105e6,
        60_000.0,
    )
    np.testing.assert_array_equal(
        recovered,
        np.array([[[1.0 + 0.25j, 0.0], [0.0, 1.0 + 0.25j]]]),
    )


class _CountingLoader:
    def __init__(self, *, fail_on: set[int] | None = None) -> None:
        self.calls: list[Path] = []
        self.fail_on = set() if fail_on is None else set(fail_on)

    def read(self, path: Path):
        self.calls.append(path)
        if len(self.calls) in self.fail_on:
            raise OSError(f"controlled load failure {len(self.calls)}")
        return build_scalar_efield_uvbeam()


def _resolved_runtime_state(
    tmp_path: Path,
    beams: dict[str, object],
    *,
    heterogeneous_diameters: bool = False,
):
    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.instrument_resolution import resolve_instrument
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config

    data = valid_config_mapping(tmp_path, beams=beams)
    if heterogeneous_diameters:
        Path(data["instrument"]["source"]["path"]).write_text(
            "Name Number BeamID E N U Diameter\n"
            "ANT0 0 0 0.0 0.0 0.0 10.0\n"
            "ANT1 1 0 14.0 0.0 0.0 20.0\n"
        )
    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )
    runtime = bundle.runtime
    instrument = resolve_instrument(runtime.instrument)
    return (
        resolve_beam_assignments(runtime.beams, instrument),
        runtime.frequency.channel_frequencies_hz,
        runtime.execution.precision,
    )


def _load_system(
    state,
    frequencies: tuple[float, ...],
    precision: PrecisionConfig,
    *,
    loader: _CountingLoader,
):
    from radiosim.core.beam.runtime import _load_beam_system

    return _load_beam_system(
        state,
        observation_frequencies_hz=frequencies,
        precision=precision,
        loader=loader,
    )


def test_beam_system_is_factory_only_final_and_exposes_immutable_state(
    tmp_path: Path,
) -> None:
    from radiosim.core.beam import BeamSystem, LoadedBeamState

    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic"},
    )
    system = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    )

    assert type(system) is BeamSystem
    assert type(system.state) is LoadedBeamState
    assert system.state.resolved is not state
    with pytest.raises(TypeError):
        BeamSystem()
    with pytest.raises(TypeError):

        class MutableBeamSystem(BeamSystem):
            pass


@pytest.mark.parametrize(
    ("model", "expected_handlers"),
    [
        (
            {
                "kind": "circular_aperture",
                "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
            },
            2,
        ),
        (
            {
                "kind": "rectangular_aperture",
                "north_length_m": 14.0,
                "east_length_m": 12.0,
            },
            1,
        ),
        (
            {
                "kind": "elliptical_aperture",
                "north_diameter_m": 14.0,
                "east_diameter_m": 12.0,
            },
            1,
        ),
    ],
)
def test_analytic_dedup_uses_only_effective_science(
    tmp_path: Path,
    model: dict[str, object],
    expected_handlers: int,
) -> None:
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic", "model": model},
        heterogeneous_diameters=True,
    )
    system = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    )

    assert len(system.state.handlers) == expected_handlers
    assert len(system.state.assignment_handler_ids) == 2


def test_analytic_scientific_fingerprint_includes_schema_and_loaded_validity(
    tmp_path: Path,
) -> None:
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic"},
    )
    handler = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    ).state.handlers[0]
    assignment = state.assignments[0]
    expected = beam_models._canonical_digest(
        {
            "schema_version": "tier3-beam-v1",
            "kind": "analytic_handler",
            "contract": "tier3-scalar-v1",
            "model": assignment.definition.model,
            "effective_dimensions": (
                beam_models._effective_assignment_dimensions(
                    assignment.definition,
                    assignment.antenna_diameter_m,
                )
            ),
            "derived_edge_taper_db": None,
            "n_radial": None,
            "observation_frequencies_hz": frequencies,
            "voltage_feature_scale_by_frequency": (
                handler.voltage_feature_scale_by_frequency
            ),
        }
    )

    assert handler.scientific_fingerprint == expected


def test_same_fits_preload_key_loads_once_per_system_and_never_globally(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shared.beamfits"
    source.write_bytes(b"transport bytes")
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": source.name},
        },
    )
    loader = _CountingLoader()

    first = _load_system(state, frequencies, precision, loader=loader)
    second = _load_system(state, frequencies, precision, loader=loader)

    assert len(first.state.handlers) == 1
    assert len(second.state.handlers) == 1
    assert len(loader.calls) == 2


def test_distinct_fits_options_and_paths_are_never_preload_deduplicated(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.beamfits"
    second = tmp_path / "second.beamfits"
    first.write_bytes(b"same transport")
    second.write_bytes(b"same transport")
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {
            "mode": "per_antenna_fits",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {
                        "kind": "fits",
                        "path": first.name,
                        "frequency_interpolation": "linear",
                    },
                },
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {
                        "kind": "fits",
                        "path": second.name,
                        "frequency_interpolation": "cubic",
                    },
                },
            ],
        },
    )
    loader = _CountingLoader()

    system = _load_system(state, frequencies, precision, loader=loader)

    assert len(loader.calls) == 2
    assert len(system.state.handlers) == 2


def test_mixed_system_lookup_is_canonical_and_unknown_is_fixed_error(
    tmp_path: Path,
) -> None:
    from radiosim.core.beam import InconsistentBeamAssignmentError
    from radiosim.core.instrument import AntennaId

    source = tmp_path / "mixed.beamfits"
    source.write_bytes(b"transport bytes")
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {
            "mode": "mixed",
            "analytic_model": {"kind": "circular_aperture"},
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "analytic"},
                },
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {"kind": "fits", "path": source.name},
                },
            ],
        },
    )
    system = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    )
    altitude = np.array([np.pi / 2.0])
    azimuth = np.array([0.0])

    analytic = system.evaluate_jones(
        AntennaId(0, "ANT0"),
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        frequency_hz=frequencies[0],
        time_mjd=60_000.0,
    )
    fits = system.evaluate_jones(
        AntennaId(1, "ANT1"),
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        frequency_hz=frequencies[0],
        time_mjd=60_000.0,
    )

    np.testing.assert_array_equal(analytic, np.eye(2)[None, ...])
    assert fits.shape == (1, 2, 2)
    assert fits[0, 0, 0] == fits[0, 1, 1]
    np.testing.assert_array_equal(fits[0, 0, 1], 0.0)
    np.testing.assert_array_equal(fits[0, 1, 0], 0.0)
    message = (
        "BeamSystem has no handler assignment for canonical antenna "
        "number=99, name='MISSING'; loaded beam state is inconsistent."
    )
    with pytest.raises(InconsistentBeamAssignmentError, match="^" + message + "$"):
        system.evaluate_jones(
            AntennaId(99, "MISSING"),
            altitude_rad=altitude,
            azimuth_rad=azimuth,
            frequency_hz=frequencies[0],
            time_mjd=60_000.0,
        )


def test_late_load_failure_publishes_nothing_and_retry_reloads_every_handler(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.beamfits"
    second = tmp_path / "second.beamfits"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {
            "mode": "per_antenna_fits",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "fits", "path": first.name},
                },
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {"kind": "fits", "path": second.name},
                },
            ],
        },
    )
    loader = _CountingLoader(fail_on={2})

    from radiosim.core.beam import BeamFileReadError

    with pytest.raises(BeamFileReadError) as caught:
        _load_system(state, frequencies, precision, loader=loader)
    assert str(caught.value.__cause__) == "controlled load failure 2"
    recovered = _load_system(state, frequencies, precision, loader=loader)

    assert len(recovered.state.handlers) == 2
    assert len(loader.calls) == 4


@pytest.mark.parametrize(
    "model",
    [
        {"kind": "circular_aperture"},
        {
            "kind": "rectangular_aperture",
            "north_length_m": 14.0,
            "east_length_m": 12.0,
        },
        {
            "kind": "elliptical_aperture",
            "north_diameter_m": 14.0,
            "east_diameter_m": 12.0,
        },
        {
            "kind": "analytical_illumination",
            "illumination": {"kind": "corrugated_horn"},
        },
        {
            "kind": "numerical_illumination",
            "illumination": {"kind": "open_waveguide"},
        },
    ],
)
def test_all_analytic_models_produce_owned_read_only_scalar_jones(
    tmp_path: Path,
    model: dict[str, object],
) -> None:
    from radiosim.core.instrument import AntennaId

    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic", "model": model},
    )
    system = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    )
    altitude = np.array([np.pi / 2.0, 0.0, -0.1])
    azimuth = np.array([0.0, np.pi / 2.0, np.pi])

    first = system.evaluate_jones(
        AntennaId(0, "ANT0"),
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        frequency_hz=frequencies[0],
        time_mjd=60_000.0,
    )
    second = system.evaluate_jones(
        AntennaId(0, "ANT0"),
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        frequency_hz=frequencies[0],
        time_mjd=60_000.0,
    )

    assert first.shape == (3, 2, 2)
    assert first.dtype == np.dtype(np.complex128)
    assert not first.flags.writeable
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first[0], np.eye(2))
    np.testing.assert_array_equal(first[2], np.zeros((2, 2)))


@pytest.mark.parametrize(
    ("altitude", "azimuth"),
    [
        (np.array(["1.0"]), np.array([0.0])),
        (np.array([1.0]), np.array([0.0 + 1.0j])),
    ],
)
def test_analytic_beam_rejects_nonreal_coordinate_dtypes(
    tmp_path: Path,
    altitude: np.ndarray,
    azimuth: np.ndarray,
) -> None:
    from radiosim.core.beam import BeamAngularDomainError
    from radiosim.core.instrument import AntennaId

    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic"},
    )
    system = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    )

    with pytest.raises(BeamAngularDomainError):
        system.evaluate_jones(
            AntennaId(0, "ANT0"),
            altitude_rad=altitude,
            azimuth_rad=azimuth,
            frequency_hz=frequencies[0],
            time_mjd=60_000.0,
        )


def test_public_factory_and_evaluation_signatures_hide_loader_seam() -> None:
    import inspect

    from radiosim.core.beam import BeamSystem, load_beam_system

    factory = inspect.signature(load_beam_system)
    assert tuple(factory.parameters) == (
        "resolved_state",
        "observation_frequencies_hz",
        "precision",
    )
    assert factory.parameters["observation_frequencies_hz"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert factory.parameters["precision"].kind is inspect.Parameter.KEYWORD_ONLY
    operation = inspect.signature(BeamSystem.evaluate_jones)
    assert tuple(operation.parameters) == (
        "self",
        "antenna_id",
        "altitude_rad",
        "azimuth_rad",
        "frequency_hz",
        "time_mjd",
        "backend",
    )
    assert operation.parameters["backend"].default is None


def test_symlink_to_same_resolved_fits_target_is_one_preload_key(
    tmp_path: Path,
) -> None:
    source = tmp_path / "target.beamfits"
    alias = tmp_path / "alias.beamfits"
    source.write_bytes(b"shared target")
    alias.symlink_to(source)
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {
            "mode": "per_antenna_fits",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "fits", "path": source.name},
                },
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {"kind": "fits", "path": alias.name},
                },
            ],
        },
    )
    loader = _CountingLoader()

    system = _load_system(state, frequencies, precision, loader=loader)

    assert len(loader.calls) == 1
    assert len(system.state.handlers) == 1


def test_same_fits_path_with_different_interpolation_is_two_preload_keys(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shared.beamfits"
    source.write_bytes(b"one path")
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {
            "mode": "per_antenna_fits",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {
                        "kind": "fits",
                        "path": source.name,
                        "frequency_interpolation": "linear",
                    },
                },
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {
                        "kind": "fits",
                        "path": source.name,
                        "frequency_interpolation": "cubic",
                    },
                },
            ],
        },
    )
    loader = _CountingLoader()

    system = _load_system(state, frequencies, precision, loader=loader)

    assert len(loader.calls) == 2
    assert len(system.state.handlers) == 2


def test_repeated_fits_evaluation_does_not_repeat_file_io(tmp_path: Path) -> None:
    from radiosim.core.instrument import AntennaId

    source = tmp_path / "shared.beamfits"
    source.write_bytes(b"transport")
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": source.name},
        },
    )
    loader = _CountingLoader()
    system = _load_system(state, frequencies, precision, loader=loader)
    arguments = {
        "altitude_rad": np.array([0.0, np.pi / 2.0]),
        "azimuth_rad": np.array([0.0, np.pi / 2.0]),
        "frequency_hz": frequencies[0],
        "time_mjd": 60_000.0,
    }

    first = system.evaluate_jones(AntennaId(0, "ANT0"), **arguments)
    second = system.evaluate_jones(AntennaId(0, "ANT0"), **arguments)

    assert len(loader.calls) == 1
    assert not np.shares_memory(first, second)
    assert not first.flags.writeable
    assert not second.flags.writeable


def test_per_antenna_fits_preserves_distinct_phase_at_native_and_interpolated_hz(
    tmp_path: Path,
) -> None:
    from radiosim.core.instrument import AntennaId

    first_path = tmp_path / "canonical.beamfits"
    second_path = tmp_path / "distinct.beamfits"
    first_path.write_bytes(b"canonical transport")
    second_path.write_bytes(b"distinct transport")
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {
            "mode": "per_antenna_fits",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "fits", "path": first_path.name},
                },
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {"kind": "fits", "path": second_path.name},
                },
            ],
        },
    )

    class VariantLoader:
        def __init__(self) -> None:
            self.calls: list[Path] = []

        def read(self, path: Path):
            self.calls.append(path)
            variant = (
                BeamScienceVariant.CANONICAL
                if len(self.calls) == 1
                else BeamScienceVariant.DISTINCT
            )
            return build_scalar_efield_uvbeam(variant=variant)

    loader = VariantLoader()
    system = _load_system(
        state,
        frequencies,
        precision,
        loader=loader,
    )
    directions = {
        "altitude_rad": np.array([0.8]),
        "azimuth_rad": np.array([0.3]),
        "time_mjd": 60_000.0,
    }

    for frequency_hz in (frequencies[0], frequencies[1]):
        canonical = system.evaluate_jones(
            AntennaId(0, "ANT0"),
            frequency_hz=frequency_hz,
            **directions,
        )
        distinct = system.evaluate_jones(
            AntennaId(1, "ANT1"),
            frequency_hz=frequency_hz,
            **directions,
        )
        assert canonical[0, 0, 0].imag * distinct[0, 0, 0].imag < 0.0
        assert canonical[0, 0, 0] != distinct[0, 0, 0]
        np.testing.assert_array_equal(canonical[:, 0, 1], 0.0)
        np.testing.assert_array_equal(distinct[:, 1, 0], 0.0)

    assert len(loader.calls) == 2


@pytest.mark.parametrize(
    ("model", "north_dimension_m", "east_dimension_m"),
    [
        (
            {
                "kind": "rectangular_aperture",
                "north_length_m": 14.0,
                "east_length_m": 7.0,
            },
            14.0,
            7.0,
        ),
        (
            {
                "kind": "elliptical_aperture",
                "north_diameter_m": 14.0,
                "east_diameter_m": 7.0,
            },
            14.0,
            7.0,
        ),
    ],
)
def test_two_axis_models_use_radiosim_north_and_east_azimuth_convention(
    tmp_path: Path,
    model: dict[str, object],
    north_dimension_m: float,
    east_dimension_m: float,
) -> None:
    from radiosim.core.instrument import AntennaId

    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic", "model": model},
        heterogeneous_diameters=True,
    )
    system = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    )
    altitude = np.array([1.4, 1.4])
    azimuth = np.array([0.0, np.pi / 2.0])
    result = system.evaluate_jones(
        AntennaId(0, "ANT0"),
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        frequency_hz=frequencies[0],
        time_mjd=60_000.0,
    )
    theta = np.pi / 2.0 - altitude[0]
    wavelength = 299_792_458.0 / frequencies[0]
    north_u = north_dimension_m * np.sin(theta) / wavelength
    east_u = east_dimension_m * np.sin(theta) / wavelength
    if model["kind"] == "rectangular_aperture":
        expected = np.array([np.sinc(north_u), np.sinc(east_u)])
    else:
        from radiosim.core.jones.beam.analytic.taper import uniform_taper

        expected = uniform_taper(np.array([north_u, east_u]))

    np.testing.assert_allclose(result[:, 0, 0].real, expected)
    assert (
        system.state.handlers[0].voltage_feature_scale_by_frequency[0][1]
        == wavelength / north_dimension_m
    )


def test_analytic_beam_precision_is_preserved_without_fallback(
    tmp_path: Path,
) -> None:
    from radiosim.core.beam import UnsupportedBeamPrecisionError
    from radiosim.core.instrument import AntennaId
    from radiosim.core.precision import COMPLEX256_AVAILABLE, JonesPrecision

    state, frequencies, _precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic"},
    )
    arguments = {
        "altitude_rad": np.array([np.pi / 2.0]),
        "azimuth_rad": np.array([0.0]),
        "frequency_hz": frequencies[0],
        "time_mjd": 60_000.0,
    }
    fast = _load_system(
        state,
        frequencies,
        PrecisionConfig.fast(),
        loader=_CountingLoader(),
    )
    assert fast.evaluate_jones(AntennaId(0, "ANT0"), **arguments).dtype == np.dtype(
        np.complex64
    )

    extended = PrecisionConfig(jones=JonesPrecision(beam="float128"))
    if COMPLEX256_AVAILABLE:
        precise = _load_system(
            state,
            frequencies,
            extended,
            loader=_CountingLoader(),
        )
        assert precise.evaluate_jones(
            AntennaId(0, "ANT0"),
            **arguments,
        ).dtype == np.dtype(np.complex256)
    else:
        with pytest.raises(UnsupportedBeamPrecisionError):
            _load_system(
                state,
                frequencies,
                extended,
                loader=_CountingLoader(),
            )


def test_backend_conversion_uses_resolved_numpy_and_numba_backends(
    tmp_path: Path,
) -> None:
    from radiosim.backends.numba_backend import NumbaBackend
    from radiosim.backends.numpy_backend import NumPyBackend
    from radiosim.core.instrument import AntennaId

    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic"},
    )
    system = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    )
    arguments = {
        "altitude_rad": np.array([np.pi / 2.0]),
        "azimuth_rad": np.array([0.0]),
        "frequency_hz": frequencies[0],
        "time_mjd": 60_000.0,
    }

    for backend in (
        NumPyBackend(precision=precision),
        NumbaBackend(mode="cpu", precision=precision),
    ):
        result = system.evaluate_jones(
            AntennaId(0, "ANT0"),
            backend=backend,
            **arguments,
        )
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.dtype(np.complex128)
        assert not result.flags.writeable
        np.testing.assert_array_equal(result, np.eye(2)[None, ...])


def test_unknown_forged_and_subclassed_antenna_ids_never_fall_back(
    tmp_path: Path,
) -> None:
    from radiosim.core.beam import InconsistentBeamAssignmentError
    from radiosim.core.instrument import AntennaId

    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic"},
    )
    system = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    )
    arguments = {
        "altitude_rad": np.array([np.pi / 2.0]),
        "azimuth_rad": np.array([0.0]),
        "frequency_hz": frequencies[0],
        "time_mjd": 60_000.0,
    }

    class AntennaSubclass(AntennaId):
        pass

    forged = object.__new__(AntennaId)
    object.__setattr__(forged, "number", -1)
    object.__setattr__(forged, "name", "FORGED")
    for invalid in (
        AntennaId(99, "MISSING"),
        AntennaSubclass(0, "ANT0"),
        forged,
    ):
        with pytest.raises(InconsistentBeamAssignmentError):
            system.evaluate_jones(invalid, **arguments)
    for raw in (0, "ANT0", {"number": 0}):
        with pytest.raises(TypeError):
            system.evaluate_jones(raw, **arguments)


def test_loaded_state_rejects_duplicate_missing_extra_and_reordered_mappings(
    tmp_path: Path,
) -> None:
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic"},
        heterogeneous_diameters=True,
    )
    loaded = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    ).state
    first_pair, second_pair = loaded.assignment_handler_ids

    with pytest.raises(ValueError, match="handler_id values must be unique"):
        replace(loaded, handlers=(loaded.handlers[0], loaded.handlers[0]))
    with pytest.raises(ValueError, match="cover every resolved assignment"):
        replace(loaded, assignment_handler_ids=(first_pair,))
    with pytest.raises(ValueError, match="canonical assignment order"):
        replace(
            loaded,
            assignment_handler_ids=(second_pair, first_pair),
        )
    with pytest.raises(ValueError, match="reference a loaded handler_id"):
        replace(
            loaded,
            assignment_handler_ids=(
                first_pair,
                (second_pair[0], "beam-9999-deadbeefdead"),
            ),
        )
    with pytest.raises(ValueError, match="first canonical assignment use"):
        replace(
            loaded,
            assignment_handler_ids=(
                first_pair,
                (second_pair[0], first_pair[1]),
            ),
        )
    with pytest.raises(
        ValueError,
        match="loaded_fingerprint does not match",
    ):
        replace(loaded, loaded_fingerprint="0" * 64)


def test_loaded_state_is_hashable_deterministic_and_snapshot_detached(
    tmp_path: Path,
) -> None:
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {"mode": "analytic"},
    )
    first = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    ).state
    second = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    ).state

    assert first == second
    assert hash(first) == hash(second)
    snapshot = first.to_snapshot()
    snapshot["assignment_handler_ids"][0][0]["name"] = "MUTATED"
    assert first.assignment_handler_ids[0][0].name == "ANT0"


def test_loaded_handler_kind_and_file_coherence_is_exact(tmp_path: Path) -> None:
    source = tmp_path / "shared.beamfits"
    source.write_bytes(b"transport")
    state, frequencies, precision = _resolved_runtime_state(
        tmp_path,
        {
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": source.name},
        },
    )
    handler = _load_system(
        state,
        frequencies,
        precision,
        loader=_CountingLoader(),
    ).state.handlers[0]

    with pytest.raises(ValueError, match="analytic handlers require file=None"):
        replace(handler, kind="analytic")
    with pytest.raises(TypeError, match="file must be an exact BeamFileProvenance"):
        replace(handler, file=None)
