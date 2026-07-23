"""Standalone scalar BeamFITS evaluator tests for Tier 3D."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    canonical_azimuth_grid,
    canonical_zenith_angle_grid,
    scalar_voltage_reference,
    write_scalar_efield_beamfits,
)


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
