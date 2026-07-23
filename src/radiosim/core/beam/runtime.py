"""Private standalone BeamFITS evaluation primitives for Tier 3D."""

from __future__ import annotations

import importlib
import math
import threading
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from radiosim.core.beam.errors import (
    BeamAngularDomainError,
    BeamDependencyError,
    BeamEvaluationError,
    BeamFileReadError,
    BeamFrequencyDomainError,
    NonFiniteBeamResponseError,
    UnsupportedBeamBasisError,
    UnsupportedBeamPrecisionError,
)

_FREQUENCY_MATCH_TOLERANCE_HZ = 1e-6


class _UVBeamLike(Protocol):
    """Structural subset of UVBeam privately consumed by Tier 3D."""

    data_array: NDArray[np.complexfloating[Any, Any]]
    basis_vector_array: NDArray[np.floating[Any]]
    freq_array: NDArray[np.floating[Any]]
    axis1_array: NDArray[np.floating[Any]]
    axis2_array: NDArray[np.floating[Any]]

    def interp(self, **kwargs: Any) -> object: ...


class _UVBeamLoaderProtocol(Protocol):  # pyright: ignore[reportUnusedClass]
    """Private injectable reader boundary for one complete BeamFITS file."""

    def read(self, path: Path) -> _UVBeamLike: ...


class _ProductionUVBeamLoader:  # pyright: ignore[reportUnusedClass]
    """Lazy pyuvdata 3.2.1 reader used only at the actual load boundary."""

    __slots__ = ()

    def read(self, path: Path) -> _UVBeamLike:
        try:
            module = importlib.import_module("pyuvdata")
        except (ImportError, ModuleNotFoundError) as exc:
            raise BeamDependencyError(
                "BeamFITS loading requires pyuvdata 3.2.1; install the pinned "
                "RadioSim dependency and retry."
            ) from exc
        beam = module.UVBeam()
        result = beam.read_beamfits(path)
        if result is not None:
            raise BeamFileReadError(
                f"BeamFITS {path}: UVBeam.read_beamfits returned {result!r}; "
                "Tier 3 requires the pinned pyuvdata 3.2.1 mutating-None contract."
            )
        return beam


class _BeamEvaluator(Protocol):  # pyright: ignore[reportUnusedClass]
    """Private evaluator contract shared by later canonical beam runtimes."""

    def evaluate_numpy(
        self,
        altitude_rad: np.ndarray,
        azimuth_rad: np.ndarray,
        frequency_hz: float,
        time_mjd: float,
    ) -> np.ndarray: ...

    def voltage_feature_scale_rad(self, frequency_hz: float) -> float: ...


def _require_exact_finite_float(value: Any, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise NonFiniteBeamResponseError(
            f"{field_name} must be an exact finite Python float; observed {value!r}."
        )
    return value


def _is_intrinsic_frequency(frequencies_hz: np.ndarray, target_hz: float) -> bool:
    return bool(
        np.min(np.abs(frequencies_hz - target_hz)) < _FREQUENCY_MATCH_TOLERANCE_HZ
    )


def _preflight_frequency(
    *,
    frequencies_hz: np.ndarray,
    target_hz: float,
    interpolation_kind: str,
    identity: str,
) -> None:
    if target_hz <= 0.0:
        raise BeamFrequencyDomainError(
            f"{identity}: frequency_hz={target_hz!r} must be finite and positive."
        )
    minimum = float(frequencies_hz[0])
    maximum = float(frequencies_hz[-1])
    if target_hz < minimum or target_hz > maximum:
        raise BeamFrequencyDomainError(
            f"{identity}: frequency_hz={target_hz!r} is outside the closed loaded "
            f"interval [{minimum!r}, {maximum!r}] Hz; select an in-domain channel."
        )
    if _is_intrinsic_frequency(frequencies_hz, target_hz):
        return
    required = 2 if interpolation_kind == "linear" else 4
    if frequencies_hz.size < required:
        raise BeamFrequencyDomainError(
            f"{identity}: {interpolation_kind} interpolation at "
            f"frequency_hz={target_hz!r} requires at least {required} intrinsic "
            f"channels; the BeamFITS source has {frequencies_hz.size}."
        )


class _UVBeamScalarEvaluator:  # pyright: ignore[reportUnusedClass]
    """Private owned, locked evaluator for the accepted scalar ``e I2`` subset."""

    __slots__ = (
        "_beam",
        "_feature_scale_rad",
        "_frequency_interpolation",
        "_frequencies_hz",
        "_identity",
        "_lock",
        "_result_dtype",
        "_scalar_atol",
        "_scalar_rtol",
    )

    def __init__(
        self,
        *,
        beam: _UVBeamLike,
        identity: str,
        frequency_interpolation: str,
        frequencies_hz: np.ndarray,
        scalar_absolute_tolerance: float,
        scalar_relative_tolerance: float,
        feature_scale_rad: float,
        result_dtype: np.dtype[Any],
    ) -> None:
        self._beam = beam
        self._identity = identity
        self._frequency_interpolation = frequency_interpolation
        self._frequencies_hz = np.array(
            frequencies_hz,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        self._frequencies_hz.setflags(write=False)
        self._scalar_atol = scalar_absolute_tolerance
        self._scalar_rtol = scalar_relative_tolerance
        self._feature_scale_rad = feature_scale_rad
        self._result_dtype = np.dtype(result_dtype)
        self._lock = threading.RLock()

    def voltage_feature_scale_rad(self, frequency_hz: float) -> float:
        """Return the fixed native-grid representation scale for an in-domain Hz."""
        target = _require_exact_finite_float(frequency_hz, "frequency_hz")
        _preflight_frequency(
            frequencies_hz=self._frequencies_hz,
            target_hz=target,
            interpolation_kind=self._frequency_interpolation,
            identity=self._identity,
        )
        return self._feature_scale_rad

    def evaluate_numpy(
        self,
        altitude_rad: np.ndarray,
        azimuth_rad: np.ndarray,
        frequency_hz: float,
        time_mjd: float,
    ) -> np.ndarray:
        """Evaluate owned read-only Jones matrices in RadioSim coordinates."""
        if type(altitude_rad) is not np.ndarray or type(azimuth_rad) is not np.ndarray:
            raise BeamAngularDomainError(
                f"{self._identity}: altitude_rad and azimuth_rad must be "
                "one-dimensional NumPy arrays."
            )
        if altitude_rad.ndim != 1 or azimuth_rad.ndim != 1:
            raise BeamAngularDomainError(
                f"{self._identity}: altitude_rad and azimuth_rad must be "
                "one-dimensional NumPy arrays."
            )
        if altitude_rad.shape != azimuth_rad.shape:
            raise BeamAngularDomainError(
                f"{self._identity}: altitude_rad shape {altitude_rad.shape!r} and "
                f"azimuth_rad shape {azimuth_rad.shape!r} must match."
            )
        try:
            if (
                altitude_rad.dtype.kind not in "fiu"
                or azimuth_rad.dtype.kind not in "fiu"
            ):
                raise TypeError("direction arrays must have real numeric dtypes")
            altitude_values = np.asarray(altitude_rad, dtype=np.float64)
            azimuth_values = np.asarray(azimuth_rad, dtype=np.float64)
            finite = bool(
                np.all(np.isfinite(altitude_values))
                and np.all(np.isfinite(azimuth_values))
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise BeamAngularDomainError(
                f"{self._identity}: direction arrays must contain real numeric "
                "radian values."
            ) from exc
        if not finite:
            raise NonFiniteBeamResponseError(
                f"{self._identity}: direction coordinates must be finite."
            )
        if np.any(altitude_values < -np.pi / 2.0) or np.any(
            altitude_values > np.pi / 2.0
        ):
            raise BeamAngularDomainError(
                f"{self._identity}: altitude values must lie inside the closed "
                "interval [-pi/2, pi/2] radians; horizon and zenith are included."
            )

        target = _require_exact_finite_float(frequency_hz, "frequency_hz")
        _ = _require_exact_finite_float(time_mjd, "time_mjd")
        _preflight_frequency(
            frequencies_hz=self._frequencies_hz,
            target_hz=target,
            interpolation_kind=self._frequency_interpolation,
            identity=self._identity,
        )

        count = altitude_rad.size
        output = np.zeros((count, 2, 2), dtype=np.complex128)
        visible = altitude_values >= 0.0
        if np.any(visible):
            visible_altitude = altitude_values[visible]
            visible_azimuth = azimuth_values[visible]
            azimuth_uv = (np.pi / 2.0 - visible_azimuth) % (2.0 * np.pi)
            zenith_angle = np.pi / 2.0 - visible_altitude
            try:
                with self._lock:
                    interpolated = self._beam.interp(
                        az_array=azimuth_uv,
                        za_array=zenith_angle,
                        interpolation_function="az_za_simple",
                        freq_array=np.array([target], dtype=np.float64),
                        freq_interp_kind=self._frequency_interpolation,
                        freq_interp_tol=_FREQUENCY_MATCH_TOLERANCE_HZ,
                        return_basis_vector=False,
                        spline_opts={"kx": 1, "ky": 1, "s": 0},
                    )
            except ValueError as exc:
                message = str(exc).lower()
                error_type = (
                    BeamFrequencyDomainError
                    if "freq" in message
                    else BeamAngularDomainError
                )
                raise error_type(
                    f"{self._identity}: pyuvdata rejected an already preflighted "
                    "interpolation domain; verify the BeamFITS native axes."
                ) from exc
            except Exception as exc:
                raise BeamEvaluationError(
                    f"{self._identity}: pyuvdata interpolation failed after RadioSim "
                    "domain preflight; inspect the chained dependency failure."
                ) from exc
            if type(interpolated) is not tuple:
                raise UnsupportedBeamBasisError(
                    f"{self._identity}: UVBeam.interp must return exact "
                    "(data, None); the installed dependency contract is unsupported."
                )
            interpolation_tuple = cast(tuple[object, ...], interpolated)
            if len(interpolation_tuple) != 2:
                raise UnsupportedBeamBasisError(
                    f"{self._identity}: UVBeam.interp must return exact "
                    "(data, None); the installed dependency contract is unsupported."
                )
            data, basis = interpolation_tuple
            if basis is not None:
                raise UnsupportedBeamBasisError(
                    f"{self._identity}: UVBeam.interp returned a basis despite "
                    "return_basis_vector=False; the installed dependency contract "
                    "is unsupported."
                )
            if not isinstance(data, np.ndarray):
                raise UnsupportedBeamBasisError(
                    f"{self._identity}: UVBeam.interp data must be a NumPy array."
                )
            data_array = cast(np.ndarray[Any, Any], data)
            if data_array.dtype != np.dtype(np.complex128):
                raise UnsupportedBeamPrecisionError(
                    f"{self._identity}: UVBeam.interp returned dtype "
                    f"{data_array.dtype.name!r}; the pinned contract requires "
                    "complex128 before the final RadioSim precision cast."
                )
            expected_shape = (2, 2, 1, int(np.count_nonzero(visible)))
            if data_array.shape != expected_shape:
                raise UnsupportedBeamBasisError(
                    f"{self._identity}: interpolated E-field shape "
                    f"{data_array.shape!r} is unsupported; expected "
                    f"{expected_shape!r}."
                )
            if not np.all(np.isfinite(data_array)):
                raise NonFiniteBeamResponseError(
                    f"{self._identity}: interpolated E-field contains NaN or Inf."
                )
            jones = np.transpose(
                np.asarray(data_array[:, :, 0, :], dtype=np.complex128),
                (2, 1, 0),
            )
            matrix_scale = np.max(np.abs(jones), axis=(1, 2))
            bound = self._scalar_atol + self._scalar_rtol * matrix_scale
            if np.any(np.abs(jones[:, 0, 1]) > bound) or np.any(
                np.abs(jones[:, 1, 0]) > bound
            ):
                raise UnsupportedBeamBasisError(
                    f"{self._identity}: interpolated Jones response contains "
                    "cross-polar terms; Tier 3 accepts only scalar e I2 response."
                )
            if np.any(np.abs(jones[:, 0, 0] - jones[:, 1, 1]) > bound):
                raise UnsupportedBeamBasisError(
                    f"{self._identity}: interpolated X/Y diagonal responses differ; "
                    "Tier 3 accepts only scalar e I2 response."
                )
            scalar = jones[:, 0, 0]
            canonical = np.zeros_like(jones)
            canonical[:, 0, 0] = scalar
            canonical[:, 1, 1] = scalar
            output[visible] = canonical

        result = np.array(output, dtype=self._result_dtype, copy=True, order="C")
        result.setflags(write=False)
        return result


__all__: list[str] = []
