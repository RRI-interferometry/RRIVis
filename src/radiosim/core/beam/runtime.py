"""Canonical per-antenna beam runtime and private evaluator primitives."""

from __future__ import annotations

import importlib
import logging
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from radiosim.core.beam.aperture import (
    ApertureSpecification,
    RuzePowerDiagnostic,
)
from radiosim.core.beam.aperture import (
    evaluate_ruze_power_diagnostic as _evaluate_ruze_power_diagnostic,
)
from radiosim.core.beam.errors import (
    BeamAngularDomainError,
    BeamDependencyError,
    BeamEvaluationError,
    BeamFileReadError,
    BeamFrequencyDomainError,
    InconsistentBeamAssignmentError,
    NonFiniteBeamResponseError,
    UnsupportedBeamBasisError,
    UnsupportedBeamPrecisionError,
)
from radiosim.core.beam.models import (
    LoadedBeamHandlerState,
    LoadedBeamState,
    ResolvedAnalyticBeamDefinition,
    ResolvedBeamState,
    ResolvedFITSBeamDefinition,
    ResolvedPointingOffset,
    ResolvedSurfaceError,
    _canonical_digest,  # pyright: ignore[reportPrivateUsage]
    _create_loaded_beam_state,  # pyright: ignore[reportPrivateUsage]
)
from radiosim.core.instrument import AntennaId
from radiosim.core.precision import PrecisionConfig

if TYPE_CHECKING:
    from radiosim.backends.base import ArrayBackend

ArrayLike = Any

_FREQUENCY_MATCH_TOLERANCE_HZ = 1e-6
_C_M_PER_S = 299_792_458.0
_LOGGER = logging.getLogger(__name__)


def ruze_power_efficiency(
    *,
    rms_surface_error_m: float,
    wavelength_m: float,
) -> float:
    """Return Ruze's random-surface **power** efficiency.

    ``eta_s = exp(-(4 pi sigma / lambda)^2)`` (Ruze 1966, *Antenna tolerance
    theory -- a review*, Proc. IEEE 54, 633). This is a gain ratio, so the
    factor RadioSim applies to its *voltage* beam is ``sqrt(eta_s)``: the
    visibility amplitude on a baseline of two antennas sharing this ``sigma``
    is then scaled by exactly ``eta_s``. That is the same voltage/power
    discipline the tropospheric opacity uses for ``exp(-tau/2)``.

    The rule of thumb quoted alongside the equation, ``lambda_min ~= 10 sigma``,
    is the wavelength at which ``eta_s`` has already fallen to
    ``exp(-(0.4 pi)^2) ~= 0.206`` -- the point past which a dish stops being
    usable rather than a point at which anything diverges, so it bounds nothing
    here and is not enforced as a rejection.

    Parameters
    ----------
    rms_surface_error_m
        Reflector surface RMS error ``sigma``, in metres. Must be finite and
        nonnegative.
    wavelength_m
        Observing wavelength, in metres. Must be finite and positive.
    """
    if type(rms_surface_error_m) is not float or not math.isfinite(rms_surface_error_m):
        raise NonFiniteBeamResponseError(
            "rms_surface_error_m must be an exact finite Python float."
        )
    if rms_surface_error_m < 0.0:
        raise ValueError("rms_surface_error_m must be >= 0")
    if type(wavelength_m) is not float or not math.isfinite(wavelength_m):
        raise NonFiniteBeamResponseError(
            "wavelength_m must be an exact finite Python float."
        )
    if wavelength_m <= 0.0:
        raise BeamFrequencyDomainError("wavelength_m must be positive.")
    return math.exp(-((4.0 * math.pi * rms_surface_error_m / wavelength_m) ** 2))


def ruze_voltage_factor(
    *,
    rms_surface_error_m: float,
    wavelength_m: float,
) -> float:
    """Return the real scalar Ruze applies to the **voltage** beam.

    This is ``sqrt(eta_s)`` of :func:`ruze_power_efficiency`, written directly
    as ``exp(-(1/2)(4 pi sigma / lambda)^2)`` rather than as a square root, so
    that a surface rough enough to underflow ``eta_s`` still carries a
    representable voltage factor. The two agree to double rounding, which is
    asserted by test rather than assumed.
    """
    efficiency = ruze_power_efficiency(
        rms_surface_error_m=rms_surface_error_m,
        wavelength_m=wavelength_m,
    )
    if efficiency == 1.0:
        return 1.0
    argument = 4.0 * math.pi * rms_surface_error_m / wavelength_m
    return math.exp(-0.5 * argument * argument)


def _rotate_into_beam_frame(
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    offset: ResolvedPointingOffset,
) -> tuple[np.ndarray, np.ndarray]:
    """Express topocentric directions in one mispointed antenna's beam frame.

    The offset is the rigid rotation ``R`` of the beam frame described by
    :class:`~radiosim.core.beam.models.ResolvedPointingOffset`; this returns the
    directions as ``R^T n``, in the ``(alt, az)`` parameterization every
    evaluator consumes. Composed, the antenna's boresight sits at topocentric
    azimuth ``azimuth_offset_rad`` and zenith angle ``elevation_offset_rad``,
    so the beam's peak has moved by exactly ``elevation_offset_rad`` of
    great-circle angle and a pure azimuth offset moves it not at all -- the
    alt-az keyhole degeneracy, which is real at a zenith-pointed mount.

    The horizon gate is *not* applied here. A rotation of the beam frame does
    not move the ground, so the caller keeps the below-horizon directions at
    their true altitude and lets the evaluator zero them.
    """
    shifted_azimuth = azimuth_rad - offset.azimuth_offset_rad
    cos_altitude = np.cos(altitude_rad)
    east = cos_altitude * np.sin(shifted_azimuth)
    north = cos_altitude * np.cos(shifted_azimuth)
    up = np.sin(altitude_rad)

    cos_tilt = math.cos(offset.elevation_offset_rad)
    sin_tilt = math.sin(offset.elevation_offset_rad)
    beam_north = north * cos_tilt - up * sin_tilt
    beam_up = north * sin_tilt + up * cos_tilt

    # ``arctan2(up, hypot(east, north))`` rather than ``arcsin(up)``: the beam's
    # own boresight lands at the pole of this frame, where ``arcsin`` is
    # ill-conditioned (an input error of eps becomes an angle error of order
    # sqrt(eps), about 1e-8 rad in float64).  The two-argument form is accurate
    # to rounding there, which is what makes "the peak moves by exactly delta"
    # exact rather than exact-to-1e-8.
    beam_altitude = np.arctan2(beam_up, np.hypot(east, beam_north))
    beam_azimuth = np.arctan2(east, beam_north)
    return beam_altitude, beam_azimuth


class _UVBeamLike(Protocol):
    """Structural subset of UVBeam privately consumed by Tier 3D."""

    data_array: NDArray[np.complexfloating[Any, Any]]
    basis_vector_array: NDArray[np.floating[Any]]
    freq_array: NDArray[np.floating[Any]]
    axis1_array: NDArray[np.floating[Any]]
    axis2_array: NDArray[np.floating[Any]]

    def interp(self, **kwargs: Any) -> object: ...


class _UVBeamLoaderProtocol(Protocol):
    """Private injectable reader boundary for one complete BeamFITS file."""

    def read(self, path: Path) -> _UVBeamLike: ...


class _ProductionUVBeamLoader:
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


class _BeamEvaluator(Protocol):
    """Private evaluator contract shared by later canonical beam runtimes."""

    def evaluate_numpy(
        self,
        altitude_rad: np.ndarray,
        azimuth_rad: np.ndarray,
        frequency_hz: float,
        time_mjd: float,
    ) -> NDArray[np.complexfloating[Any, Any]]: ...

    def voltage_feature_scale_rad(self, frequency_hz: float) -> float: ...


def _response_key(
    handler_id: str,
    pointing: ResolvedPointingOffset | None,
    surface_error: ResolvedSurfaceError | None,
) -> str:
    """Return the key two antennas share iff their responses are identical.

    It is the ``handler_id`` itself whenever the antenna carries no mount
    physics, which is what keeps the pre-7I solver cache behaviour, and its
    keys, unchanged by construction.
    """
    if pointing is None and surface_error is None:
        return handler_id
    digest = _canonical_digest(
        {
            "kind": "beam_response_key",
            "handler_id": handler_id,
            "pointing": (
                None
                if pointing is None
                else {
                    "azimuth_offset_rad": pointing.azimuth_offset_rad,
                    "elevation_offset_rad": pointing.elevation_offset_rad,
                }
            ),
            "surface_error": (
                None
                if surface_error is None
                else {"rms_surface_error_m": surface_error.rms_surface_error_m}
            ),
        }
    )
    return f"{handler_id}+{digest[:16]}"


class _BeamSystemRuntime:
    """Unpublished evaluator lookup owned by exactly one BeamSystem."""

    __slots__ = (
        "evaluator_by_handler_id",
        "handler_id_by_antenna",
        "pointing_by_antenna",
        "response_key_by_antenna",
        "ruze_diagnostic_by_antenna",
        "surface_error_by_antenna",
    )
    evaluator_by_handler_id: Mapping[str, _BeamEvaluator]
    handler_id_by_antenna: Mapping[AntennaId, str]
    pointing_by_antenna: Mapping[AntennaId, ResolvedPointingOffset]
    surface_error_by_antenna: Mapping[AntennaId, ResolvedSurfaceError]
    ruze_diagnostic_by_antenna: Mapping[AntennaId, _RuzeDiagnosticPlan]
    response_key_by_antenna: Mapping[AntennaId, str]

    def __init__(
        self,
        *,
        evaluator_by_handler_id: dict[str, _BeamEvaluator],
        handler_id_by_antenna: dict[AntennaId, str],
        pointing_by_antenna: dict[AntennaId, ResolvedPointingOffset],
        surface_error_by_antenna: dict[AntennaId, ResolvedSurfaceError],
        ruze_diagnostic_by_antenna: dict[AntennaId, _RuzeDiagnosticPlan],
    ) -> None:
        object.__setattr__(
            self,
            "evaluator_by_handler_id",
            MappingProxyType(dict(evaluator_by_handler_id)),
        )
        object.__setattr__(
            self,
            "handler_id_by_antenna",
            MappingProxyType(dict(handler_id_by_antenna)),
        )
        object.__setattr__(
            self,
            "pointing_by_antenna",
            MappingProxyType(dict(pointing_by_antenna)),
        )
        object.__setattr__(
            self,
            "surface_error_by_antenna",
            MappingProxyType(dict(surface_error_by_antenna)),
        )
        object.__setattr__(
            self,
            "ruze_diagnostic_by_antenna",
            MappingProxyType(dict(ruze_diagnostic_by_antenna)),
        )
        object.__setattr__(
            self,
            "response_key_by_antenna",
            MappingProxyType(
                {
                    antenna_id: _response_key(
                        handler_id,
                        pointing_by_antenna.get(antenna_id),
                        surface_error_by_antenna.get(antenna_id),
                    )
                    for antenna_id, handler_id in handler_id_by_antenna.items()
                }
            ),
        )

    @override
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("BeamSystem runtime attributes are immutable")


@dataclass(frozen=True, slots=True)
class _RuzeDiagnosticPlan:
    """Everything one antenna's Ruze power diagnostic needs, resolved once."""

    aperture: ApertureSpecification
    rms_surface_error_m: float
    correlation_length_m: float
    real_dtype: np.dtype[Any]
    complex_dtype: np.dtype[Any]


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


def _require_lookup_antenna_id(value: Any) -> AntennaId:
    if type(value) is not AntennaId:
        if isinstance(value, AntennaId):
            raise InconsistentBeamAssignmentError(
                "BeamSystem has no handler assignment for canonical antenna "
                f"number={value.number}, name={value.name!r}; loaded beam state "
                "is inconsistent."
            )
        raise TypeError("antenna_id must be an exact AntennaId")
    return value


def _convert_backend_result(
    backend: Any,
    host_result: NDArray[np.complexfloating[Any, Any]],
) -> Any:
    from radiosim.backends.base import ArrayBackend

    if not isinstance(backend, ArrayBackend):
        raise TypeError("backend must be an ArrayBackend or None")
    converted = backend.asarray(host_result, dtype=host_result.dtype)
    try:
        converted_shape = tuple(converted.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise BeamEvaluationError(
            "Beam backend conversion returned a value without a valid shape."
        ) from exc
    if converted_shape != host_result.shape:
        raise BeamEvaluationError(
            "Beam backend conversion returned shape "
            f"{converted_shape!r}; expected {host_result.shape!r}."
        )
    try:
        converted_dtype = np.dtype(converted.dtype)
    except (AttributeError, TypeError, ValueError) as exc:
        raise BeamEvaluationError(
            "Beam backend conversion returned a value without a valid dtype."
        ) from exc
    if converted_dtype != host_result.dtype:
        raise BeamEvaluationError(
            "Beam backend conversion returned dtype "
            f"{converted_dtype!s}; expected {host_result.dtype!s}."
        )
    if isinstance(converted, np.ndarray):
        owned = np.array(
            converted,
            dtype=host_result.dtype,
            copy=True,
            order="C",
        )
        owned.setflags(write=False)
        return owned
    return converted


class BeamSystem:
    """Final canonical per-antenna beam runtime created only by its factory."""

    __slots__ = ("__runtime", "__state")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("BeamSystem does not support subclassing")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("BeamSystem instances must be created by load_beam_system")

    @override
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("BeamSystem attributes are immutable")

    @property
    def state(self) -> LoadedBeamState:
        """Return the immutable detached loaded-state snapshot."""
        return self.__state

    def response_key(self, antenna_id: AntennaId) -> str:
        """Return the key two antennas share iff their responses are identical.

        A per-``(time, frequency)`` caller that evaluates one direction batch
        per beam handler must key its cache on this and not on ``handler_id``:
        two antennas of the same diameter and model share a handler, and Tier
        7I's per-antenna pointing offsets and surface errors are the first thing
        that makes their responses differ. With neither configured the key *is*
        the ``handler_id``, so nothing about the absent case changes.
        """
        canonical = _require_lookup_antenna_id(antenna_id)
        key = self.__runtime.response_key_by_antenna.get(canonical)
        if key is None:
            raise InconsistentBeamAssignmentError(
                "BeamSystem has no handler assignment for canonical antenna "
                f"number={canonical.number}, name={canonical.name!r}; loaded "
                "beam state is inconsistent."
            )
        return key

    def evaluate_jones(
        self,
        antenna_id: AntennaId,
        *,
        altitude_rad: np.ndarray,
        azimuth_rad: np.ndarray,
        frequency_hz: float,
        time_mjd: float,
        backend: ArrayBackend | None = None,
    ) -> np.ndarray | ArrayLike:
        """Evaluate one antenna's canonical scalar Jones response."""
        antenna_id = _require_lookup_antenna_id(antenna_id)
        try:
            canonical = AntennaId(antenna_id.number, antenna_id.name)
        except (TypeError, ValueError) as exc:
            raise InconsistentBeamAssignmentError(
                "BeamSystem has no handler assignment for canonical antenna "
                f"number={antenna_id.number!r}, name={antenna_id.name!r}; loaded "
                "beam state is inconsistent."
            ) from exc
        handler_id = self.__runtime.handler_id_by_antenna.get(canonical)
        if handler_id is None:
            raise InconsistentBeamAssignmentError(
                "BeamSystem has no handler assignment for canonical antenna "
                f"number={canonical.number}, name={canonical.name!r}; loaded "
                "beam state is inconsistent."
            )
        evaluator = cast(
            _BeamEvaluator | None,
            self.__runtime.evaluator_by_handler_id.get(handler_id),
        )
        if evaluator is None:
            raise InconsistentBeamAssignmentError(
                f"BeamSystem has no evaluator for handler_id={handler_id!r}; "
                "loaded beam runtime is inconsistent."
            )
        evaluated_altitude = altitude_rad
        evaluated_azimuth = azimuth_rad
        offset = self.__runtime.pointing_by_antenna.get(canonical)
        if offset is not None and type(altitude_rad) is np.ndarray:
            # The horizon gate belongs to the ground, not to the beam frame, so
            # only the visible directions are rotated; the rest keep their true
            # below-horizon altitude and the evaluator zeroes them exactly as it
            # would have without an offset.  A visible direction whose rotated
            # altitude falls below zero -- a band |elevation_offset_rad| wide at
            # the horizon -- is zeroed by the evaluator's own forward-hemisphere
            # domain, which is what a reflector actually does.
            visible = np.asarray(altitude_rad, dtype=np.float64) >= 0.0
            if bool(np.any(visible)):
                beam_altitude, beam_azimuth = _rotate_into_beam_frame(
                    np.asarray(altitude_rad, dtype=np.float64)[visible],
                    np.asarray(azimuth_rad, dtype=np.float64)[visible],
                    offset,
                )
                evaluated_altitude = np.array(
                    altitude_rad,
                    dtype=np.float64,
                    copy=True,
                    order="C",
                )
                evaluated_azimuth = np.array(
                    azimuth_rad,
                    dtype=np.float64,
                    copy=True,
                    order="C",
                )
                evaluated_altitude[visible] = beam_altitude
                evaluated_azimuth[visible] = beam_azimuth

        host_result = evaluator.evaluate_numpy(
            evaluated_altitude,
            evaluated_azimuth,
            frequency_hz,
            time_mjd,
        )

        surface_error = self.__runtime.surface_error_by_antenna.get(canonical)
        if surface_error is not None:
            voltage_factor = ruze_voltage_factor(
                rms_surface_error_m=surface_error.rms_surface_error_m,
                wavelength_m=_C_M_PER_S / float(frequency_hz),
            )
            real_dtype = np.empty(0, dtype=host_result.dtype).real.dtype
            scaled = np.asarray(
                host_result * real_dtype.type(voltage_factor),
                dtype=host_result.dtype,
            )
            host_result = np.array(scaled, copy=True, order="C")
            host_result.setflags(write=False)

        if backend is None:
            return host_result
        return _convert_backend_result(backend, host_result)

    def evaluate_ruze_power_diagnostic(
        self,
        antenna_id: AntennaId,
        *,
        altitude_rad: np.ndarray,
        azimuth_rad: np.ndarray,
        frequency_hz: float,
        time_mjd: float,
    ) -> RuzePowerDiagnostic:
        """Return one antenna's Ruze ensemble-power diagnostic.

        ``docs/development/sci005_beam_physics_plan.md`` Section 3.4.2 freezes
        this contract.  The whole algorithm is host-side, so the method accepts
        no backend argument; it never mutates or substitutes the matrix returned
        by :meth:`evaluate_jones`, and it creates no Jones voltage.  Unlike
        ``evaluate_jones`` it requires at least one direction, because the
        convergence maxima it retains are part of its result.
        """
        canonical = _require_lookup_antenna_id(antenna_id)
        plan = self.__runtime.ruze_diagnostic_by_antenna.get(canonical)
        if plan is None:
            raise BeamEvaluationError(
                "A Ruze power diagnostic is not configured for this antenna."
            )
        if type(altitude_rad) is not np.ndarray or altitude_rad.ndim != 1:
            raise BeamAngularDomainError(
                "altitude_rad must be a one-dimensional array."
            )
        if type(azimuth_rad) is not np.ndarray or azimuth_rad.ndim != 1:
            raise BeamAngularDomainError("azimuth_rad must be a one-dimensional array.")
        if altitude_rad.shape != azimuth_rad.shape:
            raise BeamAngularDomainError(
                "altitude_rad and azimuth_rad must have identical shapes."
            )
        frequency = _require_exact_finite_float(frequency_hz, "frequency_hz")
        time_value = _require_exact_finite_float(time_mjd, "time_mjd")
        if frequency <= 0.0:
            raise BeamFrequencyDomainError("frequency_hz must be positive.")
        return _evaluate_ruze_power_diagnostic(
            plan.aperture,
            antenna_id=AntennaId(canonical.number, canonical.name),
            altitude_rad=altitude_rad,
            azimuth_rad=azimuth_rad,
            frequency_hz=frequency,
            time_mjd=time_value,
            rms_surface_error_m=plan.rms_surface_error_m,
            correlation_length_m=plan.correlation_length_m,
            real_dtype=plan.real_dtype,
            complex_dtype=plan.complex_dtype,
        )


def _validated_observation_frequencies(
    value: tuple[float, ...],
) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise BeamFrequencyDomainError(
            "observation_frequencies_hz must be a nonempty exact tuple."
        )
    copied: list[float] = []
    previous: float | None = None
    for frequency in value:
        if type(frequency) is not float or not math.isfinite(frequency):
            raise NonFiniteBeamResponseError(
                "observation_frequencies_hz must contain exact finite Python floats."
            )
        if frequency <= 0.0:
            raise BeamFrequencyDomainError(
                "observation_frequencies_hz must contain positive frequencies."
            )
        if previous is not None and frequency <= previous:
            raise BeamFrequencyDomainError(
                "observation_frequencies_hz must be strictly increasing."
            )
        copied.append(frequency)
        previous = frequency
    return tuple(copied)


def _load_beam_system(
    resolved_state: ResolvedBeamState,
    *,
    observation_frequencies_hz: tuple[float, ...],
    precision: PrecisionConfig,
    loader: _UVBeamLoaderProtocol,
) -> BeamSystem:
    """Private injectable implementation of the atomic BeamSystem factory."""
    if type(resolved_state) is not ResolvedBeamState:
        raise TypeError("resolved_state must be an exact ResolvedBeamState")
    resolved_state.__post_init__()
    if type(precision) is not PrecisionConfig:
        raise TypeError("precision must be an exact PrecisionConfig")
    if not callable(getattr(loader, "read", None)):
        raise TypeError("loader must provide a callable read(path) method")
    frequencies = _validated_observation_frequencies(observation_frequencies_hz)

    from radiosim.core.beam.analytic import (
        _analytic_preload_key,  # pyright: ignore[reportPrivateUsage]
        _load_analytic_handler,  # pyright: ignore[reportPrivateUsage]
    )
    from radiosim.core.beam.fits import (
        _fits_preload_key,  # pyright: ignore[reportPrivateUsage]
        _load_fits_handler,  # pyright: ignore[reportPrivateUsage]
    )

    cached: dict[object, tuple[LoadedBeamHandlerState, _BeamEvaluator]] = {}
    handlers: list[LoadedBeamHandlerState] = []
    assignment_handler_ids: list[tuple[AntennaId, str]] = []
    evaluator_by_handler_id: dict[str, _BeamEvaluator] = {}
    handler_id_by_antenna: dict[AntennaId, str] = {}
    pointing_by_antenna: dict[AntennaId, ResolvedPointingOffset] = {}
    surface_error_by_antenna: dict[AntennaId, ResolvedSurfaceError] = {}
    ruze_diagnostic_by_antenna: dict[AntennaId, _RuzeDiagnosticPlan] = {}

    for assignment in resolved_state.assignments:
        definition = assignment.definition
        if type(definition) is ResolvedAnalyticBeamDefinition:
            key: object = (
                "analytic",
                _analytic_preload_key(
                    definition,
                    assignment.antenna_diameter_m,
                    assignment.aperture_physics,
                ),
            )
        elif type(definition) is ResolvedFITSBeamDefinition:
            key = ("fits", _fits_preload_key(definition))
        else:
            raise TypeError("resolved assignment contains an unsupported definition")

        cached_handler = cached.get(key)
        if cached_handler is None:
            handler_ordinal = len(handlers)
            if type(definition) is ResolvedAnalyticBeamDefinition:
                loaded = _load_analytic_handler(
                    definition,
                    antenna_diameter_m=assignment.antenna_diameter_m,
                    observation_frequencies_hz=frequencies,
                    precision=precision,
                    handler_ordinal=handler_ordinal,
                    aperture_physics=assignment.aperture_physics,
                )
            else:
                loaded = _load_fits_handler(
                    cast(ResolvedFITSBeamDefinition, definition),
                    observation_frequencies_hz=frequencies,
                    precision=precision,
                    handler_ordinal=handler_ordinal,
                    loader=loader,
                )
            cached_handler = (loaded.state, loaded.evaluator)
            cached[key] = cached_handler
            handlers.append(loaded.state)
            evaluator_by_handler_id[loaded.state.handler_id] = loaded.evaluator

        handler_state = cached_handler[0]
        antenna_id = AntennaId(
            assignment.antenna_id.number,
            assignment.antenna_id.name,
        )
        assignment_handler_ids.append((antenna_id, handler_state.handler_id))
        handler_id_by_antenna[antenna_id] = handler_state.handler_id
        # Pointing and surface errors are per-antenna and are applied around the
        # evaluator, never inside it, so they take no part in the preload
        # deduplication above: two mispointed antennas still share one handler.
        if assignment.pointing is not None:
            pointing_by_antenna[antenna_id] = assignment.pointing
        if assignment.surface_error is not None:
            surface_error_by_antenna[antenna_id] = assignment.surface_error
            diagnostic = assignment.surface_error.error_beam_diagnostic
            if diagnostic is not None:
                if type(definition) is not ResolvedAnalyticBeamDefinition:
                    raise UnsupportedBeamBasisError(
                        "A Ruze power diagnostic requires an analytic circular "
                        "pupil; configuration resolution should already have "
                        "rejected this beam family."
                    )
                from radiosim.core.beam.analytic import (
                    _diagnostic_aperture_specification,  # pyright: ignore[reportPrivateUsage]
                    _real_dtype,  # pyright: ignore[reportPrivateUsage]
                    _result_dtype,  # pyright: ignore[reportPrivateUsage]
                )

                complex_dtype = _result_dtype(precision)
                ruze_diagnostic_by_antenna[antenna_id] = _RuzeDiagnosticPlan(
                    aperture=_diagnostic_aperture_specification(
                        definition,
                        antenna_diameter_m=assignment.antenna_diameter_m,
                        aperture_physics=assignment.aperture_physics,
                    ),
                    rms_surface_error_m=(assignment.surface_error.rms_surface_error_m),
                    correlation_length_m=diagnostic.correlation_length_m,
                    real_dtype=_real_dtype(complex_dtype),
                    complex_dtype=complex_dtype,
                )

    loaded_state = _create_loaded_beam_state(
        resolved=resolved_state,
        handlers=tuple(handlers),
        assignment_handler_ids=tuple(assignment_handler_ids),
    )
    runtime = _BeamSystemRuntime(
        evaluator_by_handler_id=evaluator_by_handler_id,
        handler_id_by_antenna=handler_id_by_antenna,
        pointing_by_antenna=pointing_by_antenna,
        surface_error_by_antenna=surface_error_by_antenna,
        ruze_diagnostic_by_antenna=ruze_diagnostic_by_antenna,
    )
    system = object.__new__(BeamSystem)
    object.__setattr__(system, "_BeamSystem__state", loaded_state)
    object.__setattr__(system, "_BeamSystem__runtime", runtime)

    for handler in loaded_state.handlers:
        if handler.file is None:
            transport_path = "<analytic>"
            metadata_summary = (
                f"definition_fingerprint={handler.definition_fingerprint}"
            )
        else:
            transport_path = str(handler.file.resolved_path)
            metadata_summary = (
                f"shape={handler.file.data_shape} "
                f"frequencies_hz=[{handler.file.frequency_min_hz},"
                f"{handler.file.frequency_max_hz}] "
                f"frequency_count={handler.file.frequency_count}"
            )
        _LOGGER.info(
            "BeamSystem handler validated: handler_id=%s kind=%s "
            "transport_path=%s metadata=%s",
            handler.handler_id,
            handler.kind,
            transport_path,
            metadata_summary,
        )
    _LOGGER.info(
        "BeamSystem published: handlers=%d assignments=%d deduplicated_assignments=%d",
        len(loaded_state.handlers),
        len(loaded_state.assignment_handler_ids),
        len(loaded_state.assignment_handler_ids) - len(loaded_state.handlers),
    )
    return system


def load_beam_system(
    resolved_state: ResolvedBeamState,
    *,
    observation_frequencies_hz: tuple[float, ...],
    precision: PrecisionConfig,
) -> BeamSystem:
    """Atomically load one complete canonical per-antenna beam system."""
    return _load_beam_system(
        resolved_state,
        observation_frequencies_hz=observation_frequencies_hz,
        precision=precision,
        loader=_ProductionUVBeamLoader(),
    )


__all__ = [
    "BeamSystem",
    "load_beam_system",
    "ruze_power_efficiency",
    "ruze_voltage_factor",
]
