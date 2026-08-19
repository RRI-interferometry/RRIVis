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
    SquintFrequencyDomainError,
    SquintReceptorBasisError,
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
    ResolvedSquint,
    ResolvedSurfaceError,
    _canonical_digest,  # pyright: ignore[reportPrivateUsage]
    _create_loaded_beam_state,  # pyright: ignore[reportPrivateUsage]
)
from radiosim.core.instrument import AntennaId
from radiosim.core.precision import PrecisionConfig
from radiosim.core.receptor import ResolvedReceptorSet

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


#: Section 4.1.1's label pairing: the feed that is not the positive one takes the
#: negative displacement, and each pair belongs to exactly one receptor basis.
_REQUIRED_BASIS_BY_FEED: Mapping[str, str] = MappingProxyType(
    {"x": "linear", "y": "linear", "r": "circular", "l": "circular"}
)


@dataclass(frozen=True, slots=True)
class _SquintPlan:
    """Everything one antenna's native-feed squint needs, resolved once.

    The receptor half is read from the resolved receptor set the ``C`` term
    itself comes from, so the ``C`` inside ``E`` and the chain's ``C`` cannot
    disagree.
    """

    squint: ResolvedSquint
    receptor_basis: str
    feed_rotation_rad: float
    feed_array: tuple[str, str]


def _squint_arcsine_argument(
    *,
    reference_frequency_hz: float,
    per_feed_offset_deg_at_reference: float,
    frequency_hz: float,
) -> float:
    """Return the exact binary64 Cotton/Uson arcsine argument.

    ``docs/development/sci005_beam_physics_plan.md`` Section 4.1: the frequency
    law is ``delta(nu) = asin[(nu_ref / nu) sin delta_ref]`` (J. M. Uson and
    W. D. Cotton, *Beam squint and Stokes V with off-axis feeds*, arXiv:0807.0026,
    2008).  The setup preflight and the evaluation path compute this identical
    expression, so an argument the preflight accepted cannot leave the domain
    later.
    """
    return (reference_frequency_hz / frequency_hz) * math.sin(
        math.radians(per_feed_offset_deg_at_reference)
    )


def _wrap_to_pi(angle_rad: float) -> float:
    """Section 4.1's ``wrap`` onto the canonical ``(-pi, pi]`` interval."""
    wrapped = math.remainder(angle_rad, 2.0 * math.pi)
    return math.pi if wrapped == -math.pi else wrapped


def _squint_position_angle_rad(
    squint: ResolvedSquint,
    *,
    boresight_parallactic_rad: float,
    boresight_altitude_rad: float,
) -> float:
    """Return ``beta_squint`` for one antenna at one time step (Section 4.2.1).

    ``beta_feed = wrap(beta_mechanical + eta_p psi_p + nu_p alt_p)`` is the
    physical feed-location ray in the beam frame, evaluated at the antenna's
    resolved boresight; the squint direction is orthogonal to the
    optical-axis/feed plane, which the v1 handedness fixes as
    ``beta_feed + pi/2``.  The mount factors are the same accepted
    field-rotation factors ``P`` uses.
    """
    from radiosim.core.jones.parallactic import mount_factors

    parallactic_factor, nasmyth_factor = mount_factors(squint.mount_type)
    feed_angle = _wrap_to_pi(
        math.radians(squint.mechanical_feed_position_angle_deg)
        + parallactic_factor * boresight_parallactic_rad
        + nasmyth_factor * boresight_altitude_rad
    )
    return _wrap_to_pi(feed_angle + math.pi / 2.0)


def _displaced_beam_directions(
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    *,
    beta_squint_rad: float,
    signed_offset_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``R(-s_f delta; a_p) n`` as beam-frame ``(altitude, azimuth)``.

    Section 4.2.1's exact Rodrigues rotation about the horizontal axis
    ``a_p = sin(beta_squint) N - cos(beta_squint) E``, so that rotating the
    beam-frame zenith by ``+delta`` about ``a_p`` moves it along
    ``+u_squint``.  The components are written in the right-handed
    ``(East, North, Up)`` triad -- ``N x E = -U``, so writing the cross product
    with the memo's ``(North, East)`` ordering would reverse the rotation -- and
    the inverse is taken with the same ``arctan2`` forms the accepted pointing
    rotation uses.  Everything here is binary64, matching the accepted
    ``DirectionBatch`` and pointing-rotation contract.
    """
    cos_altitude = np.cos(altitude_rad)
    vectors = np.stack(
        (
            cos_altitude * np.sin(azimuth_rad),
            cos_altitude * np.cos(azimuth_rad),
            np.sin(altitude_rad),
        ),
        axis=-1,
    )
    axis = np.array(
        [-math.cos(beta_squint_rad), math.sin(beta_squint_rad), 0.0],
        dtype=np.float64,
    )
    angle = -signed_offset_rad
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    cross = np.cross(np.broadcast_to(axis, vectors.shape), vectors)
    projection = vectors @ axis
    rotated = (
        vectors * cos_angle
        + cross * sin_angle
        + axis * projection[..., None] * (1.0 - cos_angle)
    )
    east = rotated[..., 0]
    north = rotated[..., 1]
    up = rotated[..., 2]
    return np.arctan2(up, np.hypot(east, north)), np.arctan2(east, north)


def _squint_receptor_matrix(
    basis: str,
    chi_rad: float,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Return ``C = M(basis) @ R(chi)`` at the resolved beam dtype.

    Section 4.2.1 fixes the formulas and requires the composition to stay at the
    resolved beam width, so this is written here rather than taken from
    :func:`radiosim.core.jones.receptor.receptor_matrix`, which is defined at
    ``complex128`` and would narrow an extended-width composition.
    """
    real_dtype = np.empty(0, dtype=dtype).real.dtype
    chi = real_dtype.type(chi_rad)
    rotation = np.array(
        [
            [np.cos(chi), np.sin(chi)],
            [-np.sin(chi), np.cos(chi)],
        ],
        dtype=dtype,
    )
    if basis == "linear":
        leading = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    elif basis == "circular":
        one = real_dtype.type(1.0)
        inverse_root_two = one / np.sqrt(real_dtype.type(2.0))
        leading = inverse_root_two * np.array(
            [[1.0, 1.0j], [1.0, -1.0j]],
            dtype=dtype,
        )
    else:  # pragma: no cover - receptor resolution owns the two-member vocabulary
        raise UnsupportedBeamBasisError(
            f"receptor basis {basis!r} is not one of 'linear' or 'circular'."
        )
    return np.asarray(leading @ rotation, dtype=dtype)


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
    squint_plan: _SquintPlan | None = None,
) -> str:
    """Return the key two antennas share iff their responses are identical.

    It is the ``handler_id`` itself whenever the antenna carries no mount
    physics, which is what keeps the pre-7I solver cache behaviour, and its
    keys, unchanged by construction.

    ``docs/development/sci005_beam_physics_plan.md`` Section 4.2.1 widens the
    identity exactly when squint is present, and the receptor half is part of it
    because ``E = C^dagger D_b C`` differs between two antennas that share a
    handler and a squint record but not a receptor.  An antenna without squint
    produces a byte-identical key to today.
    """
    if pointing is None and surface_error is None and squint_plan is None:
        return handler_id
    payload: dict[str, Any] = {
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
    if squint_plan is not None:
        squint = squint_plan.squint
        payload["squint"] = {
            "reference_frequency_hz": squint.reference_frequency_hz,
            "per_feed_offset_deg_at_reference": (
                squint.per_feed_offset_deg_at_reference
            ),
            "mechanical_feed_position_angle_deg": (
                squint.mechanical_feed_position_angle_deg
            ),
            "positive_native_feed": squint.positive_native_feed,
            "mount_type": squint.mount_type,
            "receptor_basis": squint_plan.receptor_basis,
            "feed_rotation_rad": squint_plan.feed_rotation_rad,
        }
    digest = _canonical_digest(payload)
    return f"{handler_id}+{digest[:16]}"


class _BeamSystemRuntime:
    """Unpublished evaluator lookup owned by exactly one BeamSystem."""

    __slots__ = (
        "evaluator_by_handler_id",
        "handler_id_by_antenna",
        "pointing_by_antenna",
        "response_key_by_antenna",
        "ruze_diagnostic_by_antenna",
        "squint_by_antenna",
        "surface_error_by_antenna",
    )
    evaluator_by_handler_id: Mapping[str, _BeamEvaluator]
    handler_id_by_antenna: Mapping[AntennaId, str]
    pointing_by_antenna: Mapping[AntennaId, ResolvedPointingOffset]
    surface_error_by_antenna: Mapping[AntennaId, ResolvedSurfaceError]
    ruze_diagnostic_by_antenna: Mapping[AntennaId, _RuzeDiagnosticPlan]
    squint_by_antenna: Mapping[AntennaId, _SquintPlan]
    response_key_by_antenna: Mapping[AntennaId, str]

    def __init__(
        self,
        *,
        evaluator_by_handler_id: dict[str, _BeamEvaluator],
        handler_id_by_antenna: dict[AntennaId, str],
        pointing_by_antenna: dict[AntennaId, ResolvedPointingOffset],
        surface_error_by_antenna: dict[AntennaId, ResolvedSurfaceError],
        ruze_diagnostic_by_antenna: dict[AntennaId, _RuzeDiagnosticPlan],
        squint_by_antenna: dict[AntennaId, _SquintPlan] | None = None,
    ) -> None:
        # Absent squint is the accepted pre-Stage-2 runtime, so the new map
        # defaults to empty rather than making every construction site pass one.
        squint_plans: dict[AntennaId, _SquintPlan] = dict(squint_by_antenna or {})
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
            "squint_by_antenna",
            MappingProxyType(squint_plans),
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
                        squint_plans.get(antenna_id),
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


def _require_boresight_pair(
    squint_plan: _SquintPlan | None,
    *,
    boresight_parallactic_rad: float | None,
    boresight_altitude_rad: float | None,
) -> None:
    """Enforce Section 4.2.1's two-sided boresight rule.

    When the resolved antenna carries squint both values must be exact finite
    Python floats; when it does not, both must be ``None``.
    """
    supplied = (
        ("boresight_parallactic_rad", boresight_parallactic_rad),
        ("boresight_altitude_rad", boresight_altitude_rad),
    )
    if squint_plan is None:
        for name, value in supplied:
            if value is not None:
                raise BeamEvaluationError(
                    f"{name} must be None for an antenna that carries no beam "
                    f"squint; observed {value!r}."
                )
        return
    for name, value in supplied:
        if type(value) is not float or not math.isfinite(value):
            raise BeamEvaluationError(
                f"{name} must be an exact finite Python float for an antenna "
                f"that carries beam squint; observed {value!r}."
            )


def _evaluate_squinted_response(
    evaluator: _BeamEvaluator,
    squint_plan: _SquintPlan,
    *,
    true_altitude_rad: Any,
    beam_altitude_rad: Any,
    beam_azimuth_rad: Any,
    frequency_hz: float,
    time_mjd: float,
    boresight_parallactic_rad: float,
    boresight_altitude_rad: float,
) -> np.ndarray:
    """Return one antenna's composed ``E = C^dagger D_b C`` (Section 4.2.1).

    The two native feeds sample the antenna's existing scalar evaluator at
    oppositely displaced directions: the common pointing rotation has already
    expressed the true visible directions in the beam frame, and each feed adds
    its own exact great-circle rotation about the resolved boresight.  The
    horizon gate stays on true topocentric altitude, so only visible directions
    are rotated and the evaluator's own domain behaviour applies to a displaced
    direction exactly as it does to a pointing-rotated one.
    """
    if type(beam_altitude_rad) is not np.ndarray or (
        type(beam_azimuth_rad) is not np.ndarray
    ):
        raise BeamAngularDomainError(
            "altitude_rad and azimuth_rad must be one-dimensional NumPy arrays."
        )
    squint = squint_plan.squint
    argument = _squint_arcsine_argument(
        reference_frequency_hz=squint.reference_frequency_hz,
        per_feed_offset_deg_at_reference=(squint.per_feed_offset_deg_at_reference),
        frequency_hz=float(frequency_hz),
    )
    if not -1.0 <= argument <= 1.0:
        # The setup preflight evaluates this identical binary64 expression over
        # every observation channel, so reaching here is an internal failure.
        raise BeamEvaluationError(
            "Beam squint arcsine argument left [-1, 1] at evaluation time after "
            f"the load preflight accepted it; observed {argument!r}."
        )
    offset_rad = math.asin(argument)
    beta_squint_rad = _squint_position_angle_rad(
        squint,
        boresight_parallactic_rad=boresight_parallactic_rad,
        boresight_altitude_rad=boresight_altitude_rad,
    )
    visible = np.asarray(true_altitude_rad, dtype=np.float64) >= 0.0

    samples: list[np.ndarray] = []
    for label in squint_plan.feed_array:
        sign = 1.0 if label == squint.positive_native_feed else -1.0
        feed_altitude = np.array(
            beam_altitude_rad,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        feed_azimuth = np.array(
            beam_azimuth_rad,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if bool(np.any(visible)):
            displaced_altitude, displaced_azimuth = _displaced_beam_directions(
                feed_altitude[visible],
                feed_azimuth[visible],
                beta_squint_rad=beta_squint_rad,
                signed_offset_rad=sign * offset_rad,
            )
            feed_altitude[visible] = displaced_altitude
            feed_azimuth[visible] = displaced_azimuth
        sampled = evaluator.evaluate_numpy(
            feed_altitude,
            feed_azimuth,
            frequency_hz,
            time_mjd,
        )
        samples.append(np.asarray(sampled)[:, 0, 0])

    result_dtype = np.dtype(samples[0].dtype)
    diagonal = np.zeros((samples[0].size, 2, 2), dtype=result_dtype, order="C")
    diagonal[:, 0, 0] = samples[0]
    diagonal[:, 1, 1] = samples[1]
    receptor = _squint_receptor_matrix(
        squint_plan.receptor_basis,
        squint_plan.feed_rotation_rad,
        result_dtype,
    )
    composed = np.array(
        receptor.conj().T @ diagonal @ receptor,
        dtype=result_dtype,
        copy=True,
        order="C",
    )
    composed.setflags(write=False)
    return composed


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
        boresight_parallactic_rad: float | None = None,
        boresight_altitude_rad: float | None = None,
    ) -> np.ndarray | ArrayLike:
        """Evaluate one antenna's canonical Jones response.

        The response is the accepted scalar ``e I2`` unless the antenna carries
        SCI-005 Stage-2 native-feed squint, in which case it is the composed
        ``E = C^dagger D_b C`` of ``docs/development/sci005_beam_physics_plan.md``
        Section 4.2.1 and the two boresight values are required: the parallactic
        angle and true altitude of the antenna's resolved boresight, which the
        private solver adapter owns and supplies once per antenna and time step.
        An antenna without squint requires both to be ``None`` and its call
        surface, behaviour, and results are unchanged.
        """
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
        squint_plan = self.__runtime.squint_by_antenna.get(canonical)
        _require_boresight_pair(
            squint_plan,
            boresight_parallactic_rad=boresight_parallactic_rad,
            boresight_altitude_rad=boresight_altitude_rad,
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

        if squint_plan is None:
            host_result = evaluator.evaluate_numpy(
                evaluated_altitude,
                evaluated_azimuth,
                frequency_hz,
                time_mjd,
            )
        else:
            host_result = _evaluate_squinted_response(
                evaluator,
                squint_plan,
                true_altitude_rad=altitude_rad,
                beam_altitude_rad=evaluated_altitude,
                beam_azimuth_rad=evaluated_azimuth,
                frequency_hz=frequency_hz,
                time_mjd=time_mjd,
                boresight_parallactic_rad=cast(float, boresight_parallactic_rad),
                boresight_altitude_rad=cast(float, boresight_altitude_rad),
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


def _squint_plans(
    resolved_state: ResolvedBeamState,
    *,
    observation_frequencies_hz: tuple[float, ...],
    receptors: ResolvedReceptorSet | None,
) -> dict[AntennaId, _SquintPlan]:
    """Resolve and preflight every squint-carrying antenna (Section 4.1.1).

    Both rejections are raised before any handler evaluation: the exact
    Cotton/Uson arcsine domain over every observation channel, which rejects and
    never clips, and the receptor-basis membership of the authored feed label,
    which needs the resolved receptor set and therefore cannot be a document
    check.
    """
    squinting = tuple(
        assignment
        for assignment in resolved_state.assignments
        if assignment.squint is not None
    )
    if not squinting:
        return {}
    if receptors is None:
        raise TypeError(
            "load_beam_system requires the resolved receptor set whenever any "
            "resolved antenna carries beams.squint; pass receptors="
            "<ResolvedReceptorSet>."
        )
    if type(receptors) is not ResolvedReceptorSet:
        raise TypeError("receptors must be an exact ResolvedReceptorSet")

    for assignment in squinting:
        squint = cast(ResolvedSquint, assignment.squint)
        antenna_id = assignment.antenna_id
        for frequency_hz in observation_frequencies_hz:
            argument = _squint_arcsine_argument(
                reference_frequency_hz=squint.reference_frequency_hz,
                per_feed_offset_deg_at_reference=(
                    squint.per_feed_offset_deg_at_reference
                ),
                frequency_hz=frequency_hz,
            )
            if not -1.0 <= argument <= 1.0:
                raise SquintFrequencyDomainError(
                    "beams.squint: canonical antenna "
                    f"number={antenna_id.number}, name={antenna_id.name!r} has "
                    "no real feed displacement at observation frequency "
                    f"{frequency_hz!r} Hz: the exact Cotton/Uson argument "
                    f"(reference_frequency_hz={squint.reference_frequency_hz!r} "
                    f"/ {frequency_hz!r}) * sin(radians("
                    "per_feed_offset_deg_at_reference="
                    f"{squint.per_feed_offset_deg_at_reference!r})) is "
                    f"{argument!r}, which is outside [-1, 1]; RadioSim rejects "
                    "the observation rather than clipping the displacement."
                )

    plans: dict[AntennaId, _SquintPlan] = {}
    for assignment in squinting:
        squint = cast(ResolvedSquint, assignment.squint)
        antenna_id = assignment.antenna_id
        receptor = receptors.receptor_by_antenna.get(antenna_id)
        if receptor is None:
            raise InconsistentBeamAssignmentError(
                "The resolved receptor set has no receptor for canonical "
                f"antenna number={antenna_id.number}, name={antenna_id.name!r}; "
                "receptor and beam resolution disagree."
            )
        required_basis = _REQUIRED_BASIS_BY_FEED[squint.positive_native_feed]
        if receptor.basis != required_basis:
            raise SquintReceptorBasisError(
                "beams.squint: canonical antenna "
                f"number={antenna_id.number}, name={antenna_id.name!r} declares "
                f"positive_native_feed={squint.positive_native_feed!r}, which "
                f"requires the {required_basis!r} receptor basis, but the "
                f"antenna's resolved receptor basis is {receptor.basis!r}; "
                "'x'/'y' belong to 'linear' and 'r'/'l' to 'circular'."
            )
        plans[AntennaId(antenna_id.number, antenna_id.name)] = _SquintPlan(
            squint=squint,
            receptor_basis=receptor.basis,
            feed_rotation_rad=receptor.feed_rotation_rad,
            feed_array=receptor.feed_array,
        )
    return plans


def _load_beam_system(
    resolved_state: ResolvedBeamState,
    *,
    observation_frequencies_hz: tuple[float, ...],
    precision: PrecisionConfig,
    loader: _UVBeamLoaderProtocol,
    receptors: ResolvedReceptorSet | None = None,
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
    squint_by_antenna = _squint_plans(
        resolved_state,
        observation_frequencies_hz=frequencies,
        receptors=receptors,
    )

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
        squint_by_antenna=squint_by_antenna,
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
    receptors: ResolvedReceptorSet | None = None,
) -> BeamSystem:
    """Atomically load one complete canonical per-antenna beam system.

    ``receptors`` is the already-resolved receptor set
    (``docs/development/sci005_beam_physics_plan.md`` Section 4.2.1).  It is
    required whenever any resolved antenna carries ``beams.squint``, because the
    composed ``E = C^dagger D_b C`` is built from the antenna's own resolved
    receptor basis and static feed rotation -- the same authority the solver's
    ``C`` term comes from, so the two cannot disagree.  Receptor resolution is
    unchanged, and a run with no squint neither needs nor reads it.
    """
    return _load_beam_system(
        resolved_state,
        observation_frequencies_hz=observation_frequencies_hz,
        precision=precision,
        loader=_ProductionUVBeamLoader(),
        receptors=receptors,
    )


__all__ = [
    "BeamSystem",
    "load_beam_system",
    "ruze_power_efficiency",
    "ruze_voltage_factor",
]
