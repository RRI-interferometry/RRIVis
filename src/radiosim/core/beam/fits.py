"""Private strict standalone BeamFITS loading for Tier 3D."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from radiosim.core.beam.errors import (
    BeamAngularDomainError,
    BeamDependencyError,
    BeamError,
    BeamFileChangedError,
    BeamFileReadError,
    BeamFrequencyDomainError,
    BeamNormalizationError,
    BeamSamplingDerivationError,
    NonFiniteBeamResponseError,
    UnsupportedBeamBasisError,
    UnsupportedBeamCoordinateError,
    UnsupportedBeamFeedError,
    UnsupportedBeamPrecisionError,
    UnsupportedBeamTypeError,
)
from radiosim.core.beam.models import (
    BeamFileProvenance,
    LoadedBeamHandlerState,
    ResolvedFITSBeamDefinition,
    _canonical_digest,  # pyright: ignore[reportPrivateUsage]
)
from radiosim.core.beam.runtime import (
    _FREQUENCY_MATCH_TOLERANCE_HZ,  # pyright: ignore[reportPrivateUsage]
    _FULL_EFIELD_NORMALIZATION,  # pyright: ignore[reportPrivateUsage]
    _FULL_EFIELD_SUBSET_VERSION,  # pyright: ignore[reportPrivateUsage]
    _converted_native_jones,  # pyright: ignore[reportPrivateUsage]
    _EfieldReceptorExpectation,  # pyright: ignore[reportPrivateUsage]
    _preflight_frequency,  # pyright: ignore[reportPrivateUsage]
    _ProductionUVBeamLoader,  # pyright: ignore[reportPrivateUsage]
    _UVBeamEfieldEvaluator,  # pyright: ignore[reportPrivateUsage]
    _UVBeamLike,  # pyright: ignore[reportPrivateUsage]
    _UVBeamLoaderProtocol,  # pyright: ignore[reportPrivateUsage]
    _UVBeamScalarEvaluator,  # pyright: ignore[reportPrivateUsage]
    _zenith_spin_rotation,  # pyright: ignore[reportPrivateUsage]
)
from radiosim.core.precision import PrecisionConfig

_ACCEPTED_SUBSET_VERSION = "tier3-scalar-v1"
_BASIS_TOLERANCE = 1e-12
_FEED_ANGLE_TOLERANCE_RAD = 1e-12
_GRID_TOLERANCE_RAD = 1e-12
_AZIMUTH_CLOSURE_TOLERANCE_RAD = 1e-10
_HORIZON_COVERAGE_TOLERANCE_RAD = 1e-10
_STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")

#: SCI-005 Stage-3 convention literals (Sections 5.2.1 and 8.1).  The
#: basis-vector literal names the conversion from the native ``az_za``
#: components into the chain's own mixed-sign spherical tangent pair, whose
#: frozen definition is the constant ``M`` of :func:`_chain_conversion`; the
#: zenith literal names the de-spin predicate together with the
#: ``az_radiosim = 0`` selection whose limit is ``-(N_hat, E_hat)``.
_BASIS_VECTOR_CONVENTION = "uvbeam_theta_phi_chain_tangent_v1"
_FACTORIZATION_CONVENTION = "receptor_conjugated_native_efield_v1"
_ZENITH_LIMIT_CONVENTION = "north_east_tangent_limit_v1"

#: Corrected Section 5.2.1's frozen wrap-continuity margin factor: the seam
#: second difference is compared against eight times the interior maximum.
_WRAP_SECOND_DIFFERENCE_FACTOR = 8.0

#: Section 5.1.1 item 5's two accepted ordered native feed pairs.
_ACCEPTED_FEED_PAIRS = (("x", "y"), ("r", "l"))

#: pyuvdata 3.2.1's nine mount literals; only ``fixed`` declares that the
#: stored ``az_za`` pattern is intrinsic to the antenna beam frame (item 8).
_ACCEPTED_MOUNT_TYPE = "fixed"


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    source_stat: tuple[int, int, int, int, int]
    sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedEfieldFacts:
    """The Stage-3 facts Section 5.2.1 retains beside the shared ones."""

    feed_array: tuple[str, str]
    feed_angle_rad: tuple[float, float]
    x_orientation: str | None
    derived_x_orientation_verdict: str
    stored_grid_peak_by_frequency: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class _ValidatedBeam:
    beam: _UVBeamLike
    native_dtype: str
    data_shape: tuple[int, ...]
    frequencies_hz: np.ndarray
    azimuth_rad: np.ndarray
    zenith_angle_rad: np.ndarray
    azimuth_step_rad: float
    zenith_angle_step_rad: float
    feature_scale_rad: float
    scalar_absolute_tolerance: float
    scalar_relative_tolerance: float
    normalization_absolute_tolerance: float
    efield: _ValidatedEfieldFacts | None = None


@dataclass(frozen=True, slots=True)
class _LoadedFITSHandler:
    """Private publication pair for one standalone loaded definition."""

    state: LoadedBeamHandlerState
    evaluator: _UVBeamScalarEvaluator | _UVBeamEfieldEvaluator


def _accepted_subset_version(definition: ResolvedFITSBeamDefinition) -> str:
    """Return the internal accepted-subset version literal for one definition.

    ``docs/development/sci005_beam_physics_plan.md`` Section 5.1.1: it is
    exactly ``tier3-scalar-v1`` for ``peak`` and exactly
    ``sci005-stage3-full-efield-v1`` for ``uvbeam_peak_common_v1``, and it
    enters the handler pre-load key and the handler scientific fingerprint
    exactly where the scalar literal does today.
    """
    if definition.normalization == _FULL_EFIELD_NORMALIZATION:
        return _FULL_EFIELD_SUBSET_VERSION
    return _ACCEPTED_SUBSET_VERSION


def _fits_preload_key(  # pyright: ignore[reportUnusedFunction]
    definition: ResolvedFITSBeamDefinition,
) -> tuple[Path, str, str, str, float, str]:
    """Return the exact per-factory key used before BeamFITS I/O."""
    if type(definition) is not ResolvedFITSBeamDefinition:
        raise TypeError("definition must be an exact ResolvedFITSBeamDefinition")
    definition.__post_init__()
    return (
        definition.path.resolve(strict=False),
        definition.normalization,
        definition.angular_interpolation,
        definition.frequency_interpolation,
        _FREQUENCY_MATCH_TOLERANCE_HZ,
        _accepted_subset_version(definition),
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return cast(
        tuple[int, int, int, int, int],
        tuple(int(getattr(value, field)) for field in _STAT_FIELDS),
    )


def _changed_error(path: Path) -> BeamFileChangedError:
    return BeamFileChangedError(
        f"BeamFITS {path}: the source changed during snapshot, dependency read, or "
        "scientific validation; retry with a stable local file."
    )


def _snapshot_source(path: Path, directory: Path) -> _Snapshot:
    snapshot_path = directory / "beam.beamfits"
    try:
        source_fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise BeamFileReadError(
            f"BeamFITS {path}: cannot open the resolved local source for reading; "
            "verify permissions and replace it with a readable regular file."
        ) from exc

    try:
        try:
            before = os.fstat(source_fd)
        except OSError as exc:
            raise BeamFileReadError(
                f"BeamFITS {path}: cannot inspect the opened source descriptor; "
                "verify the local file and storage."
            ) from exc
        if not stat.S_ISREG(before.st_mode):
            raise BeamFileReadError(
                f"BeamFITS {path}: source is not a regular local file; provide a "
                "readable regular BeamFITS file."
            )
        digest = hashlib.sha256()
        try:
            target_fd = os.open(
                snapshot_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(target_fd, "wb", closefd=True) as target:
                while True:
                    chunk = os.read(source_fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    _ = target.write(chunk)
                target.flush()
            os.chmod(snapshot_path, 0o600)
        except OSError as exc:
            raise BeamFileReadError(
                f"BeamFITS {path}: failed while copying the source into the private "
                "atomic snapshot; verify local storage and permissions."
            ) from exc
        try:
            after = os.fstat(source_fd)
        except OSError as exc:
            raise BeamFileReadError(
                f"BeamFITS {path}: cannot re-inspect the opened source descriptor; "
                "verify the local file and storage."
            ) from exc
        if _stat_identity(before) != _stat_identity(after):
            raise _changed_error(path)
    finally:
        os.close(source_fd)
    return _Snapshot(
        path=snapshot_path,
        source_stat=_stat_identity(before),
        sha256=digest.hexdigest(),
    )


def _require_attr(beam: Any, name: str, path: Path) -> Any:
    try:
        return getattr(beam, name)
    except (AttributeError, TypeError) as exc:
        raise BeamFileReadError(
            f"BeamFITS {path}: dependency object is missing required {name!r} "
            "metadata; regenerate the file with pyuvdata 3.2.1."
        ) from exc


def _regular_step(
    values: Any,
    *,
    name: str,
    path: Path,
) -> tuple[np.ndarray, float]:
    try:
        array = np.asarray(values)
    except Exception as exc:
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: {name} cannot be represented as a numeric "
            "one-dimensional axis."
        ) from exc
    if array.ndim != 1 or array.size < 2:
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: {name} must be a one-dimensional nondegenerate "
            "regular axis; regenerate a full regular az/ZA grid."
        )
    try:
        finite = bool(np.all(np.isfinite(array)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: {name} must contain real numeric radian values."
        ) from exc
    if not finite:
        raise NonFiniteBeamResponseError(
            f"BeamFITS {path}: {name} contains NaN or Inf; regenerate finite "
            "coordinate metadata."
        )
    try:
        owned = np.array(array, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: {name} must contain float-convertible radian values."
        ) from exc
    differences = np.diff(owned)
    if np.any(differences <= 0.0):
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: {name} must be strictly increasing with a positive "
            "step; regenerate the regular axis."
        )
    step = float(differences[0])
    if not np.allclose(
        differences,
        step,
        rtol=0.0,
        atol=_GRID_TOLERANCE_RAD,
    ):
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: {name} spacing is irregular; Tier 3 accepts only a "
            "regular az/ZA grid."
        )
    return owned, step


def _validate_frequency_axis(value: Any, path: Path) -> np.ndarray:
    try:
        frequencies = np.asarray(value)
    except Exception as exc:
        raise BeamFrequencyDomainError(
            f"BeamFITS {path}: freq_array cannot be represented as a numeric "
            "one-dimensional Hz axis."
        ) from exc
    if frequencies.ndim != 1 or frequencies.size < 1:
        raise BeamFrequencyDomainError(
            f"BeamFITS {path}: freq_array must be a nonempty one-dimensional Hz "
            "axis; regenerate the file with complete intrinsic channels."
        )
    try:
        finite = bool(np.all(np.isfinite(frequencies)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise BeamFrequencyDomainError(
            f"BeamFITS {path}: freq_array must contain real numeric Hz values."
        ) from exc
    if not finite:
        raise NonFiniteBeamResponseError(
            f"BeamFITS {path}: freq_array contains NaN or Inf; regenerate finite "
            "positive frequency metadata."
        )
    try:
        owned = np.array(frequencies, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise BeamFrequencyDomainError(
            f"BeamFITS {path}: freq_array must contain float-convertible Hz values."
        ) from exc
    if np.any(owned <= 0.0):
        raise BeamFrequencyDomainError(
            f"BeamFITS {path}: intrinsic frequencies must be positive Hz values."
        )
    if owned.size > 1 and np.any(np.diff(owned) <= 0.0):
        raise BeamFrequencyDomainError(
            f"BeamFITS {path}: intrinsic frequencies must be unique and strictly "
            "increasing; reorder or regenerate the file."
        )
    return owned


def _derive_feature_scale(
    *,
    zenith_angle_rad: np.ndarray,
    zenith_angle_step_rad: float,
    azimuth_step_rad: float,
    path: Path,
) -> float:
    positive_visible = zenith_angle_rad[
        (zenith_angle_rad > 0.0)
        & (zenith_angle_rad <= np.pi / 2.0 + _HORIZON_COVERAGE_TOLERANCE_RAD)
    ]
    if positive_visible.size == 0:
        raise BeamSamplingDerivationError(
            f"BeamFITS {path}: no positive visible ZA row remains for native-grid "
            "representation-scale derivation; regenerate a nondegenerate grid."
        )
    cosine = np.cos(positive_visible)
    sine = np.sin(positive_visible)
    argument = cosine**2 + sine**2 * np.cos(azimuth_step_rad)
    horizontal = np.arccos(np.clip(argument, -1.0, 1.0))
    native = float(min(zenith_angle_step_rad, float(np.min(horizontal))))
    scale = 2.0 * native
    if not math.isfinite(scale) or scale <= 0.0:
        raise BeamSamplingDerivationError(
            f"BeamFITS {path}: native-grid voltage feature scale {scale!r} is not "
            "finite and positive; regenerate a nondegenerate regular grid."
        )
    return scale


def _owned_uvbeam(
    beam: Any,
    data: np.ndarray,
    basis: np.ndarray,
    path: Path,
) -> _UVBeamLike:
    try:
        owned = copy.deepcopy(beam)
        if owned is beam:
            raise TypeError("dependency deepcopy returned the mutable source object")
        array_names = (
            "basis_vector_array",
            "freq_array",
            "axis1_array",
            "axis2_array",
            "feed_array",
            "feed_angle",
            "bandpass_array",
        )
        for name in array_names:
            value = basis if name == "basis_vector_array" else getattr(beam, name, None)
            if value is not None:
                private = np.array(value, copy=True, order="C")
                private.setflags(write=False)
                setattr(owned, name, private)
                published = getattr(owned, name)
                if (
                    type(published) is not np.ndarray
                    or not published.flags.owndata
                    or not published.flags.c_contiguous
                    or published.flags.writeable
                    or np.shares_memory(published, np.asarray(value))
                ):
                    raise TypeError(
                        f"dependency did not retain private owned read-only {name}"
                    )
        owned.data_array = np.array(
            data,
            dtype=np.complex128,
            copy=True,
            order="C",
        )
        owned.data_array.setflags(write=False)
        if (
            type(owned.data_array) is not np.ndarray
            or not owned.data_array.flags.owndata
            or not owned.data_array.flags.c_contiguous
            or owned.data_array.flags.writeable
            or np.shares_memory(owned.data_array, data)
        ):
            raise TypeError(
                "dependency did not retain private owned read-only data_array"
            )
    except Exception as exc:
        raise BeamFileReadError(
            f"BeamFITS {path}: could not detach dependency arrays into private "
            "owned storage; regenerate the file and retry."
        ) from exc
    return cast(_UVBeamLike, owned)


def _classify_dependency_check_failure(beam: Any, path: Path) -> None:
    """Preserve RadioSim's more specific taxonomy after dependency rejection."""
    for name in (
        "basis_vector_array",
        "freq_array",
        "axis1_array",
        "axis2_array",
        "bandpass_array",
        "data_array",
    ):
        try:
            value = getattr(beam, name, None)
        except (AttributeError, TypeError):
            continue
        if value is not None:
            try:
                finite = bool(np.all(np.isfinite(np.asarray(value))))
            except (TypeError, ValueError, OverflowError):
                continue
            if not finite:
                raise NonFiniteBeamResponseError(
                    f"BeamFITS {path}: {name} contains NaN or Inf; regenerate "
                    "finite BeamFITS science and metadata."
                )
    # Section 5.2.1's one sanctioned classifier extension: pyuvdata's own
    # ``check()`` refuses a complex ``basis_vector_array`` with an untyped
    # ``ValueError`` reading "UVParameter _basis_vector_array is not the
    # appropriate type", which would otherwise surface as ``BeamFileReadError``
    # and leave the frozen ``UnsupportedBeamBasisError`` unreachable.  It runs
    # after the finiteness sweep, which is the frozen precedence.
    try:
        basis_value = getattr(beam, "basis_vector_array", None)
    except (AttributeError, TypeError):
        basis_value = None
    if basis_value is not None:
        try:
            basis_kind = np.asarray(basis_value).dtype.kind
        except (TypeError, ValueError, OverflowError):
            basis_kind = "f"
        if basis_kind != "f":
            raise UnsupportedBeamBasisError(
                f"BeamFITS {path}: basis_vector_array has dtype kind "
                f"{basis_kind!r}; pyuvdata declares this parameter real valued, "
                "so RadioSim requires a real floating stored array."
            )
    for name, label in (
        ("axis1_array", "azimuth axis"),
        ("axis2_array", "zenith-angle axis"),
    ):
        try:
            value = getattr(beam, name, None)
        except (AttributeError, TypeError):
            continue
        if value is not None:
            _ = _regular_step(value, name=label, path=path)


def _run_dependency_check(beam: Any, path: Path) -> None:
    """Run Section 5.1.1 item 1's pinned dependency structure check."""
    try:
        check = _require_attr(beam, "check", path)
        check_result = check(check_extra=True, run_check_acceptability=True)
        if check_result is not True:
            raise ValueError(
                f"UVBeam.check returned {check_result!r}, not exact success True"
            )
    except BeamFileReadError:
        raise
    except Exception as exc:
        _classify_dependency_check_failure(beam, path)
        raise BeamFileReadError(
            f"BeamFITS {path}: pyuvdata rejected the file structure or metadata; "
            "regenerate a valid pyuvdata 3.2.1 BeamFITS file."
        ) from exc


def _validate_beam_family(beam: Any, path: Path) -> None:
    """Run items 2 and 3: beam type, antenna type, and pixel coordinates."""
    beam_type = _require_attr(beam, "beam_type", path)
    antenna_type = _require_attr(beam, "antenna_type", path)
    if type(beam_type) is not str:
        cause = TypeError("beam_type must be an exact dependency string")
        raise UnsupportedBeamTypeError(
            f"BeamFITS {path}: beam_type has unsupported container type "
            f"{type(beam_type).__name__!r}; Tier 3 requires exact 'efield'."
        ) from cause
    if type(antenna_type) is not str:
        cause = TypeError("antenna_type must be an exact dependency string")
        raise UnsupportedBeamTypeError(
            f"BeamFITS {path}: antenna_type has unsupported container type "
            f"{type(antenna_type).__name__!r}; Tier 3 requires exact 'simple'."
        ) from cause
    if beam_type != "efield":
        raise UnsupportedBeamTypeError(
            f"BeamFITS {path}: beam_type={beam_type!r} is unsupported; Tier 3 "
            "requires beam_type='efield', antenna_type='simple', scalar X/Y "
            "identity-basis response."
        )
    if antenna_type != "simple":
        raise UnsupportedBeamTypeError(
            f"BeamFITS {path}: antenna_type={antenna_type!r} is unsupported; Tier 3 "
            "requires antenna_type='simple' without phased-array coupling."
        )

    coordinate_system = _require_attr(beam, "pixel_coordinate_system", path)
    if type(coordinate_system) is not str:
        cause = TypeError("pixel_coordinate_system must be an exact dependency string")
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: pixel_coordinate_system has unsupported container "
            f"type {type(coordinate_system).__name__!r}; Tier 3 requires 'az_za'."
        ) from cause
    if coordinate_system != "az_za":
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: pixel_coordinate_system={coordinate_system!r} is "
            "unsupported; Tier 3 requires a regular full-horizon 'az_za' grid."
        )


def _validate_axes(
    beam: Any,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Run item 9: the frequency axis and the regular closed ``az_za`` grid."""
    frequencies = _validate_frequency_axis(
        _require_attr(beam, "freq_array", path), path
    )
    azimuth, azimuth_step = _regular_step(
        _require_attr(beam, "axis1_array", path),
        name="azimuth axis",
        path=path,
    )
    zenith_angle, zenith_step = _regular_step(
        _require_attr(beam, "axis2_array", path),
        name="zenith-angle axis",
        path=path,
    )
    if abs(float(azimuth[0])) > _GRID_TOLERANCE_RAD:
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: azimuth axis starts at {azimuth[0]!r}, not zero; "
            "regenerate an endpoint-excluded zero-origin regular grid."
        )
    closure = float(azimuth[-1] + azimuth_step)
    if abs(closure - 2.0 * np.pi) > _AZIMUTH_CLOSURE_TOLERANCE_RAD:
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: azimuth closure is {closure!r} radians; last + step "
            "must close 2*pi within 1e-10 radians."
        )
    if abs(float(zenith_angle[0])) > _GRID_TOLERANCE_RAD:
        raise UnsupportedBeamCoordinateError(
            f"BeamFITS {path}: zenith-angle axis starts at "
            f"{zenith_angle[0]!r}, not zero."
        )
    if zenith_angle[-1] < np.pi / 2.0 - _HORIZON_COVERAGE_TOLERANCE_RAD:
        raise BeamAngularDomainError(
            f"BeamFITS {path}: zenith-angle maximum {zenith_angle[-1]!r} does not "
            "reach the horizon; provide complete visible-hemisphere coverage."
        )
    return frequencies, azimuth, azimuth_step, zenith_angle, zenith_step


def _dtype_tolerances(native_dtype: np.dtype[Any]) -> tuple[float, float, float]:
    """Return the accepted dtype-derived ``(atol, rtol, normalization atol)``."""
    epsilon = (
        float(np.finfo(np.float32).eps)
        if native_dtype == np.dtype(np.complex64)
        else float(np.finfo(np.float64).eps)
    )
    return (
        max(1e-12, 32.0 * epsilon),
        max(1e-10, 32.0 * epsilon),
        max(1e-12, 32.0 * epsilon),
    )


def _validate_beam(beam: Any, path: Path) -> _ValidatedBeam:
    _run_dependency_check(beam, path)
    _validate_beam_family(beam, path)

    feed_value = _require_attr(beam, "feed_array", path)
    try:
        feed_array = np.asarray(feed_value)
        if feed_array.ndim != 1:
            raise TypeError("feed_array must be one-dimensional")
        feeds = tuple(feed_array.tolist())
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: feed_array must be a one-dimensional dependency "
            "array containing exact ordered feeds ('x', 'y')."
        ) from exc
    if feeds != ("x", "y"):
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: feed_array={feeds!r} is unsupported; Tier 3 requires "
            "exact ordered feeds ('x', 'y')."
        )
    try:
        feed_angle = np.asarray(_require_attr(beam, "feed_angle", path))
        if feed_angle.shape != (2,):
            raise TypeError("feed_angle must have shape (2,)")
        feed_angle_finite = bool(np.all(np.isfinite(feed_angle)))
    except BeamError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: feed_angle must be a finite numeric dependency "
            "array with shape (2,)."
        ) from exc
    if not feed_angle_finite:
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: feed_angle={feed_angle!r} is unsupported; Tier 3 "
            "requires finite angles (pi/2, 0) radians."
        )
    x_orientation = _require_attr(beam, "x_orientation", path)
    mount_type = _require_attr(beam, "mount_type", path)
    if type(x_orientation) is not str:
        cause = TypeError("x_orientation must be an exact dependency string")
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: x_orientation has unsupported container type "
            f"{type(x_orientation).__name__!r}; Tier 3 requires exact 'east'."
        ) from cause
    if x_orientation != "east":
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: x_orientation={x_orientation!r} is unsupported; "
            "Tier 3 requires x_orientation='east'."
        )
    if not np.allclose(
        feed_angle,
        np.array([np.pi / 2.0, 0.0]),
        rtol=0.0,
        atol=_FEED_ANGLE_TOLERANCE_RAD,
    ):
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: feed_angle={tuple(feed_angle)!r} is unsupported; "
            "Tier 3 requires (pi/2, 0) radians within 1e-12."
        )
    if type(mount_type) is not str:
        cause = TypeError("mount_type must be an exact dependency string")
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: mount_type has unsupported container type "
            f"{type(mount_type).__name__!r}; Tier 3 requires exact 'fixed'."
        ) from cause
    if mount_type != "fixed":
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: mount_type={mount_type!r} is unsupported; Tier 3 "
            "requires mount_type='fixed'."
        )

    naxes = _require_attr(beam, "Naxes_vec", path)
    ncomponents = _require_attr(beam, "Ncomponents_vec", path)
    if type(naxes) is not int or type(ncomponents) is not int:
        cause = TypeError("vector dimensions must be exact dependency integers")
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: vector dimensions have unsupported container types; "
            "Tier 3 requires exact integers Naxes_vec=2 and Ncomponents_vec=2."
        ) from cause
    try:
        basis = np.asarray(_require_attr(beam, "basis_vector_array", path))
        basis_finite = bool(np.all(np.isfinite(basis)))
    except BeamError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: basis_vector_array must be a finite numeric array."
        ) from exc
    if naxes != 2 or ncomponents != 2:
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: Naxes_vec={naxes!r}, "
            f"Ncomponents_vec={ncomponents!r} is unsupported; Tier 3 requires 2x2 "
            "identity-basis E-field vectors."
        )
    if not basis_finite:
        raise NonFiniteBeamResponseError(
            f"BeamFITS {path}: basis_vector_array contains NaN or Inf; regenerate "
            "finite identity-basis metadata."
        )

    frequencies, azimuth, azimuth_step, zenith_angle, zenith_step = _validate_axes(
        beam,
        path,
    )

    expected_basis_shape = (2, 2, zenith_angle.size, azimuth.size)
    if basis.shape != expected_basis_shape:
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: basis_vector_array shape {basis.shape!r} is "
            f"unsupported; expected {expected_basis_shape!r}."
        )
    identity = np.zeros(expected_basis_shape, dtype=np.float64)
    identity[0, 0] = 1.0
    identity[1, 1] = 1.0
    if not np.allclose(basis, identity, rtol=0.0, atol=_BASIS_TOLERANCE):
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: basis_vector_array is not identity within 1e-12; "
            "Tier 3 defers basis transforms to Tier 5."
        )

    normalization = _require_attr(beam, "data_normalization", path)
    if type(normalization) is not str:
        cause = TypeError("data_normalization must be an exact dependency string")
        raise BeamNormalizationError(
            f"BeamFITS {path}: data_normalization has unsupported container type "
            f"{type(normalization).__name__!r}; Tier 3 requires exact 'peak'."
        ) from cause
    if normalization != "peak":
        raise BeamNormalizationError(
            f"BeamFITS {path}: data_normalization={normalization!r} is unsupported; "
            "provide an already normalized 'peak' beam."
        )
    data_value = _require_attr(beam, "data_array", path)
    if not isinstance(data_value, np.ndarray):
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: data_array must be a NumPy array with scalar E-field "
            "shape (2, 2, Nfreq, Nza, Naz)."
        )
    data_array = cast(np.ndarray[Any, Any], data_value)
    native_dtype = data_array.dtype
    if native_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise UnsupportedBeamPrecisionError(
            f"BeamFITS {path}: native data dtype {native_dtype.name!r} is "
            "unsupported; write complex64 or complex128 E-field samples."
        )
    expected_data_shape = (
        2,
        2,
        frequencies.size,
        zenith_angle.size,
        azimuth.size,
    )
    if data_array.shape != expected_data_shape:
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: data_array shape {data_array.shape!r} is unsupported; "
            f"expected {expected_data_shape!r}."
        )
    if not np.all(np.isfinite(data_array)):
        raise NonFiniteBeamResponseError(
            f"BeamFITS {path}: native E-field data contains NaN or Inf."
        )

    scalar_atol, scalar_rtol, normalization_atol = _dtype_tolerances(native_dtype)
    canonical_data = np.array(data_array, dtype=np.complex128, copy=True, order="C")
    canonical_basis = np.array(basis, dtype=np.float64, copy=True, order="C")
    jones = np.einsum(
        "afqzy,aczy->qzyfc",
        canonical_data,
        canonical_basis,
        optimize=True,
    )
    matrix_scale = np.max(np.abs(jones), axis=(-2, -1))
    bound = scalar_atol + scalar_rtol * matrix_scale
    if np.any(np.abs(jones[..., 0, 1]) > bound) or np.any(
        np.abs(jones[..., 1, 0]) > bound
    ):
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: native Jones response contains cross-polar terms; "
            "Tier 3 accepts only scalar e I2 response."
        )
    if np.any(np.abs(jones[..., 0, 0] - jones[..., 1, 1]) > bound):
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: native X/Y diagonal responses differ; Tier 3 "
            "accepts only scalar e I2 response."
        )

    try:
        bandpass = np.asarray(_require_attr(beam, "bandpass_array", path))
        bandpass_finite = bool(np.all(np.isfinite(bandpass)))
    except BeamError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise BeamNormalizationError(
            f"BeamFITS {path}: bandpass_array must be a finite numeric dependency "
            "array of unit values."
        ) from exc
    if bandpass.shape != (frequencies.size,):
        raise BeamNormalizationError(
            f"BeamFITS {path}: bandpass_array shape {bandpass.shape!r} is invalid; "
            f"expected ({frequencies.size},) unit values."
        )
    if not bandpass_finite:
        raise NonFiniteBeamResponseError(
            f"BeamFITS {path}: bandpass_array contains NaN or Inf."
        )
    if not np.allclose(
        bandpass,
        np.ones(frequencies.size),
        rtol=0.0,
        atol=normalization_atol,
    ):
        raise BeamNormalizationError(
            f"BeamFITS {path}: peak beam has non-unit bandpass_array; Tier 3 E-Jones "
            "requires unit bandpass and defers spectral bandpass to Tier 7."
        )

    visible = zenith_angle <= np.pi / 2.0 + _HORIZON_COVERAGE_TOLERANCE_RAD
    scalar = jones[..., 0, 0]
    for index in range(frequencies.size):
        peak = float(np.max(np.abs(scalar[index, visible, :])))
        if not math.isfinite(peak):
            raise NonFiniteBeamResponseError(
                f"BeamFITS {path}: native scalar peak at frequency "
                f"{frequencies[index]!r} Hz is non-finite."
            )
        if peak <= 0.0 or not math.isclose(
            peak,
            1.0,
            rel_tol=0.0,
            abs_tol=normalization_atol,
        ):
            raise BeamNormalizationError(
                f"BeamFITS {path}: data_normalization='peak' but native scalar "
                f"maximum at frequency {frequencies[index]!r} Hz is {peak!r}; "
                "provide explicit positive unit-peak science."
            )

    feature_scale = _derive_feature_scale(
        zenith_angle_rad=zenith_angle,
        zenith_angle_step_rad=zenith_step,
        azimuth_step_rad=azimuth_step,
        path=path,
    )
    scalar_data = np.array(jones[..., 0, 0], dtype=np.complex128, copy=True, order="C")
    canonical_data.fill(0.0)
    canonical_data[0, 0] = scalar_data
    canonical_data[1, 1] = scalar_data
    canonical_basis = identity
    owned = _owned_uvbeam(beam, canonical_data, canonical_basis, path)
    return _ValidatedBeam(
        beam=owned,
        native_dtype=native_dtype.name,
        data_shape=tuple(int(item) for item in expected_data_shape),
        frequencies_hz=frequencies,
        azimuth_rad=azimuth,
        zenith_angle_rad=zenith_angle,
        azimuth_step_rad=azimuth_step,
        zenith_angle_step_rad=zenith_step,
        feature_scale_rad=feature_scale,
        scalar_absolute_tolerance=scalar_atol,
        scalar_relative_tolerance=scalar_rtol,
        normalization_absolute_tolerance=normalization_atol,
    )


def _derive_x_orientation(
    feed_array: tuple[str, str],
    feed_angle_rad: tuple[float, float],
    path: Path,
) -> str | None:
    """Return pyuvdata 3.2.1's ``get_x_orientation_from_feeds`` verdict exactly.

    Section 5.1.1 item 7 requires the *exact* verdict implied by one feed pair
    and its angles, so the dependency's own helper is called with its default
    exact tolerances rather than with a UVParameter's ``isclose`` window.  It is
    consistency metadata and is never applied as a second rotation. ``None`` is
    a legal, common result: it is what the function returns for a rotated linear
    receptor and for a circular receptor whose static rotation is neither ``0``
    nor ``pi/2``.
    """
    try:
        module = importlib.import_module("pyuvdata.utils.pol")
        derive = module.get_x_orientation_from_feeds
    except Exception as exc:
        raise BeamDependencyError(
            f"BeamFITS {path}: the pinned pyuvdata 3.2.1 dependency does not "
            "expose utils.pol.get_x_orientation_from_feeds."
        ) from exc
    verdict = derive(
        feed_array=np.array(list(feed_array)),
        feed_angle=np.array(list(feed_angle_rad), dtype=np.float64),
    )
    if verdict is not None and verdict not in {"east", "north"}:
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: pyuvdata derived x_orientation {verdict!r} from "
            f"feed_array={feed_array!r} and feed_angle={feed_angle_rad!r}; the "
            "pinned contract returns exactly 'east', 'north', or None."
        )
    return cast("str | None", verdict)


def _validate_efield_receptor_agreement(
    *,
    path: Path,
    feeds: tuple[str, str],
    feed_angle: tuple[float, float],
    receptors: tuple[_EfieldReceptorExpectation, ...],
) -> str:
    """Run items 5b, 6, and 7 against every antenna the file is assigned to."""
    file_verdict = _derive_x_orientation(feeds, feed_angle, path)
    for expectation in receptors:
        antenna = expectation.antenna_id
        if expectation.feed_array != feeds:
            raise UnsupportedBeamFeedError(
                f"BeamFITS {path}: feed_array={feeds!r} does not equal the "
                f"resolved receptor feed pair {expectation.feed_array!r} of "
                f"canonical antenna number={antenna.number}, "
                f"name={antenna.name!r}; the accepted full-efield subset "
                "requires the file row labels to be exactly that antenna's own."
            )
        expected_angles = expectation.feed_angle_rad
        difference = np.abs(
            np.remainder(
                np.asarray(feed_angle, dtype=np.float64)
                - np.asarray(expected_angles, dtype=np.float64)
                + np.pi,
                2.0 * np.pi,
            )
            - np.pi
        )
        if float(np.max(difference)) > _FEED_ANGLE_TOLERANCE_RAD:
            raise UnsupportedBeamFeedError(
                f"BeamFITS {path}: feed_angle={feed_angle!r} does not equal the "
                f"resolved receptor feed angles {expected_angles!r} of canonical "
                f"antenna number={antenna.number}, name={antenna.name!r} modulo "
                f"2*pi within {_FEED_ANGLE_TOLERANCE_RAD!r} radians; the "
                f"{expectation.basis!r} receptor with feed_rotation_rad="
                f"{expectation.feed_rotation_rad!r} implies those angles."
            )
        receptor_verdict = _derive_x_orientation(
            expectation.feed_array,
            expected_angles,
            path,
        )
        if receptor_verdict != file_verdict:
            raise UnsupportedBeamFeedError(
                f"BeamFITS {path}: the derived x_orientation of the file's own "
                f"feed_array={feeds!r} and feed_angle={feed_angle!r} is "
                f"{file_verdict!r}, but canonical antenna "
                f"number={antenna.number}, name={antenna.name!r} implies "
                f"{receptor_verdict!r}; it is consistency metadata and is never "
                "applied as a second rotation."
            )
    return "none" if file_verdict is None else file_verdict


def _validate_stored_basis(
    beam: Any,
    path: Path,
    *,
    expected_shape: tuple[int, ...],
) -> None:
    """Run item 10's stored-basis contract in its frozen precedence.

    ``UVBeam._prepare_basis_vector_array`` raises a bare untyped
    ``NotImplementedError`` whenever any stored off-diagonal entry is strictly
    positive, and in every other case discards the stored array and rebuilds the
    exact native identity per interpolation point.  A stored non-identity basis
    therefore either crashes evaluation outside every typed rejection or is
    silently replaced by a different basis than the file declares, so rejecting
    it at load is the only outcome that is both typed and truthful.
    """
    stored = _require_attr(beam, "basis_vector_array", path)
    if stored is None:
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: basis_vector_array is absent; the accepted "
            "full-efield subset requires the stored native identity basis."
        )
    try:
        basis = np.asarray(stored)
        finite = bool(np.all(np.isfinite(basis)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: basis_vector_array must be a finite real floating array."
        ) from exc
    if not finite:
        raise NonFiniteBeamResponseError(
            f"BeamFITS {path}: basis_vector_array contains NaN or Inf; "
            "regenerate finite native-identity basis metadata."
        )
    # Judged by kind and width, never by a byte-order-qualified comparison:
    # BeamFITS round-trips this array as big-endian '>f8' or '>f4', and both
    # stored widths are accepted because 1.0 and 0.0 are exactly representable
    # and round-trip bit-exactly in each.
    if basis.dtype.kind != "f" or basis.dtype.itemsize not in {4, 8}:
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: basis_vector_array dtype {basis.dtype.name!r} is "
            "unsupported; pyuvdata declares this parameter real valued, so the "
            "accepted subset requires stored float32 or float64 entries."
        )
    if basis.shape != expected_shape:
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: basis_vector_array shape {basis.shape!r} is "
            f"unsupported; expected {expected_shape!r}."
        )
    identity = np.zeros(expected_shape, dtype=basis.dtype)
    identity[0, 0] = 1.0
    identity[1, 1] = 1.0
    if not np.array_equal(basis, identity):
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: basis_vector_array is not exactly the native "
            "identity at every stored grid point; entries [0, 0] and [1, 1] must "
            "be exactly 1.0 and entries [0, 1] and [1, 0] exactly 0.0, because "
            "pyuvdata either refuses or silently discards any other stored basis."
        )


def _validate_converted_continuity(
    *,
    path: Path,
    canonical_data: np.ndarray,
    azimuth_rad: np.ndarray,
    frequencies_hz: np.ndarray,
    scalar_atol: float,
    scalar_rtol: float,
) -> None:
    """Run corrected Section 5.2.1's two frozen file-grid continuity predicates.

    Both are evaluated on the converted matrices at every intrinsic frequency,
    before any frequency interpolation, and neither touches the accepted scalar
    ``peak`` path.

    The **zenith** predicate is the de-spin form.  Requiring the converted
    matrices of the ``za = 0`` row to be *equal* is withdrawn: the chain tangent
    pair spins with the azimuth coordinate at the pole while the physical
    response is one fixed map, so under any constant ``M`` a perfectly valid
    file's converted zenith row spreads by the full response scale.  What is
    single-valued is ``J(az_uv) R(az_uv)^T``, equivalently
    ``J(az_uv) = J(az_ref) R(az_uv - az_ref)``.

    The **wrap** predicate is a second difference compared against the interior
    **maximum**, because only the second difference is scale-consistent: for a
    twice-differentiable periodic row sampled at step ``h`` every
    ``Delta^2_k`` equals ``h^2 J''(xi_k)``, so seam and interior second
    differences are the same order at every sampling density.  A first
    difference has no such property -- it holds only when the sampling is
    symmetric about the seam, an accident of one particular grid.
    """
    despin = _zenith_spin_rotation(np.asarray(azimuth_rad, dtype=np.float64))
    for index in range(int(frequencies_hz.size)):
        matrices = _converted_native_jones(canonical_data[:, :, index])

        zenith_row = matrices[0]
        de_spun = zenith_row @ np.transpose(despin, (0, 2, 1))
        zenith_scale = float(np.max(np.abs(de_spun)))
        zenith_bound = scalar_atol + scalar_rtol * zenith_scale
        spread = float(np.max(np.abs(de_spun - de_spun[0])))
        if spread > zenith_bound:
            raise UnsupportedBeamCoordinateError(
                f"BeamFITS {path}: the de-spun zenith-angle row at frequency "
                f"{float(frequencies_hz[index])!r} Hz is not constant; its "
                f"{zenith_row.shape[0]} azimuth samples of "
                f"J(az_uv) R(az_uv)^T spread by {spread!r} against the accepted "
                f"bound {zenith_bound!r}. The first row is one physical "
                "direction whose azimuth is arbitrary, so a file whose physical "
                "matrix depends on it is rejected rather than averaged."
            )

        count = int(matrices.shape[1])
        second = (
            np.roll(matrices, -1, axis=1)
            - 2.0 * matrices
            + np.roll(matrices, 1, axis=1)
        )
        seam = np.max(np.abs(second[:, 0]), axis=(1, 2))
        interior_samples = second[:, 2 : count - 1]
        interior = (
            np.max(np.abs(interior_samples), axis=(1, 2, 3))
            if interior_samples.shape[1] > 0
            else np.zeros(matrices.shape[0], dtype=np.float64)
        )
        scale = np.max(np.abs(matrices), axis=(1, 2, 3))
        bound = scalar_atol + scalar_rtol * scale
        excess = seam - _WRAP_SECOND_DIFFERENCE_FACTOR * interior - bound
        if bool(np.any(excess > 0.0)):
            row = int(np.argmax(excess))
            raise UnsupportedBeamCoordinateError(
                f"BeamFITS {path}: the converted matrix is discontinuous across "
                f"the endpoint-excluded azimuth wrap at frequency "
                f"{float(frequencies_hz[index])!r} Hz and zenith-angle row "
                f"{row}: the seam second difference {float(seam[row])!r} exceeds "
                f"{_WRAP_SECOND_DIFFERENCE_FACTOR!r} times the interior maximum "
                f"{float(interior[row])!r} by more than the accepted bound "
                f"{float(bound[row])!r}."
            )


def _validate_efield_beam(
    beam: Any,
    path: Path,
    *,
    receptors: tuple[_EfieldReceptorExpectation, ...],
) -> _ValidatedBeam:
    """Validate one file against Section 5.1.1's ordered full-efield contract.

    The thirteen items are evaluated in exactly the frozen order, so that the
    first recorded rejection is deterministic, with Section 5.2.1's two
    converted-grid continuity predicates between the data and normalization
    items.  Item 13 keeps its accepted site at the top of
    :func:`_load_fits_handler`, where its byte-frozen ``float128`` message is
    raised before any dependency read.
    """
    if not receptors:
        raise TypeError(
            "the full-efield accepted subset requires the resolved receptor set "
            "of every assigned antenna; pass receptors=<ResolvedReceptorSet> to "
            "load_beam_system."
        )
    # 1. Dependency and structure.
    _run_dependency_check(beam, path)
    # 2. Beam and antenna type.  3. Pixel coordinate system.
    _validate_beam_family(beam, path)

    # 4. Vector dimensions.
    naxes = _require_attr(beam, "Naxes_vec", path)
    ncomponents = _require_attr(beam, "Ncomponents_vec", path)
    if type(naxes) is not int or type(ncomponents) is not int:
        cause = TypeError("vector dimensions must be exact dependency integers")
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: vector dimensions have unsupported container types; "
            "the accepted full-efield subset requires exact integers Naxes_vec=2 "
            "and Ncomponents_vec=2."
        ) from cause
    if naxes != 2 or ncomponents != 2:
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: Naxes_vec={naxes!r}, "
            f"Ncomponents_vec={ncomponents!r} is unsupported; a three-component "
            "field has no two-column tangent image and no frozen conversion in "
            "the accepted full-efield subset."
        )

    # 5. Feed identity.
    feed_count = _require_attr(beam, "Nfeeds", path)
    if type(feed_count) is not int or feed_count != 2:
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: Nfeeds={feed_count!r} is unsupported; the accepted "
            "full-efield subset requires exactly two native feeds."
        )
    feed_value = _require_attr(beam, "feed_array", path)
    try:
        feed_array = np.asarray(feed_value)
        if feed_array.ndim != 1:
            raise TypeError("feed_array must be one-dimensional")
        feeds = cast(tuple[str, ...], tuple(feed_array.tolist()))
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: feed_array must be a one-dimensional dependency "
            "array containing the ordered pair ('x', 'y') or ('r', 'l')."
        ) from exc
    if feeds not in _ACCEPTED_FEED_PAIRS:
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: feed_array={feeds!r} is unsupported; the accepted "
            "full-efield subset requires exactly the ordered pair ('x', 'y') or "
            "('r', 'l')."
        )
    ordered_feeds = cast(tuple[str, str], feeds)

    # 6. Feed angles.  7. Derived orientation consistency.
    try:
        feed_angle_array = np.asarray(_require_attr(beam, "feed_angle", path))
        if feed_angle_array.shape != (2,):
            raise TypeError("feed_angle must have shape (2,)")
        feed_angle_finite = bool(np.all(np.isfinite(feed_angle_array)))
        feed_angle = (
            float(feed_angle_array[0]),
            float(feed_angle_array[1]),
        )
    except BeamError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: feed_angle must be a finite numeric dependency "
            "array with shape (2,)."
        ) from exc
    if not feed_angle_finite:
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: feed_angle={tuple(feed_angle_array)!r} is "
            "unsupported; the accepted full-efield subset requires two finite "
            "radian angles."
        )
    verdict = _validate_efield_receptor_agreement(
        path=path,
        feeds=ordered_feeds,
        feed_angle=feed_angle,
        receptors=receptors,
    )
    legacy_orientation = _require_attr(beam, "x_orientation", path)
    if legacy_orientation is not None:
        if type(legacy_orientation) is not str:
            cause = TypeError("x_orientation must be an exact dependency string")
            raise UnsupportedBeamFeedError(
                f"BeamFITS {path}: x_orientation has unsupported container type "
                f"{type(legacy_orientation).__name__!r}."
            ) from cause
        # ``verdict`` already renders the agreed ``None`` as ``"none"``, which no
        # legacy value can equal, so a file exposing one while the feeds imply
        # none is the disagreement item 7 rejects.
        if legacy_orientation != verdict:
            raise UnsupportedBeamFeedError(
                f"BeamFITS {path}: the legacy x_orientation "
                f"{legacy_orientation!r} disagrees with the value "
                f"{verdict!r} derived from feed_array={ordered_feeds!r} and "
                f"feed_angle={feed_angle!r}."
            )

    # 8. Mount.
    mount_type = _require_attr(beam, "mount_type", path)
    if type(mount_type) is not str:
        cause = TypeError("mount_type must be an exact dependency string")
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: mount_type has unsupported container type "
            f"{type(mount_type).__name__!r}; the accepted full-efield subset "
            f"requires exact {_ACCEPTED_MOUNT_TYPE!r}."
        ) from cause
    if mount_type != _ACCEPTED_MOUNT_TYPE:
        raise UnsupportedBeamFeedError(
            f"BeamFITS {path}: mount_type={mount_type!r} is unsupported; exact "
            f"{_ACCEPTED_MOUNT_TYPE!r} declares that the stored az_za pattern is "
            "intrinsic to the antenna beam frame, and RadioSim's instrument plus "
            "the P term remain the sole owners of field rotation."
        )

    # 9. Grid.
    frequencies, azimuth, azimuth_step, zenith_angle, zenith_step = _validate_axes(
        beam,
        path,
    )

    # 10. Basis-vector array.
    expected_basis_shape = (2, 2, int(zenith_angle.size), int(azimuth.size))
    _validate_stored_basis(beam, path, expected_shape=expected_basis_shape)

    # 11. Data.
    data_value = _require_attr(beam, "data_array", path)
    if not isinstance(data_value, np.ndarray):
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: data_array must be a NumPy array with full "
            "E-field shape (2, 2, Nfreq, Nza, Naz)."
        )
    data_array = cast(np.ndarray[Any, Any], data_value)
    native_dtype = data_array.dtype
    if native_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise UnsupportedBeamPrecisionError(
            f"BeamFITS {path}: native data dtype {native_dtype.name!r} is "
            "unsupported; write complex64 or complex128 E-field samples."
        )
    expected_data_shape = (
        2,
        2,
        int(frequencies.size),
        int(zenith_angle.size),
        int(azimuth.size),
    )
    if data_array.shape != expected_data_shape:
        raise UnsupportedBeamBasisError(
            f"BeamFITS {path}: data_array shape {data_array.shape!r} is "
            f"unsupported; expected {expected_data_shape!r}."
        )
    if not np.all(np.isfinite(data_array)):
        raise NonFiniteBeamResponseError(
            f"BeamFITS {path}: native E-field data contains NaN or Inf."
        )

    scalar_atol, scalar_rtol, normalization_atol = _dtype_tolerances(native_dtype)
    canonical_data = np.array(data_array, dtype=np.complex128, copy=True, order="C")

    _validate_converted_continuity(
        path=path,
        canonical_data=canonical_data,
        azimuth_rad=azimuth,
        frequencies_hz=frequencies,
        scalar_atol=scalar_atol,
        scalar_rtol=scalar_rtol,
    )

    # 12. Normalization.
    normalization = _require_attr(beam, "data_normalization", path)
    if type(normalization) is not str:
        cause = TypeError("data_normalization must be an exact dependency string")
        raise BeamNormalizationError(
            f"BeamFITS {path}: data_normalization has unsupported container type "
            f"{type(normalization).__name__!r}; the accepted full-efield subset "
            "requires exact 'peak'."
        ) from cause
    if normalization != "peak":
        raise BeamNormalizationError(
            f"BeamFITS {path}: data_normalization={normalization!r} is "
            "unsupported; provide an already normalized 'peak' beam."
        )
    try:
        bandpass = np.asarray(_require_attr(beam, "bandpass_array", path))
        bandpass_finite = bool(np.all(np.isfinite(bandpass)))
    except BeamError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise BeamNormalizationError(
            f"BeamFITS {path}: bandpass_array must be a finite numeric dependency "
            "array of unit values."
        ) from exc
    if bandpass.shape != (frequencies.size,):
        raise BeamNormalizationError(
            f"BeamFITS {path}: bandpass_array shape {bandpass.shape!r} is "
            f"invalid; expected ({frequencies.size},) unit values."
        )
    if not bandpass_finite:
        raise NonFiniteBeamResponseError(
            f"BeamFITS {path}: bandpass_array contains NaN or Inf."
        )
    if not np.allclose(
        bandpass,
        np.ones(frequencies.size),
        rtol=0.0,
        atol=normalization_atol,
    ):
        raise BeamNormalizationError(
            f"BeamFITS {path}: peak beam has non-unit bandpass_array; the "
            "accepted full-efield subset requires unit bandpass and defers "
            "direction-independent spectral gain to the configured B term."
        )
    stored_peaks: list[tuple[float, float]] = []
    for index in range(int(frequencies.size)):
        # The maximum over both vector axes, both feeds, and the *complete*
        # stored grid, below-horizon rows included: exactly the common scalar
        # factor UVBeam.peak_normalize divides by.  RadioSim never calls that
        # mutating operation and never renormalizes over the visible rows alone.
        peak = float(np.max(np.abs(canonical_data[:, :, index])))
        if not math.isfinite(peak):
            raise NonFiniteBeamResponseError(
                f"BeamFITS {path}: full-stored-grid maximum at frequency "
                f"{frequencies[index]!r} Hz is non-finite."
            )
        if peak <= 0.0 or not math.isclose(
            peak,
            1.0,
            rel_tol=0.0,
            abs_tol=normalization_atol,
        ):
            raise BeamNormalizationError(
                f"BeamFITS {path}: data_normalization='peak' but the "
                "full-stored-grid maximum over both vector axes and both feeds "
                f"at frequency {frequencies[index]!r} Hz is {peak!r}; the "
                "accepted full-efield subset requires one common unit maximum "
                "including any stored below-horizon rows."
            )
        stored_peaks.append((float(frequencies[index]), peak))

    feature_scale = _derive_feature_scale(
        zenith_angle_rad=zenith_angle,
        zenith_angle_step_rad=zenith_step,
        azimuth_step_rad=azimuth_step,
        path=path,
    )
    canonical_basis = np.zeros(expected_basis_shape, dtype=np.float64)
    canonical_basis[0, 0] = 1.0
    canonical_basis[1, 1] = 1.0
    owned = _owned_uvbeam(beam, canonical_data, canonical_basis, path)
    return _ValidatedBeam(
        beam=owned,
        native_dtype=native_dtype.name,
        data_shape=tuple(int(item) for item in expected_data_shape),
        frequencies_hz=frequencies,
        azimuth_rad=azimuth,
        zenith_angle_rad=zenith_angle,
        azimuth_step_rad=azimuth_step,
        zenith_angle_step_rad=zenith_step,
        feature_scale_rad=feature_scale,
        scalar_absolute_tolerance=scalar_atol,
        scalar_relative_tolerance=scalar_rtol,
        normalization_absolute_tolerance=normalization_atol,
        efield=_ValidatedEfieldFacts(
            feed_array=ordered_feeds,
            feed_angle_rad=feed_angle,
            x_orientation=cast("str | None", legacy_orientation),
            derived_x_orientation_verdict=verdict,
            stored_grid_peak_by_frequency=tuple(stored_peaks),
        ),
    )


def _read_and_validate(
    *,
    loader: _UVBeamLoaderProtocol,
    snapshot: _Snapshot,
    source_path: Path,
    efield_receptors: tuple[_EfieldReceptorExpectation, ...] | None,
) -> _ValidatedBeam:
    try:
        beam = loader.read(snapshot.path)
    except (BeamDependencyError, BeamFileReadError):
        raise
    except BeamError:
        raise
    except Exception as exc:
        raise BeamFileReadError(
            f"BeamFITS {source_path}: pyuvdata could not read the private snapshot; "
            "regenerate a valid BeamFITS file with pyuvdata 3.2.1."
        ) from exc
    if efield_receptors is None:
        return _validate_beam(beam, source_path)
    return _validate_efield_beam(beam, source_path, receptors=efield_receptors)


def _validate_observation_frequencies(
    value: Any,
    *,
    native_frequencies_hz: np.ndarray,
    interpolation_kind: str,
    identity: str,
) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise BeamFrequencyDomainError(
            "observation_frequencies_hz must be a nonempty exact tuple of Python "
            "floats."
        )
    source = cast(tuple[Any, ...], value)
    copied: list[float] = []
    previous: float | None = None
    for index, item in enumerate(source):
        frequency = cast(float, item)
        if type(frequency) is not float or not math.isfinite(frequency):
            raise NonFiniteBeamResponseError(
                f"observation_frequencies_hz[{index}] must be an exact finite "
                f"Python float; observed {frequency!r}."
            )
        if previous is not None and frequency <= previous:
            raise BeamFrequencyDomainError(
                "observation_frequencies_hz must be strictly increasing and is "
                "never reordered by the BeamFITS loader."
            )
        previous = frequency
        _preflight_frequency(
            frequencies_hz=native_frequencies_hz,
            target_hz=frequency,
            interpolation_kind=interpolation_kind,
            identity=identity,
        )
        copied.append(frequency)
    return tuple(copied)


def _scientific_fingerprint(
    *,
    definition: ResolvedFITSBeamDefinition,
    file_sha256: str,
    pyuvdata_version: str,
    validated: _ValidatedBeam,
    observation_frequencies_hz: tuple[float, ...],
    feature_scales: tuple[tuple[float, float], ...],
) -> str:
    """Return the handler's scientific fingerprint.

    Section 5.2.1 extends the existing ``validated_metadata`` and ``contracts``
    blocks for the full-efield subset -- with the resolved feed pair, the
    derived orientation, the mount, the three convention literals, and the
    realized per-frequency common maxima -- and adds no new fingerprint path.
    The accepted scalar payload is unchanged byte for byte.
    """
    efield = validated.efield
    payload: dict[str, Any] = {
        "schema_version": "tier3-beam-v1",
        "kind": "fits_handler",
        "accepted_subset_version": _accepted_subset_version(definition),
        "pyuvdata_version": pyuvdata_version,
        "fits_content_sha256": file_sha256,
        "validated_metadata": {
            "beam_type": "efield",
            "antenna_type": "simple",
            "pixel_coordinate_system": "az_za",
            "mount_type": "fixed",
            "data_normalization": "peak",
            "feed_array": ("x", "y") if efield is None else efield.feed_array,
            "x_orientation": (
                "east" if efield is None else efield.derived_x_orientation_verdict
            ),
            "feed_angle_rad": (
                (math.pi / 2.0, 0.0) if efield is None else efield.feed_angle_rad
            ),
            "data_shape": validated.data_shape,
            "native_dtype": validated.native_dtype,
            "native_frequencies_hz": tuple(
                float(item) for item in validated.frequencies_hz
            ),
            "azimuth_start_rad": float(validated.azimuth_rad[0]),
            "azimuth_step_rad": validated.azimuth_step_rad,
            "azimuth_count": int(validated.azimuth_rad.size),
            "zenith_angle_start_rad": float(validated.zenith_angle_rad[0]),
            "zenith_angle_step_rad": validated.zenith_angle_step_rad,
            "zenith_angle_max_rad": float(validated.zenith_angle_rad[-1]),
            "zenith_angle_count": int(validated.zenith_angle_rad.size),
        },
        "contracts": {
            "basis": "finite_identity_2x2",
            "scalar_jones": "e_i2_no_conjugation",
            "normalization": "positive_unit_peak_and_unit_bandpass",
            "basis_tolerance": _BASIS_TOLERANCE,
            "feed_angle_tolerance_rad": _FEED_ANGLE_TOLERANCE_RAD,
            "scalar_absolute_tolerance": validated.scalar_absolute_tolerance,
            "scalar_relative_tolerance": validated.scalar_relative_tolerance,
            "normalization_absolute_tolerance": (
                validated.normalization_absolute_tolerance
            ),
            "frequency_match_tolerance_hz": _FREQUENCY_MATCH_TOLERANCE_HZ,
            "azimuth_closure_tolerance_rad": (_AZIMUTH_CLOSURE_TOLERANCE_RAD),
            "horizon_coverage_tolerance_rad": (_HORIZON_COVERAGE_TOLERANCE_RAD),
        },
        "load_options": {
            "normalization": definition.normalization,
            "angular_interpolation": definition.angular_interpolation,
            "frequency_interpolation": definition.frequency_interpolation,
            "interpolation_function": "az_za_simple",
            "spline_opts": {"kx": 1, "ky": 1, "s": 0},
        },
        "observation_frequencies_hz": observation_frequencies_hz,
        "native_grid_representation_scales": feature_scales,
    }
    if efield is not None:
        payload["validated_metadata"]["resolved_feed_array"] = efield.feed_array
        payload["validated_metadata"]["derived_x_orientation_verdict"] = (
            efield.derived_x_orientation_verdict
        )
        payload["validated_metadata"]["stored_grid_peak_by_frequency"] = (
            efield.stored_grid_peak_by_frequency
        )
        payload["contracts"]["basis_vector_convention"] = _BASIS_VECTOR_CONVENTION
        payload["contracts"]["factorization_convention"] = _FACTORIZATION_CONVENTION
        payload["contracts"]["zenith_limit_convention"] = _ZENITH_LIMIT_CONVENTION
    return _canonical_digest(payload)


def _cleanup_temporary_directory(
    temporary: tempfile.TemporaryDirectory[str],
    source_path: Path,
) -> None:
    try:
        temporary.cleanup()
    except Exception as exc:
        raise BeamFileReadError(
            f"BeamFITS {source_path}: private snapshot cleanup failed before "
            "handler publication; repair temporary storage and retry."
        ) from exc


def _load_fits_handler(  # pyright: ignore[reportUnusedFunction]
    definition: ResolvedFITSBeamDefinition,
    *,
    observation_frequencies_hz: tuple[float, ...],
    precision: PrecisionConfig,
    handler_ordinal: int,
    loader: _UVBeamLoaderProtocol | None = None,
    efield_receptors: tuple[_EfieldReceptorExpectation, ...] = (),
) -> _LoadedFITSHandler:
    """Load one already-resolved FITS definition into a private evaluator.

    ``efield_receptors`` carries the resolved receptor of every antenna the
    definition is assigned to, which Section 5.1.1 items 5 through 7 compare the
    file's own feed labels and angles against.  It is read only when the
    definition selects the ``uvbeam_peak_common_v1`` accepted subset.
    """
    if type(definition) is not ResolvedFITSBeamDefinition:
        raise TypeError("definition must be an exact ResolvedFITSBeamDefinition")
    definition.__post_init__()
    if type(precision) is not PrecisionConfig:
        raise TypeError("precision must be an exact PrecisionConfig")
    if type(handler_ordinal) is not int or not 0 <= handler_ordinal <= 9999:
        raise ValueError("handler_ordinal must be an exact integer in [0, 9999]")
    beam_precision = precision.jones.beam
    if beam_precision == "float128":
        raise UnsupportedBeamPrecisionError(
            f"BeamFITS {definition.path}: beam precision 'float128' would require "
            "complex256, but accepted files and pyuvdata interpolation provide at "
            "most complex128; select beam float32 or float64."
        )
    result_dtype = np.dtype(
        np.complex64 if beam_precision == "float32" else np.complex128
    )
    selected_loader: _UVBeamLoaderProtocol = (
        _ProductionUVBeamLoader() if loader is None else loader
    )
    try:
        temporary = tempfile.TemporaryDirectory(prefix="radiosim-beam-")
    except Exception as exc:
        raise BeamFileReadError(
            f"BeamFITS {definition.path}: could not create the private snapshot "
            "directory; repair temporary storage and retry."
        ) from exc
    try:
        snapshot = _snapshot_source(definition.path, Path(temporary.name))
        validated = _read_and_validate(
            loader=selected_loader,
            snapshot=snapshot,
            source_path=definition.path,
            efield_receptors=(
                efield_receptors
                if definition.normalization == _FULL_EFIELD_NORMALIZATION
                else None
            ),
        )
        try:
            current_stat = _stat_identity(os.stat(definition.path))
        except OSError as exc:
            raise _changed_error(definition.path) from exc
        if current_stat != snapshot.source_stat:
            raise _changed_error(definition.path)

        try:
            pyuvdata_version = importlib.metadata.version("pyuvdata")
        except importlib.metadata.PackageNotFoundError as exc:
            raise BeamDependencyError(
                "BeamFITS loading requires the pinned pyuvdata 3.2.1 dependency."
            ) from exc
        except Exception as exc:
            raise BeamDependencyError(
                "BeamFITS loading could not verify the required pyuvdata 3.2.1 "
                "dependency version."
            ) from exc
        if pyuvdata_version != "3.2.1":
            raise BeamDependencyError(
                f"BeamFITS {definition.path}: installed pyuvdata version "
                f"{pyuvdata_version!r} is unsupported; install exactly '3.2.1'."
            )

        observations = _validate_observation_frequencies(
            observation_frequencies_hz,
            native_frequencies_hz=validated.frequencies_hz,
            interpolation_kind=definition.frequency_interpolation,
            identity=f"BeamFITS {definition.path}",
        )
        feature_scales = tuple(
            (frequency, validated.feature_scale_rad) for frequency in observations
        )
        efield_facts = validated.efield
        file_provenance = BeamFileProvenance(
            resolved_path=definition.path,
            size_bytes=snapshot.source_stat[2],
            sha256=snapshot.sha256,
            pyuvdata_version=pyuvdata_version,
            beam_type="efield",
            antenna_type="simple",
            pixel_coordinate_system="az_za",
            mount_type="fixed",
            data_normalization="peak",
            feed_array=(
                ("x", "y") if efield_facts is None else efield_facts.feed_array
            ),
            x_orientation=(
                "east" if efield_facts is None else efield_facts.x_orientation
            ),
            data_shape=validated.data_shape,
            native_dtype=validated.native_dtype,
            frequency_min_hz=float(validated.frequencies_hz[0]),
            frequency_max_hz=float(validated.frequencies_hz[-1]),
            frequency_count=int(validated.frequencies_hz.size),
            azimuth_step_rad=validated.azimuth_step_rad,
            zenith_angle_step_rad=validated.zenith_angle_step_rad,
            zenith_angle_max_rad=float(validated.zenith_angle_rad[-1]),
            basis_tolerance=_BASIS_TOLERANCE,
            scalar_absolute_tolerance=validated.scalar_absolute_tolerance,
            scalar_relative_tolerance=validated.scalar_relative_tolerance,
            normalization_absolute_tolerance=(
                validated.normalization_absolute_tolerance
            ),
            accepted_subset_version=(
                None if efield_facts is None else _FULL_EFIELD_SUBSET_VERSION
            ),
            radiosim_normalization=(
                None if efield_facts is None else _FULL_EFIELD_NORMALIZATION
            ),
            resolved_feed_array=(
                None if efield_facts is None else efield_facts.feed_array
            ),
            derived_x_orientation_verdict=(
                None
                if efield_facts is None
                else efield_facts.derived_x_orientation_verdict
            ),
            basis_vector_convention=(
                None if efield_facts is None else _BASIS_VECTOR_CONVENTION
            ),
            factorization_convention=(
                None if efield_facts is None else _FACTORIZATION_CONVENTION
            ),
            stored_grid_peak_by_frequency=(
                None
                if efield_facts is None
                else efield_facts.stored_grid_peak_by_frequency
            ),
        )
        scientific_fingerprint = _scientific_fingerprint(
            definition=definition,
            file_sha256=snapshot.sha256,
            pyuvdata_version=pyuvdata_version,
            validated=validated,
            observation_frequencies_hz=observations,
            feature_scales=feature_scales,
        )
        state = LoadedBeamHandlerState(
            handler_id=(f"beam-{handler_ordinal:04d}-{scientific_fingerprint[:12]}"),
            kind="fits",
            definition_fingerprint=definition.definition_fingerprint,
            scientific_fingerprint=scientific_fingerprint,
            file=file_provenance,
            voltage_feature_scale_by_frequency=feature_scales,
        )
        evaluator: _UVBeamScalarEvaluator | _UVBeamEfieldEvaluator
        if efield_facts is None:
            evaluator = _UVBeamScalarEvaluator(
                beam=validated.beam,
                identity=state.handler_id,
                frequency_interpolation=definition.frequency_interpolation,
                frequencies_hz=validated.frequencies_hz,
                scalar_absolute_tolerance=validated.scalar_absolute_tolerance,
                scalar_relative_tolerance=validated.scalar_relative_tolerance,
                feature_scale_rad=validated.feature_scale_rad,
                result_dtype=result_dtype,
            )
        else:
            evaluator = _UVBeamEfieldEvaluator(
                beam=validated.beam,
                identity=state.handler_id,
                frequency_interpolation=definition.frequency_interpolation,
                frequencies_hz=validated.frequencies_hz,
                feature_scale_rad=validated.feature_scale_rad,
                result_dtype=result_dtype,
            )
        loaded = _LoadedFITSHandler(state=state, evaluator=evaluator)
    except BaseException as primary:
        try:
            _cleanup_temporary_directory(temporary, definition.path)
        except BeamFileReadError as cleanup_error:
            primary.add_note(str(cleanup_error))
        raise
    _cleanup_temporary_directory(temporary, definition.path)
    return loaded


__all__: list[str] = []
