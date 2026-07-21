"""Deterministic, offline BeamFITS fixtures for pyuvdata contract tests.

The helpers in this module deliberately expose only test infrastructure. Every
builder returns independently owned mutable dependency state, and every file writer
requires a caller-owned temporary directory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyuvdata import UVBeam


class BeamScienceVariant(str, Enum):
    """Analytical scalar-voltage models available to test fixtures."""

    CANONICAL = "canonical"
    DISTINCT = "distinct"


class BeamVariantClassification(str, Enum):
    """Scientific role of a deliberately non-canonical UVBeam fixture."""

    DEPENDENCY_VALID_UNSUPPORTED = "dependency_valid_unsupported"
    DEPENDENCY_INVALID = "dependency_invalid"
    VALID_INSUFFICIENT_COVERAGE = "valid_insufficient_coverage"


class UnsupportedBeamVariant(str, Enum):
    """Metadata and value variants needed by later BeamFITS validation tests."""

    POWER = "power"
    CIRCULAR_FEEDS = "circular_feeds"
    NONIDENTITY_BASIS = "nonidentity_basis"
    CROSS_POLAR = "cross_polar"
    UNEQUAL_DIAGONALS = "unequal_diagonals"
    NON_PEAK_NORMALIZATION = "non_peak_normalization"
    NON_UNIT_BANDPASS = "non_unit_bandpass"
    HEALPIX = "healpix"
    SHORT_ZA = "short_za"
    OUT_OF_FREQUENCY_COVERAGE = "out_of_frequency_coverage"
    NONFINITE_FREQUENCY = "nonfinite_frequency"
    NONFINITE_DATA = "nonfinite_data"
    INVALID_DATA_SHAPE = "invalid_data_shape"


@dataclass(frozen=True, slots=True)
class WrittenBeamFITS:
    """Description of one freshly generated BeamFITS transport."""

    path: Path
    sha256: str
    native_dtype: str
    science_variant: BeamScienceVariant


@dataclass(frozen=True, slots=True)
class BeamVariantFixture:
    """One independently owned non-canonical UVBeam and its intended role."""

    variant: UnsupportedBeamVariant
    classification: BeamVariantClassification
    beam: UVBeam


def canonical_azimuth_grid() -> NDArray[np.float64]:
    """Return the eight-sample endpoint-excluded UVBeam azimuth grid."""
    return np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float64)


def canonical_zenith_angle_grid() -> NDArray[np.float64]:
    """Return the five-sample grid from zenith through the horizon."""
    return np.linspace(0.0, np.pi / 2.0, 5, dtype=np.float64)


def canonical_frequency_grid() -> NDArray[np.float64]:
    """Return the four exact intrinsic fixture frequencies in Hz."""
    return np.array(
        [100_000_000.0, 110_000_000.0, 120_000_000.0, 130_000_000.0],
        dtype=np.float64,
    )


def scalar_voltage_reference(
    *,
    azimuth_uv_rad: ArrayLike,
    zenith_angle_rad: ArrayLike,
    frequency_index: ArrayLike,
    variant: BeamScienceVariant = BeamScienceVariant.CANONICAL,
) -> NDArray[np.complex128]:
    """Evaluate an analytical scalar voltage without consulting pyuvdata.

    Parameters
    ----------
    azimuth_uv_rad
        UVBeam azimuth in radians, zero at East and increasing through North.
    zenith_angle_rad
        Zenith angle in radians.
    frequency_index
        Zero-based intrinsic-frequency index. Inputs broadcast by NumPy rules.
    variant
        Canonical ``cos(ZA)^2`` science or the distinct ``cos(ZA)^3`` science.

    Returns
    -------
    numpy.ndarray
        Broadcast complex128 scalar voltage values. Scalar inputs produce a
        zero-dimensional array.
    """
    azimuth = np.asarray(azimuth_uv_rad, dtype=np.float64)
    zenith_angle = np.asarray(zenith_angle_rad, dtype=np.float64)
    index = np.asarray(frequency_index, dtype=np.float64)
    exponent = 2 if variant is BeamScienceVariant.CANONICAL else 3
    phase_sign = 1.0 if variant is BeamScienceVariant.CANONICAL else -1.0
    phase = phase_sign * (0.03 * np.sin(azimuth) + 0.01 * index)
    return np.asarray(np.cos(zenith_angle) ** exponent * np.exp(1j * phase))


def _normalise_complex_dtype(dtype: Any) -> np.dtype[Any]:
    """Return one of the two native complex dtypes accepted by the fixture."""
    normalized = np.dtype(dtype)
    allowed = (np.dtype(np.complex64), np.dtype(np.complex128))
    if normalized not in allowed:
        raise ValueError("dtype must be numpy complex64 or complex128")
    return normalized


def _regular_scalar_data(
    *,
    dtype: Any,
    variant: BeamScienceVariant,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Create the canonical array in pyuvdata's vector/feed/frequency order."""
    azimuth = canonical_azimuth_grid()
    zenith_angle = canonical_zenith_angle_grid()
    frequencies = canonical_frequency_grid()
    scalar = scalar_voltage_reference(
        azimuth_uv_rad=azimuth[np.newaxis, np.newaxis, :],
        zenith_angle_rad=zenith_angle[np.newaxis, :, np.newaxis],
        frequency_index=np.arange(frequencies.size)[:, np.newaxis, np.newaxis],
        variant=variant,
    )
    data = np.zeros(
        (2, 2, frequencies.size, zenith_angle.size, azimuth.size),
        dtype=_normalise_complex_dtype(dtype),
    )
    data[0, 0] = scalar
    data[1, 1] = scalar
    return data


def build_scalar_efield_uvbeam(
    *,
    dtype: Any = np.complex128,
    variant: BeamScienceVariant = BeamScienceVariant.CANONICAL,
) -> UVBeam:
    """Build one independently owned canonical scalar E-field UVBeam.

    The explicit ``pixel_coordinate_system="az_za"`` argument is intentionally
    omitted. pyuvdata 3.2.1 checks uninitialized coordinate state when that
    redundant literal is supplied. The working initializer path derives ``az_za``
    from the two regular axes, and the postcondition below pins that result.
    """
    frequencies = canonical_frequency_grid()
    beam = UVBeam.new(
        telescope_name="RadioSim deterministic fixture",
        data_normalization="peak",
        freq_array=frequencies,
        feed_name="RadioSim X/Y",
        feed_version="tier3-scalar-v1",
        model_name=f"RadioSim {variant.value} scalar beam",
        model_version="tier3-scalar-v1",
        feed_array=np.array(["x", "y"]),
        x_orientation="east",
        mount_type="fixed",
        axis1_array=canonical_azimuth_grid(),
        axis2_array=canonical_zenith_angle_grid(),
        bandpass_array=np.ones(frequencies.size, dtype=np.float64),
        data_array=_regular_scalar_data(dtype=dtype, variant=variant),
        history="Deterministic offline RadioSim BeamFITS fixture. ",
    )
    assert beam.pixel_coordinate_system == "az_za"
    return beam


def sha256_from_file(path: Path) -> str:
    """Calculate SHA-256 from the bytes currently stored at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_scalar_efield_beamfits(
    directory: Path,
    *,
    dtype: Any = np.complex128,
    variant: BeamScienceVariant = BeamScienceVariant.CANONICAL,
    filename: str | None = None,
) -> WrittenBeamFITS:
    """Write a fresh BeamFITS file below a caller-provided temporary directory.

    Parameters
    ----------
    directory
        Existing temporary directory that will own the generated file.
    dtype
        Native complex64 or complex128 source dtype.
    variant
        Canonical or scientifically distinct scalar-voltage model.
    filename
        Optional single basename. Directory traversal and overwriting are rejected.

    Returns
    -------
    WrittenBeamFITS
        Path, runtime digest from actual bytes, native dtype, and science identity.
    """
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    native_dtype = _normalise_complex_dtype(dtype)
    target_name = filename or f"{variant.value}-{native_dtype.name}.beamfits"
    if not target_name or Path(target_name).name != target_name:
        raise ValueError("filename must be one non-empty basename")
    target = root / target_name
    if target.exists():
        raise FileExistsError(target)

    beam = build_scalar_efield_uvbeam(dtype=native_dtype, variant=variant)
    result = beam.write_beamfits(target, clobber=True)
    assert result is None
    return WrittenBeamFITS(
        path=target,
        sha256=sha256_from_file(target),
        native_dtype=native_dtype.name,
        science_variant=variant,
    )


def _build_healpix_scalar_uvbeam(*, dtype: Any) -> UVBeam:
    """Build a dependency-valid HEALPix object without angular interpolation."""
    native_dtype = _normalise_complex_dtype(dtype)
    frequencies = canonical_frequency_grid()
    nside = 1
    pixel_count = 12 * nside**2
    data = np.zeros((2, 2, frequencies.size, pixel_count), dtype=native_dtype)
    data[0, 0] = 1.0
    data[1, 1] = 1.0
    return UVBeam.new(
        telescope_name="RadioSim deterministic HEALPix fixture",
        data_normalization="peak",
        freq_array=frequencies,
        feed_array=np.array(["x", "y"]),
        x_orientation="east",
        mount_type="fixed",
        nside=nside,
        ordering="ring",
        bandpass_array=np.ones(frequencies.size, dtype=np.float64),
        data_array=data,
        history="Deterministic offline RadioSim HEALPix fixture. ",
    )


def build_beam_variant(
    variant: UnsupportedBeamVariant,
    *,
    dtype: Any = np.complex128,
) -> BeamVariantFixture:
    """Build a deterministic invalid, unsupported, or insufficient UVBeam state.

    Dependency-valid unsupported states are valid pyuvdata objects that the accepted
    RadioSim scalar subset will later reject. Dependency-invalid states are retained
    in memory because pyuvdata correctly refuses them. Coverage fixtures remain valid
    files but do not cover all planned RadioSim observation coordinates.
    """
    if variant is UnsupportedBeamVariant.HEALPIX:
        beam = _build_healpix_scalar_uvbeam(dtype=dtype)
    else:
        beam = build_scalar_efield_uvbeam(dtype=dtype)

    classification = BeamVariantClassification.DEPENDENCY_VALID_UNSUPPORTED
    if variant is UnsupportedBeamVariant.POWER:
        converted = beam.efield_to_power(inplace=False)
        assert isinstance(converted, UVBeam)
        beam = converted
        beam.feed_array = None
        beam.feed_angle = None
        beam.Nfeeds = None
    elif variant is UnsupportedBeamVariant.CIRCULAR_FEEDS:
        beam.feed_array = np.array(["r", "l"])
        beam.feed_angle = np.array([0.0, 0.0])
        beam.x_orientation = None
    elif variant is UnsupportedBeamVariant.NONIDENTITY_BASIS:
        basis = np.zeros_like(beam.basis_vector_array)
        basis[0, 1] = 1.0
        basis[1, 0] = -1.0
        beam.basis_vector_array = basis
    elif variant is UnsupportedBeamVariant.CROSS_POLAR:
        beam.data_array[0, 1] = 0.05 * beam.data_array[0, 0]
    elif variant is UnsupportedBeamVariant.UNEQUAL_DIAGONALS:
        beam.data_array[1, 1] *= 0.9
    elif variant is UnsupportedBeamVariant.NON_PEAK_NORMALIZATION:
        beam.data_normalization = "physical"
    elif variant is UnsupportedBeamVariant.NON_UNIT_BANDPASS:
        beam.bandpass_array = np.array([1.0, 1.1, 1.2, 1.3])
    elif variant is UnsupportedBeamVariant.SHORT_ZA:
        selected = beam.select(axis2_inds=np.arange(4), inplace=False)
        assert isinstance(selected, UVBeam)
        beam = selected
        classification = BeamVariantClassification.VALID_INSUFFICIENT_COVERAGE
    elif variant is UnsupportedBeamVariant.OUT_OF_FREQUENCY_COVERAGE:
        selected = beam.select(freq_chans=np.arange(3), inplace=False)
        assert isinstance(selected, UVBeam)
        beam = selected
        classification = BeamVariantClassification.VALID_INSUFFICIENT_COVERAGE
    elif variant is UnsupportedBeamVariant.NONFINITE_FREQUENCY:
        frequencies = np.array(beam.freq_array, copy=True)
        frequencies[1] = np.nan
        beam.freq_array = frequencies
    elif variant is UnsupportedBeamVariant.NONFINITE_DATA:
        data = np.array(beam.data_array, copy=True)
        data[0, 0, 0, 0, 0] = complex(np.nan, 0.0)
        beam.data_array = data
    elif variant is UnsupportedBeamVariant.INVALID_DATA_SHAPE:
        beam.data_array = np.array(beam.data_array[..., :-1], copy=True)
        classification = BeamVariantClassification.DEPENDENCY_INVALID

    return BeamVariantFixture(
        variant=variant,
        classification=classification,
        beam=beam,
    )


@dataclass(slots=True)
class CountingBeamFITSLoader:
    """Test-only UVBeam reader with per-instance attempt and failure tracking."""

    fail_on_attempts: frozenset[int] = frozenset()
    attempts: int = field(init=False, default=0)
    _requested_paths: list[Path] = field(init=False, default_factory=list)

    @property
    def requested_paths(self) -> tuple[Path, ...]:
        """Return an immutable snapshot of paths requested so far."""
        return tuple(self._requested_paths)

    def read(self, path: Path) -> UVBeam:
        """Record and read ``path``, failing on configured one-based attempts."""
        requested = Path(path)
        self.attempts += 1
        self._requested_paths.append(requested)
        if self.attempts in self.fail_on_attempts:
            raise RuntimeError(
                f"injected BeamFITS read failure on attempt {self.attempts}"
            )
        beam = UVBeam()
        result = beam.read_beamfits(requested)
        assert result is None
        return beam
