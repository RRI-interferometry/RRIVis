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
    WRONG_FEED_ORDER = "wrong_feed_order"
    WRONG_X_ORIENTATION = "wrong_x_orientation"
    WRONG_FEED_ANGLES = "wrong_feed_angles"
    NON_FIXED_MOUNT = "non_fixed_mount"
    NONIDENTITY_BASIS = "nonidentity_basis"
    NONFINITE_BASIS = "nonfinite_basis"
    CROSS_POLAR = "cross_polar"
    UNEQUAL_DIAGONALS = "unequal_diagonals"
    NON_PEAK_NORMALIZATION = "non_peak_normalization"
    NON_UNIT_BANDPASS = "non_unit_bandpass"
    PEAK_LABEL_NONPEAK_DATA = "peak_label_nonpeak_data"
    HEALPIX = "healpix"
    WRONG_COORDINATE_METADATA = "wrong_coordinate_metadata"
    IRREGULAR_AZIMUTH = "irregular_azimuth"
    IRREGULAR_ZENITH_ANGLE = "irregular_zenith_angle"
    INCOMPLETE_AZIMUTH_CLOSURE = "incomplete_azimuth_closure"
    SHORT_ZA = "short_za"
    OUT_OF_FREQUENCY_COVERAGE = "out_of_frequency_coverage"
    DUPLICATE_FREQUENCY = "duplicate_frequency"
    DECREASING_FREQUENCY = "decreasing_frequency"
    NONPOSITIVE_FREQUENCY = "nonpositive_frequency"
    NONFINITE_FREQUENCY = "nonfinite_frequency"
    NONFINITE_DATA = "nonfinite_data"
    WRONG_NATIVE_DTYPE = "wrong_native_dtype"
    INVALID_DATA_SHAPE = "invalid_data_shape"


def _require_science_variant(variant: Any) -> BeamScienceVariant:
    """Return an exact science enum member or reject malformed identity."""
    if not isinstance(variant, BeamScienceVariant):
        raise TypeError("variant must be a BeamScienceVariant member")
    return variant


def _require_unsupported_variant(variant: Any) -> UnsupportedBeamVariant:
    """Return an exact unsupported enum member or reject malformed identity."""
    if not isinstance(variant, UnsupportedBeamVariant):
        raise TypeError("variant must be an UnsupportedBeamVariant member")
    return variant


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
    variant = _require_science_variant(variant)
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
    variant = _require_science_variant(variant)
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
    variant = _require_science_variant(variant)
    root = Path(directory)
    # Created rather than required, matching ``write_efield_beamfits``: a
    # caller writing one comparison pair into two sibling subdirectories should
    # not have to remember which of the two writers creates its own.
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    native_dtype = _normalise_complex_dtype(dtype)
    target_name = (
        f"{variant.value}-{native_dtype.name}.beamfits"
        if filename is None
        else filename
    )
    if not target_name or Path(target_name).name != target_name:
        raise ValueError("filename must be one non-empty basename")
    target = root / target_name
    if target.exists() or target.is_symlink():
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
    variant = _require_unsupported_variant(variant)
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
    elif variant is UnsupportedBeamVariant.WRONG_FEED_ORDER:
        beam.feed_array = np.array(["y", "x"])
        beam.feed_angle = np.array([0.0, np.pi / 2.0])
    elif variant is UnsupportedBeamVariant.WRONG_X_ORIENTATION:
        beam.x_orientation = "north"
        beam.feed_angle = np.array([0.0, np.pi / 2.0])
    elif variant is UnsupportedBeamVariant.WRONG_FEED_ANGLES:
        beam.feed_angle = np.array([np.pi / 2.0 + 1e-3, 0.0])
    elif variant is UnsupportedBeamVariant.NON_FIXED_MOUNT:
        beam.mount_type = "alt-az"
    elif variant is UnsupportedBeamVariant.NONIDENTITY_BASIS:
        basis = np.zeros_like(beam.basis_vector_array)
        basis[0, 1] = 1.0
        basis[1, 0] = -1.0
        beam.basis_vector_array = basis
    elif variant is UnsupportedBeamVariant.NONFINITE_BASIS:
        basis = np.array(beam.basis_vector_array, copy=True)
        basis[0, 0, 0, 0] = np.nan
        beam.basis_vector_array = basis
        classification = BeamVariantClassification.DEPENDENCY_INVALID
    elif variant is UnsupportedBeamVariant.CROSS_POLAR:
        beam.data_array[0, 1] = 0.05 * beam.data_array[0, 0]
    elif variant is UnsupportedBeamVariant.UNEQUAL_DIAGONALS:
        beam.data_array[1, 1] *= 0.9
    elif variant is UnsupportedBeamVariant.NON_PEAK_NORMALIZATION:
        beam.data_normalization = "physical"
    elif variant is UnsupportedBeamVariant.NON_UNIT_BANDPASS:
        beam.bandpass_array = np.array([1.0, 1.1, 1.2, 1.3])
    elif variant is UnsupportedBeamVariant.PEAK_LABEL_NONPEAK_DATA:
        beam.data_array *= 0.8
    elif variant is UnsupportedBeamVariant.WRONG_COORDINATE_METADATA:
        beam.pixel_coordinate_system = "orthoslant_zenith"
    elif variant is UnsupportedBeamVariant.IRREGULAR_AZIMUTH:
        axis = np.array(beam.axis1_array, copy=True)
        axis[3] += 0.01
        beam.axis1_array = axis
        classification = BeamVariantClassification.DEPENDENCY_INVALID
    elif variant is UnsupportedBeamVariant.IRREGULAR_ZENITH_ANGLE:
        axis = np.array(beam.axis2_array, copy=True)
        axis[2] += 0.01
        beam.axis2_array = axis
        classification = BeamVariantClassification.DEPENDENCY_INVALID
    elif variant is UnsupportedBeamVariant.INCOMPLETE_AZIMUTH_CLOSURE:
        selected = beam.select(axis1_inds=np.arange(7), inplace=False)
        assert isinstance(selected, UVBeam)
        beam = selected
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
    elif variant is UnsupportedBeamVariant.DUPLICATE_FREQUENCY:
        frequencies = np.array(beam.freq_array, copy=True)
        frequencies[1] = frequencies[0]
        beam.freq_array = frequencies
    elif variant is UnsupportedBeamVariant.DECREASING_FREQUENCY:
        frequencies = np.array(beam.freq_array, copy=True)
        frequencies[1], frequencies[2] = frequencies[2], frequencies[1]
        beam.freq_array = frequencies
    elif variant is UnsupportedBeamVariant.NONPOSITIVE_FREQUENCY:
        frequencies = np.array(beam.freq_array, copy=True)
        frequencies[0] = 0.0
        beam.freq_array = frequencies
    elif variant is UnsupportedBeamVariant.NONFINITE_FREQUENCY:
        frequencies = np.array(beam.freq_array, copy=True)
        frequencies[1] = np.nan
        beam.freq_array = frequencies
    elif variant is UnsupportedBeamVariant.NONFINITE_DATA:
        data = np.array(beam.data_array, copy=True)
        data[0, 0, 0, 0, 0] = complex(np.nan, 0.0)
        beam.data_array = data
    elif variant is UnsupportedBeamVariant.WRONG_NATIVE_DTYPE:
        beam.data_array = np.array(beam.data_array, dtype=object, copy=True)
    elif variant is UnsupportedBeamVariant.INVALID_DATA_SHAPE:
        beam.data_array = np.array(beam.data_array[..., :-1], copy=True)
        classification = BeamVariantClassification.DEPENDENCY_INVALID

    return BeamVariantFixture(
        variant=variant,
        classification=classification,
        beam=beam,
    )


# ==============================================================================
# SCI-005 Stage 3: full efield UVBeam fixtures
# ==============================================================================
#
# ``docs/development/sci005_beam_physics_plan.md`` Sections 5.1.1 and 5.2.1
# freeze a second accepted subset of the same ``beam_type == "efield"`` files:
# ``normalization: uvbeam_peak_common_v1``. Its files carry a *generally full*
# complex ``data_array`` rather than the equal-diagonal cross-hand-free scalar
# science the accepted ``peak`` subset requires, and their unit peak is taken
# over the complete stored grid rather than the visible rows alone.
#
# Two analytic sciences are provided, matching the two oracles Section 5.6
# names.
#
# * :func:`crossed_ideal_dipole_components` is the classical crossed pair of
#   infinitesimal dipoles. Its stored components are **bit-identical** to
#   pyuvdata 3.2.1's own ``ShortDipoleBeam._efield_eval``, which is the
#   independent published implementation of the same model, so the sign and
#   row-order content of the fixture is not RadioSim's own opinion. It is real
#   valued, which is what makes it the sign oracle.
# * :func:`quadrupolar_native_jones` is the phenomenological cross-polar model
#   ``docs/development/beam_physics_scope.md`` states:
#   ``epsilon(theta) = epsilon_0 (theta / theta_ref)**2`` with
#   ``cross = epsilon(theta) sin(2 phi)``, assembled as
#   ``[[co, cross], [-cross, co]]`` so the two feed rows carry opposite parity
#   and both cross-hands vanish on the principal planes. A deterministic
#   zenith-angle-only phase makes it complex, so a conjugation mistake in the
#   conversion is observable, while leaving the zenith row single valued.
#
# Both sciences reach exactly ``1.0`` as their full-stored-grid maximum on the
# canonical grids, so no fixture ever calls ``UVBeam.peak_normalize`` -- Section
# 5.1.1: "a producer may run the official operation before writing, but the
# simulator accepts or rejects the committed bytes as authored".


class EfieldScienceVariant(str, Enum):
    """Analytic full-efield sciences available to Stage-3 fixtures."""

    CROSSED_IDEAL_DIPOLE = "crossed_ideal_dipole"
    QUADRUPOLAR = "quadrupolar"


class UnsupportedEfieldVariant(str, Enum):
    """Rejection probes for the Section 5.1.1 ordered load contract.

    Every member is spelled exactly as the ``probe_kind`` literal Section 8.1's
    ``efield_file_contracts`` array freezes for it, so an evidence row and the
    fixture that produced it carry one name.
    """

    POWER_BEAM = "power_beam"
    PHASED_ARRAY_ANTENNA = "phased_array_antenna"
    HEALPIX_PIXELS = "healpix_pixels"
    VECTOR_DIMENSION = "vector_dimension"
    FEED_PAIR = "feed_pair"
    FEED_ANGLE = "feed_angle"
    DERIVED_ORIENTATION = "derived_orientation"
    MOUNT = "mount"
    GRID_COVERAGE = "grid_coverage"
    WRAP_CONTINUITY = "wrap_continuity"
    BASIS_VECTOR_NOT_IDENTITY = "basis_vector_not_identity"
    BASIS_VECTOR_COMPLEX = "basis_vector_complex"
    BASIS_VECTOR_NON_FINITE = "basis_vector_non_finite"
    DATA_DTYPE = "data_dtype"
    DATA_NON_FINITE = "data_non_finite"
    DATA_NORMALIZATION = "data_normalization"
    BANDPASS = "bandpass"
    VISIBLE_ONLY_PEAK = "visible_only_peak"


#: ``epsilon_0`` and ``theta_ref`` of the scope document's quadrupolar model.
#: ``0.2`` at ``theta_ref = pi/2`` keeps ``sqrt(co**2 + cross**2) < 1`` at every
#: positive zenith angle, so the stored grid maximum is attained at the zenith
#: and is exactly one.
QUADRUPOLAR_EPSILON_0 = 0.2
QUADRUPOLAR_THETA_REF_RAD = np.pi / 2.0

#: Deterministic per-feed-row phase slope in radians per radian of zenith
#: angle. It is a function of zenith angle alone, so the zenith row stays a
#: single physical matrix while every off-zenith sample is genuinely complex.
QUADRUPOLAR_ROW_PHASE_SLOPE: tuple[float, float] = (0.37, 0.48)


#: Stored bases that pyuvdata 3.2.1 itself tolerates and that corrected
#: Section 5.1.1 item 10 nevertheless rejects, because they are not *exactly*
#: the native identity. The memo names the first two by hand -- "including one
#: pyuvdata itself would tolerate, such as ``0.5*I`` or a negative
#: off-diagonal" -- and the third is the fixture the retired
#: ``basis_vector_degenerate`` probe already shipped.
NON_IDENTITY_STORED_BASES: dict[str, tuple[tuple[float, float], ...]] = {
    "half_identity": ((0.5, 0.0), (0.0, 0.5)),
    "negative_off_diagonal": ((0.9553, -0.2955), (-0.2955, 0.9553)),
    "rank_one": ((1.0, 0.0), (1.0, 0.0)),
    "anti_diagonal": ((0.0, -1.0), (-1.0, 0.0)),
}


def canonical_full_sphere_zenith_angle_grid() -> NDArray[np.float64]:
    """Return a nine-sample grid from the zenith through the nadir.

    Section 5.1.1 item 12: "Below-horizon samples, when stored, participate in
    this maximum." A file that stores them is the only way to exhibit the
    difference between the accepted full-stored-grid predicate and the
    ``peak`` subset's visible-row one.
    """
    return np.linspace(0.0, np.pi, 9, dtype=np.float64)


def native_identity_basis_vector_array(
    *,
    dtype: Any = np.float64,
    zenith_angle_rad: NDArray[np.float64] | None = None,
    azimuth_uv_rad: NDArray[np.float64] | None = None,
) -> NDArray[np.floating[Any]]:
    """Return the exact native identity ``basis_vector_array`` of one grid.

    Corrected Section 5.1.1 item 10 requires entries ``[0, 0]`` and ``[1, 1]``
    to be exactly ``1.0`` and ``[0, 1]`` and ``[1, 0]`` exactly ``0.0``, and
    accepts either real floating stored width, "because the identity values
    ``1.0`` and ``0.0`` are exactly representable and round-trip bit-exactly
    in each".
    """
    normalized = np.dtype(dtype)
    if normalized.kind != "f":
        raise ValueError("dtype must be a real floating NumPy dtype")
    azimuth = canonical_azimuth_grid() if azimuth_uv_rad is None else azimuth_uv_rad
    zenith_angle = (
        canonical_zenith_angle_grid() if zenith_angle_rad is None else zenith_angle_rad
    )
    basis = np.zeros((2, 2, zenith_angle.size, azimuth.size), dtype=normalized)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    return basis


def constant_basis_vector_array(
    matrix: Any,
    *,
    dtype: Any = np.float64,
    zenith_angle_rad: NDArray[np.float64] | None = None,
    azimuth_uv_rad: NDArray[np.float64] | None = None,
) -> NDArray[np.floating[Any]]:
    """Return a stored ``basis_vector_array`` holding one constant 2x2 matrix."""
    normalized = np.dtype(dtype)
    azimuth = canonical_azimuth_grid() if azimuth_uv_rad is None else azimuth_uv_rad
    zenith_angle = (
        canonical_zenith_angle_grid() if zenith_angle_rad is None else zenith_angle_rad
    )
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (2, 2):
        raise ValueError("matrix must be exactly 2x2")
    basis = np.zeros((2, 2, zenith_angle.size, azimuth.size), dtype=normalized)
    for row in range(2):
        for column in range(2):
            basis[row, column, :, :] = values[row, column]
    return basis


class ForgedInterpolationBasisUVBeam(UVBeam):
    """A UVBeam whose interpolation returns a non-identity basis.

    Corrected Section 5.2.1 requires the evaluator to *verify* the returned
    basis rather than compose it, and rules that a returned array which is not
    the identity "means the pinned dependency contract has changed beneath
    RadioSim. It is therefore an **internal failure**, not a file rejection".
    No committed file can express that state, because pyuvdata builds the
    returned array from ``numpy.ones`` and ``numpy.zeros`` itself, so the only
    honest probe is a dependency object that violates the pinned return
    contract on purpose.

    The class is applied to an already-built beam by ``__class__``
    reassignment in :func:`forge_interpolation_basis`, so every load-stage
    predicate still sees an ordinary accepted file.
    """

    #: The value written into the returned ``[0, 1]`` basis entry.
    forged_off_diagonal: float = 0.25

    def interp(self, **kwargs: Any) -> Any:
        """Delegate to pyuvdata, then corrupt the returned basis vectors."""
        result = super().interp(**kwargs)
        if isinstance(result, tuple) and len(result) == 2 and result[1] is not None:
            basis = np.array(result[1], copy=True)
            basis[0, 1, :] = type(self).forged_off_diagonal
            return (result[0], basis)
        return result


def forge_interpolation_basis(beam: UVBeam) -> UVBeam:
    """Return ``beam`` re-typed so its interpolation violates the contract."""
    if not isinstance(beam, UVBeam):
        raise TypeError("beam must be a UVBeam")
    beam.__class__ = ForgedInterpolationBasisUVBeam
    return beam


def crossed_ideal_dipole_components(
    *,
    azimuth_uv_rad: ArrayLike,
    zenith_angle_rad: ArrayLike,
) -> NDArray[np.complex128]:
    """Return stored ``data[vector_axis, feed]`` for two crossed ideal dipoles.

    The first axis is ``[azimuth, zenith angle]`` in that order and the second
    is ``[east-aligned feed, north-aligned feed]``, which is pyuvdata 3.2.1's
    own documented ``data_array`` convention for an ``az_za`` efield beam (see
    ``pyuvdata.analytic_beam.ShortDipoleBeam._efield_eval``, whose four lines
    these four reproduce). ``azimuth_uv_rad`` is UVBeam azimuth, zero at East
    and increasing through North.

    Returns
    -------
    numpy.ndarray
        ``complex128`` array of shape ``(2, 2) + broadcast_shape``.
    """
    azimuth = np.asarray(azimuth_uv_rad, dtype=np.float64)
    zenith_angle = np.asarray(zenith_angle_rad, dtype=np.float64)
    azimuth, zenith_angle = np.broadcast_arrays(azimuth, zenith_angle)
    components = np.zeros((2, 2) + azimuth.shape, dtype=np.complex128)
    components[0, 0] = -np.sin(azimuth)
    components[0, 1] = np.cos(azimuth)
    components[1, 0] = np.cos(zenith_angle) * np.cos(azimuth)
    components[1, 1] = np.cos(zenith_angle) * np.sin(azimuth)
    return components


def _radiosim_azimuth_rad(azimuth_uv_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``phi``, the accepted North-through-East RadioSim azimuth."""
    return np.asarray((np.pi / 2.0 - azimuth_uv_rad) % (2.0 * np.pi))


def quadrupolar_native_jones(
    *,
    azimuth_uv_rad: ArrayLike,
    zenith_angle_rad: ArrayLike,
) -> NDArray[np.complex128]:
    """Return the quadrupolar ``J_native[feed, tangent_component]``.

    ``docs/development/beam_physics_scope.md``: "The Y feed's cross-polar
    response has the opposite parity to the X feed's, so the assembled Jones
    matrix is ``[[co, cross], [-cross, co]]``", with
    ``cross_pol = epsilon(theta) sin(2 phi)`` "vanishing on the principal
    planes (``phi = 0, pi/2``) and peaking at ``phi = pi/4``".

    The tangent components are the Ludwig-3 co-polar and cross-polar pair of
    the design's Section 5.2, whose ``theta -> 0`` limit at ``phi = 0`` is
    ``(North, East)``.
    """
    azimuth = np.asarray(azimuth_uv_rad, dtype=np.float64)
    zenith_angle = np.asarray(zenith_angle_rad, dtype=np.float64)
    azimuth, zenith_angle = np.broadcast_arrays(azimuth, zenith_angle)
    phi = _radiosim_azimuth_rad(azimuth)
    co = np.cos(zenith_angle)
    cross = (
        QUADRUPOLAR_EPSILON_0
        * (zenith_angle / QUADRUPOLAR_THETA_REF_RAD) ** 2
        * np.sin(2.0 * phi)
    )
    jones = np.zeros((2, 2) + azimuth.shape, dtype=np.complex128)
    jones[0, 0] = co
    jones[0, 1] = cross
    jones[1, 0] = -cross
    jones[1, 1] = co
    for feed, slope in enumerate(QUADRUPOLAR_ROW_PHASE_SLOPE):
        jones[:, feed] *= np.exp(1j * slope * zenith_angle)
    return jones


def quadrupolar_components(
    *,
    azimuth_uv_rad: ArrayLike,
    zenith_angle_rad: ArrayLike,
) -> NDArray[np.complex128]:
    """Return stored ``data[vector_axis, feed]`` for the quadrupolar science.

    The Ludwig-3 pair is written back into the file's own
    ``[azimuth, zenith angle]`` components through the geometric identities
    ``e_co = theta_hat cos(phi) - phi_hat sin(phi)``,
    ``e_cross = theta_hat sin(phi) + phi_hat cos(phi)`` and
    ``e_az_uv = -phi_hat``. No production symbol takes part.
    """
    azimuth = np.asarray(azimuth_uv_rad, dtype=np.float64)
    zenith_angle = np.asarray(zenith_angle_rad, dtype=np.float64)
    azimuth, zenith_angle = np.broadcast_arrays(azimuth, zenith_angle)
    phi = _radiosim_azimuth_rad(azimuth)
    jones = quadrupolar_native_jones(
        azimuth_uv_rad=azimuth,
        zenith_angle_rad=zenith_angle,
    )
    co = jones[:, 0]
    cross = jones[:, 1]
    components = np.zeros((2, 2) + azimuth.shape, dtype=np.complex128)
    # data[1, f] is the zenith-angle (theta) component and data[0, f] the
    # azimuth one, which is minus the phi component.
    components[1] = co * np.cos(phi) + cross * np.sin(phi)
    components[0] = co * np.sin(phi) - cross * np.cos(phi)
    return components


_EFIELD_SCIENCE = {
    EfieldScienceVariant.CROSSED_IDEAL_DIPOLE: crossed_ideal_dipole_components,
    EfieldScienceVariant.QUADRUPOLAR: quadrupolar_components,
}

_EFIELD_FEED_ANGLES = {
    ("x", "y"): lambda chi: (np.pi / 2.0 + chi, chi),
    ("r", "l"): lambda chi: (chi, chi),
}


def _require_efield_science(variant: Any) -> EfieldScienceVariant:
    if not isinstance(variant, EfieldScienceVariant):
        raise TypeError("variant must be an EfieldScienceVariant member")
    return variant


def _require_unsupported_efield_variant(variant: Any) -> UnsupportedEfieldVariant:
    if not isinstance(variant, UnsupportedEfieldVariant):
        raise TypeError("variant must be an UnsupportedEfieldVariant member")
    return variant


def efield_grid_data(
    *,
    science: EfieldScienceVariant = EfieldScienceVariant.CROSSED_IDEAL_DIPOLE,
    dtype: Any = np.complex128,
    zenith_angle_rad: NDArray[np.float64] | None = None,
    azimuth_uv_rad: NDArray[np.float64] | None = None,
    frequencies_hz: NDArray[np.float64] | None = None,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Evaluate one science over the full ``(2, 2, Nfreq, Nza, Naz)`` grid."""
    science = _require_efield_science(science)
    azimuth = canonical_azimuth_grid() if azimuth_uv_rad is None else azimuth_uv_rad
    zenith_angle = (
        canonical_zenith_angle_grid() if zenith_angle_rad is None else zenith_angle_rad
    )
    frequencies = (
        canonical_frequency_grid() if frequencies_hz is None else frequencies_hz
    )
    plane = _EFIELD_SCIENCE[science](
        azimuth_uv_rad=azimuth[np.newaxis, :],
        zenith_angle_rad=zenith_angle[:, np.newaxis],
    )
    data = np.zeros(
        (2, 2, frequencies.size, zenith_angle.size, azimuth.size),
        dtype=_normalise_complex_dtype(dtype),
    )
    for index in range(frequencies.size):
        data[:, :, index] = plane
    return data


def build_efield_uvbeam(
    *,
    science: EfieldScienceVariant = EfieldScienceVariant.CROSSED_IDEAL_DIPOLE,
    dtype: Any = np.complex128,
    feed_array: tuple[str, str] = ("x", "y"),
    feed_rotation_rad: float = 0.0,
    feed_angle_rad: tuple[float, float] | None = None,
    mount_type: str = "fixed",
    data_normalization: str = "peak",
    zenith_angle_rad: NDArray[np.float64] | None = None,
    basis_vector_array: NDArray[Any] | None = None,
    bandpass_array: NDArray[np.float64] | None = None,
    data_array: NDArray[Any] | None = None,
) -> UVBeam:
    """Build one independently owned full-efield ``az_za`` UVBeam.

    ``feed_angle_rad`` defaults to the exact pair Section 5.1.1 item 6 requires
    of a receptor with static feed rotation ``chi``: ``(pi/2 + chi, chi)`` for a
    linear ``("x", "y")`` pair and ``(chi, chi)`` for a circular ``("r", "l")``
    one.
    """
    science = _require_efield_science(science)
    feeds = tuple(feed_array)
    if feeds not in _EFIELD_FEED_ANGLES:
        raise ValueError("feed_array must be exactly ('x', 'y') or ('r', 'l')")
    frequencies = canonical_frequency_grid()
    azimuth = canonical_azimuth_grid()
    zenith_angle = (
        canonical_zenith_angle_grid() if zenith_angle_rad is None else zenith_angle_rad
    )
    angles = (
        _EFIELD_FEED_ANGLES[feeds](float(feed_rotation_rad))
        if feed_angle_rad is None
        else feed_angle_rad
    )
    if basis_vector_array is None:
        basis_vector_array = native_identity_basis_vector_array(
            zenith_angle_rad=zenith_angle,
            azimuth_uv_rad=azimuth,
        )
    if data_array is None:
        data_array = efield_grid_data(
            science=science,
            dtype=dtype,
            zenith_angle_rad=zenith_angle,
            azimuth_uv_rad=azimuth,
            frequencies_hz=frequencies,
        )
    if bandpass_array is None:
        bandpass_array = np.ones(frequencies.size, dtype=np.float64)
    beam = UVBeam.new(
        telescope_name="RadioSim deterministic efield fixture",
        data_normalization=data_normalization,
        freq_array=frequencies,
        feed_name="RadioSim full efield",
        feed_version="sci005-stage3-full-efield-v1",
        model_name=f"RadioSim {science.value} efield beam",
        model_version="sci005-stage3-full-efield-v1",
        feed_array=np.array(list(feeds)),
        feed_angle=np.array(list(angles), dtype=np.float64),
        mount_type=mount_type,
        axis1_array=azimuth,
        axis2_array=zenith_angle,
        bandpass_array=bandpass_array,
        basis_vector_array=basis_vector_array,
        data_array=data_array,
        history="Deterministic offline RadioSim full-efield BeamFITS fixture. ",
    )
    assert beam.pixel_coordinate_system == "az_za"
    assert beam.beam_type == "efield"
    return beam


@dataclass(frozen=True, slots=True)
class WrittenEfieldBeamFITS:
    """Description of one freshly generated full-efield BeamFITS transport."""

    path: Path
    sha256: str
    native_dtype: str
    science: EfieldScienceVariant
    feed_array: tuple[str, str]
    feed_angle_rad: tuple[float, float]


def write_efield_beamfits(
    directory: Path,
    *,
    science: EfieldScienceVariant = EfieldScienceVariant.CROSSED_IDEAL_DIPOLE,
    dtype: Any = np.complex128,
    feed_array: tuple[str, str] = ("x", "y"),
    feed_rotation_rad: float = 0.0,
    filename: str | None = None,
    **build_kwargs: Any,
) -> WrittenEfieldBeamFITS:
    """Write one fresh full-efield BeamFITS file below ``directory``."""
    science = _require_efield_science(science)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    native_dtype = _normalise_complex_dtype(dtype)
    feeds = tuple(feed_array)
    target_name = (
        f"efield-{science.value}-{''.join(feeds)}-{native_dtype.name}.beamfits"
        if filename is None
        else filename
    )
    if not target_name or Path(target_name).name != target_name:
        raise ValueError("filename must be one non-empty basename")
    target = root / target_name
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)

    beam = build_efield_uvbeam(
        science=science,
        dtype=native_dtype,
        feed_array=feeds,
        feed_rotation_rad=feed_rotation_rad,
        **build_kwargs,
    )
    result = beam.write_beamfits(target, clobber=True)
    assert result is None
    return WrittenEfieldBeamFITS(
        path=target,
        sha256=sha256_from_file(target),
        native_dtype=native_dtype.name,
        science=science,
        feed_array=(str(feeds[0]), str(feeds[1])),
        feed_angle_rad=tuple(float(value) for value in beam.feed_angle),
    )


@dataclass(frozen=True, slots=True)
class EfieldVariantFixture:
    """One independently owned non-accepted full-efield UVBeam and its role."""

    variant: UnsupportedEfieldVariant
    classification: BeamVariantClassification
    beam: UVBeam


def _phased_array_efield_uvbeam(*, dtype: Any) -> UVBeam:
    """Build a dependency-valid phased-array antenna UVBeam."""
    native_dtype = _normalise_complex_dtype(dtype)
    frequencies = canonical_frequency_grid()
    elements = 4
    return UVBeam.new(
        telescope_name="RadioSim deterministic phased fixture",
        data_normalization="peak",
        freq_array=frequencies,
        feed_array=np.array(["x", "y"]),
        feed_angle=np.array([np.pi / 2.0, 0.0], dtype=np.float64),
        mount_type="phased",
        axis1_array=canonical_azimuth_grid(),
        axis2_array=canonical_zenith_angle_grid(),
        bandpass_array=np.ones(frequencies.size, dtype=np.float64),
        element_location_array=np.zeros((2, elements)) + np.arange(elements),
        element_coordinate_system="x-y",
        delay_array=np.zeros(elements, dtype=np.float64),
        gain_array=np.ones(elements, dtype=np.float64),
        coupling_matrix=np.zeros(
            (elements, elements, 2, 2, frequencies.size),
            dtype=np.complex128,
        ),
        data_array=efield_grid_data(dtype=native_dtype),
        history="Deterministic offline RadioSim phased-array fixture. ",
    )


def _healpix_efield_uvbeam(*, dtype: Any) -> UVBeam:
    """Build a dependency-valid HEALPix full-efield UVBeam."""
    native_dtype = _normalise_complex_dtype(dtype)
    frequencies = canonical_frequency_grid()
    nside = 4
    pixels = 12 * nside**2
    data = np.zeros((2, 2, frequencies.size, pixels), dtype=native_dtype)
    data[0, 0] = 1.0
    data[1, 1] = 1.0
    return UVBeam.new(
        telescope_name="RadioSim deterministic HEALPix efield fixture",
        data_normalization="peak",
        freq_array=frequencies,
        feed_array=np.array(["x", "y"]),
        feed_angle=np.array([np.pi / 2.0, 0.0], dtype=np.float64),
        mount_type="fixed",
        nside=nside,
        ordering="ring",
        bandpass_array=np.ones(frequencies.size, dtype=np.float64),
        data_array=data,
        history="Deterministic offline RadioSim HEALPix efield fixture. ",
    )


def build_efield_variant(
    variant: UnsupportedEfieldVariant,
    *,
    dtype: Any = np.complex128,
    science: EfieldScienceVariant = EfieldScienceVariant.CROSSED_IDEAL_DIPOLE,
) -> EfieldVariantFixture:
    """Build one deliberately non-accepted full-efield UVBeam state.

    ``DEPENDENCY_VALID_UNSUPPORTED`` states pass
    ``UVBeam.check(check_extra=True, run_check_acceptability=True)`` -- Section
    5.1.1 item 1 -- and are rejected by RadioSim's own ordered contract.
    ``DEPENDENCY_INVALID`` states are refused by pyuvdata itself, so RadioSim
    reaches them only through its accepted dependency-failure classification.
    """
    variant = _require_unsupported_efield_variant(variant)
    classification = BeamVariantClassification.DEPENDENCY_VALID_UNSUPPORTED
    frequencies = canonical_frequency_grid()

    if variant is UnsupportedEfieldVariant.PHASED_ARRAY_ANTENNA:
        return EfieldVariantFixture(
            variant=variant,
            classification=classification,
            beam=_phased_array_efield_uvbeam(dtype=dtype),
        )
    if variant is UnsupportedEfieldVariant.HEALPIX_PIXELS:
        return EfieldVariantFixture(
            variant=variant,
            classification=classification,
            beam=_healpix_efield_uvbeam(dtype=dtype),
        )
    if variant is UnsupportedEfieldVariant.FEED_PAIR:
        beam = build_efield_uvbeam(
            science=science,
            dtype=dtype,
            feed_array=("x", "y"),
            feed_angle_rad=(0.0, np.pi / 2.0),
        )
        beam.feed_array = np.array(["y", "x"])
        return EfieldVariantFixture(variant, classification, beam)
    if variant is UnsupportedEfieldVariant.FEED_ANGLE:
        return EfieldVariantFixture(
            variant,
            classification,
            build_efield_uvbeam(
                science=science,
                dtype=dtype,
                feed_angle_rad=(np.pi / 2.0 + 1e-3, 0.0),
            ),
        )
    if variant is UnsupportedEfieldVariant.DERIVED_ORIENTATION:
        # Exactly one full turn on each feed: Section 5.1.1 item 6 compares
        # modulo 2*pi and passes, while pyuvdata 3.2.1's
        # ``get_x_orientation_from_feeds`` is not modular and returns ``None``
        # here against the resolved receptor's ``"east"``.
        return EfieldVariantFixture(
            variant,
            classification,
            build_efield_uvbeam(
                science=science,
                dtype=dtype,
                feed_angle_rad=(np.pi / 2.0 + 2.0 * np.pi, 2.0 * np.pi),
            ),
        )
    if variant is UnsupportedEfieldVariant.MOUNT:
        return EfieldVariantFixture(
            variant,
            classification,
            build_efield_uvbeam(science=science, dtype=dtype, mount_type="alt-az"),
        )
    if variant is UnsupportedEfieldVariant.GRID_COVERAGE:
        return EfieldVariantFixture(
            variant,
            BeamVariantClassification.VALID_INSUFFICIENT_COVERAGE,
            build_efield_uvbeam(
                science=science,
                dtype=dtype,
                zenith_angle_rad=np.linspace(0.0, 1.0, 5, dtype=np.float64),
            ),
        )
    if variant is UnsupportedEfieldVariant.WRAP_CONTINUITY:
        beam = build_efield_uvbeam(science=science, dtype=dtype)
        data = np.array(beam.data_array, copy=True)
        data[..., -1] = data[..., -1] + 0.5
        beam.data_array = data
        return EfieldVariantFixture(variant, classification, beam)
    if variant is UnsupportedEfieldVariant.BASIS_VECTOR_NOT_IDENTITY:
        return EfieldVariantFixture(
            variant,
            classification,
            build_efield_uvbeam(
                science=science,
                dtype=dtype,
                basis_vector_array=constant_basis_vector_array(
                    NON_IDENTITY_STORED_BASES["rank_one"]
                ),
            ),
        )
    if variant is UnsupportedEfieldVariant.BASIS_VECTOR_COMPLEX:
        beam = build_efield_uvbeam(science=science, dtype=dtype)
        beam.basis_vector_array = np.array(
            beam.basis_vector_array,
            dtype=np.complex128,
            copy=True,
        )
        return EfieldVariantFixture(
            variant,
            BeamVariantClassification.DEPENDENCY_INVALID,
            beam,
        )
    if variant is UnsupportedEfieldVariant.BASIS_VECTOR_NON_FINITE:
        beam = build_efield_uvbeam(science=science, dtype=dtype)
        basis = np.array(beam.basis_vector_array, copy=True)
        basis[0, 0, 0, 0] = np.nan
        beam.basis_vector_array = basis
        return EfieldVariantFixture(
            variant,
            BeamVariantClassification.DEPENDENCY_INVALID,
            beam,
        )
    if variant is UnsupportedEfieldVariant.VECTOR_DIMENSION:
        beam = build_efield_uvbeam(science=science, dtype=dtype)
        azimuth = canonical_azimuth_grid()
        zenith_angle = canonical_zenith_angle_grid()
        basis = np.zeros((3, 2, zenith_angle.size, azimuth.size), dtype=np.float64)
        basis[0, 0] = 1.0
        basis[1, 1] = 1.0
        data = np.zeros(
            (3, 2, frequencies.size, zenith_angle.size, azimuth.size),
            dtype=beam.data_array.dtype,
        )
        data[:2] = beam.data_array
        beam.Naxes_vec = 3
        beam.basis_vector_array = basis
        beam.data_array = data
        return EfieldVariantFixture(variant, classification, beam)
    if variant is UnsupportedEfieldVariant.DATA_DTYPE:
        beam = build_efield_uvbeam(science=science, dtype=dtype)
        beam.data_array = np.array(beam.data_array, dtype=object, copy=True)
        return EfieldVariantFixture(variant, classification, beam)
    if variant is UnsupportedEfieldVariant.DATA_NON_FINITE:
        beam = build_efield_uvbeam(science=science, dtype=dtype)
        data = np.array(beam.data_array, copy=True)
        data[0, 0, 0, 1, 1] = complex(np.nan, 0.0)
        beam.data_array = data
        return EfieldVariantFixture(variant, classification, beam)
    if variant is UnsupportedEfieldVariant.DATA_NORMALIZATION:
        return EfieldVariantFixture(
            variant,
            classification,
            build_efield_uvbeam(
                science=science, dtype=dtype, data_normalization="physical"
            ),
        )
    if variant is UnsupportedEfieldVariant.BANDPASS:
        return EfieldVariantFixture(
            variant,
            classification,
            build_efield_uvbeam(
                science=science,
                dtype=dtype,
                bandpass_array=np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float64),
            ),
        )
    if variant is UnsupportedEfieldVariant.VISIBLE_ONLY_PEAK:
        zenith_angle = canonical_full_sphere_zenith_angle_grid()
        data = efield_grid_data(
            science=science,
            dtype=dtype,
            zenith_angle_rad=zenith_angle,
        )
        # Unit peak over the visible rows; a stored below-horizon row exceeds
        # one, so only the full-stored-grid predicate rejects the file.
        data[:, :, :, -1, :] = data[:, :, :, -1, :] + 3.0
        return EfieldVariantFixture(
            variant,
            classification,
            build_efield_uvbeam(
                science=science,
                dtype=dtype,
                zenith_angle_rad=zenith_angle,
                data_array=data,
            ),
        )
    if variant is UnsupportedEfieldVariant.POWER_BEAM:
        converted = build_efield_uvbeam(science=science, dtype=dtype).efield_to_power(
            inplace=False
        )
        assert isinstance(converted, UVBeam)
        return EfieldVariantFixture(variant, classification, converted)
    raise AssertionError(f"unhandled variant {variant!r}")  # pragma: no cover


@dataclass(slots=True)
class CountingBeamFITSLoader:
    """Test-only UVBeam reader with per-instance attempt and failure tracking."""

    fail_on_attempts: frozenset[int] = frozenset()
    attempts: int = field(init=False, default=0)
    _requested_paths: list[Path] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Validate an immutable schedule of positive one-based attempts."""
        message = "fail_on_attempts must be a frozenset of positive one-based integers"
        if type(self.fail_on_attempts) is not frozenset:
            raise TypeError(message)
        if any(
            type(attempt) is not int or attempt < 1 for attempt in self.fail_on_attempts
        ):
            raise ValueError(message)

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
