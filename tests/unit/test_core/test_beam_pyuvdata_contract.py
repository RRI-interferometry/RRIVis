"""Dependency contract tests for deterministic pyuvdata BeamFITS fixtures."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pyuvdata import UVBeam

from tests.fixtures import beamfits as beamfits_helpers
from tests.fixtures.beamfits import (
    BeamScienceVariant,
    BeamVariantClassification,
    CountingBeamFITSLoader,
    UnsupportedBeamVariant,
    build_beam_variant,
    build_scalar_efield_uvbeam,
    canonical_azimuth_grid,
    canonical_frequency_grid,
    canonical_zenith_angle_grid,
    scalar_voltage_reference,
    sha256_from_file,
    write_scalar_efield_beamfits,
)

NATIVE_DTYPES = (np.complex64, np.complex128)


def _read_beamfits(path: Path, **kwargs: Any) -> UVBeam:
    """Read a BeamFITS file while pinning the mutating public call contract."""
    beam = UVBeam()
    result = beam.read_beamfits(path, **kwargs)
    assert result is None
    return beam


def _interp(
    beam: UVBeam,
    *,
    azimuth: np.ndarray,
    zenith_angle: np.ndarray,
    frequencies: np.ndarray,
    frequency_kind: str = "linear",
    return_basis: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Call the exact pyuvdata interpolation surface selected for Tier 3."""
    result = beam.interp(
        az_array=azimuth,
        za_array=zenith_angle,
        interpolation_function="az_za_simple",
        freq_array=frequencies,
        freq_interp_kind=frequency_kind,
        freq_interp_tol=1e-6,
        return_basis_vector=return_basis,
        spline_opts={"kx": 1, "ky": 1, "s": 0},
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    return result


def test_pyuvdata_version_is_the_pinned_contract() -> None:
    """Prevent silent dependency drift away from the characterized release."""
    assert importlib.metadata.version("pyuvdata") == "3.2.1"


def test_scalar_voltage_reference_matches_the_accepted_formula() -> None:
    """Pin the independently evaluable Tier 3 scalar-voltage equation."""
    azimuth = canonical_azimuth_grid()
    zenith_angle = canonical_zenith_angle_grid()
    frequency = canonical_frequency_grid()

    value = scalar_voltage_reference(
        azimuth_uv_rad=azimuth[1],
        zenith_angle_rad=zenith_angle[2],
        frequency_index=1,
        variant=BeamScienceVariant.CANONICAL,
    )
    expected = np.cos(zenith_angle[2]) ** 2 * np.exp(
        1j * (0.03 * np.sin(azimuth[1]) + 0.01)
    )

    assert frequency[1] == 110_000_000.0
    assert value == expected


def test_distinct_scalar_voltage_reference_has_cubed_opposite_phase() -> None:
    """Keep the second science fixture numerically and analytically distinct."""
    azimuth = 3.0 * np.pi / 4.0
    zenith_angle = np.pi / 4.0
    index = 2
    value = scalar_voltage_reference(
        azimuth_uv_rad=azimuth,
        zenith_angle_rad=zenith_angle,
        frequency_index=index,
        variant=BeamScienceVariant.DISTINCT,
    )
    expected = np.cos(zenith_angle) ** 3 * np.exp(
        -1j * (0.03 * np.sin(azimuth) + 0.01 * index)
    )

    assert value == expected


@pytest.mark.parametrize(
    "variant",
    ("canonical", "unexpected", UnsupportedBeamVariant.POWER, None),
    ids=("raw-valid", "raw-unexpected", "other-enum", "none"),
)
def test_science_helpers_reject_non_enum_variants(
    tmp_path: Path,
    variant: Any,
) -> None:
    """Keep malformed science identity from selecting either analytical model."""
    with pytest.raises(
        TypeError,
        match="^variant must be a BeamScienceVariant member$",
    ):
        scalar_voltage_reference(
            azimuth_uv_rad=0.2,
            zenith_angle_rad=0.3,
            frequency_index=1,
            variant=variant,
        )
    with pytest.raises(
        TypeError,
        match="^variant must be a BeamScienceVariant member$",
    ):
        build_scalar_efield_uvbeam(variant=variant)
    with pytest.raises(
        TypeError,
        match="^variant must be a BeamScienceVariant member$",
    ):
        write_scalar_efield_beamfits(
            tmp_path,
            variant=variant,
            filename="invalid-science.beamfits",
        )


@pytest.mark.parametrize(
    "variant",
    ("power", "unexpected", BeamScienceVariant.CANONICAL, None),
    ids=("raw-valid", "raw-unexpected", "other-enum", "none"),
)
def test_unsupported_fixture_builder_rejects_non_enum_variants(
    variant: Any,
) -> None:
    """Prevent malformed negative-fixture identity from returning canonical science."""
    with pytest.raises(
        TypeError,
        match="^variant must be an UnsupportedBeamVariant member$",
    ):
        build_beam_variant(variant)


@pytest.mark.parametrize("dtype", (np.float64, object(), "complex256"))
def test_fixture_builders_reject_unsupported_dtypes(dtype: Any) -> None:
    """Accept only native complex64 or complex128 fixture storage."""
    with pytest.raises((TypeError, ValueError)):
        build_scalar_efield_uvbeam(dtype=dtype)


def test_canonical_grids_have_exact_values_and_fresh_ownership() -> None:
    """Pin dimensions, endpoints, exact Hz values, and caller ownership."""
    azimuth = canonical_azimuth_grid()
    zenith_angle = canonical_zenith_angle_grid()
    frequencies = canonical_frequency_grid()

    np.testing.assert_allclose(
        azimuth,
        np.arange(8, dtype=np.float64) * (np.pi / 4.0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        zenith_angle,
        np.arange(5, dtype=np.float64) * (np.pi / 8.0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        frequencies,
        np.array([100e6, 110e6, 120e6, 130e6], dtype=np.float64),
    )
    assert azimuth[-1] == 2.0 * np.pi - np.pi / 4.0
    assert zenith_angle[0] == 0.0
    assert zenith_angle[-1] == np.pi / 2.0

    azimuth[0] = -1.0
    zenith_angle[-1] = -1.0
    frequencies[0] = -1.0
    assert canonical_azimuth_grid()[0] == 0.0
    assert canonical_zenith_angle_grid()[-1] == np.pi / 2.0
    assert canonical_frequency_grid()[0] == 100e6


def test_uvbeam_new_explicit_az_za_literal_reproduces_local_defect() -> None:
    """Characterize why the production fixture helper omits a redundant literal."""
    message = (
        "pixel_coordinate_system must be one of "
        "['az_za', 'orthoslant_zenith', 'healpix']"
    )
    # pyuvdata 3.2.1 validates still-unset coordinate state before assigning the
    # supplied literal. The fixture uses the supported axis-derived default instead.
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        UVBeam.new(
            telescope_name="explicit coordinate defect",
            data_normalization="peak",
            freq_array=canonical_frequency_grid(),
            feed_array=np.array(["x", "y"]),
            x_orientation="east",
            mount_type="fixed",
            pixel_coordinate_system="az_za",
            axis1_array=canonical_azimuth_grid(),
            axis2_array=canonical_zenith_angle_grid(),
        )


@pytest.mark.parametrize("dtype", NATIVE_DTYPES, ids=("complex64", "complex128"))
def test_beamfits_public_write_and_read_mutate_as_characterized(
    tmp_path: Path,
    dtype: Any,
) -> None:
    """Pin the public write/read return values and receiver mutation."""
    source = build_scalar_efield_uvbeam(dtype=dtype)
    target = tmp_path / f"public-{np.dtype(dtype).name}.beamfits"

    assert source.write_beamfits(target, clobber=True) is None
    receiver = UVBeam()
    assert receiver.data_array is None
    assert receiver.read_beamfits(target) is None
    assert receiver.data_array is not None
    assert receiver.data_array.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", NATIVE_DTYPES, ids=("complex64", "complex128"))
def test_beamfits_round_trip_preserves_canonical_structure(
    tmp_path: Path,
    dtype: Any,
) -> None:
    """Pin every structural field later scalar BeamFITS validation will consume."""
    written = write_scalar_efield_beamfits(tmp_path, dtype=dtype)
    beam = _read_beamfits(written.path)

    assert beam.check(check_extra=True, run_check_acceptability=True)
    assert beam.data_array.shape == (2, 2, 4, 5, 8)
    assert beam.data_array.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(beam.freq_array, canonical_frequency_grid())
    np.testing.assert_allclose(beam.axis1_array, canonical_azimuth_grid())
    np.testing.assert_allclose(beam.axis2_array, canonical_zenith_angle_grid())
    np.testing.assert_array_equal(beam.feed_array, np.array(["x", "y"]))
    np.testing.assert_allclose(beam.feed_angle, np.array([np.pi / 2.0, 0.0]))
    assert beam.polarization_array is None
    assert beam.beam_type == "efield"
    assert beam.antenna_type == "simple"
    assert beam.pixel_coordinate_system == "az_za"
    assert beam.pixel_array is None
    assert beam.nside is None
    assert beam.ordering is None
    assert beam.mount_type == "fixed"
    assert beam.x_orientation == "east"
    assert beam.basis_vector_array.shape == (2, 2, 5, 8)
    expected_basis = np.zeros((2, 2, 5, 8), dtype=np.float64)
    expected_basis[0, 0] = 1.0
    expected_basis[1, 1] = 1.0
    np.testing.assert_array_equal(beam.basis_vector_array, expected_basis)
    assert beam.data_normalization == "peak"
    assert np.all(np.isfinite(beam.bandpass_array))
    np.testing.assert_array_equal(beam.bandpass_array, np.ones(4))
    assert np.all(np.isfinite(beam.data_array))
    np.testing.assert_array_equal(beam.data_array[0, 1], 0.0)
    np.testing.assert_array_equal(beam.data_array[1, 0], 0.0)
    np.testing.assert_array_equal(beam.data_array[0, 0], beam.data_array[1, 1])


@pytest.mark.parametrize("dtype", NATIVE_DTYPES, ids=("complex64", "complex128"))
@pytest.mark.parametrize(
    "variant", tuple(BeamScienceVariant), ids=lambda variant: variant.value
)
def test_native_scalar_science_matches_independent_axis_ordered_formula(
    dtype: Any,
    variant: BeamScienceVariant,
) -> None:
    """Verify both science variants at zenith, interior nodes, and the horizon."""
    beam = build_scalar_efield_uvbeam(dtype=dtype, variant=variant)
    azimuth = canonical_azimuth_grid()
    zenith_angle = canonical_zenith_angle_grid()
    exponent = 2 if variant is BeamScienceVariant.CANONICAL else 3
    sign = 1.0 if variant is BeamScienceVariant.CANONICAL else -1.0
    phase = sign * (
        0.03 * np.sin(azimuth)[np.newaxis, np.newaxis, :]
        + 0.01 * np.arange(4)[:, np.newaxis, np.newaxis]
    )
    expected = np.cos(zenith_angle)[np.newaxis, :, np.newaxis] ** exponent * np.exp(
        1j * phase
    )
    tolerance = 2e-7 if np.dtype(dtype) == np.dtype(np.complex64) else 1e-13

    np.testing.assert_allclose(
        beam.data_array[0, 0], expected, rtol=tolerance, atol=tolerance
    )
    for frequency_index, za_index, az_index in (
        (0, 0, 0),
        (1, 2, 3),
        (3, 4, 7),
    ):
        np.testing.assert_allclose(
            beam.data_array[0, 0, frequency_index, za_index, az_index],
            expected[frequency_index, za_index, az_index],
            rtol=tolerance,
            atol=tolerance,
        )
    horizon = np.cos(np.pi / 2.0) ** exponent
    real_dtype = np.float32 if np.dtype(dtype) == np.dtype(np.complex64) else np.float64
    np.testing.assert_allclose(
        np.abs(beam.data_array[0, 0, :, -1]),
        horizon,
        rtol=tolerance,
        atol=np.finfo(real_dtype).tiny,
    )


def test_builders_return_independently_owned_dependency_state() -> None:
    """Prevent mutable UVBeam or ndarray state from leaking between builders."""
    first = build_scalar_efield_uvbeam(dtype=np.complex64)
    second = build_scalar_efield_uvbeam(dtype=np.complex64)

    assert first is not second
    for first_array, second_array in (
        (first.data_array, second.data_array),
        (first.basis_vector_array, second.basis_vector_array),
        (first.freq_array, second.freq_array),
        (first.axis1_array, second.axis1_array),
        (first.axis2_array, second.axis2_array),
        (first.feed_array, second.feed_array),
        (first.bandpass_array, second.bandpass_array),
    ):
        assert not np.shares_memory(first_array, second_array)

    expected = second.data_array[0, 0, 0, 0, 0]
    first.data_array[0, 0, 0, 0, 0] = 99.0 + 12.0j
    first.basis_vector_array[0, 0, 0, 0] = -1.0
    first.freq_array[0] = -1.0
    assert second.data_array[0, 0, 0, 0, 0] == expected
    assert second.basis_vector_array[0, 0, 0, 0] == 1.0
    assert second.freq_array[0] == 100e6


def test_helper_module_has_no_mutable_module_level_uvbeam_cache() -> None:
    """Keep fixture construction independent of test order and global state."""
    module_values = vars(beamfits_helpers).values()
    assert not any(isinstance(value, UVBeam) for value in module_values)


@pytest.mark.parametrize(
    "filename",
    ("", "../outside.beamfits", "/tmp/outside.beamfits", "nested/file.beamfits"),
    ids=("empty", "traversal", "absolute", "nested"),
)
def test_writer_rejects_empty_or_nonbasename_filenames(
    tmp_path: Path,
    filename: str,
) -> None:
    """Ensure fixture transports remain fresh and below the caller's directory."""
    with pytest.raises(ValueError, match="one non-empty basename"):
        write_scalar_efield_beamfits(tmp_path, filename=filename)


def test_writer_rejects_overwrite(tmp_path: Path) -> None:
    """Keep each generated transport fresh even though pyuvdata permits clobbering."""

    written = write_scalar_efield_beamfits(tmp_path, filename="fresh.beamfits")
    assert written.path.parent == tmp_path.resolve()
    with pytest.raises(FileExistsError):
        write_scalar_efield_beamfits(tmp_path, filename="fresh.beamfits")


def test_writer_rejects_dangling_symlink_escape(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Prevent a broken basename symlink from redirecting bytes outside the root."""
    outside = tmp_path_factory.mktemp("beamfits-outside")
    escaped = outside / "escaped.beamfits"
    link = tmp_path / "dangling.beamfits"
    link.symlink_to(escaped)

    with pytest.raises(FileExistsError):
        write_scalar_efield_beamfits(tmp_path, filename=link.name)
    assert not escaped.exists()


def test_hash_is_calculated_from_actual_written_bytes(tmp_path: Path) -> None:
    """Prove fixture digests are runtime byte hashes rather than constants."""
    written = write_scalar_efield_beamfits(tmp_path)
    independent = hashlib.sha256(written.path.read_bytes()).hexdigest()

    assert written.sha256 == independent
    assert sha256_from_file(written.path) == independent
    with written.path.open("ab") as stream:
        stream.write(b"runtime digest probe")
    assert sha256_from_file(written.path) != independent


@pytest.mark.parametrize("dtype", NATIVE_DTYPES, ids=("complex64", "complex128"))
def test_repeated_files_preserve_science_without_assuming_transport_bytes(
    tmp_path: Path,
    dtype: Any,
) -> None:
    """Compare scientific content while treating FITS transport bytes as opaque."""
    first = write_scalar_efield_beamfits(
        tmp_path, dtype=dtype, filename=f"first-{np.dtype(dtype).name}.beamfits"
    )
    second = write_scalar_efield_beamfits(
        tmp_path, dtype=dtype, filename=f"second-{np.dtype(dtype).name}.beamfits"
    )
    first_beam = _read_beamfits(first.path)
    second_beam = _read_beamfits(second.path)

    assert first.sha256 == hashlib.sha256(first.path.read_bytes()).hexdigest()
    assert second.sha256 == hashlib.sha256(second.path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(first_beam.data_array, second_beam.data_array)
    np.testing.assert_array_equal(first_beam.freq_array, second_beam.freq_array)
    np.testing.assert_array_equal(first_beam.axis1_array, second_beam.axis1_array)
    np.testing.assert_array_equal(first_beam.axis2_array, second_beam.axis2_array)


def test_science_variants_differ_numerically(tmp_path: Path) -> None:
    """Keep the second BeamFITS fixture useful for assignment/dedup tests."""
    canonical = write_scalar_efield_beamfits(
        tmp_path,
        variant=BeamScienceVariant.CANONICAL,
        filename="canonical.beamfits",
    )
    distinct = write_scalar_efield_beamfits(
        tmp_path,
        variant=BeamScienceVariant.DISTINCT,
        filename="distinct.beamfits",
    )
    canonical_beam = _read_beamfits(canonical.path)
    distinct_beam = _read_beamfits(distinct.path)

    assert not np.array_equal(canonical_beam.data_array, distinct_beam.data_array)
    interior = (0, 0, 2, 2, 1)
    assert canonical_beam.data_array[interior] != distinct_beam.data_array[interior]


@pytest.mark.parametrize("dtype", NATIVE_DTYPES, ids=("complex64", "complex128"))
def test_interp_tuple_shapes_basis_dtype_and_nonmutation(dtype: Any) -> None:
    """Pin tuple arity, array shapes, basis behavior, dtype, and ownership."""
    beam = build_scalar_efield_uvbeam(dtype=dtype)
    original = np.array(beam.data_array, copy=True)
    azimuth = np.array([np.pi / 8.0, 7.0 * np.pi / 8.0])
    zenith_angle = np.array([np.pi / 16.0, 5.0 * np.pi / 16.0])
    frequencies = np.array([100e6, 120e6])

    data_without_basis, absent_basis = _interp(
        beam,
        azimuth=azimuth,
        zenith_angle=zenith_angle,
        frequencies=frequencies,
        return_basis=False,
    )
    data_with_basis, basis = _interp(
        beam,
        azimuth=azimuth,
        zenith_angle=zenith_angle,
        frequencies=frequencies,
        return_basis=True,
    )

    assert data_without_basis.shape == (2, 2, 2, 2)
    assert data_with_basis.shape == (2, 2, 2, 2)
    assert data_without_basis.dtype == np.dtype(np.complex128)
    assert data_with_basis.dtype == np.dtype(np.complex128)
    assert absent_basis is None
    assert basis is not None
    assert basis.shape == (2, 2, 2)
    expected_basis = np.repeat(np.eye(2)[:, :, np.newaxis], 2, axis=2)
    np.testing.assert_allclose(basis, expected_basis, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(beam.data_array, original)
    assert beam.data_array.dtype == np.dtype(dtype)


def test_bilinear_midpoint_matches_explicit_neighbor_weights() -> None:
    """Derive the controlled angular midpoint from four native neighbors."""
    beam = build_scalar_efield_uvbeam(dtype=np.complex128)
    azimuth = canonical_azimuth_grid()
    zenith_angle = canonical_zenith_angle_grid()
    az_index = 2
    za_index = 1
    midpoint_az = (azimuth[az_index] + azimuth[az_index + 1]) / 2.0
    midpoint_za = (zenith_angle[za_index] + zenith_angle[za_index + 1]) / 2.0

    data, _ = _interp(
        beam,
        azimuth=np.array([midpoint_az]),
        zenith_angle=np.array([midpoint_za]),
        frequencies=np.array([120e6]),
    )
    native = beam.data_array[0, 0, 2]
    expected = 0.25 * (
        native[za_index, az_index]
        + native[za_index, az_index + 1]
        + native[za_index + 1, az_index]
        + native[za_index + 1, az_index + 1]
    )

    np.testing.assert_allclose(data[0, 0, 0, 0], expected, rtol=0.0, atol=2e-15)


def test_jones_index_mapping_is_transpose_without_conjugation() -> None:
    """Pin J[feed, component] = data[component, feed] for identity basis."""
    beam = build_scalar_efield_uvbeam(dtype=np.complex128)
    beam.data_array[0, 0] = 1.0 + 2.0j
    beam.data_array[0, 1] = 3.0 + 4.0j
    beam.data_array[1, 0] = 5.0 + 6.0j
    beam.data_array[1, 1] = 7.0 + 8.0j
    data, basis = _interp(
        beam,
        azimuth=np.array([canonical_azimuth_grid()[1]]),
        zenith_angle=np.array([canonical_zenith_angle_grid()[2]]),
        frequencies=np.array([120e6]),
        return_basis=True,
    )
    expected_data = np.array(
        [
            [1.0 + 2.0j, 3.0 + 4.0j],
            [5.0 + 6.0j, 7.0 + 8.0j],
        ]
    )
    np.testing.assert_array_equal(data[:, :, 0, 0], expected_data)
    jones = data[:, :, 0, 0].transpose(1, 0)

    assert tuple(beam.feed_array) == ("x", "y")
    np.testing.assert_array_equal(
        jones,
        np.array(
            [
                [1.0 + 2.0j, 5.0 + 6.0j],
                [3.0 + 4.0j, 7.0 + 8.0j],
            ]
        ),
    )
    assert jones[0, 1] == data[1, 0, 0, 0]
    assert jones[1, 0] == data[0, 1, 0, 0]
    assert jones[0, 1] != data[0, 1, 0, 0]
    assert jones[0, 1] != np.conjugate(data[1, 0, 0, 0])
    assert basis is not None
    np.testing.assert_allclose(basis[:, :, 0], np.eye(2), rtol=0.0, atol=1e-15)


def test_radiosim_to_uvbeam_azimuth_conversion() -> None:
    """Prove North/East/South/West and modulo wrap without production code."""
    radiosim_azimuth = np.array(
        [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi, -np.pi / 2.0]
    )
    converted = (np.pi / 2.0 - radiosim_azimuth) % (2.0 * np.pi)
    expected = np.array(
        [
            np.pi / 2.0,
            0.0,
            3.0 * np.pi / 2.0,
            np.pi,
            np.pi / 2.0,
            np.pi,
        ]
    )

    np.testing.assert_allclose(converted, expected, rtol=0.0, atol=1e-15)


def test_exact_intrinsic_frequencies_select_native_values() -> None:
    """Pin exact channel selection before non-exact interpolation behavior."""
    beam = build_scalar_efield_uvbeam(dtype=np.complex64)
    az_index = 1
    za_index = 2
    data, _ = _interp(
        beam,
        azimuth=np.full(4, beam.axis1_array[az_index]),
        zenith_angle=np.full(4, beam.axis2_array[za_index]),
        frequencies=canonical_frequency_grid(),
        frequency_kind="cubic",
    )

    np.testing.assert_array_equal(
        data[0, 0, :, 0],
        beam.data_array[0, 0, :, za_index, az_index],
    )


def test_representable_frequency_tolerance_edge_is_strict() -> None:
    """Distinguish the measured representable values below and above 1e-6 Hz."""
    beam = build_scalar_efield_uvbeam(dtype=np.complex64)
    base = beam.freq_array[0]
    nominal_boundary = base + 1e-6
    below = np.nextafter(nominal_boundary, base)
    at_or_above = np.nextafter(nominal_boundary, np.inf)
    below_distance = below - base
    boundary_distance = at_or_above - base

    assert below_distance < 1e-6
    assert boundary_distance >= 1e-6
    assert boundary_distance - below_distance == 2.0 * np.spacing(base)

    az_index = 1
    za_index = 2
    coordinates = {
        "azimuth": np.array([beam.axis1_array[az_index]]),
        "zenith_angle": np.array([beam.axis2_array[za_index]]),
    }
    snapped, _ = _interp(
        beam,
        frequencies=np.array([below]),
        frequency_kind="linear",
        **coordinates,
    )
    interpolated, _ = _interp(
        beam,
        frequencies=np.array([at_or_above]),
        frequency_kind="linear",
        **coordinates,
    )
    native = complex(beam.data_array[0, 0, 0, za_index, az_index])

    assert snapped[0, 0, 0, 0] == native
    assert interpolated[0, 0, 0, 0] != native


def test_linear_frequency_interpolation_works_with_two_channels() -> None:
    """Pin explicit linear interpolation without relying on cubic fallback."""
    beam = build_scalar_efield_uvbeam(dtype=np.complex128)
    selected = beam.select(freq_chans=np.array([0, 1]), inplace=False)
    assert isinstance(selected, UVBeam)
    az_index = 3
    za_index = 1
    data, _ = _interp(
        selected,
        azimuth=np.array([selected.axis1_array[az_index]]),
        zenith_angle=np.array([selected.axis2_array[za_index]]),
        frequencies=np.array([105e6]),
        frequency_kind="linear",
    )
    expected = 0.5 * (
        beam.data_array[0, 0, 0, za_index, az_index]
        + beam.data_array[0, 0, 1, za_index, az_index]
    )

    assert data.dtype == np.dtype(np.complex128)
    np.testing.assert_allclose(data[0, 0, 0, 0], expected, rtol=0.0, atol=2e-15)


def test_cubic_frequency_interpolation_requires_four_channels() -> None:
    """Pin successful cubic output and the actual insufficient-channel error."""
    beam = build_scalar_efield_uvbeam(dtype=np.complex128)
    azimuth = np.array([beam.axis1_array[2]])
    zenith_angle = np.array([beam.axis2_array[2]])
    data, _ = _interp(
        beam,
        azimuth=azimuth,
        zenith_angle=zenith_angle,
        frequencies=np.array([105e6]),
        frequency_kind="cubic",
    )
    assert data.dtype == np.dtype(np.complex128)
    assert np.all(np.isfinite(data))

    three_channels = beam.select(freq_chans=np.array([0, 1, 2]), inplace=False)
    assert isinstance(three_channels, UVBeam)
    with pytest.raises(
        ValueError,
        match="number of derivatives at boundaries does not match",
    ):
        _interp(
            three_channels,
            azimuth=azimuth,
            zenith_angle=zenith_angle,
            frequencies=np.array([105e6]),
            frequency_kind="cubic",
        )


@pytest.mark.parametrize("frequency", (99e6, 131e6), ids=("below", "above"))
def test_frequency_interpolation_rejects_out_of_domain_values(
    frequency: float,
) -> None:
    """Pin fail-closed dependency behavior below and above the native interval."""
    beam = build_scalar_efield_uvbeam()
    with pytest.raises(
        ValueError,
        match="outside of the UVBeam freq_array range",
    ):
        _interp(
            beam,
            azimuth=np.array([0.0]),
            zenith_angle=np.array([0.0]),
            frequencies=np.array([frequency]),
            frequency_kind="linear",
        )


def test_angular_endpoints_wrap_and_in_domain_midpoint() -> None:
    """Pin normal angular interpolation plus zenith, horizon, and azimuth wrap."""
    beam = build_scalar_efield_uvbeam(dtype=np.complex128)
    azimuth = np.array([0.0, np.pi / 8.0, 2.0 * np.pi])
    zenith_angle = np.array([0.0, np.pi / 4.0, np.pi / 2.0])
    data, _ = _interp(
        beam,
        azimuth=azimuth,
        zenith_angle=zenith_angle,
        frequencies=np.full(3, 110e6),
    )

    assert data.shape == (2, 2, 3, 3)
    assert np.all(np.isfinite(data))
    np.testing.assert_allclose(
        data[:, :, 0, 0],
        beam.data_array[:, :, 1, 0, 0],
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        data[:, :, 2, 2],
        beam.data_array[:, :, 1, -1, 0],
        rtol=0.0,
        atol=2e-15,
    )


def test_angular_margin_can_extrapolate_but_clear_outlier_fails() -> None:
    """Characterize dependency margin that future RadioSim validation must narrow."""
    beam = build_scalar_efield_uvbeam(dtype=np.complex128)
    azimuth = np.array([0.2])
    inside_dependency_margin = 3.0 * np.pi / 4.0
    extrapolated, _ = _interp(
        beam,
        azimuth=azimuth,
        zenith_angle=np.array([inside_dependency_margin]),
        frequencies=np.array([110e6]),
    )
    horizon, _ = _interp(
        beam,
        azimuth=azimuth,
        zenith_angle=np.array([np.pi / 2.0]),
        frequencies=np.array([110e6]),
    )

    # az_za_simple clamps/extrapolates inside pyuvdata's two-grid-step margin.
    np.testing.assert_allclose(extrapolated, horizon, rtol=0.0, atol=2e-15)
    with pytest.raises(
        ValueError,
        match="zenith angles values are outside UVBeam coverage",
    ):
        _interp(
            beam,
            azimuth=azimuth,
            zenith_angle=np.array([np.pi + 0.01]),
            frequencies=np.array([110e6]),
        )


def test_peak_normalize_mutates_data_and_moves_frequency_peaks() -> None:
    """Characterize why canonical fixture construction never normalizes at runtime."""
    beam = build_scalar_efield_uvbeam(dtype=np.complex128)
    factors = np.array([2.0, 3.0, 4.0, 5.0])
    beam.data_normalization = "physical"
    beam.data_array *= factors[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    original = np.array(beam.data_array, copy=True)

    assert beam.peak_normalize() is None
    assert not np.array_equal(beam.data_array, original)
    assert beam.data_normalization == "peak"
    np.testing.assert_allclose(beam.bandpass_array, factors, rtol=0.0, atol=1e-15)
    peaks = np.max(np.abs(beam.data_array), axis=(0, 1, 3, 4))
    np.testing.assert_allclose(peaks, np.ones(4), rtol=0.0, atol=1e-15)


def test_canonical_builder_never_calls_peak_normalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect the fixture from hiding source amplitude through mutation."""

    def forbidden_peak_normalize(self: UVBeam) -> None:
        raise AssertionError(f"unexpected normalization of {self.telescope_name}")

    monkeypatch.setattr(UVBeam, "peak_normalize", forbidden_peak_normalize)
    beam = build_scalar_efield_uvbeam()
    assert beam.data_normalization == "peak"
    np.testing.assert_array_equal(beam.bandpass_array, np.ones(4))


def test_full_and_partial_beamfits_reads_pin_range_units_and_endpoints(
    tmp_path: Path,
) -> None:
    """Characterize Hz frequency ranges and inclusive degree angular ranges."""
    written = write_scalar_efield_beamfits(tmp_path)
    full = _read_beamfits(written.path)
    frequency_slice = _read_beamfits(
        written.path,
        freq_range=(105e6, 120e6),
    )
    azimuth_slice = _read_beamfits(
        written.path,
        az_range=(45.0, 180.0),
    )
    zenith_slice = _read_beamfits(
        written.path,
        za_range=(22.5, 67.5),
    )

    np.testing.assert_array_equal(full.freq_array, canonical_frequency_grid())
    np.testing.assert_allclose(full.axis1_array, canonical_azimuth_grid())
    np.testing.assert_allclose(full.axis2_array, canonical_zenith_angle_grid())
    np.testing.assert_array_equal(frequency_slice.freq_array, np.array([110e6, 120e6]))
    np.testing.assert_allclose(
        np.rad2deg(azimuth_slice.axis1_array),
        np.array([45.0, 90.0, 135.0, 180.0]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.rad2deg(zenith_slice.axis2_array),
        np.array([22.5, 45.0, 67.5]),
        rtol=0.0,
        atol=1e-12,
    )
    # These partial reads are characterized only; Tier 3 intentionally loads full
    # axes so coverage and interpolation validity can be checked deterministically.


@pytest.mark.parametrize("variant", tuple(UnsupportedBeamVariant), ids=str)
def test_invalid_and_deferred_variant_metadata_is_constructed(
    variant: UnsupportedBeamVariant,
) -> None:
    """Prove every future negative fixture expresses its intended condition."""
    fixture = build_beam_variant(variant)
    beam = fixture.beam

    if variant is UnsupportedBeamVariant.POWER:
        assert beam.beam_type == "power"
        assert beam.feed_array is None
        assert beam.polarization_array is not None
    elif variant is UnsupportedBeamVariant.CIRCULAR_FEEDS:
        assert tuple(beam.feed_array) == ("r", "l")
    elif variant is UnsupportedBeamVariant.WRONG_FEED_ORDER:
        assert tuple(beam.feed_array) == ("y", "x")
    elif variant is UnsupportedBeamVariant.WRONG_X_ORIENTATION:
        assert beam.x_orientation == "north"
    elif variant is UnsupportedBeamVariant.WRONG_FEED_ANGLES:
        assert not np.allclose(beam.feed_angle, np.array([np.pi / 2.0, 0.0]))
    elif variant is UnsupportedBeamVariant.NON_FIXED_MOUNT:
        assert beam.mount_type != "fixed"
    elif variant is UnsupportedBeamVariant.NONIDENTITY_BASIS:
        assert not np.array_equal(beam.basis_vector_array[:, :, 0, 0], np.eye(2))
    elif variant is UnsupportedBeamVariant.NONFINITE_BASIS:
        assert np.any(~np.isfinite(beam.basis_vector_array))
    elif variant is UnsupportedBeamVariant.CROSS_POLAR:
        assert np.any(beam.data_array[0, 1] != 0.0)
    elif variant is UnsupportedBeamVariant.UNEQUAL_DIAGONALS:
        assert not np.array_equal(beam.data_array[0, 0], beam.data_array[1, 1])
    elif variant is UnsupportedBeamVariant.NON_PEAK_NORMALIZATION:
        assert beam.data_normalization == "physical"
    elif variant is UnsupportedBeamVariant.NON_UNIT_BANDPASS:
        assert not np.array_equal(beam.bandpass_array, np.ones(4))
    elif variant is UnsupportedBeamVariant.PEAK_LABEL_NONPEAK_DATA:
        assert beam.data_normalization == "peak"
        assert np.max(np.abs(beam.data_array)) < 1.0
    elif variant is UnsupportedBeamVariant.HEALPIX:
        assert beam.pixel_coordinate_system == "healpix"
        assert beam.nside == 1
        assert beam.pixel_array is not None
        assert beam.axis1_array is None
        assert beam.axis2_array is None
    elif variant is UnsupportedBeamVariant.WRONG_COORDINATE_METADATA:
        assert beam.pixel_coordinate_system == "orthoslant_zenith"
    elif variant is UnsupportedBeamVariant.IRREGULAR_AZIMUTH:
        assert not np.allclose(np.diff(beam.axis1_array), np.diff(beam.axis1_array)[0])
    elif variant is UnsupportedBeamVariant.IRREGULAR_ZENITH_ANGLE:
        assert not np.allclose(np.diff(beam.axis2_array), np.diff(beam.axis2_array)[0])
    elif variant is UnsupportedBeamVariant.INCOMPLETE_AZIMUTH_CLOSURE:
        assert beam.axis1_array.size == 7
        assert beam.axis1_array[-1] + np.diff(beam.axis1_array)[0] < 2.0 * np.pi
    elif variant is UnsupportedBeamVariant.SHORT_ZA:
        assert beam.axis2_array[-1] < np.pi / 2.0
    elif variant is UnsupportedBeamVariant.OUT_OF_FREQUENCY_COVERAGE:
        assert beam.freq_array[-1] == 120e6
        assert 130e6 > beam.freq_array[-1]
    elif variant is UnsupportedBeamVariant.DUPLICATE_FREQUENCY:
        assert np.any(np.diff(beam.freq_array) == 0.0)
    elif variant is UnsupportedBeamVariant.DECREASING_FREQUENCY:
        assert np.any(np.diff(beam.freq_array) < 0.0)
    elif variant is UnsupportedBeamVariant.NONPOSITIVE_FREQUENCY:
        assert np.any(beam.freq_array <= 0.0)
    elif variant is UnsupportedBeamVariant.NONFINITE_FREQUENCY:
        assert np.any(~np.isfinite(beam.freq_array))
    elif variant is UnsupportedBeamVariant.NONFINITE_DATA:
        assert np.any(~np.isfinite(beam.data_array))
    elif variant is UnsupportedBeamVariant.WRONG_NATIVE_DTYPE:
        assert beam.data_array.dtype == np.dtype(object)
    elif variant is UnsupportedBeamVariant.INVALID_DATA_SHAPE:
        assert beam.data_array.shape == (2, 2, 4, 5, 7)


def test_variant_classification_distinguishes_dependency_and_coverage() -> None:
    """Separate pyuvdata validity from the future RadioSim accepted subset."""
    for variant in (
        UnsupportedBeamVariant.INVALID_DATA_SHAPE,
        UnsupportedBeamVariant.NONFINITE_BASIS,
        UnsupportedBeamVariant.IRREGULAR_AZIMUTH,
        UnsupportedBeamVariant.IRREGULAR_ZENITH_ANGLE,
    ):
        dependency_invalid = build_beam_variant(variant)
        with pytest.raises(ValueError):
            dependency_invalid.beam.check(
                check_extra=True,
                run_check_acceptability=True,
            )
        assert (
            dependency_invalid.classification
            is BeamVariantClassification.DEPENDENCY_INVALID
        )

    for variant in (
        UnsupportedBeamVariant.SHORT_ZA,
        UnsupportedBeamVariant.OUT_OF_FREQUENCY_COVERAGE,
    ):
        fixture = build_beam_variant(variant)
        assert fixture.beam.check(check_extra=True, run_check_acceptability=True)
        assert (
            fixture.classification
            is BeamVariantClassification.VALID_INSUFFICIENT_COVERAGE
        )

    for variant in (
        UnsupportedBeamVariant.POWER,
        UnsupportedBeamVariant.CIRCULAR_FEEDS,
        UnsupportedBeamVariant.WRONG_FEED_ORDER,
        UnsupportedBeamVariant.WRONG_X_ORIENTATION,
        UnsupportedBeamVariant.WRONG_FEED_ANGLES,
        UnsupportedBeamVariant.NON_FIXED_MOUNT,
        UnsupportedBeamVariant.NONIDENTITY_BASIS,
        UnsupportedBeamVariant.CROSS_POLAR,
        UnsupportedBeamVariant.UNEQUAL_DIAGONALS,
        UnsupportedBeamVariant.NON_PEAK_NORMALIZATION,
        UnsupportedBeamVariant.NON_UNIT_BANDPASS,
        UnsupportedBeamVariant.PEAK_LABEL_NONPEAK_DATA,
        UnsupportedBeamVariant.HEALPIX,
        UnsupportedBeamVariant.WRONG_COORDINATE_METADATA,
        UnsupportedBeamVariant.INCOMPLETE_AZIMUTH_CLOSURE,
        UnsupportedBeamVariant.DUPLICATE_FREQUENCY,
        UnsupportedBeamVariant.DECREASING_FREQUENCY,
        UnsupportedBeamVariant.NONPOSITIVE_FREQUENCY,
        UnsupportedBeamVariant.NONFINITE_FREQUENCY,
        UnsupportedBeamVariant.NONFINITE_DATA,
        UnsupportedBeamVariant.WRONG_NATIVE_DTYPE,
    ):
        fixture = build_beam_variant(variant)
        assert fixture.beam.check(check_extra=True, run_check_acceptability=True)
        assert (
            fixture.classification
            is BeamVariantClassification.DEPENDENCY_VALID_UNSUPPORTED
        )


def test_counting_loader_tracks_failures_retries_and_instance_state(
    tmp_path: Path,
) -> None:
    """Provide a deterministic seam for later deduplication and atomic retry tests."""
    written = write_scalar_efield_beamfits(tmp_path)
    loader = CountingBeamFITSLoader(fail_on_attempts=frozenset({1, 2}))

    for attempt in (1, 2):
        with pytest.raises(
            RuntimeError,
            match=f"^injected BeamFITS read failure on attempt {attempt}$",
        ):
            loader.read(written.path)
    loaded = loader.read(written.path)

    assert loader.attempts == 3
    assert loader.requested_paths == (written.path, written.path, written.path)
    assert loaded.data_array.shape == (2, 2, 4, 5, 8)
    independent = CountingBeamFITSLoader()
    assert independent.attempts == 0
    assert independent.requested_paths == ()
    independently_loaded = independent.read(written.path)
    assert independent.attempts == 1
    assert independently_loaded is not loaded
    assert loader.attempts == 3


@pytest.mark.parametrize(
    "fail_on_attempts",
    (None, {1}, [1], (1,)),
    ids=("none", "mutable-set", "list", "tuple"),
)
def test_counting_loader_rejects_non_frozenset_failure_schedules(
    fail_on_attempts: Any,
) -> None:
    """Reject mutable or wrongly typed schedules instead of retaining aliases."""
    with pytest.raises(
        TypeError,
        match=("^fail_on_attempts must be a frozenset of positive one-based integers$"),
    ):
        CountingBeamFITSLoader(fail_on_attempts=fail_on_attempts)


@pytest.mark.parametrize(
    "fail_on_attempts",
    (
        frozenset({0}),
        frozenset({-1}),
        frozenset({True}),
        frozenset({1.5}),
    ),
    ids=("zero", "negative", "boolean", "non-integer"),
)
def test_counting_loader_rejects_invalid_one_based_attempts(
    fail_on_attempts: frozenset[Any],
) -> None:
    """Require exact positive integer attempt numbers for deterministic failures."""
    with pytest.raises(
        ValueError,
        match=("^fail_on_attempts must be a frozenset of positive one-based integers$"),
    ):
        CountingBeamFITSLoader(fail_on_attempts=fail_on_attempts)


# ==============================================================================
# SCI-005 Stage 3: the pinned dependency's stored-basis behaviour
# ==============================================================================
#
# ``UVBeam._prepare_basis_vector_array`` -- the single site both ``az_za``
# interpolation functions use to build the returned basis -- either raises a
# bare untyped ``NotImplementedError``, whenever any stored off-diagonal entry
# is strictly positive, or discards the stored array entirely and rebuilds the
# exact native identity at every requested direction. The three tests below pin
# that behaviour, and the exact round-trip of a stored identity in both real
# floating widths, as observed dependency fact.
#
# These measurements are the ones the accepted bounded basis-vector and
# provenance correction rests on. ``docs/development/sci005_beam_physics_plan.md``
# Section 5.1.1 item 10 now requires a committed ``basis_vector_array`` to be
# **exactly** the native identity at a real floating stored dtype judged by kind
# and width, and corrected Section 5.2.1 keeps ``return_basis_vector=True`` only
# in order to *verify* the returned identity, applying RadioSim's own
# ``T(phi)`` to the native components instead of composing the stored array. The
# round-trip control is what makes an exactness predicate -- rather than a
# tolerance -- the correct and reachable one.


def _stage3_constant_basis_beam(matrix: np.ndarray) -> UVBeam:
    """Build one scalar fixture whose stored basis is this constant matrix."""
    beam = build_scalar_efield_uvbeam(dtype=np.complex128)
    basis = np.zeros_like(np.asarray(beam.basis_vector_array, dtype=np.float64))
    for row in range(2):
        for column in range(2):
            basis[row, column, :, :] = float(matrix[row][column])
    beam.basis_vector_array = basis
    assert beam.check(check_extra=True, run_check_acceptability=True) is True
    return beam


def test_interp_refuses_a_stored_basis_with_a_positive_off_diagonal_entry() -> None:
    """A legal real non-identity basis can make ``interp`` unusable.

    ``UVBeam.check`` accepts the file -- pyuvdata declares
    ``basis_vector_array`` real with ``acceptable_range`` ``(-1, 1)`` -- but
    every ``az_za`` interpolation of it raises ``NotImplementedError``. The
    rotation below is the ordinary ``[[cos, -sin], [sin, cos]]`` at ``0.3``
    radians, whose ``[1, 0]`` entry is positive.
    """
    rotation = [
        [np.cos(0.3), -np.sin(0.3)],
        [np.sin(0.3), np.cos(0.3)],
    ]
    beam = _stage3_constant_basis_beam(rotation)

    with pytest.raises(NotImplementedError, match="not aligned to the"):
        _interp(
            beam,
            azimuth=np.array([0.1, 0.7]),
            zenith_angle=np.array([0.2, 0.4]),
            frequencies=np.array([100e6]),
            return_basis=True,
        )


@pytest.mark.parametrize(
    ("label", "matrix"),
    [
        ("identity", [[1.0, 0.0], [0.0, 1.0]]),
        ("negative_off_diagonals", [[0.9553, -0.2955], [-0.2955, 0.9553]]),
        ("uniformly_scaled", [[0.5, 0.0], [0.0, 0.5]]),
        ("anti_diagonal", [[0.0, -1.0], [-1.0, 0.0]]),
    ],
)
def test_interp_returns_the_identity_basis_whatever_the_file_stored(
    label: str,
    matrix: list[list[float]],
) -> None:
    """Every basis ``interp`` does accept is replaced by the identity.

    ``return_basis_vector=True`` never reports the committed array: the
    dependency rebuilds ``[[1, 0], [0, 1]]`` per point from
    ``theta hat``/``phi hat``. A Stage-3 conversion that multiplies the
    *returned* basis therefore cannot see a file's stored one at all.
    """
    beam = _stage3_constant_basis_beam(matrix)

    _data, basis = _interp(
        beam,
        azimuth=np.array([0.1, 0.7]),
        zenith_angle=np.array([0.2, 0.4]),
        frequencies=np.array([100e6]),
        return_basis=True,
    )

    assert basis is not None
    assert basis.shape == (2, 2, 2)
    assert basis.dtype == np.dtype(np.float64)
    expected = np.repeat(np.eye(2)[:, :, np.newaxis], 2, axis=2)
    np.testing.assert_array_equal(basis, expected)
    stored = np.asarray(beam.basis_vector_array, dtype=np.float64)
    identical = bool(np.allclose(np.asarray(matrix), np.eye(2)))
    assert identical == bool(np.allclose(stored[:, :, 0, 0], np.eye(2)))


@pytest.mark.parametrize("stored_dtype", [np.float32, np.float64])
def test_a_stored_identity_basis_round_trips_bit_exactly_in_both_widths(
    tmp_path: Path,
    stored_dtype: Any,
) -> None:
    """Corrected Section 5.1.1 item 10's justification, measured.

    "Both stored widths are accepted, because the identity values ``1.0`` and
    ``0.0`` are exactly representable and round-trip bit-exactly in each --
    which is why the predicate above is exact equality rather than a
    tolerance." The same measurement shows why the dtype must be judged by kind
    and width: BeamFITS returns the array big-endian, so an equality test
    against ``numpy.float64`` would reject every committed beam.
    """
    from tests.fixtures.beamfits import (
        build_efield_uvbeam,
        native_identity_basis_vector_array,
    )

    stored = native_identity_basis_vector_array(dtype=stored_dtype)
    beam = build_efield_uvbeam(basis_vector_array=stored)
    target = tmp_path / f"identity-{np.dtype(stored_dtype).name}.beamfits"
    assert beam.write_beamfits(target, clobber=True) is None

    read = _read_beamfits(target)
    observed = np.asarray(read.basis_vector_array)

    np.testing.assert_array_equal(observed, stored)
    assert observed.dtype.kind == "f"
    assert observed.dtype.itemsize == np.dtype(stored_dtype).itemsize
    assert observed.dtype.name == np.dtype(stored_dtype).name
    # Exactly the native identity, entry by entry, after the round trip.
    assert bool(np.all(observed[0, 0] == 1.0))
    assert bool(np.all(observed[1, 1] == 1.0))
    assert bool(np.all(observed[0, 1] == 0.0))
    assert bool(np.all(observed[1, 0] == 0.0))
    # The byte-order-qualified comparison the correction retires.
    assert observed.dtype != np.dtype(stored_dtype)
