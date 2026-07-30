"""Canonical UVFITS projection, validation, and atomic publication."""

from __future__ import annotations

import os
import stat
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np

from radiosim.core.result import SimulationResult
from radiosim.io.atomic_paths import (
    create_sibling_temporary,
    fsync_directory,
    fsync_file,
    open_parent_directory,
    publish_no_clobber,
    publish_replace,
    unlink_temporary,
    validate_input_regular_file,
    validate_output_target,
)
from radiosim.io.result_errors import (
    AtomicWriteError,
    FormatRepresentationError,
    OptionalResultDependencyError,
    PartialCleanupError,
    ResultIOError,
    UnsafeResultInputError,
    UnsupportedPolarizationBasisError,
)
from radiosim.io.standard_visibility import (
    StandardReadLimits,
    StandardVisibilityData,
    enforce_standard_read_limits,
    normalize_autocorrelations,
    project_simulation_result,
    projection_record_from_history,
    require_polarization_basis,
    standard_visibility_from_uvdata,
    validate_projection_result,
    validate_standard_metadata,
)

_FITS_BLOCK_BYTES = 2880
_FITS_CARD_BYTES = 80
_MAX_PRIMARY_HEADER_BYTES = 262_144
_MAX_PROJECTION_HISTORY_BYTES = 16_000


def _pyuvdata_version() -> str:
    try:
        return version("pyuvdata")
    except PackageNotFoundError:
        return "unavailable"


def _import_pyuvdata() -> type[Any]:
    """Import pyuvdata only after pure UVFITS preflight succeeds."""
    installed = _pyuvdata_version()
    try:
        module = import_module("pyuvdata")
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalResultDependencyError(
            "format=uvfits missing_package=pyuvdata "
            f"pyuvdata_version={installed} install_extra=radiosim"
        ) from exc
    return module.UVData


def _validate_uvfits_representability(result: object) -> SimulationResult:
    """Aggregate every pure UVFITS representation constraint."""
    failures: list[str] = []
    try:
        typed = validate_projection_result(result, format_name="uvfits")
    except (TypeError, FormatRepresentationError) as exc:
        if type(result) is not SimulationResult:
            raise
        typed = result
        failures.append(str(exc))

    antenna_numbers = [antenna.id.number for antenna in typed.instrument.antennas]
    if not 1 <= len(antenna_numbers) <= 255:
        failures.append("UVFITS requires 1 through 255 antennas")
    if any(number < 0 or number > 254 for number in antenna_numbers):
        failures.append("UVFITS antenna numbers must be in 0..254")
    if len(set(antenna_numbers)) != len(antenna_numbers):
        failures.append("UVFITS antenna numbers must be unique")

    frequencies = np.asarray(typed.frequencies_hz)
    widths = np.asarray(typed.channel_widths_hz)
    if not np.all(np.isfinite(frequencies)):
        failures.append("UVFITS frequency centers must be finite")
    if not np.all(np.isfinite(widths)):
        failures.append("UVFITS channel widths must be finite")
    if frequencies.size > 0 and widths.size > 0:
        scale = float(np.max(np.abs(frequencies)))
        tolerance = 32.0 * np.finfo(np.float64).eps * scale
        if frequencies.size >= 2:
            spacings = np.diff(frequencies)
            if not np.allclose(
                spacings,
                spacings[0],
                rtol=0.0,
                atol=tolerance,
            ):
                failures.append("UVFITS frequency centers must be evenly spaced")
        if not np.allclose(
            widths,
            widths[0],
            rtol=0.0,
            atol=tolerance,
        ):
            failures.append("UVFITS requires equal channel widths")
        if frequencies.size >= 2:
            if not np.allclose(
                np.diff(frequencies),
                widths[0],
                rtol=0.0,
                atol=tolerance,
            ):
                failures.append("UVFITS frequency spacing must equal channel width")
    try:
        _ = require_polarization_basis(typed.correlations)
    except UnsupportedPolarizationBasisError as exc:
        failures.append(f"UVFITS requires an accepted correlation set: {exc}")
    else:
        try:
            _ = normalize_autocorrelations(typed)
        except FormatRepresentationError as exc:
            failures.append(str(exc))

    metadata_arrays = (
        typed.visibilities,
        typed.flags.astype(np.float32, copy=False),
        typed.weights,
        typed.frequencies_hz,
        typed.channel_widths_hz,
        typed.time_grid.utc_jd1,
        typed.time_grid.utc_jd2,
        typed.time_grid.integration_time_seconds,
        np.asarray(typed.instrument.location.itrs_xyz_m),
        np.asarray([antenna.position_enu_m for antenna in typed.instrument.antennas]),
        np.asarray([antenna.diameter_m for antenna in typed.instrument.antennas]),
        np.asarray([baseline.vector_enu_m for baseline in typed.selection.baselines]),
    )
    if any(not np.all(np.isfinite(array)) for array in metadata_arrays):
        failures.append(
            "UVFITS data, coordinates, weights, and metadata must be finite"
        )

    if failures:
        unique = tuple(dict.fromkeys(failures))
        raise FormatRepresentationError(
            "UVFITS representability constraints failed: "
            + "; ".join(unique)
            + ". Use HDF5 or Measurement Set for unsupported results."
        )
    return typed


def _write_uvfits(uvdata: Any, path: Path, **kwargs: object) -> None:
    uvdata.write_uvfits(str(path), **kwargs)


def _read_uvfits(path: Path, *, read_data: bool) -> Any:
    uvdata_class = _import_pyuvdata()
    uvdata = uvdata_class()
    try:
        uvdata.read_uvfits(
            str(path),
            read_data=read_data,
            run_check=read_data,
        )
    except ResultIOError:
        raise
    except Exception as exc:
        raise UnsafeResultInputError(
            f"could not read validated UVFITS path: {path}"
        ) from exc
    return uvdata


def _bounded_primary_projection_history(path: Path) -> dict[str, object]:
    """Scan the primary header incrementally before constructing an Astropy header."""
    descriptor: int | None = None
    history = bytearray()
    end_found = False
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise UnsafeResultInputError(
                "UVFITS input changed to a non-regular file during preflight"
            )
        if status.st_size % _FITS_BLOCK_BYTES != 0:
            raise UnsafeResultInputError(
                "UVFITS input has partial trailing FITS block bytes"
            )
        consumed = 0
        while consumed + _FITS_BLOCK_BYTES <= _MAX_PRIMARY_HEADER_BYTES:
            block = os.read(descriptor, _FITS_BLOCK_BYTES)
            if len(block) != _FITS_BLOCK_BYTES:
                raise UnsafeResultInputError(
                    "UVFITS primary header is truncated before END"
                )
            consumed += len(block)
            for offset in range(0, _FITS_BLOCK_BYTES, _FITS_CARD_BYTES):
                card = block[offset : offset + _FITS_CARD_BYTES]
                keyword = card[:8]
                if keyword == b"HISTORY ":
                    value = card[8:].rstrip(b" ")
                    additional = len(value) + (1 if history else 0)
                    if len(history) + additional > _MAX_PROJECTION_HISTORY_BYTES:
                        raise UnsafeResultInputError(
                            "UVFITS projection HISTORY exceeds 16000 bytes"
                        )
                    if history:
                        history.extend(b"\n")
                    history.extend(value)
                if keyword == b"END     ":
                    end_found = True
                    break
            if end_found:
                break
        if not end_found:
            raise UnsafeResultInputError(
                "UVFITS primary header exceeds the bounded size or lacks END"
            )
    except ResultIOError:
        raise
    except OSError as exc:
        raise UnsafeResultInputError(
            f"UVFITS primary header could not be read safely: {path}"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    try:
        history_text = bytes(history).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise UnsafeResultInputError(
            "UVFITS projection HISTORY is not strict UTF-8"
        ) from exc
    record, _lines = projection_record_from_history(history_text)
    return record


def _inspect_uvfits_headers(
    path: Path,
    limits: StandardReadLimits,
) -> dict[str, object]:
    """Bound FITS random-group dimensions before pyuvdata science allocation."""
    if type(limits) is not StandardReadLimits:
        raise TypeError("limits must be an exact StandardReadLimits")
    projection_record = _bounded_primary_projection_history(path)
    try:
        fits = import_module("astropy.io.fits")
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            lazy_load_hdus=True,
        ) as handle:
            if len(handle) < 2:
                raise UnsafeResultInputError("UVFITS input lacks the antenna table")
            last_info = handle.fileinfo(len(handle) - 1)
            if not isinstance(last_info, dict):
                raise UnsafeResultInputError(
                    "UVFITS input lacks a bounded final HDU location"
                )
            last_info = cast(dict[str, object], last_info)
            try:
                dat_loc = last_info.get("datLoc")
                dat_span = last_info.get("datSpan")
                if type(dat_loc) is not int or type(dat_span) is not int:
                    raise ValueError("FITS HDU offsets must be integers")
                final_hdu_end = dat_loc + dat_span
                file_size = int(path.stat().st_size)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise UnsafeResultInputError(
                    "UVFITS input has unsafe final HDU metadata"
                ) from exc
            if final_hdu_end != file_size:
                raise UnsafeResultInputError(
                    "UVFITS input has trailing bytes after its final HDU"
                )
            primary = handle[0].header
            if primary.get("SIMPLE") is not True or primary.get("GROUPS") is not True:
                raise UnsafeResultInputError(
                    "UVFITS input is not a FITS random-groups file"
                )
            group_count = int(primary.get("GCOUNT", 0))
            polarization_count = abs(int(primary.get("NAXIS3", 0)))
            frequency_count = int(primary.get("NAXIS4", 0))
            if min(group_count, polarization_count, frequency_count) <= 0:
                raise UnsafeResultInputError(
                    "UVFITS input has nonpositive random-group dimensions"
                )
            if polarization_count != 4:
                raise FormatRepresentationError(
                    "UVFITS input must contain exactly four polarizations"
                )
            if frequency_count > limits.max_frequencies:
                raise UnsafeResultInputError("standard input exceeds max_frequencies")
            elements = group_count * frequency_count * polarization_count
            if elements > limits.max_visibility_elements:
                raise UnsafeResultInputError(
                    "standard input exceeds max_visibility_elements"
                )
            if elements * 21 > limits.max_data_bytes:
                raise UnsafeResultInputError("standard input exceeds max_data_bytes")
            antenna_hdus = [hdu for hdu in handle[1:] if hdu.name == "AIPS AN"]
            if len(antenna_hdus) != 1:
                raise UnsafeResultInputError(
                    "UVFITS input requires one AIPS AN antenna table"
                )
            antenna_count = int(antenna_hdus[0].header.get("NAXIS2", 0))
            if antenna_count <= 0:
                raise UnsafeResultInputError("UVFITS input has no antennas")
            if antenna_count > limits.max_antennas:
                raise UnsafeResultInputError("standard input exceeds max_antennas")
    except (ResultIOError, TypeError):
        raise
    except Exception as exc:
        raise UnsafeResultInputError(
            f"UVFITS header inspection failed: {path}"
        ) from exc
    return projection_record


def _assert_round_trip(
    expected: StandardVisibilityData,
    observed: StandardVisibilityData,
) -> None:
    if observed.format != "uvfits":
        raise AtomicWriteError("temporary UVFITS readback has the wrong format")
    for field_name in (
        "correlations",
        "source_scientific_sha256",
        "source_provenance_sha256",
    ):
        if getattr(observed, field_name) != getattr(expected, field_name):
            raise AtomicWriteError(f"temporary UVFITS readback changed {field_name}")
    for field_name in (
        "flags",
        "antenna1_numbers",
        "antenna2_numbers",
    ):
        if not np.array_equal(
            getattr(observed, field_name),
            getattr(expected, field_name),
        ):
            raise AtomicWriteError(f"temporary UVFITS readback changed {field_name}")
    visibility_tolerance = (
        (2e-6, 2e-6)
        if expected.visibilities.dtype == np.dtype("complex64")
        else (5e-13, 5e-13)
    )
    comparisons = (
        ("visibilities", *visibility_tolerance),
        ("weights", 5e-6, 1e-7),
        ("frequencies_hz", 0.0, 1e-6),
        ("channel_widths_hz", 0.0, 1e-9),
        ("exposure_seconds", 0.0, 1e-9),
        ("uvw_m", 0.0, 1e-6),
    )
    for field_name, relative, absolute in comparisons:
        if not np.allclose(
            getattr(observed, field_name),
            getattr(expected, field_name),
            rtol=relative,
            atol=absolute,
        ):
            raise AtomicWriteError(f"temporary UVFITS readback changed {field_name}")
    if not np.allclose(
        observed.utc_jd1 + observed.utc_jd2,
        expected.utc_jd1 + expected.utc_jd2,
        rtol=0.0,
        atol=5e-10,
    ):
        raise AtomicWriteError("temporary UVFITS readback changed time coordinates")
    if observed.phase_center != expected.phase_center:
        raise AtomicWriteError("temporary UVFITS readback changed projected phase")
    if not _telescope_metadata_equal(
        expected.telescope_snapshot,
        observed.telescope_snapshot,
    ):
        raise AtomicWriteError("temporary UVFITS readback changed telescope metadata")


def _telescope_metadata_equal(expected: Any, observed: Any) -> bool:
    if (
        expected["name"] != observed["name"]
        or expected["instrument"] != observed["instrument"]
        or not np.allclose(
            expected["location_itrs_xyz_m"],
            observed["location_itrs_xyz_m"],
            rtol=0.0,
            atol=1e-6,
        )
    ):
        return False
    expected_antennas = expected["antennas"]
    observed_antennas = observed["antennas"]
    if len(expected_antennas) != len(observed_antennas):
        return False
    for expected_antenna, observed_antenna in zip(
        expected_antennas,
        observed_antennas,
        strict=True,
    ):
        if (
            expected_antenna["number"] != observed_antenna["number"]
            or expected_antenna["name"] != observed_antenna["name"]
            or expected_antenna["diameter_m"] != observed_antenna["diameter_m"]
            or not np.allclose(
                expected_antenna["position_enu_m"],
                observed_antenna["position_enu_m"],
                rtol=0.0,
                atol=1e-6,
            )
        ):
            return False
    return True


def _verify_temporary_uvfits(
    expected: StandardVisibilityData,
    temporary: Path,
) -> None:
    projection_record = _inspect_uvfits_headers(
        temporary,
        StandardReadLimits(),
    )
    metadata = _read_uvfits(temporary, read_data=False)
    enforce_standard_read_limits(metadata, StandardReadLimits())
    validate_standard_metadata(metadata)
    loaded = _read_uvfits(temporary, read_data=True)
    observed = standard_visibility_from_uvdata(
        loaded,
        format="uvfits",
        expected_projection_record=projection_record,
    )
    _assert_round_trip(expected, observed)


def _cleanup_temporary(
    temporary: Path,
    parent_fd: int,
    cause: BaseException,
) -> None:
    try:
        unlink_temporary(temporary, parent_fd)
    except Exception as cleanup_error:
        error = PartialCleanupError(temporary)
        error.add_note(f"cleanup failure: {cleanup_error!r}")
        raise error from cause


def write_uvfits(
    result: SimulationResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Project and atomically publish one canonical result as UVFITS."""
    typed = _validate_uvfits_representability(result)
    final = validate_output_target(
        path,
        extension=".uvfits",
        overwrite=overwrite,
    )
    _ = _import_pyuvdata()
    projected = project_simulation_result(typed, format="uvfits")

    parent_fd: int | None = None
    temporary_fd: int | None = None
    temporary: Path | None = None
    published = False
    try:
        parent_fd = open_parent_directory(final.parent, create=True)
        temporary_fd, temporary = create_sibling_temporary(final, parent_fd)
        os.close(temporary_fd)
        temporary_fd = None
        _write_uvfits(projected.uvdata, temporary, force_phase=False)
        temporary_fd = os.open(
            temporary.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        fsync_file(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        _verify_temporary_uvfits(projected.data, temporary)
        if overwrite:
            publish_replace(temporary, final, parent_fd)
            temporary = None
        else:
            publish_no_clobber(temporary, final, parent_fd)
            published = True
            unlink_temporary(temporary, parent_fd)
            temporary = None
        published = True
        try:
            fsync_directory(parent_fd)
        except OSError as exc:
            raise AtomicWriteError(
                f"published UVFITS but directory fsync failed: {final}"
            ) from exc
    except PartialCleanupError:
        raise
    except Exception as exc:
        if temporary is not None and parent_fd is not None:
            _cleanup_temporary(temporary, parent_fd, exc)
        if isinstance(exc, ResultIOError):
            raise
        if published:
            raise AtomicWriteError(
                f"atomic UVFITS publication completed with an error: {final}"
            ) from exc
        raise AtomicWriteError(
            f"atomic UVFITS transaction failed before publication: {final}"
        ) from exc
    finally:
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except OSError:
                pass
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass
    return final


def read_uvfits(
    path: str | Path,
    *,
    limits: StandardReadLimits = StandardReadLimits(),
) -> StandardVisibilityData:
    """Read and validate one bounded canonical UVFITS view."""
    if type(limits) is not StandardReadLimits:
        raise TypeError("limits must be an exact StandardReadLimits")
    try:
        source = validate_input_regular_file(path)
    except ResultIOError as exc:
        raise UnsafeResultInputError(str(exc)) from exc
    projection_record = _inspect_uvfits_headers(source, limits)
    _ = _import_pyuvdata()
    metadata = _read_uvfits(source, read_data=False)
    enforce_standard_read_limits(metadata, limits)
    validate_standard_metadata(metadata)
    loaded = _read_uvfits(source, read_data=True)
    try:
        return standard_visibility_from_uvdata(
            loaded,
            format="uvfits",
            expected_projection_record=projection_record,
        )
    except ResultIOError:
        raise
    except Exception as exc:
        raise UnsafeResultInputError(
            f"UVFITS failed canonical validation: {source}"
        ) from exc


__all__ = ["read_uvfits", "write_uvfits"]
