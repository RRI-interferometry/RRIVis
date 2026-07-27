"""Canonical Measurement Set projection, validation, and atomic publication."""

from __future__ import annotations

import os
import re
import stat
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from radiosim.core.result import SimulationResult
from radiosim.io.atomic_paths import (
    create_sibling_temporary_directory,
    exchange_directories,
    fsync_directory,
    open_parent_directory,
    publish_directory_no_clobber,
    remove_temporary_directory,
    require_atomic_directory_support,
    validate_input_directory,
    validate_output_directory_target,
)
from radiosim.io.result_errors import (
    AtomicWriteError,
    OptionalResultDependencyError,
    OutputCollisionError,
    PartialCleanupError,
    ResultIOError,
    UnsafeResultInputError,
)
from radiosim.io.standard_visibility import (
    StandardReadLimits,
    StandardVisibilityData,
    enforce_standard_read_limits,
    normalize_autocorrelations,
    project_simulation_result,
    standard_visibility_from_uvdata,
    validate_projection_result,
    validate_standard_metadata,
)

_COLUMN_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")


def _pyuvdata_version() -> str:
    try:
        return version("pyuvdata")
    except PackageNotFoundError:
        return "unavailable"


def _import_standard_dependencies() -> type[Any]:
    """Import MS dependencies only after pure operation preflight succeeds."""
    pyuvdata_version = _pyuvdata_version()
    try:
        pyuvdata = import_module("pyuvdata")
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalResultDependencyError(
            "format=ms missing_package=pyuvdata "
            f"pyuvdata_version={pyuvdata_version} "
            "install_extra=radiosim[ms]"
        ) from exc
    try:
        _ = import_module("casacore.tables")
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalResultDependencyError(
            "format=ms missing_package=python-casacore "
            f"pyuvdata_version={pyuvdata_version} "
            "install_extra=radiosim[ms]"
        ) from exc
    return pyuvdata.UVData


def _validate_data_column(value: object) -> str:
    if type(value) is not str or _COLUMN_NAME.fullmatch(value) is None:
        raise TypeError("data_column must be an uppercase Measurement Set column name")
    return value


def _read_ms(
    path: Path,
    *,
    data_column: str,
    read_data: bool,
) -> Any:
    uvdata_class = _import_standard_dependencies()
    try:
        if not read_data:
            return _read_ms_metadata(path, data_column=data_column)
        uvdata = uvdata_class()
        uvdata.read_ms(
            str(path),
            data_column=data_column,
            background_lsts=False,
            ignore_single_chan=False,
            run_check=True,
            fix_autos=False,
        )
    except ResultIOError:
        raise
    except Exception as exc:
        raise UnsafeResultInputError(
            f"could not read validated Measurement Set path: {path}"
        ) from exc
    return uvdata


def _table_column(table: Any, name: str) -> np.ndarray:
    if name not in table.colnames():
        raise UnsafeResultInputError(
            f"Measurement Set table lacks required column {name}"
        )
    return np.asarray(table.getcol(name))


def _read_ms_metadata(path: Path, *, data_column: str) -> SimpleNamespace:
    """Inspect bounded MS coordinates and subtables without reading DATA."""
    tables = import_module("casacore.tables")
    opened: list[Any] = []

    def open_table(location: Path) -> Any:
        handle = tables.table(
            str(location),
            readonly=True,
            ack=False,
        )
        opened.append(handle)
        return handle

    try:
        main = open_table(path)
        if data_column not in main.colnames():
            raise UnsafeResultInputError(
                f"Measurement Set lacks requested column {data_column}"
            )
        antenna1 = _table_column(main, "ANTENNA1").astype(np.int64, copy=False)
        antenna2 = _table_column(main, "ANTENNA2").astype(np.int64, copy=False)
        time_seconds = _table_column(main, "TIME").astype(
            np.float64,
            copy=False,
        )
        uvw = _table_column(main, "UVW").astype(np.float64, copy=False)
        exposures = _table_column(main, "EXPOSURE").astype(
            np.float64,
            copy=False,
        )
        if not (
            antenna1.shape
            == antenna2.shape
            == time_seconds.shape
            == exposures.shape
            == (int(main.nrows()),)
        ):
            raise UnsafeResultInputError(
                "Measurement Set MAIN metadata shapes are inconsistent"
            )
        if uvw.shape != (int(main.nrows()), 3):
            raise UnsafeResultInputError(
                "Measurement Set UVW metadata shape is invalid"
            )

        spectral = open_table(path / "SPECTRAL_WINDOW")
        if int(spectral.nrows()) != 1:
            frequency_array = np.empty(0, dtype=np.float64)
            width_array = np.empty(0, dtype=np.float64)
        else:
            frequency_array = np.asarray(
                spectral.getcell("CHAN_FREQ", 0),
                dtype=np.float64,
            )
            width_array = np.asarray(
                spectral.getcell("CHAN_WIDTH", 0),
                dtype=np.float64,
            )

        polarization = open_table(path / "POLARIZATION")
        if int(polarization.nrows()) != 1:
            polarization_array = np.empty(0, dtype=np.int64)
        else:
            casa_codes = np.asarray(
                polarization.getcell("CORR_TYPE", 0),
                dtype=np.int64,
            )
            casa_to_aips = {9: -5, 10: -7, 11: -8, 12: -6}
            polarization_array = np.array(
                [casa_to_aips.get(int(code), 0) for code in casa_codes],
                dtype=np.int64,
            )
        expected_shape = (
            f"[{int(frequency_array.size)}, {int(polarization_array.size)}]"
        )
        data_shapes = main.getcolshapestring(data_column)
        if len(data_shapes) != int(main.nrows()) or any(
            shape != expected_shape for shape in data_shapes
        ):
            raise UnsafeResultInputError(
                "Measurement Set DATA cell shapes disagree with its subtables"
            )

        antennas = open_table(path / "ANTENNA")
        all_antenna_names = _table_column(antennas, "NAME").astype(str)
        all_antenna_positions = _table_column(antennas, "POSITION").astype(
            np.float64,
            copy=False,
        )
        antenna_numbers = np.flatnonzero(all_antenna_names != "").astype(
            np.int64,
            copy=False,
        )
        used_antenna_numbers = np.unique(np.concatenate((antenna1, antenna2)))
        if (
            np.any(used_antenna_numbers < 0)
            or np.any(used_antenna_numbers >= int(antennas.nrows()))
            or not set(used_antenna_numbers.tolist()).issubset(
                set(antenna_numbers.tolist())
            )
        ):
            raise UnsafeResultInputError(
                "Measurement Set baselines reference unknown antennas"
            )
        antenna_names = all_antenna_names[antenna_numbers]
        antenna_positions = all_antenna_positions[antenna_numbers]

        field = open_table(path / "FIELD")
        phase_directions = _table_column(field, "PHASE_DIR").astype(
            np.float64,
            copy=False,
        )
        phase_keywords = field.getcolkeywords("PHASE_DIR")
        measure_info = phase_keywords.get("MEASINFO", {})
        reference: object = None
        if isinstance(measure_info, dict):
            reference = cast(dict[str, object], measure_info).get("Ref")
        if phase_directions.size == 2 and reference in {"ICRS", "J2000"}:
            longitude, latitude = phase_directions.reshape(-1)
            phase_catalog: dict[int, dict[str, object]] = {
                0: {
                    "cat_type": "sidereal",
                    "cat_frame": "icrs",
                    "cat_lon": float(longitude),
                    "cat_lat": float(latitude),
                }
            }
        else:
            phase_catalog = {}

        pairs = set(zip(antenna1.tolist(), antenna2.tolist(), strict=True))
        return SimpleNamespace(
            Ntimes=int(np.unique(time_seconds).size),
            Nbls=len(pairs),
            Nfreqs=int(frequency_array.size),
            Npols=int(polarization_array.size),
            Nblts=int(main.nrows()),
            Nspws=int(spectral.nrows()),
            polarization_array=polarization_array,
            freq_array=frequency_array,
            channel_width=width_array,
            phase_center_catalog=phase_catalog,
            time_array=time_seconds / 86400.0 + 2400000.5,
            integration_time=exposures,
            uvw_array=uvw,
            telescope=SimpleNamespace(
                Nants=int(antenna_numbers.size),
                antenna_numbers=antenna_numbers,
                antenna_names=antenna_names,
                antenna_positions=antenna_positions,
            ),
        )
    finally:
        for handle in reversed(opened):
            handle.close()


def _write_ms(uvdata: Any, path: Path, **kwargs: object) -> None:
    uvdata.write_ms(str(path), **kwargs)


def _assert_round_trip(
    expected: StandardVisibilityData,
    observed: StandardVisibilityData,
) -> None:
    """Require the declared MS storage contract after writer readback."""
    if observed.format != "ms":
        raise AtomicWriteError("temporary MS readback has the wrong format")
    for field_name in (
        "correlations",
        "source_scientific_sha256",
        "source_provenance_sha256",
    ):
        if getattr(observed, field_name) != getattr(expected, field_name):
            raise AtomicWriteError(f"temporary MS readback changed {field_name}")
    for field_name in (
        "flags",
        "antenna1_numbers",
        "antenna2_numbers",
    ):
        if not np.array_equal(
            getattr(observed, field_name),
            getattr(expected, field_name),
        ):
            raise AtomicWriteError(f"temporary MS readback changed {field_name}")
    comparisons = (
        ("visibilities", 5e-6, 1e-7),
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
            raise AtomicWriteError(f"temporary MS readback changed {field_name}")
    expected_jd = expected.utc_jd1 + expected.utc_jd2
    observed_jd = observed.utc_jd1 + observed.utc_jd2
    if not np.allclose(observed_jd, expected_jd, rtol=0.0, atol=5e-10):
        raise AtomicWriteError("temporary MS readback changed time coordinates")
    if observed.phase_center != expected.phase_center:
        raise AtomicWriteError("temporary MS readback changed projected phase")
    if not _telescope_metadata_equal(
        expected.telescope_snapshot,
        observed.telescope_snapshot,
    ):
        raise AtomicWriteError("temporary MS readback changed telescope metadata")


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


def _verify_temporary_measurement_set(
    expected: StandardVisibilityData,
    temporary: Path,
) -> None:
    metadata = _read_ms(temporary, data_column="DATA", read_data=False)
    enforce_standard_read_limits(metadata, StandardReadLimits())
    validate_standard_metadata(metadata)
    loaded = _read_ms(temporary, data_column="DATA", read_data=True)
    observed = standard_visibility_from_uvdata(loaded, format="ms")
    _assert_round_trip(expected, observed)


def _cleanup_temporary(temporary: Path, cause: BaseException) -> None:
    try:
        remove_temporary_directory(temporary)
    except Exception as cleanup_error:
        error = PartialCleanupError(temporary)
        error.add_note(f"cleanup failure: {cleanup_error!r}")
        raise error from cause


def write_measurement_set(
    result: SimulationResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Project and atomically publish one canonical result as Measurement Set."""
    typed = validate_projection_result(result, format_name="ms")
    _ = normalize_autocorrelations(typed)
    require_atomic_directory_support()
    final = validate_output_directory_target(
        path,
        extension=".ms",
        overwrite=overwrite,
    )
    _ = _import_standard_dependencies()
    projected = project_simulation_result(typed, format="ms")

    parent_fd: int | None = None
    temporary: Path | None = None
    temporary_fd: int | None = None
    published = False
    try:
        parent_fd = open_parent_directory(final.parent, create=True)
        temporary = create_sibling_temporary_directory(final, parent_fd)
        payload = temporary / "payload.ms"
        _write_ms(
            projected.uvdata,
            payload,
            clobber=False,
            force_phase=False,
        )
        _verify_temporary_measurement_set(projected.data, payload)
        temporary_fd = open_parent_directory(temporary, create=False)
        try:
            target_status = os.stat(
                final.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_status = None
        if target_status is not None and not stat.S_ISDIR(target_status.st_mode):
            raise OutputCollisionError(
                f"publication target is no longer a directory: {final}"
            )
        if overwrite and target_status is not None:
            exchange_directories(
                payload,
                final,
                parent_fd,
                source_parent_fd=temporary_fd,
            )
            published = True
        else:
            publish_directory_no_clobber(
                payload,
                final,
                parent_fd,
                source_parent_fd=temporary_fd,
            )
            published = True
        os.close(temporary_fd)
        temporary_fd = None
        try:
            fsync_directory(parent_fd)
        except OSError as exc:
            raise AtomicWriteError(
                f"published Measurement Set but directory fsync failed: {final}"
            ) from exc
        try:
            remove_temporary_directory(temporary)
        except Exception as exc:
            raise PartialCleanupError(temporary) from exc
        temporary = None
        try:
            fsync_directory(parent_fd)
        except OSError as exc:
            raise AtomicWriteError(
                f"published Measurement Set but temporary cleanup fsync failed: {final}"
            ) from exc
    except PartialCleanupError:
        raise
    except Exception as exc:
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except OSError:
                pass
            temporary_fd = None
        if temporary is not None:
            _cleanup_temporary(temporary, exc)
        if isinstance(exc, ResultIOError):
            raise
        if published:
            raise AtomicWriteError(
                f"atomic Measurement Set publication completed with an error: {final}"
            ) from exc
        raise AtomicWriteError(
            f"atomic Measurement Set transaction failed before publication: {final}"
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


def read_measurement_set(
    path: str | Path,
    *,
    data_column: str = "DATA",
    limits: StandardReadLimits = StandardReadLimits(),
) -> StandardVisibilityData:
    """Read and validate one bounded canonical Measurement Set view."""
    column = _validate_data_column(data_column)
    if type(limits) is not StandardReadLimits:
        raise TypeError("limits must be an exact StandardReadLimits")
    source = validate_input_directory(path)
    _ = _import_standard_dependencies()
    metadata = _read_ms(source, data_column=column, read_data=False)
    enforce_standard_read_limits(metadata, limits)
    validate_standard_metadata(metadata)
    loaded = _read_ms(source, data_column=column, read_data=True)
    try:
        return standard_visibility_from_uvdata(loaded, format="ms")
    except ResultIOError:
        raise
    except Exception as exc:
        raise UnsafeResultInputError(
            f"Measurement Set failed canonical validation: {source}"
        ) from exc


__all__ = ["read_measurement_set", "write_measurement_set"]
