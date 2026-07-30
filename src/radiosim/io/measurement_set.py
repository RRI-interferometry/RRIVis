"""Canonical Measurement Set projection, validation, and atomic publication."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Final, cast

import numpy as np

from radiosim.core.polarization_basis import (
    AIPS_CODES_CANONICAL,
    PolarizationBasis,
)
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
    projected_phase_from_uvdata,
    projection_record_from_history,
    standard_visibility_from_uvdata,
    validate_projection_result,
    validate_standard_metadata,
)

_COLUMN_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")

# The casacore ``Stokes`` enumeration runs in the same row-major correlation
# order as ``CORRELATION_LABELS``, starting at 5 for a circular basis and 9 for a
# linear one (Section 14.3): RR=5, RL=6, LR=7, LL=8, XX=9, XY=10, YX=11, YY=12.
# RadioSim never writes ``CORR_TYPE`` itself -- pyuvdata derives it -- so this
# map exists only to translate a read axis back onto the shared AIPS table.
_CASA_STOKES_FIRST: Final[Mapping[PolarizationBasis, int]] = MappingProxyType(
    {
        "circular_rl": 5,
        "linear_xy": 9,
    }
)
_CASA_TO_AIPS: Final[Mapping[int, int]] = MappingProxyType(
    {
        first + offset: AIPS_CODES_CANONICAL[basis][offset]
        for basis, first in _CASA_STOKES_FIRST.items()
        for offset in range(4)
    }
)
_MS_METADATA_CHUNK_ROWS = 4096
_MS_HISTORY_ROWS_LIMIT = 1024
_MS_HISTORY_STORAGE_LIMIT = 262_144
_MS_HISTORY_ENTRY_LIMIT = 4096


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
    limits: StandardReadLimits | None = None,
) -> Any:
    uvdata_class = _import_standard_dependencies()
    try:
        if not read_data:
            if type(limits) is not StandardReadLimits:
                raise TypeError("metadata inspection requires exact StandardReadLimits")
            return _read_ms_metadata(
                path,
                data_column=data_column,
                limits=limits,
            )
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


def _table_scalar(table: Any, name: str, *, row: int = 0) -> object:
    if name not in table.colnames():
        raise UnsafeResultInputError(
            f"Measurement Set table lacks required column {name}"
        )
    descriptor = table.getcoldesc(name)
    if descriptor.get("ndim") not in (None, 0):
        raise UnsafeResultInputError(f"Measurement Set column {name} must be scalar")
    return table.getcell(name, row)


def _table_cell_shape(table: Any, name: str, *, row: int = 0) -> str:
    if name not in table.colnames():
        raise UnsafeResultInputError(
            f"Measurement Set table lacks required column {name}"
        )
    shapes: object = table.getcolshapestring(name, row, 1, 1)
    if type(shapes) is not list:
        raise UnsafeResultInputError(
            f"Measurement Set column {name} has unsafe shape metadata"
        )
    typed_shapes = cast(list[object], shapes)
    if len(typed_shapes) != 1 or type(typed_shapes[0]) is not str:
        raise UnsafeResultInputError(
            f"Measurement Set column {name} has unsafe shape metadata"
        )
    return typed_shapes[0]


def _validate_ms_column_descriptor(
    table: Any,
    name: str,
    *,
    value_types: set[str],
    ndim: int,
    shape: tuple[int, ...] | None = None,
) -> None:
    if name not in table.colnames():
        raise UnsafeResultInputError(
            f"Measurement Set table lacks required column {name}"
        )
    raw_descriptor: object = table.getcoldesc(name)
    if not isinstance(raw_descriptor, dict):
        raise UnsafeResultInputError(
            f"Measurement Set column {name} has unsafe descriptor metadata"
        )
    descriptor = cast(dict[str, object], raw_descriptor)
    if descriptor.get("valueType") not in value_types:
        raise UnsafeResultInputError(
            f"Measurement Set column {name} has an unsupported value type"
        )
    descriptor_ndim = descriptor.get("ndim")
    if ndim == 0:
        if descriptor_ndim not in (None, 0):
            raise UnsafeResultInputError(
                f"Measurement Set column {name} must be scalar"
            )
    elif descriptor_ndim != ndim:
        raise UnsafeResultInputError(
            f"Measurement Set column {name} has an unsupported rank"
        )
    if shape is not None and descriptor.get("shape") is not None:
        declared_shape = descriptor.get("shape")
        try:
            shape_tuple = tuple(
                int(item) for item in np.asarray(declared_shape).reshape(-1)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise UnsafeResultInputError(
                f"Measurement Set column {name} has unsafe shape metadata"
            ) from exc
        if shape_tuple != shape:
            raise UnsafeResultInputError(
                f"Measurement Set column {name} has an unsupported shape"
            )


def _bounded_history_storage(path: Path) -> None:
    """Bound variable-length HISTORY storage before any string value access."""
    pending = [path]
    total = 0
    entries = 0
    while pending:
        directory = pending.pop()
        try:
            iterator = os.scandir(directory)
        except OSError as exc:
            raise UnsafeResultInputError(
                "Measurement Set HISTORY storage cannot be inspected safely"
            ) from exc
        with iterator:
            for entry in iterator:
                entries += 1
                if entries > _MS_HISTORY_ENTRY_LIMIT:
                    raise UnsafeResultInputError(
                        "Measurement Set HISTORY has too many storage entries"
                    )
                try:
                    status = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise UnsafeResultInputError(
                        "Measurement Set HISTORY storage cannot be inspected safely"
                    ) from exc
                if stat.S_ISLNK(status.st_mode):
                    raise UnsafeResultInputError(
                        "Measurement Set HISTORY storage contains a symbolic link"
                    )
                if stat.S_ISDIR(status.st_mode):
                    pending.append(Path(entry.path))
                    continue
                if not stat.S_ISREG(status.st_mode):
                    raise UnsafeResultInputError(
                        "Measurement Set HISTORY storage contains a special file"
                    )
                total += int(status.st_size)
                if total > _MS_HISTORY_STORAGE_LIMIT:
                    raise UnsafeResultInputError(
                        "Measurement Set HISTORY storage exceeds the bounded limit"
                    )


def _bounded_ms_history(
    path: Path,
    history: Any,
) -> tuple[dict[str, object], tuple[str, ...]]:
    _bounded_history_storage(path / "HISTORY")
    row_count = int(history.nrows())
    if row_count <= 0:
        raise UnsafeResultInputError("Measurement Set HISTORY is empty")
    if row_count > _MS_HISTORY_ROWS_LIMIT:
        raise UnsafeResultInputError(
            "Measurement Set HISTORY exceeds the bounded row limit"
        )
    if "MESSAGE" not in history.colnames():
        raise UnsafeResultInputError(
            "Measurement Set HISTORY lacks required MESSAGE column"
        )
    descriptor = history.getcoldesc("MESSAGE")
    if (
        descriptor.get("valueType") != "string"
        or descriptor.get("ndim") not in (None, 0)
        or descriptor.get("dataManagerType") != "StandardStMan"
    ):
        raise UnsafeResultInputError(
            "Measurement Set HISTORY MESSAGE storage is not safely bounded"
        )
    max_length = descriptor.get("maxlen", 0)
    if type(max_length) is not int or max_length < 0 or max_length > 16_000:
        raise UnsafeResultInputError(
            "Measurement Set HISTORY MESSAGE has an unsafe declared length"
        )
    messages: list[str] = []
    encoded_total = 0
    for row in range(row_count):
        message = history.getcell("MESSAGE", row)
        if type(message) is not str:
            raise UnsafeResultInputError(
                "Measurement Set HISTORY MESSAGE must be an exact string"
            )
        if "\x00" in message:
            raise UnsafeResultInputError("Measurement Set HISTORY MESSAGE contains NUL")
        try:
            encoded = message.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise UnsafeResultInputError(
                "Measurement Set HISTORY MESSAGE is not strict UTF-8"
            ) from exc
        encoded_total += len(encoded) + (1 if messages else 0)
        if encoded_total > 16_000:
            raise UnsafeResultInputError(
                "Measurement Set projection HISTORY exceeds 16000 UTF-8 bytes"
            )
        messages.append(message)
    record, _lines = projection_record_from_history("\n".join(messages))
    return record, tuple(messages)


def _chunked_column(
    table: Any,
    name: str,
    *,
    rows: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    if name not in table.colnames():
        raise UnsafeResultInputError(
            f"Measurement Set table lacks required column {name}"
        )
    chunks: list[np.ndarray] = []
    for start in range(0, rows, _MS_METADATA_CHUNK_ROWS):
        count = min(_MS_METADATA_CHUNK_ROWS, rows - start)
        chunks.append(
            np.asarray(
                table.getcol(name, start, count, 1),
                dtype=dtype,
            )
        )
    if not chunks:
        return np.empty(0, dtype=dtype)
    return np.concatenate(chunks, axis=0)


def _read_ms_metadata(
    path: Path,
    *,
    data_column: str,
    limits: StandardReadLimits,
) -> SimpleNamespace:
    """Inspect bounded MS coordinates and subtables without reading science."""
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
        main_rows = int(main.nrows())
        if main_rows <= 0:
            raise UnsafeResultInputError("Measurement Set MAIN has no rows")
        if main_rows > int(limits.max_times) * int(limits.max_baselines):
            raise UnsafeResultInputError(
                "Measurement Set MAIN rows exceed max_times * max_baselines"
            )
        spectral = open_table(path / "SPECTRAL_WINDOW")
        polarization = open_table(path / "POLARIZATION")
        antennas = open_table(path / "ANTENNA")
        field = open_table(path / "FIELD")
        history_table = open_table(path / "HISTORY")
        data_description = open_table(path / "DATA_DESCRIPTION")
        feed = open_table(path / "FEED")
        observation = open_table(path / "OBSERVATION")
        if int(spectral.nrows()) != 1:
            raise UnsafeResultInputError(
                "Measurement Set must contain one SPECTRAL_WINDOW row"
            )
        if int(polarization.nrows()) != 1:
            raise UnsafeResultInputError(
                "Measurement Set must contain one POLARIZATION row"
            )
        if int(field.nrows()) != 1:
            raise UnsafeResultInputError("Measurement Set must contain one FIELD row")
        if int(data_description.nrows()) != 1:
            raise UnsafeResultInputError(
                "Measurement Set must contain one DATA_DESCRIPTION row"
            )
        if int(observation.nrows()) != 1:
            raise UnsafeResultInputError(
                "Measurement Set must contain one OBSERVATION row"
            )
        antenna_rows = int(antennas.nrows())
        if antenna_rows <= 0:
            raise UnsafeResultInputError("Measurement Set ANTENNA table is empty")
        if antenna_rows > limits.max_antennas:
            raise UnsafeResultInputError("standard input exceeds max_antennas")
        feed_rows = int(feed.nrows())
        if feed_rows != antenna_rows:
            raise UnsafeResultInputError(
                "Measurement Set FEED must contain one row per ANTENNA row"
            )
        for column in (
            "SPECTRAL_WINDOW_ID",
            "POLARIZATION_ID",
        ):
            _validate_ms_column_descriptor(
                data_description,
                column,
                value_types={"int"},
                ndim=0,
            )
        _validate_ms_column_descriptor(
            data_description,
            "FLAG_ROW",
            value_types={"boolean"},
            ndim=0,
        )
        _validate_ms_column_descriptor(
            observation,
            "TIME_RANGE",
            value_types={"double"},
            ndim=1,
            shape=(2,),
        )
        _validate_ms_column_descriptor(
            observation,
            "FLAG_ROW",
            value_types={"boolean"},
            ndim=0,
        )
        for column in ("OBSERVER", "PROJECT", "TELESCOPE_NAME"):
            _validate_ms_column_descriptor(
                observation,
                column,
                value_types={"string"},
                ndim=0,
            )
        for column in (
            "ANTENNA_ID",
            "FEED_ID",
            "NUM_RECEPTORS",
            "SPECTRAL_WINDOW_ID",
        ):
            _validate_ms_column_descriptor(
                feed,
                column,
                value_types={"int"},
                ndim=0,
            )
        for column in ("TIME", "INTERVAL"):
            _validate_ms_column_descriptor(
                feed,
                column,
                value_types={"double"},
                ndim=0,
            )
        _validate_ms_column_descriptor(
            feed,
            "POSITION",
            value_types={"double"},
            ndim=1,
            shape=(3,),
        )
        _validate_ms_column_descriptor(
            feed,
            "POLARIZATION_TYPE",
            value_types={"string"},
            ndim=1,
        )
        _validate_ms_column_descriptor(
            feed,
            "POL_RESPONSE",
            value_types={"complex", "dcomplex"},
            ndim=2,
        )
        _validate_ms_column_descriptor(
            feed,
            "RECEPTOR_ANGLE",
            value_types={"double"},
            ndim=1,
        )
        if (
            _table_scalar(data_description, "SPECTRAL_WINDOW_ID") != 0
            or _table_scalar(data_description, "POLARIZATION_ID") != 0
            or _table_scalar(data_description, "FLAG_ROW") is not False
        ):
            raise UnsafeResultInputError(
                "Measurement Set DATA_DESCRIPTION row is unsupported"
            )
        for row in range(feed_rows):
            if (
                _table_cell_shape(feed, "POSITION", row=row) != "[3]"
                or _table_cell_shape(feed, "POLARIZATION_TYPE", row=row) != "[2]"
                or _table_cell_shape(feed, "POL_RESPONSE", row=row) != "[2, 2]"
                or _table_cell_shape(feed, "RECEPTOR_ANGLE", row=row) != "[2]"
            ):
                raise UnsafeResultInputError(
                    "Measurement Set FEED cell shapes are unsupported"
                )
        for column in (
            "ANTENNA1",
            "ANTENNA2",
            "FIELD_ID",
            "DATA_DESC_ID",
            "ARRAY_ID",
            "SCAN_NUMBER",
        ):
            _validate_ms_column_descriptor(
                main,
                column,
                value_types={"int"},
                ndim=0,
            )
        for column in ("TIME", "EXPOSURE"):
            _validate_ms_column_descriptor(
                main,
                column,
                value_types={"double"},
                ndim=0,
            )
        _validate_ms_column_descriptor(
            main,
            "UVW",
            value_types={"double"},
            ndim=1,
            shape=(3,),
        )
        _validate_ms_column_descriptor(
            main,
            data_column,
            value_types={"complex", "dcomplex"},
            ndim=2,
        )
        _validate_ms_column_descriptor(
            main,
            "FLAG",
            value_types={"boolean"},
            ndim=2,
        )
        if "WEIGHT_SPECTRUM" in main.colnames():
            weight_column = "WEIGHT_SPECTRUM"
            _validate_ms_column_descriptor(
                main,
                weight_column,
                value_types={"float", "double"},
                ndim=2,
            )
        else:
            weight_column = "WEIGHT"
            _validate_ms_column_descriptor(
                main,
                weight_column,
                value_types={"float", "double"},
                ndim=1,
            )
        _validate_ms_column_descriptor(
            spectral,
            "NUM_CHAN",
            value_types={"int"},
            ndim=0,
        )
        _validate_ms_column_descriptor(
            spectral,
            "CHAN_FREQ",
            value_types={"double"},
            ndim=1,
        )
        _validate_ms_column_descriptor(
            spectral,
            "CHAN_WIDTH",
            value_types={"double"},
            ndim=1,
        )
        _validate_ms_column_descriptor(
            polarization,
            "NUM_CORR",
            value_types={"int"},
            ndim=0,
        )
        _validate_ms_column_descriptor(
            polarization,
            "CORR_TYPE",
            value_types={"int"},
            ndim=1,
        )
        _validate_ms_column_descriptor(
            field,
            "PHASE_DIR",
            value_types={"double"},
            ndim=2,
        )
        frequency_count = _table_scalar(spectral, "NUM_CHAN")
        polarization_count = _table_scalar(polarization, "NUM_CORR")
        if type(frequency_count) is not int or frequency_count <= 0:
            raise UnsafeResultInputError(
                "Measurement Set has an invalid declared channel count"
            )
        if frequency_count > limits.max_frequencies:
            raise UnsafeResultInputError("standard input exceeds max_frequencies")
        if type(polarization_count) is not int or polarization_count != 4:
            raise UnsafeResultInputError(
                "Measurement Set must declare exactly four correlations"
            )
        expected_data_shape = (int(frequency_count), int(polarization_count))
        _validate_ms_column_descriptor(
            main,
            data_column,
            value_types={"complex", "dcomplex"},
            ndim=2,
            shape=expected_data_shape,
        )
        _validate_ms_column_descriptor(
            main,
            "FLAG",
            value_types={"boolean"},
            ndim=2,
            shape=expected_data_shape,
        )
        _validate_ms_column_descriptor(
            main,
            weight_column,
            value_types={"float", "double"},
            ndim=2 if weight_column == "WEIGHT_SPECTRUM" else 1,
            shape=(
                expected_data_shape
                if weight_column == "WEIGHT_SPECTRUM"
                else (int(polarization_count),)
            ),
        )
        expected_frequency_shape = f"[{frequency_count}]"
        if (
            _table_cell_shape(spectral, "CHAN_FREQ") != expected_frequency_shape
            or _table_cell_shape(spectral, "CHAN_WIDTH") != expected_frequency_shape
            or _table_cell_shape(polarization, "CORR_TYPE") != f"[{polarization_count}]"
            or _table_cell_shape(field, "PHASE_DIR") != "[1, 2]"
        ):
            raise UnsafeResultInputError(
                "Measurement Set subtable cell shapes disagree with declared counts"
            )
        antenna_name_descriptor = antennas.getcoldesc("NAME")
        antenna_position_descriptor = antennas.getcoldesc("POSITION")
        if (
            antenna_name_descriptor.get("valueType") != "string"
            or antenna_name_descriptor.get("ndim") not in (None, 0)
            or antenna_position_descriptor.get("valueType") != "double"
            or antenna_position_descriptor.get("ndim") != 1
            or tuple(
                int(item)
                for item in np.asarray(
                    antenna_position_descriptor.get("shape"),
                ).reshape(-1)
            )
            != (3,)
        ):
            raise UnsafeResultInputError(
                "Measurement Set ANTENNA column shapes are unsupported"
            )
        visibility_elements = (
            int(main_rows) * int(frequency_count) * int(polarization_count)
        )
        if visibility_elements > limits.max_visibility_elements:
            raise UnsafeResultInputError(
                "standard input exceeds max_visibility_elements"
            )
        potential_data_bytes = visibility_elements * (
            np.dtype("complex128").itemsize
            + np.dtype("bool").itemsize
            + np.dtype("float32").itemsize
        )
        if potential_data_bytes > limits.max_data_bytes:
            raise UnsafeResultInputError("standard input exceeds max_data_bytes")
        projection_record, history_messages = _bounded_ms_history(
            path,
            history_table,
        )

        unique_times: set[float] = set()
        unique_pairs: set[tuple[int, int]] = set()
        time_pair_rows: set[tuple[float, int, int]] = set()
        antenna1_chunks: list[np.ndarray] = []
        antenna2_chunks: list[np.ndarray] = []
        time_chunks: list[np.ndarray] = []
        expected_shape = f"[{frequency_count}, {polarization_count}]"
        expected_weight_shape = (
            expected_shape
            if weight_column == "WEIGHT_SPECTRUM"
            else f"[{polarization_count}]"
        )
        for start in range(0, main_rows, _MS_METADATA_CHUNK_ROWS):
            count = min(_MS_METADATA_CHUNK_ROWS, main_rows - start)
            antenna1_chunk = np.asarray(
                main.getcol("ANTENNA1", start, count, 1),
                dtype=np.int64,
            )
            antenna2_chunk = np.asarray(
                main.getcol("ANTENNA2", start, count, 1),
                dtype=np.int64,
            )
            time_chunk = np.asarray(
                main.getcol("TIME", start, count, 1),
                dtype=np.float64,
            )
            if (
                antenna1_chunk.shape != (count,)
                or antenna2_chunk.shape != (count,)
                or time_chunk.shape != (count,)
                or not np.all(np.isfinite(time_chunk))
            ):
                raise UnsafeResultInputError(
                    "Measurement Set MAIN identity metadata is malformed"
                )
            for time_value, first, second in zip(
                time_chunk.tolist(),
                antenna1_chunk.tolist(),
                antenna2_chunk.tolist(),
                strict=True,
            ):
                time_key = float(time_value)
                pair = (int(first), int(second))
                unique_times.add(time_key)
                unique_pairs.add(pair)
                key = (time_key, pair[0], pair[1])
                if key in time_pair_rows:
                    raise UnsafeResultInputError(
                        "Measurement Set contains a duplicate time-baseline row"
                    )
                time_pair_rows.add(key)
                if len(unique_times) > limits.max_times:
                    raise UnsafeResultInputError("standard input exceeds max_times")
                if len(unique_pairs) > limits.max_baselines:
                    raise UnsafeResultInputError("standard input exceeds max_baselines")
            shape_checks = (
                (data_column, expected_shape),
                ("FLAG", expected_shape),
                (weight_column, expected_weight_shape),
            )
            for column, expected in shape_checks:
                descriptor = main.getcoldesc(column)
                if descriptor.get("shape") is None:
                    shapes = main.getcolshapestring(column, start, count, 1)
                    if len(shapes) != count or any(
                        shape != expected for shape in shapes
                    ):
                        raise UnsafeResultInputError(
                            f"Measurement Set {column} cell shapes disagree with "
                            "its subtables"
                        )
            antenna1_chunks.append(antenna1_chunk)
            antenna2_chunks.append(antenna2_chunk)
            time_chunks.append(time_chunk)
        if main_rows != len(unique_times) * len(unique_pairs):
            raise UnsafeResultInputError(
                "Measurement Set is not rectangular time-by-baseline data"
            )
        if len(time_pair_rows) != main_rows:
            raise UnsafeResultInputError(
                "Measurement Set has incomplete rectangular coverage"
            )

        antenna1 = np.concatenate(antenna1_chunks)
        antenna2 = np.concatenate(antenna2_chunks)
        time_seconds = np.concatenate(time_chunks)
        uvw = _chunked_column(
            main,
            "UVW",
            rows=main_rows,
            dtype=np.dtype("float64"),
        )
        exposures = _chunked_column(
            main,
            "EXPOSURE",
            rows=main_rows,
            dtype=np.dtype("float64"),
        )
        if not (
            antenna1.shape
            == antenna2.shape
            == time_seconds.shape
            == exposures.shape
            == (main_rows,)
        ) or uvw.shape != (main_rows, 3):
            raise UnsafeResultInputError(
                "Measurement Set MAIN metadata shapes are inconsistent"
            )

        frequency_array = np.asarray(
            spectral.getcell("CHAN_FREQ", 0),
            dtype=np.float64,
        )
        width_array = np.asarray(
            spectral.getcell("CHAN_WIDTH", 0),
            dtype=np.float64,
        )
        if frequency_array.shape != (frequency_count,) or width_array.shape != (
            frequency_count,
        ):
            raise UnsafeResultInputError(
                "Measurement Set spectral metadata shapes are inconsistent"
            )
        casa_codes = np.asarray(
            polarization.getcell("CORR_TYPE", 0),
            dtype=np.int64,
        )
        if casa_codes.shape != (polarization_count,):
            raise UnsafeResultInputError(
                "Measurement Set polarization metadata shape is inconsistent"
            )
        polarization_array = np.array(
            [_CASA_TO_AIPS.get(int(code), 0) for code in casa_codes],
            dtype=np.int64,
        )

        all_antenna_names = _table_column(antennas, "NAME").astype(str)
        all_antenna_positions = _table_column(antennas, "POSITION").astype(
            np.float64,
            copy=False,
        )
        if all_antenna_names.shape != (antenna_rows,) or (
            all_antenna_positions.shape != (antenna_rows, 3)
        ):
            raise UnsafeResultInputError(
                "Measurement Set ANTENNA metadata shapes are inconsistent"
            )
        antenna_numbers = np.flatnonzero(all_antenna_names != "").astype(
            np.int64,
            copy=False,
        )
        used_antenna_numbers = np.unique(np.concatenate((antenna1, antenna2)))
        if (
            np.any(used_antenna_numbers < 0)
            or np.any(used_antenna_numbers >= antenna_rows)
            or not set(used_antenna_numbers.tolist()).issubset(
                set(antenna_numbers.tolist())
            )
        ):
            raise UnsafeResultInputError(
                "Measurement Set baselines reference unknown antennas"
            )
        antenna_names = all_antenna_names[antenna_numbers]
        antenna_positions = all_antenna_positions[antenna_numbers]

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

        metadata = SimpleNamespace(
            Ntimes=len(unique_times),
            Nbls=len(unique_pairs),
            Nfreqs=int(frequency_array.size),
            Npols=int(polarization_array.size),
            Nblts=main_rows,
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
            history="\n".join(history_messages),
            projection_record=projection_record,
        )
        _ = projected_phase_from_uvdata(metadata, projection_record)
        return metadata
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
    metadata = _read_ms(
        temporary,
        data_column="DATA",
        read_data=False,
        limits=StandardReadLimits(),
    )
    enforce_standard_read_limits(metadata, StandardReadLimits())
    validate_standard_metadata(metadata)
    loaded = _read_ms(temporary, data_column="DATA", read_data=True)
    observed = standard_visibility_from_uvdata(
        loaded,
        format="ms",
        expected_projection_record=metadata.projection_record,
    )
    _assert_round_trip(expected, observed)


def _cleanup_temporary(temporary: Path, cause: BaseException) -> None:
    try:
        remove_temporary_directory(temporary)
    except Exception as cleanup_error:
        error = PartialCleanupError(temporary)
        error.add_note(f"cleanup failure: {cleanup_error!r}")
        raise error from cause


def _cleanup_published_directory(
    temporary: Path,
    *,
    old_measurement_set: Path | None,
) -> None:
    """Remove post-publication residue while reporting its exact live path."""
    if old_measurement_set is not None:
        try:
            remove_temporary_directory(old_measurement_set)
        except Exception as exc:
            if old_measurement_set.exists():
                error = PartialCleanupError(old_measurement_set)
                error.add_note(
                    "old Measurement Set cleanup was incomplete; the reported "
                    "residual may be only partially intact"
                )
                raise error from exc
    try:
        remove_temporary_directory(temporary)
    except Exception as exc:
        if temporary.exists():
            raise PartialCleanupError(temporary) from exc


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
    old_measurement_set: Path | None = None
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
            old_measurement_set = payload
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
            if old_measurement_set is not None and old_measurement_set.exists():
                error = PartialCleanupError(old_measurement_set)
                error.add_note(
                    "parent-directory fsync failed after exchange; the old "
                    "Measurement Set was retained at the reported path"
                )
                raise error from exc
            raise AtomicWriteError(
                f"published Measurement Set but directory fsync failed: {final}"
            ) from exc
        _cleanup_published_directory(
            temporary,
            old_measurement_set=old_measurement_set,
        )
        temporary = None
        old_measurement_set = None
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
    metadata = _read_ms(
        source,
        data_column=column,
        read_data=False,
        limits=limits,
    )
    enforce_standard_read_limits(metadata, limits)
    validate_standard_metadata(metadata)
    loaded = _read_ms(source, data_column=column, read_data=True)
    try:
        return standard_visibility_from_uvdata(
            loaded,
            format="ms",
            expected_projection_record=metadata.projection_record,
        )
    except ResultIOError:
        raise
    except Exception as exc:
        raise UnsafeResultInputError(
            f"Measurement Set failed canonical validation: {source}"
        ) from exc


__all__ = ["read_measurement_set", "write_measurement_set"]
