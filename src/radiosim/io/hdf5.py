"""Versioned, bounded, and atomically published RadioSim HDF5 results."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from radiosim.core.phase_center import PhaseCenter
from radiosim.core.polarization_basis import (
    AIPS_CODES_CANONICAL,
    CORRELATION_LABELS,
    POLARIZATION_BASES,
    PolarizationBasis,
)
from radiosim.core.result import (
    MMODE_SOLVER_SNAPSHOT_KEYS,
    InvalidResultError,
    LoadedSimulationResult,
    ResultPerformance,
    SimulationResult,
    SolverResultProvenance,
    build_loaded_simulation_result,
)
from radiosim.core.time_grid import (
    ObservationTimeGrid,
    build_observation_time_grid,
)
from radiosim.io.atomic_paths import (
    create_sibling_temporary as _create_sibling_temporary,
)
from radiosim.io.atomic_paths import fsync_directory as _fsync_directory
from radiosim.io.atomic_paths import fsync_file as _fsync_file
from radiosim.io.atomic_paths import open_parent_directory as _open_parent_directory
from radiosim.io.atomic_paths import publish_no_clobber as _publish_no_clobber
from radiosim.io.atomic_paths import publish_replace as _publish_replace
from radiosim.io.atomic_paths import unlink_temporary as _unlink_temporary
from radiosim.io.atomic_paths import (
    validate_input_regular_file as _validate_input_regular_file,
)
from radiosim.io.atomic_paths import validate_output_target as _validate_output_target
from radiosim.io.result_errors import (
    AtomicWriteError,
    FormatRepresentationError,
    LegacyHDF5Error,
    OptionalResultDependencyError,
    OutputPathError,
    PartialCleanupError,
    ResultIOError,
    UnsafeResultInputError,
    UnsupportedSchemaVersionError,
)

SCHEMA_NAME: Final = "radiosim.visibility"
SCHEMA_VERSION: Final = "4.0.0"
DIMENSION_ORDER: Final = "time,baseline,frequency,correlation"
VISIBILITY_UNIT: Final = "Jy"
_RECEPTOR_BASES: Final = ("linear", "circular")
HDF5_SIGNATURE: Final = b"\x89HDF\r\n\x1a\n"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_ATTRIBUTES: Final = {
    "schema_name",
    "schema_version",
    "radiosim_version",
    "scientific_sha256",
    "provenance_sha256",
    "dimension_order",
    "visibility_unit",
}
_ROOT_ATTRIBUTE_READ_ORDER: Final = (
    "schema_name",
    "schema_version",
    "radiosim_version",
    "scientific_sha256",
    "provenance_sha256",
    "dimension_order",
    "visibility_unit",
)
_GROUPS: Final = {
    "coordinates",
    "coordinates/baseline",
    "coordinates/correlation",
    "coordinates/frequency",
    "coordinates/time",
    "data",
    "instrument",
    "instrument/antenna",
    "instrument/location",
    "phase_center",
    "provenance",
    "receptors",
}
#: Groups a conforming file may omit.  ``jones/`` is the only one: a run that
#: configured no optional Jones or baseline term has nothing to record there,
#: while solver-owned ``H``, ``C``, and ``E`` remain represented elsewhere;
#: ``Tier7JonesSciencePlan.md`` Section 25.2 requires a reader to accept such a
#: file by treating it as "no optional terms enabled" rather than as corrupt.
#: Every other group is mandatory, and the allowlist is still exact -- a file
#: may omit this group entirely or carry all of it, never part of it.
_OPTIONAL_GROUPS: Final = {"jones"}
_DATASETS: Final = {
    "coordinates/baseline/antenna1_number",
    "coordinates/baseline/antenna2_number",
    "coordinates/baseline/vector_enu_m",
    "coordinates/correlation/aips_codes",
    "coordinates/correlation/basis",
    "coordinates/correlation/labels",
    "coordinates/frequency/center_hz",
    "coordinates/frequency/channel_width_hz",
    "coordinates/time/integration_time_seconds",
    "coordinates/time/utc_jd1",
    "coordinates/time/utc_jd2",
    "data/flags",
    "data/visibilities",
    "data/weights",
    "instrument/antenna/diameter_m",
    "instrument/antenna/name",
    "instrument/antenna/number",
    "instrument/antenna/position_enu_m",
    "instrument/location/geodetic_lon_lat_height",
    "instrument/location/itrs_xyz_m",
    "instrument/name",
    "phase_center/altitude_rad",
    "phase_center/azimuth_rad",
    "phase_center/frame",
    "phase_center/geometric_phase_sign",
    "phase_center/kind",
    "phase_center/time_dependent",
    "phase_center/w_reference",
    "provenance/backend_json",
    "provenance/beam_json",
    "provenance/configuration_source_json",
    "provenance/history_json",
    "provenance/instrument_json",
    "provenance/performance_json",
    "provenance/resolved_config_json",
    "provenance/selection_json",
    "provenance/solver_json",
    "receptors/antenna_name",
    "receptors/antenna_number",
    "receptors/basis",
    "receptors/feed_angle_rad",
    "receptors/feed_rotation_rad",
    "receptors/output_basis",
    "receptors/receptor_sha256",
}
_JONES_DATASETS: Final = {
    "jones/chain_order",
    "jones/enabled_terms",
    "jones/jones_sha256",
    "jones/mount_types_json",
    "jones/term_snapshots_json",
}
_JSON_PATHS: Final = {
    "provenance/instrument_json": "instrument",
    "provenance/selection_json": "selection",
    "provenance/beam_json": "beam",
    "provenance/backend_json": "backend",
    "provenance/solver_json": "solver",
    "provenance/resolved_config_json": "resolved_config",
    "provenance/configuration_source_json": "configuration_source",
    "provenance/performance_json": "performance",
    "provenance/history_json": "history",
}
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
#: A schema-``4.0.0`` result solves at most the two components of
#: ``Tier6HybridRuntimePlan.md`` Section 8.3, so a longer component list is
#: forged.  The bound is checked before the sequence is walked, so a hostile
#: file cannot make the reader iterate an attacker-sized list.
_MAX_SOLVED_COMPONENTS: Final = 2


@dataclass(frozen=True, slots=True)
class HDF5ReadLimits:
    """Pre-allocation limits for untrusted ``radiosim.visibility`` files."""

    max_time: int = 10_000_000
    max_baseline: int = 10_000_000
    max_frequency: int = 1_000_000
    max_antenna: int = 1_000_000
    max_visibility_elements: int = 100_000_000
    max_single_dataset_bytes: int = 2_147_483_648
    max_total_json_bytes: int = 67_108_864
    max_single_string_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int:
                raise TypeError(f"{field.name} must be an exact built-in integer")
            if value <= 0:
                raise ValueError(f"{field.name} must be positive")


@dataclass(frozen=True, slots=True)
class _DatasetSpec:
    shape: tuple[int, ...]
    dtype: str
    dimensions: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...] = ()
    storage: str = "plain"


@dataclass(frozen=True, slots=True)
class _PreparedResult:
    loaded: LoadedSimulationResult
    instrument: dict[str, object]
    selection: dict[str, object]
    beam: dict[str, object]
    backend: dict[str, object]
    solver: dict[str, object]
    resolved_config: dict[str, object]
    configuration_source: dict[str, object] | None
    performance: dict[str, object]
    history: list[str]
    text_payloads: dict[str, bytes | tuple[bytes, ...]]


def _import_h5py() -> Any:
    try:
        return import_module("h5py")
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalResultDependencyError(
            "HDF5 result I/O requires the installed h5py dependency"
        ) from exc


def _package_version() -> str:
    try:
        return version("radiosim")
    except PackageNotFoundError:
        return "unknown"


def _json_tree(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_tree(item)
            for key, item in cast(Mapping[object, object], value).items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_tree(item) for item in cast(Sequence[object], value)]
    if isinstance(value, np.generic):
        return cast(object, value.item())
    return value


def _mapping_tree(value: object, *, field_name: str) -> dict[str, object]:
    converted = _json_tree(value)
    if type(converted) is not dict:
        raise TypeError(f"{field_name} must be a JSON mapping")
    return cast(dict[str, object], converted)


def _contains_nul(value: object) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, Mapping):
        return any(
            _contains_nul(key) or _contains_nul(item)
            for key, item in cast(Mapping[object, object], value).items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_nul(item) for item in cast(Sequence[object], value))
    return False


def _encode_text(value: object, *, field_name: str) -> bytes:
    if type(value) is not str:
        raise FormatRepresentationError(f"{field_name} must be text")
    try:
        payload = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise FormatRepresentationError(f"{field_name} is not strict UTF-8") from exc
    if b"\x00" in payload:
        raise FormatRepresentationError(f"{field_name} contains a NUL byte")
    return payload


def _encode_json(value: object, *, field_name: str) -> bytes:
    converted = _json_tree(value)
    if _contains_nul(converted):
        raise FormatRepresentationError(f"{field_name} contains a NUL byte")
    try:
        encoded = json.dumps(
            converted,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        )
        payload = encoded.encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FormatRepresentationError(
            f"{field_name} cannot be represented as finite UTF-8 JSON"
        ) from exc
    return payload


def _jones_text_payloads(snapshot: Mapping[str, object]) -> dict[str, bytes]:
    """Encode one non-empty Jones snapshot into the ``jones/`` group's datasets.

    ``enabled_terms`` and ``chain_order`` are stored as JSON arrays rather than
    as variable-length string datasets, so the whole group is five *scalar*
    datasets whose shapes do not depend on how many terms a run enabled.  A
    shape that varies with the science is a shape a reader has to be told about
    before it can allocate, and there is nothing here worth that.
    """
    return {
        "jones/enabled_terms": _encode_json(
            list(cast(Sequence[str], snapshot["enabled_terms"])),
            field_name="jones/enabled_terms",
        ),
        "jones/chain_order": _encode_json(
            list(cast(Sequence[str], snapshot["chain_order"])),
            field_name="jones/chain_order",
        ),
        "jones/term_snapshots_json": _encode_json(
            snapshot["term_snapshots"],
            field_name="jones/term_snapshots_json",
        ),
        "jones/mount_types_json": _encode_json(
            snapshot["mount_types"],
            field_name="jones/mount_types_json",
        ),
        "jones/jones_sha256": _encode_text(
            cast(str, snapshot["jones_sha256"]),
            field_name="jones/jones_sha256",
        ),
    }


def _prepare_text_payloads(
    result: SimulationResult,
    instrument: dict[str, object],
    selection: dict[str, object],
    beam: dict[str, object],
    backend: dict[str, object],
    solver: dict[str, object],
    resolved_config: dict[str, object],
    configuration_source: dict[str, object] | None,
    performance: dict[str, object],
    history: list[str],
) -> dict[str, bytes | tuple[bytes, ...]]:
    payloads: dict[str, bytes | tuple[bytes, ...]] = {
        "instrument/name": _encode_text(
            result.instrument.name,
            field_name="instrument/name",
        ),
        "instrument/antenna/name": tuple(
            _encode_text(
                antenna.id.name,
                field_name=f"instrument/antenna/name[{index}]",
            )
            for index, antenna in enumerate(result.instrument.antennas)
        ),
        "phase_center/kind": _encode_text(
            result.phase_center.kind,
            field_name="phase_center/kind",
        ),
        "phase_center/frame": _encode_text(
            result.phase_center.frame,
            field_name="phase_center/frame",
        ),
        "phase_center/w_reference": _encode_text(
            result.phase_center.w_reference,
            field_name="phase_center/w_reference",
        ),
        "coordinates/correlation/basis": _encode_text(
            result.polarization_basis,
            field_name="coordinates/correlation/basis",
        ),
        "receptors/output_basis": _encode_text(
            result.receptors.output_basis,
            field_name="receptors/output_basis",
        ),
        "receptors/receptor_sha256": _encode_text(
            result.receptors.provenance.receptor_sha256,
            field_name="receptors/receptor_sha256",
        ),
        "receptors/antenna_name": tuple(
            _encode_text(
                antenna.id.name,
                field_name=f"receptors/antenna_name[{index}]",
            )
            for index, antenna in enumerate(result.instrument.antennas)
        ),
        "receptors/basis": tuple(
            _encode_text(
                result.receptors.receptor_by_antenna[antenna.id].basis,
                field_name=f"receptors/basis[{index}]",
            )
            for index, antenna in enumerate(result.instrument.antennas)
        ),
    }
    jones_snapshot = dict(result.jones)
    if jones_snapshot:
        payloads.update(_jones_text_payloads(jones_snapshot))
    provenance_values = {
        "instrument_json": instrument,
        "selection_json": selection,
        "beam_json": beam,
        "backend_json": backend,
        "solver_json": solver,
        "resolved_config_json": resolved_config,
        "configuration_source_json": configuration_source,
        "performance_json": performance,
        "history_json": history,
    }
    payloads.update(
        {
            f"provenance/{name}": _encode_json(value, field_name=name)
            for name, value in provenance_values.items()
        }
    )
    return payloads


def _prepare_result(result: object) -> _PreparedResult:
    if type(result) is not SimulationResult:
        raise TypeError("result must be an exact SimulationResult")
    typed = result
    if typed.schema_version != "radiosim.result.v1":
        raise FormatRepresentationError("result schema is not radiosim.result.v1")
    dtype = typed.visibilities.dtype
    if dtype.kind == "c" and dtype.itemsize == 32:
        raise FormatRepresentationError(
            "complex256 is not representable in radiosim.visibility HDF5 4.0.0"
        )
    if dtype not in {np.dtype("complex64"), np.dtype("complex128")}:
        raise FormatRepresentationError(
            "HDF5 4.0.0 requires complex64 or complex128 visibilities"
        )
    instrument = _mapping_tree(
        typed.instrument.to_snapshot(),
        field_name="instrument snapshot",
    )
    selection = _mapping_tree(
        typed.selection.to_snapshot(),
        field_name="selection snapshot",
    )
    beam = _mapping_tree(typed.beam_state.to_snapshot(), field_name="beam snapshot")
    backend = _mapping_tree(typed.backend.to_snapshot(), field_name="backend snapshot")
    solver = _mapping_tree(typed.solver.to_snapshot(), field_name="solver snapshot")
    resolved_config = _mapping_tree(
        typed.resolved_config,
        field_name="resolved configuration",
    )
    configuration_source = (
        None
        if typed.configuration_provenance is None
        else _mapping_tree(
            typed.configuration_provenance,
            field_name="configuration provenance",
        )
    )
    performance = _mapping_tree(
        typed.performance.to_snapshot(),
        field_name="performance snapshot",
    )
    history = list(typed.history)
    try:
        loaded = build_loaded_simulation_result(
            visibilities=typed.visibilities,
            flags=typed.flags,
            weights=typed.weights,
            time_grid=typed.time_grid,
            frequencies_hz=typed.frequencies_hz,
            channel_widths_hz=typed.channel_widths_hz,
            correlations=typed.correlations,
            phase_center=typed.phase_center,
            instrument_snapshot=instrument,
            selection_snapshot=selection,
            beam_snapshot=beam,
            receptors_snapshot=typed.receptors.to_snapshot(),
            jones_snapshot=dict(typed.jones) or None,
            backend_snapshot=backend,
            solver_snapshot=solver,
            resolved_config_snapshot=resolved_config,
            configuration_provenance_snapshot=configuration_source,
            performance_snapshot=performance,
            history=history,
            expected_scientific_sha256=typed.scientific_sha256,
            expected_provenance_sha256=typed.provenance_sha256,
        )
    except (TypeError, ValueError, InvalidResultError) as exc:
        raise FormatRepresentationError(
            "SimulationResult failed canonical HDF5 preflight validation"
        ) from exc
    if not loaded.scientifically_equal(typed):
        raise FormatRepresentationError(
            "SimulationResult failed canonical scientific equality preflight"
        )
    text_payloads = _prepare_text_payloads(
        typed,
        instrument,
        selection,
        beam,
        backend,
        solver,
        resolved_config,
        configuration_source,
        performance,
        history,
    )
    return _PreparedResult(
        loaded=loaded,
        instrument=instrument,
        selection=selection,
        beam=beam,
        backend=backend,
        solver=solver,
        resolved_config=resolved_config,
        configuration_source=configuration_source,
        performance=performance,
        history=history,
        text_payloads=text_payloads,
    )


def _create_groups(handle: Any) -> None:
    for path in (
        "data",
        "coordinates",
        "coordinates/time",
        "coordinates/frequency",
        "coordinates/correlation",
        "coordinates/baseline",
        "instrument",
        "instrument/antenna",
        "instrument/location",
        "phase_center",
        "provenance",
        "receptors",
    ):
        handle.create_group(path, track_order=True)


def _set_dimensions(dataset: Any, dimensions: tuple[str, ...]) -> None:
    for axis, label in enumerate(dimensions):
        dataset.dims[axis].label = label


def _create_dataset(
    handle: Any,
    path: str,
    *,
    shape: tuple[int, ...],
    dtype: object,
    options: Mapping[str, object],
) -> Any:
    return handle.create_dataset(path, shape=shape, dtype=dtype, **options)


def _write_dataset_payload(dataset: Any, payload: object) -> None:
    dataset[...] = payload


def _write_fixed_string_payload(
    dataset: Any,
    h5py: Any,
    payload: np.ndarray,
) -> None:
    if payload.size == 0:
        return
    memory_type = h5py.h5t.C_S1.copy()
    file_space = dataset.id.get_space()
    memory_space = (
        h5py.h5s.create(h5py.h5s.SCALAR)
        if payload.shape == ()
        else h5py.h5s.create_simple(payload.shape)
    )
    try:
        memory_type.set_size(int(payload.dtype.itemsize))
        memory_type.set_cset(h5py.h5t.CSET_UTF8)
        memory_type.set_strpad(h5py.h5t.STR_NULLPAD)
        dataset.id.write(memory_space, file_space, payload, mtype=memory_type)
    finally:
        memory_space.close()
        file_space.close()
        memory_type.close()


def _set_attribute(owner: Any, name: str, value: object) -> None:
    owner.attrs[name] = value


def _set_root_text_attribute(
    owner: Any,
    h5py: Any,
    name: str,
    value: object,
) -> None:
    if type(value) is not str:
        raise FormatRepresentationError(f"root attribute {name} must be text")
    try:
        payload = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise FormatRepresentationError(
            f"root attribute {name} is not strict UTF-8"
        ) from exc
    if not payload or b"\x00" in payload:
        raise FormatRepresentationError(
            f"root attribute {name} violates the text contract"
        )
    dtype = h5py.string_dtype(encoding="utf-8", length=len(payload))
    owner.attrs.create(
        name,
        np.bytes_(payload),
        shape=(),
        dtype=dtype,
    )


def _numeric_dataset(
    handle: Any,
    path: str,
    data: object,
    *,
    spec: _DatasetSpec,
) -> Any:
    array = np.asarray(data, dtype=np.dtype(spec.dtype))
    options: dict[str, object] = {}
    if spec.storage.startswith("science"):
        options.update(
            chunks=(
                min(array.shape[0], 16),
                min(array.shape[1], 64),
                min(array.shape[2], 64),
                4,
            ),
            compression="gzip",
            compression_opts=4,
            shuffle=spec.storage != "science_flags",
            fletcher32=True,
        )
    elif spec.storage == "coordinate":
        options["chunks"] = (
            (min(array.shape[0], 4096),)
            if array.ndim == 1
            else (min(array.shape[0], 4096), array.shape[1])
        )
        options["fletcher32"] = True
    dataset = _create_dataset(
        handle,
        path,
        shape=tuple(array.shape),
        dtype=np.dtype(spec.dtype),
        options=options,
    )
    _write_dataset_payload(dataset, array)
    _set_dimensions(dataset, spec.dimensions)
    for key, value in spec.attributes:
        _set_attribute(dataset, key, value)
    return dataset


def _string_dataset(
    handle: Any,
    h5py: Any,
    path: str,
    data: bytes | Sequence[bytes],
    *,
    spec: _DatasetSpec,
) -> Any:
    if isinstance(data, bytes):
        width = max(1, len(data))
        payload = np.asarray(data, dtype=np.dtype(f"S{width}"))
        shape: tuple[int, ...] = ()
    else:
        values = tuple(data)
        width = max(1, max((len(value) for value in values), default=0))
        payload = np.asarray(values, dtype=np.dtype(f"S{width}"))
        shape = (len(values),)
    dtype = h5py.string_dtype(encoding="utf-8", length=width)
    dataset = _create_dataset(
        handle,
        path,
        shape=shape,
        dtype=dtype,
        options={},
    )
    _write_fixed_string_payload(dataset, h5py, payload)
    _set_dimensions(dataset, spec.dimensions)
    return dataset


def _write_hdf5_content(
    handle: Any,
    h5py: Any,
    result: SimulationResult,
    prepared: _PreparedResult,
) -> None:
    time_count, baseline_count, frequency_count, _correlation_count = (
        int(value) for value in result.visibilities.shape
    )
    specs = _metadata_specs(
        time_count=time_count,
        baseline_count=baseline_count,
        frequency_count=frequency_count,
        antenna_count=len(result.instrument.antennas),
        visibility_dtype=("<c8" if result.visibilities.dtype.itemsize == 8 else "<c16"),
    )
    _create_groups(handle)
    for key, value in (
        ("schema_name", SCHEMA_NAME),
        ("schema_version", SCHEMA_VERSION),
        ("radiosim_version", _package_version()),
        ("scientific_sha256", result.scientific_sha256),
        ("provenance_sha256", result.provenance_sha256),
        ("dimension_order", DIMENSION_ORDER),
        ("visibility_unit", VISIBILITY_UNIT),
    ):
        _set_root_text_attribute(handle, h5py, key, value)

    _numeric_dataset(
        handle,
        "data/visibilities",
        result.visibilities,
        spec=specs["data/visibilities"],
    )
    _numeric_dataset(
        handle,
        "data/flags",
        result.flags,
        spec=specs["data/flags"],
    )
    _numeric_dataset(
        handle,
        "data/weights",
        result.weights,
        spec=specs["data/weights"],
    )
    for path, values in (
        ("coordinates/time/utc_jd1", result.time_grid.utc_jd1),
        ("coordinates/time/utc_jd2", result.time_grid.utc_jd2),
        (
            "coordinates/time/integration_time_seconds",
            result.time_grid.integration_time_seconds,
        ),
    ):
        _numeric_dataset(
            handle,
            path,
            values,
            spec=specs[path],
        )
    for path, values in (
        ("coordinates/frequency/center_hz", result.frequencies_hz),
        (
            "coordinates/frequency/channel_width_hz",
            result.channel_widths_hz,
        ),
    ):
        _numeric_dataset(
            handle,
            path,
            values,
            spec=specs[path],
        )
    basis = result.polarization_basis
    _numeric_dataset(
        handle,
        "coordinates/correlation/labels",
        CORRELATION_LABELS[basis],
        spec=specs["coordinates/correlation/labels"],
    )
    _numeric_dataset(
        handle,
        "coordinates/correlation/aips_codes",
        AIPS_CODES_CANONICAL[basis],
        spec=specs["coordinates/correlation/aips_codes"],
    )
    _string_dataset(
        handle,
        h5py,
        "coordinates/correlation/basis",
        cast(bytes, prepared.text_payloads["coordinates/correlation/basis"]),
        spec=specs["coordinates/correlation/basis"],
    )
    antenna1 = [baseline.ant1.number for baseline in result.selection.baselines]
    antenna2 = [baseline.ant2.number for baseline in result.selection.baselines]
    vectors = [baseline.vector_enu_m for baseline in result.selection.baselines]
    for path, values in (
        ("coordinates/baseline/antenna1_number", antenna1),
        ("coordinates/baseline/antenna2_number", antenna2),
    ):
        _numeric_dataset(
            handle,
            path,
            values,
            spec=specs[path],
        )
    _numeric_dataset(
        handle,
        "coordinates/baseline/vector_enu_m",
        vectors,
        spec=specs["coordinates/baseline/vector_enu_m"],
    )

    _string_dataset(
        handle,
        h5py,
        "instrument/name",
        prepared.text_payloads["instrument/name"],
        spec=specs["instrument/name"],
    )
    antennas = result.instrument.antennas
    _numeric_dataset(
        handle,
        "instrument/antenna/number",
        [antenna.id.number for antenna in antennas],
        spec=specs["instrument/antenna/number"],
    )
    _string_dataset(
        handle,
        h5py,
        "instrument/antenna/name",
        prepared.text_payloads["instrument/antenna/name"],
        spec=specs["instrument/antenna/name"],
    )
    _numeric_dataset(
        handle,
        "instrument/antenna/position_enu_m",
        [antenna.position_enu_m for antenna in antennas],
        spec=specs["instrument/antenna/position_enu_m"],
    )
    _numeric_dataset(
        handle,
        "instrument/antenna/diameter_m",
        [antenna.diameter_m for antenna in antennas],
        spec=specs["instrument/antenna/diameter_m"],
    )
    location = result.instrument.location
    _numeric_dataset(
        handle,
        "instrument/location/itrs_xyz_m",
        location.itrs_xyz_m,
        spec=specs["instrument/location/itrs_xyz_m"],
    )
    _numeric_dataset(
        handle,
        "instrument/location/geodetic_lon_lat_height",
        (location.longitude_deg, location.latitude_deg, location.height_m),
        spec=specs["instrument/location/geodetic_lon_lat_height"],
    )

    phase = result.phase_center
    for path, value in (
        ("phase_center/kind", prepared.text_payloads["phase_center/kind"]),
        ("phase_center/frame", prepared.text_payloads["phase_center/frame"]),
        (
            "phase_center/w_reference",
            prepared.text_payloads["phase_center/w_reference"],
        ),
    ):
        _string_dataset(handle, h5py, path, value, spec=specs[path])
    for path, value in (
        ("phase_center/azimuth_rad", phase.azimuth_rad),
        ("phase_center/altitude_rad", phase.altitude_rad),
        ("phase_center/time_dependent", phase.time_dependent),
        ("phase_center/geometric_phase_sign", phase.geometric_phase_sign),
    ):
        _numeric_dataset(
            handle,
            path,
            value,
            spec=specs[path],
        )

    receptors = result.receptors
    ordered_receptors = [
        receptors.receptor_by_antenna[antenna.id] for antenna in antennas
    ]
    for path in ("receptors/output_basis", "receptors/receptor_sha256"):
        _string_dataset(
            handle,
            h5py,
            path,
            cast(bytes, prepared.text_payloads[path]),
            spec=specs[path],
        )
    _numeric_dataset(
        handle,
        "receptors/antenna_number",
        [antenna.id.number for antenna in antennas],
        spec=specs["receptors/antenna_number"],
    )
    for path in ("receptors/antenna_name", "receptors/basis"):
        _string_dataset(
            handle,
            h5py,
            path,
            cast(tuple[bytes, ...], prepared.text_payloads[path]),
            spec=specs[path],
        )
    _numeric_dataset(
        handle,
        "receptors/feed_rotation_rad",
        [receptor.feed_rotation_rad for receptor in ordered_receptors],
        spec=specs["receptors/feed_rotation_rad"],
    )
    _numeric_dataset(
        handle,
        "receptors/feed_angle_rad",
        [list(receptor.feed_angle_rad) for receptor in ordered_receptors],
        spec=specs["receptors/feed_angle_rad"],
    )

    # The ``jones/`` group exists only when an optional Jones or baseline term
    # was configured.  A run with no ``jones:`` section writes a file with no
    # such group, and the reader treats its absence as "no optional terms
    # enabled" (Section 25.2).  This preserves the optional group's structural
    # absence; it does not claim that visibility values or scientific
    # fingerprints match pre-SCI-006 output.
    if "jones/jones_sha256" in prepared.text_payloads:
        handle.create_group("jones", track_order=True)
        for path in sorted(_JONES_DATASETS):
            _string_dataset(
                handle,
                h5py,
                path,
                cast(bytes, prepared.text_payloads[path]),
                spec=specs[path],
            )

    for name in (
        "instrument_json",
        "selection_json",
        "beam_json",
        "backend_json",
        "solver_json",
        "resolved_config_json",
        "configuration_source_json",
        "performance_json",
        "history_json",
    ):
        _string_dataset(
            handle,
            h5py,
            f"provenance/{name}",
            cast(bytes, prepared.text_payloads[f"provenance/{name}"]),
            spec=specs[f"provenance/{name}"],
        )


def _flush_hdf5(handle: Any) -> None:
    handle.flush()


def _close_hdf5(handle: Any) -> None:
    handle.close()


def _write_temporary_file(
    h5py: Any,
    descriptor: int,
    result: SimulationResult,
    prepared: _PreparedResult,
) -> None:
    stream = os.fdopen(os.dup(descriptor), "r+b", buffering=0)
    handle: Any | None = None
    try:
        handle = h5py.File(stream, "w", track_order=True)
        _write_hdf5_content(handle, h5py, result, prepared)
        _flush_hdf5(handle)
        _fsync_file(descriptor)
        _close_hdf5(handle)
        handle = None
    finally:
        if handle is not None:
            try:
                _close_hdf5(handle)
            except Exception:
                pass
        stream.close()


def _reopen_temporary_result(path: Path) -> LoadedSimulationResult:
    return load_result_hdf5(path)


def _verify_temporary_result(
    result: SimulationResult,
    path: Path,
) -> LoadedSimulationResult:
    loaded = _reopen_temporary_result(path)
    if (
        loaded.scientific_sha256 != result.scientific_sha256
        or loaded.provenance_sha256 != result.provenance_sha256
        or not loaded.scientifically_equal(result)
    ):
        raise UnsafeResultInputError(
            "temporary HDF5 result failed fingerprint or scientific verification"
        )
    return loaded


def write_result_hdf5(
    result: SimulationResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a complete ``radiosim.visibility`` 4.0.0 result atomically.

    Parameters
    ----------
    result
        Exact canonical in-memory simulation result.
    path
        Final regular-file path with one exact ``.h5`` extension.
    overwrite
        Atomically replace an existing regular file when true.

    Returns
    -------
    pathlib.Path
        The normalized final path.
    """
    prepared = _prepare_result(result)
    final = _validate_output_target(path, extension=".h5", overwrite=overwrite)
    h5py = _import_h5py()
    parent_fd = -1
    temporary_fd = -1
    temporary: Path | None = None
    published = False
    try:
        parent_fd = _open_parent_directory(final.parent, create=True)
        temporary_fd, temporary = _create_sibling_temporary(final, parent_fd)
        try:
            _write_temporary_file(h5py, temporary_fd, result, prepared)
            os.close(temporary_fd)
            temporary_fd = -1
            _ = _verify_temporary_result(result, temporary)
            if overwrite:
                _publish_replace(temporary, final, parent_fd)
                published = True
            else:
                _publish_no_clobber(temporary, final, parent_fd)
                published = True
                try:
                    _unlink_temporary(temporary, parent_fd)
                except OSError as exc:
                    raise PartialCleanupError(temporary) from exc
            try:
                _fsync_directory(parent_fd)
            except OSError as exc:
                raise AtomicWriteError(
                    f"published HDF5 result but directory fsync failed: {final.parent}"
                ) from exc
            return final
        except Exception as exc:
            if not published:
                try:
                    _unlink_temporary(temporary, parent_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    cleanup = PartialCleanupError(temporary)
                    raise cleanup from exc
            if isinstance(exc, ResultIOError):
                raise
            raise AtomicWriteError(
                f"atomic HDF5 transaction failed before publication: {final}"
            ) from exc
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _root_text(value: object, *, name: str, limit: int) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, (bytes, np.bytes_)):
        try:
            text = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise UnsafeResultInputError(
                f"root attribute {name} is not strict UTF-8"
            ) from exc
    else:
        raise UnsafeResultInputError(f"root attribute {name} must be scalar text")
    try:
        encoded = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise UnsafeResultInputError(
            f"root attribute {name} is not strict UTF-8"
        ) from exc
    if not encoded or len(encoded) > limit or b"\x00" in encoded:
        raise UnsafeResultInputError(
            f"root attribute {name} violates the bounded string contract"
        )
    return text


def _root_attribute_exists(handle: Any, h5py: Any, name: str) -> bool:
    return bool(h5py.h5a.exists(handle.id, name.encode("utf-8")))


def _read_bounded_root_text(
    handle: Any,
    h5py: Any,
    *,
    name: str,
    limit: int,
) -> str:
    attribute_id = handle.attrs.get_id(name)
    try:
        if attribute_id.shape != ():
            raise UnsafeResultInputError(f"root attribute {name} must be a scalar")
        type_id = attribute_id.get_type()
        try:
            if type_id.get_class() != h5py.h5t.STRING:
                raise UnsafeResultInputError(f"root attribute {name} must be text")
            if type_id.is_variable_str():
                raise UnsafeResultInputError(
                    f"root attribute {name} uses an unbounded variable-length string"
                )
            if type_id.get_cset() != h5py.h5t.CSET_UTF8:
                raise UnsafeResultInputError(
                    f"root attribute {name} must use fixed-length UTF-8 storage"
                )
            declared_size = int(type_id.get_size())
            storage_size = int(attribute_id.get_storage_size())
            if (
                declared_size <= 0
                or declared_size > limit
                or storage_size != declared_size
            ):
                raise UnsafeResultInputError(
                    f"root attribute {name} violates the bounded string contract"
                )
        finally:
            type_id.close()
    finally:
        attribute_id.close()
    return _root_text(
        handle.attrs[name],
        name=name,
        limit=limit,
    )


def _read_root_attributes(
    handle: Any,
    h5py: Any,
    limits: HDF5ReadLimits,
) -> dict[str, str]:
    if not _root_attribute_exists(handle, h5py, "schema_name"):
        raise LegacyHDF5Error()
    values = {
        "schema_name": _read_bounded_root_text(
            handle,
            h5py,
            name="schema_name",
            limit=limits.max_single_string_bytes,
        )
    }
    if values["schema_name"] != SCHEMA_NAME:
        raise UnsafeResultInputError("HDF5 schema_name is not radiosim.visibility")
    if not _root_attribute_exists(handle, h5py, "schema_version"):
        raise UnsafeResultInputError("HDF5 schema_version root attribute is missing")
    values["schema_version"] = _read_bounded_root_text(
        handle,
        h5py,
        name="schema_version",
        limit=limits.max_single_string_bytes,
    )
    if values["schema_version"] != SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(values["schema_version"])
    if int(h5py.h5a.get_num_attrs(handle.id)) != len(_ROOT_ATTRIBUTES) or any(
        not _root_attribute_exists(handle, h5py, name) for name in _ROOT_ATTRIBUTES
    ):
        raise UnsafeResultInputError("HDF5 root attribute allowlist mismatch")
    for name in _ROOT_ATTRIBUTE_READ_ORDER[2:]:
        values[name] = _read_bounded_root_text(
            handle,
            h5py,
            name=name,
            limit=limits.max_single_string_bytes,
        )
    if values["dimension_order"] != DIMENSION_ORDER:
        raise UnsafeResultInputError("HDF5 dimension_order is invalid")
    if values["visibility_unit"] != VISIBILITY_UNIT:
        raise UnsafeResultInputError("HDF5 visibility_unit is invalid")
    for name in ("scientific_sha256", "provenance_sha256"):
        if _SHA256.fullmatch(values[name]) is None:
            raise UnsafeResultInputError(f"HDF5 {name} is not a lower-case SHA-256")
    return values


def _object_address(h5py: Any, value: Any) -> int:
    return int(h5py.h5o.get_info(value.id).addr)


def _inspect_tree(handle: Any, h5py: Any) -> dict[str, Any]:
    groups: set[str] = set()
    datasets: dict[str, Any] = {}
    seen = {_object_address(h5py, handle): "/"}

    def walk(group: Any, prefix: str) -> None:
        if prefix and set(group.attrs):
            raise UnsafeResultInputError(
                f"HDF5 group /{prefix} has unexpected attributes"
            )
        for name in group.keys():
            path = f"{prefix}/{name}" if prefix else name
            link = group.get(name, getlink=True)
            if type(link) is not h5py.HardLink:
                raise UnsafeResultInputError(f"unsafe HDF5 link at /{path}")
            value = group[name]
            address = _object_address(h5py, value)
            if address in seen:
                raise UnsafeResultInputError(
                    f"duplicate HDF5 hard-link alias at /{path}"
                )
            seen[address] = path
            if isinstance(value, h5py.Group):
                groups.add(path)
                walk(value, path)
            elif isinstance(value, h5py.Dataset):
                if value.is_virtual:
                    raise UnsafeResultInputError(
                        f"virtual HDF5 dataset is not accepted at /{path}"
                    )
                if h5py.check_dtype(ref=value.dtype) is not None:
                    raise UnsafeResultInputError(
                        f"HDF5 references are not accepted at /{path}"
                    )
                datasets[path] = value
            else:
                raise UnsafeResultInputError(f"unknown HDF5 object at /{path}")

    walk(handle, "")
    optional_present = groups & _OPTIONAL_GROUPS
    expected_datasets = set(_DATASETS)
    if "jones" in optional_present:
        expected_datasets |= _JONES_DATASETS
    if groups != _GROUPS | optional_present or set(datasets) != expected_datasets:
        raise UnsafeResultInputError("HDF5 object allowlist mismatch")
    return datasets


def _metadata_specs(
    *,
    time_count: int,
    baseline_count: int,
    frequency_count: int,
    antenna_count: int,
    visibility_dtype: str,
) -> dict[str, _DatasetSpec]:
    science_shape = (time_count, baseline_count, frequency_count, 4)
    weight_dtype = "<f4" if visibility_dtype == "<c8" else "<f8"
    specs = {
        "data/visibilities": _DatasetSpec(
            science_shape,
            visibility_dtype,
            ("time", "baseline", "frequency", "correlation"),
            storage="science_visibilities",
        ),
        "data/flags": _DatasetSpec(
            science_shape,
            "|b1",
            ("time", "baseline", "frequency", "correlation"),
            storage="science_flags",
        ),
        "data/weights": _DatasetSpec(
            science_shape,
            weight_dtype,
            ("time", "baseline", "frequency", "correlation"),
            storage="science_weights",
        ),
        "coordinates/time/utc_jd1": _DatasetSpec(
            (time_count,),
            "<f8",
            ("time",),
            (("unit", "day"), ("scale", "UTC")),
            "coordinate",
        ),
        "coordinates/time/utc_jd2": _DatasetSpec(
            (time_count,),
            "<f8",
            ("time",),
            (("unit", "day"), ("scale", "UTC")),
            "coordinate",
        ),
        "coordinates/time/integration_time_seconds": _DatasetSpec(
            (time_count,),
            "<f8",
            ("time",),
            (("unit", "second"),),
            "coordinate",
        ),
        "coordinates/frequency/center_hz": _DatasetSpec(
            (frequency_count,),
            "<f8",
            ("frequency",),
            (("unit", "Hz"),),
            "coordinate",
        ),
        "coordinates/frequency/channel_width_hz": _DatasetSpec(
            (frequency_count,),
            "<f8",
            ("frequency",),
            (("unit", "Hz"),),
            "coordinate",
        ),
        "coordinates/correlation/labels": _DatasetSpec(
            (4,),
            "|S2",
            ("correlation",),
        ),
        "coordinates/correlation/aips_codes": _DatasetSpec(
            (4,),
            "<i4",
            ("correlation",),
            storage="coordinate",
        ),
        "coordinates/correlation/basis": _DatasetSpec((), "utf8", ()),
        "receptors/output_basis": _DatasetSpec((), "utf8", ()),
        "receptors/receptor_sha256": _DatasetSpec((), "utf8", ()),
        "jones/enabled_terms": _DatasetSpec((), "utf8", ()),
        "jones/chain_order": _DatasetSpec((), "utf8", ()),
        "jones/term_snapshots_json": _DatasetSpec((), "utf8", ()),
        "jones/mount_types_json": _DatasetSpec((), "utf8", ()),
        "jones/jones_sha256": _DatasetSpec((), "utf8", ()),
        "receptors/antenna_number": _DatasetSpec(
            (antenna_count,),
            "<i8",
            ("antenna",),
            storage="coordinate",
        ),
        "receptors/antenna_name": _DatasetSpec(
            (antenna_count,),
            "utf8",
            ("antenna",),
        ),
        "receptors/basis": _DatasetSpec(
            (antenna_count,),
            "utf8",
            ("antenna",),
        ),
        "receptors/feed_rotation_rad": _DatasetSpec(
            (antenna_count,),
            "<f8",
            ("antenna",),
            (("unit", "radian"),),
            "coordinate",
        ),
        "receptors/feed_angle_rad": _DatasetSpec(
            (antenna_count, 2),
            "<f8",
            ("antenna", "feed"),
            (("unit", "radian"),),
            "coordinate",
        ),
        "coordinates/baseline/antenna1_number": _DatasetSpec(
            (baseline_count,),
            "<i8",
            ("baseline",),
            storage="coordinate",
        ),
        "coordinates/baseline/antenna2_number": _DatasetSpec(
            (baseline_count,),
            "<i8",
            ("baseline",),
            storage="coordinate",
        ),
        "coordinates/baseline/vector_enu_m": _DatasetSpec(
            (baseline_count, 3),
            "<f8",
            ("baseline", "enu_component"),
            (("unit", "metre"),),
            "coordinate",
        ),
        "instrument/name": _DatasetSpec((), "utf8", ()),
        "instrument/antenna/number": _DatasetSpec(
            (antenna_count,),
            "<i8",
            ("antenna",),
            storage="coordinate",
        ),
        "instrument/antenna/name": _DatasetSpec(
            (antenna_count,),
            "utf8",
            ("antenna",),
        ),
        "instrument/antenna/position_enu_m": _DatasetSpec(
            (antenna_count, 3),
            "<f8",
            ("antenna", "enu_component"),
            (("unit", "metre"),),
            "coordinate",
        ),
        "instrument/antenna/diameter_m": _DatasetSpec(
            (antenna_count,),
            "<f8",
            ("antenna",),
            (("unit", "metre"),),
            "coordinate",
        ),
        "instrument/location/itrs_xyz_m": _DatasetSpec(
            (3,),
            "<f8",
            ("itrs_component",),
            (("unit", "metre"),),
            "coordinate",
        ),
        "instrument/location/geodetic_lon_lat_height": _DatasetSpec(
            (3,),
            "<f8",
            ("geodetic_component",),
            (("units", "degree, degree, metre"),),
            "coordinate",
        ),
        "phase_center/kind": _DatasetSpec((), "utf8", ()),
        "phase_center/frame": _DatasetSpec((), "utf8", ()),
        "phase_center/azimuth_rad": _DatasetSpec((), "<f8", ()),
        "phase_center/altitude_rad": _DatasetSpec((), "<f8", ()),
        "phase_center/time_dependent": _DatasetSpec((), "|b1", ()),
        "phase_center/geometric_phase_sign": _DatasetSpec((), "|i1", ()),
        "phase_center/w_reference": _DatasetSpec((), "utf8", ()),
    }
    specs.update({path: _DatasetSpec((), "utf8", ()) for path in _JSON_PATHS})
    return specs


def _filter_ids(dataset: Any) -> tuple[int, ...]:
    properties = dataset.id.get_create_plist()
    return tuple(
        int(properties.get_filter(index)[0])
        for index in range(properties.get_nfilters())
    )


def _fixed_utf8_itemsize(dataset: Any, h5py: Any, *, path: str) -> int:
    string_info = h5py.check_string_dtype(dataset.dtype)
    if string_info is None or string_info.encoding != "utf-8":
        raise UnsafeResultInputError(
            f"HDF5 dataset /{path} must use fixed UTF-8 storage"
        )
    itemsize = string_info.length
    if itemsize is None:
        raise UnsafeResultInputError(
            f"HDF5 dataset /{path} uses variable-length UTF-8; "
            "fixed-width UTF-8 is required"
        )
    if type(itemsize) is not int or itemsize <= 0:
        raise UnsafeResultInputError(
            f"HDF5 dataset /{path} has an invalid fixed UTF-8 item size"
        )
    type_id = dataset.id.get_type()
    try:
        if type_id.get_class() != h5py.h5t.STRING:
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} must use fixed UTF-8 storage"
            )
        if type_id.is_variable_str():
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} uses variable-length UTF-8; "
                "fixed-width UTF-8 is required"
            )
        if type_id.get_cset() != h5py.h5t.CSET_UTF8:
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} must use fixed UTF-8 storage"
            )
        declared_size = type_id.get_size()
        if type(declared_size) is not int or declared_size <= 0:
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} has an invalid fixed UTF-8 item size"
            )
        if type_id.get_strpad() != h5py.h5t.STR_NULLPAD:
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} must use trailing NUL padding"
            )
    finally:
        type_id.close()
    if declared_size != itemsize or int(dataset.dtype.itemsize) != itemsize:
        raise UnsafeResultInputError(
            f"HDF5 dataset /{path} has inconsistent fixed UTF-8 item size"
        )
    return itemsize


def _validate_dataset_metadata(
    dataset: Any,
    spec: _DatasetSpec,
    h5py: Any,
    *,
    path: str,
    limits: HDF5ReadLimits | None = None,
) -> None:
    if tuple(dataset.shape) != spec.shape:
        raise UnsafeResultInputError(f"HDF5 dataset /{path} has an invalid shape")
    if spec.dtype == "utf8":
        itemsize = _fixed_utf8_itemsize(dataset, h5py, path=path)
        if limits is not None and itemsize > limits.max_single_string_bytes:
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} exceeds max_single_string_bytes"
            )
    elif dataset.dtype.str != spec.dtype:
        raise UnsafeResultInputError(f"HDF5 dataset /{path} has an invalid dtype")
    if tuple(dimension.label for dimension in dataset.dims) != spec.dimensions:
        raise UnsafeResultInputError(
            f"HDF5 dataset /{path} has invalid dimension labels"
        )
    expected_attribute_names = {name for name, _value in spec.attributes}
    if spec.dimensions:
        expected_attribute_names.add("DIMENSION_LABELS")
    if set(dataset.attrs) != expected_attribute_names:
        raise UnsafeResultInputError(
            f"HDF5 dataset /{path} has an invalid attribute allowlist"
        )
    for name, expected in spec.attributes:
        actual = _root_text(dataset.attrs[name], name=f"/{path}:{name}", limit=256)
        if actual != expected:
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} has invalid {name} metadata"
            )

    if spec.storage.startswith("science"):
        expected_chunks = (
            min(spec.shape[0], 16),
            min(spec.shape[1], 64),
            min(spec.shape[2], 64),
            4,
        )
        if (
            dataset.chunks != expected_chunks
            or dataset.compression != "gzip"
            or dataset.compression_opts != 4
            or dataset.shuffle != (spec.storage != "science_flags")
            or dataset.fletcher32 is not True
            or dataset.scaleoffset is not None
        ):
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} has invalid science filters"
            )
        expected_filters = (
            (
                h5py.h5z.FILTER_DEFLATE,
                h5py.h5z.FILTER_FLETCHER32,
            )
            if spec.storage == "science_flags"
            else (
                h5py.h5z.FILTER_SHUFFLE,
                h5py.h5z.FILTER_DEFLATE,
                h5py.h5z.FILTER_FLETCHER32,
            )
        )
    elif spec.storage == "coordinate":
        expected_chunks = (
            (min(spec.shape[0], 4096),)
            if len(spec.shape) == 1
            else (min(spec.shape[0], 4096), spec.shape[1])
        )
        if (
            dataset.chunks != expected_chunks
            or dataset.compression is not None
            or dataset.shuffle is not False
            or dataset.fletcher32 is not True
            or dataset.scaleoffset is not None
        ):
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} has invalid coordinate filters"
            )
        expected_filters = (h5py.h5z.FILTER_FLETCHER32,)
    else:
        if (
            dataset.chunks is not None
            or dataset.compression is not None
            or dataset.shuffle is not False
            or dataset.fletcher32 is not False
            or dataset.scaleoffset is not None
        ):
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} must be contiguous and unfiltered"
            )
        expected_filters = ()
    if _filter_ids(dataset) != expected_filters:
        raise UnsafeResultInputError(f"HDF5 dataset /{path} has an undeclared filter")


def _checked_axis_counts(
    datasets: Mapping[str, Any],
) -> tuple[int, int, int, int, str]:
    visibility = datasets["data/visibilities"]
    if len(visibility.shape) != 4:
        raise UnsafeResultInputError("HDF5 visibility dataset rank is invalid")
    time_count, baseline_count, frequency_count, correlations = (
        int(value) for value in visibility.shape
    )
    antenna_dataset = datasets["instrument/antenna/number"]
    if len(antenna_dataset.shape) != 1:
        raise UnsafeResultInputError("HDF5 antenna-number dataset rank is invalid")
    antenna_count = int(antenna_dataset.shape[0])
    if min(time_count, baseline_count, frequency_count, antenna_count) <= 0:
        raise UnsafeResultInputError("HDF5 canonical axes must be nonempty")
    if correlations != 4:
        raise UnsafeResultInputError("HDF5 correlation axis must contain four values")
    if visibility.dtype.str not in {"<c8", "<c16"}:
        raise UnsafeResultInputError("HDF5 visibility dtype is unsupported")
    return (
        time_count,
        baseline_count,
        frequency_count,
        antenna_count,
        visibility.dtype.str,
    )


def _enforce_axis_limits(
    counts: tuple[int, int, int, int, str],
    limits: HDF5ReadLimits,
) -> None:
    time_count, baseline_count, frequency_count, antenna_count, _dtype = counts
    for count, limit, name in (
        (time_count, limits.max_time, "max_time"),
        (baseline_count, limits.max_baseline, "max_baseline"),
        (frequency_count, limits.max_frequency, "max_frequency"),
        (antenna_count, limits.max_antenna, "max_antenna"),
    ):
        if count > limit:
            raise UnsafeResultInputError(f"HDF5 input exceeds {name}")
    elements = time_count * baseline_count * frequency_count * 4
    if elements > limits.max_visibility_elements:
        raise UnsafeResultInputError("HDF5 input exceeds max_visibility_elements")


def _enforce_dataset_byte_limits(
    datasets: Mapping[str, Any],
    limits: HDF5ReadLimits,
) -> None:
    for path, dataset in datasets.items():
        element_count = math.prod(int(value) for value in dataset.shape)
        byte_count = element_count * int(dataset.dtype.itemsize)
        if byte_count > limits.max_single_dataset_bytes:
            raise UnsafeResultInputError(
                f"HDF5 dataset /{path} exceeds max_single_dataset_bytes"
            )


def _enforce_json_byte_limits(
    datasets: Mapping[str, Any],
    limits: HDF5ReadLimits,
) -> None:
    declared_total = 0
    for path in _JSON_PATHS:
        dataset = datasets[path]
        element_count = math.prod(int(value) for value in dataset.shape)
        declared_total += element_count * int(dataset.dtype.itemsize)
    if declared_total > limits.max_total_json_bytes:
        raise UnsafeResultInputError(
            "HDF5 JSON declared fixed-width storage exceeds max_total_json_bytes"
        )


def _bounded_dataset_text(
    dataset: Any,
    *,
    path: str,
    limit: int,
    index: int | None = None,
) -> tuple[str, int]:
    if type(limit) is not int or limit <= 0:
        raise UnsafeResultInputError(
            f"HDF5 string /{path} has an invalid max_single_string_bytes"
        )
    try:
        shape = tuple(int(value) for value in dataset.shape)
    except Exception as exc:
        raise UnsafeResultInputError(
            f"HDF5 string /{path} has invalid shape metadata"
        ) from exc
    if index is None:
        if shape != ():
            raise UnsafeResultInputError(
                f"HDF5 string /{path} must use a scalar dataset"
            )
    elif type(index) is not int or len(shape) != 1 or index < 0 or index >= shape[0]:
        raise UnsafeResultInputError(
            f"HDF5 string /{path} has an invalid indexed selection"
        )

    h5py = _import_h5py()
    itemsize = _fixed_utf8_itemsize(dataset, h5py, path=path)
    if itemsize > limit:
        raise UnsafeResultInputError(
            f"HDF5 string /{path} exceeds max_single_string_bytes"
        )
    file_space: Any = None
    memory_space: Any = None
    memory_type: Any = None
    primary_error: BaseException | None = None
    try:
        file_space = dataset.id.get_space()
        expected_space_type = h5py.h5s.SCALAR if index is None else h5py.h5s.SIMPLE
        if file_space.get_simple_extent_type() != expected_space_type:
            raise UnsafeResultInputError(
                f"HDF5 string /{path} has an invalid dataspace"
            )

        try:
            destination_dtype = np.dtype(f"S{itemsize}")
            destination = np.zeros(
                () if index is None else (1,),
                dtype=destination_dtype,
            )
        except (MemoryError, OverflowError, TypeError, ValueError) as exc:
            raise UnsafeResultInputError(
                f"HDF5 string /{path} limit cannot be allocated safely"
            ) from exc

        memory_type = h5py.h5t.C_S1.copy()
        memory_type.set_size(itemsize)
        memory_type.set_cset(h5py.h5t.CSET_UTF8)
        memory_type.set_strpad(h5py.h5t.STR_NULLPAD)
        if index is None:
            memory_space = h5py.h5s.create(h5py.h5s.SCALAR)
        else:
            memory_space = h5py.h5s.create_simple((1,))
            file_space.select_hyperslab((index,), (1,))

        try:
            dataset.id.read(
                memory_space,
                file_space,
                destination,
                mtype=memory_type,
            )
        except (
            MemoryError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise UnsafeResultInputError(
                f"HDF5 string /{path} could not be read safely"
            ) from exc

        raw = memoryview(cast(Any, destination)).cast("B")
        terminator: int | None = None
        for offset, value in enumerate(raw):
            if value == 0:
                terminator = offset
                break
        if terminator is not None and any(
            value != 0 for value in raw[terminator + 1 :]
        ):
            raise UnsafeResultInputError(
                f"HDF5 string /{path} contains invalid NUL padding"
            )
        payload = bytes(raw if terminator is None else raw[:terminator])
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise UnsafeResultInputError(
                f"HDF5 string /{path} is not strict UTF-8"
            ) from exc
        return text, len(payload)
    except UnsafeResultInputError as exc:
        primary_error = exc
        raise
    except (
        MemoryError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        wrapped = UnsafeResultInputError(
            f"HDF5 string /{path} could not be read safely"
        )
        primary_error = wrapped
        raise wrapped from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error: Exception | None = None
        for handle in (file_space, memory_space, memory_type):
            if handle is None:
                continue
            try:
                handle.close()
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None and primary_error is None:
            raise UnsafeResultInputError(
                f"HDF5 string /{path} handle cleanup failed"
            ) from cleanup_error


def _parse_json(text: str, *, path: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite constant {value}")

    try:
        return json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise UnsafeResultInputError(f"HDF5 JSON /{path} is invalid") from exc


def _read_json_snapshots(
    datasets: Mapping[str, Any],
    limits: HDF5ReadLimits,
) -> dict[str, object]:
    values: dict[str, object] = {}
    total = 0
    for path, name in _JSON_PATHS.items():
        text, byte_count = _bounded_dataset_text(
            datasets[path],
            path=path,
            limit=limits.max_single_string_bytes,
        )
        total += byte_count
        if total > limits.max_total_json_bytes:
            raise UnsafeResultInputError("HDF5 JSON exceeds max_total_json_bytes")
        values[name] = _parse_json(text, path=path)
    for name in (
        "instrument",
        "selection",
        "beam",
        "backend",
        "solver",
        "resolved_config",
        "performance",
    ):
        if type(values[name]) is not dict:
            raise UnsafeResultInputError(f"HDF5 {name}_json must be a JSON object")
    if (
        values["configuration_source"] is not None
        and type(values["configuration_source"]) is not dict
    ):
        raise UnsafeResultInputError(
            "HDF5 configuration_source_json must be an object or null"
        )
    history = values["history"]
    if type(history) is not list:
        raise UnsafeResultInputError("HDF5 history_json must be an array of strings")
    history_items = cast(list[object], history)
    if any(type(item) is not str for item in history_items):
        raise UnsafeResultInputError("HDF5 history_json must be an array of strings")
    return values


def _read_numeric(dataset: Any, *, path: str) -> np.ndarray:
    try:
        return np.array(dataset[...], copy=True, order="C", subok=False)
    except Exception as exc:
        raise UnsafeResultInputError(
            f"HDF5 numeric payload /{path} could not be read or verified"
        ) from exc


def _snapshot_mapping(
    snapshot: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> dict[str, object]:
    value = snapshot.get(key)
    if type(value) is not dict:
        raise UnsafeResultInputError(f"HDF5 {context}.{key} must be an object")
    return cast(dict[str, object], value)


def _snapshot_list(
    snapshot: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> list[object]:
    value = snapshot.get(key)
    if type(value) is not list:
        raise UnsafeResultInputError(f"HDF5 {context}.{key} must be an array")
    return cast(list[object], value)


def _build_time_grid(
    resolved_config: Mapping[str, object],
    utc_jd1: np.ndarray,
    utc_jd2: np.ndarray,
    integration: np.ndarray,
) -> ObservationTimeGrid:
    observation = _snapshot_mapping(
        resolved_config,
        "observation",
        context="resolved_config",
    )
    snapshot = _snapshot_mapping(observation, "time_grid", context="observation")
    required = {
        "schema_version",
        "interval_semantics",
        "start_time_iso",
        "duration_seconds",
        "cadence_seconds",
        "utc_jd1",
        "utc_jd2",
        "integration_time_seconds",
    }
    if set(snapshot) != required:
        raise UnsafeResultInputError("HDF5 time-grid snapshot fields are invalid")
    try:
        if (
            snapshot["schema_version"] != "radiosim.time-grid.v1"
            or snapshot["interval_semantics"] != "half_open_sample_centers"
        ):
            raise ValueError("invalid time-grid identity")
        snapshot_jd1 = np.asarray(snapshot["utc_jd1"], dtype=np.float64)
        snapshot_jd2 = np.asarray(snapshot["utc_jd2"], dtype=np.float64)
        snapshot_integration = np.asarray(
            snapshot["integration_time_seconds"],
            dtype=np.float64,
        )
        if (
            not np.array_equal(snapshot_jd1, utc_jd1)
            or not np.array_equal(snapshot_jd2, utc_jd2)
            or not np.array_equal(snapshot_integration, integration)
        ):
            raise ValueError("structured time coordinates disagree with snapshot")
        grid = build_observation_time_grid(
            start_time=cast(str, snapshot["start_time_iso"]),
            duration_seconds=cast(float, snapshot["duration_seconds"]),
            cadence_seconds=cast(float, snapshot["cadence_seconds"]),
        )
    except Exception as exc:
        raise UnsafeResultInputError("HDF5 time-grid contract is invalid") from exc
    if (
        not np.array_equal(grid.utc_jd1, utc_jd1)
        or not np.array_equal(grid.utc_jd2, utc_jd2)
        or not np.array_equal(grid.integration_time_seconds, integration)
    ):
        raise UnsafeResultInputError(
            "HDF5 structured time coordinates do not reconstruct exactly"
        )
    return grid


def _read_jones_group(
    datasets: Mapping[str, Any],
    limits: HDF5ReadLimits,
) -> dict[str, object] | None:
    """Read the optional ``jones/`` group, or return ``None`` when it is absent.

    Absence is a legitimate, common state and not an error: it is what a run
    with no ``jones:`` section writes (``Tier7JonesSciencePlan.md``
    Section 25.2).  When the group *is* present it is fully validated -- the
    digest shape, and that the recorded term list is not empty, since an empty
    list is what absence already means and a file asserting both is
    inconsistent.
    """
    if "jones/jones_sha256" not in datasets:
        return None
    digest, _ = _bounded_dataset_text(
        datasets["jones/jones_sha256"],
        path="jones/jones_sha256",
        limit=limits.max_single_string_bytes,
    )
    if _SHA256.fullmatch(digest) is None:
        raise UnsafeResultInputError("HDF5 jones_sha256 is not a lower-case SHA-256")
    parsed: dict[str, object] = {}
    for path, key in (
        ("jones/enabled_terms", "enabled_terms"),
        ("jones/chain_order", "chain_order"),
        ("jones/term_snapshots_json", "term_snapshots"),
        ("jones/mount_types_json", "mount_types"),
    ):
        text, _ = _bounded_dataset_text(
            datasets[path],
            path=path,
            limit=limits.max_single_string_bytes,
        )
        parsed[key] = _parse_json(text, path=path)
    enabled = parsed["enabled_terms"]
    if not isinstance(enabled, list) or not enabled:
        raise UnsafeResultInputError(
            "HDF5 jones/enabled_terms must be a nonempty list; a run with no "
            "enabled term omits the group entirely"
        )
    from radiosim.core.jones_terms import JONES_SCHEMA_VERSION

    parsed["schema_version"] = JONES_SCHEMA_VERSION
    parsed["jones_sha256"] = digest
    return parsed


def _read_receptor_group(
    datasets: Mapping[str, Any],
    limits: HDF5ReadLimits,
    *,
    basis: PolarizationBasis,
    antenna_numbers: Sequence[int],
    antenna_names: Sequence[str],
) -> dict[str, object]:
    """Read and fully validate the ``receptors/`` group (Section 21).

    The returned mapping is exactly the result-model receptor snapshot: the
    per-antenna rows are cross-validated against the instrument identity and
    the recorded output basis must agree with the correlation coordinates, so
    no unauthenticated receptor state reaches the canonical model.
    """
    output_basis, _ = _bounded_dataset_text(
        datasets["receptors/output_basis"],
        path="receptors/output_basis",
        limit=limits.max_single_string_bytes,
    )
    if output_basis != basis:
        raise UnsafeResultInputError(
            "HDF5 receptor output basis disagrees with the correlation basis"
        )
    receptor_sha256, _ = _bounded_dataset_text(
        datasets["receptors/receptor_sha256"],
        path="receptors/receptor_sha256",
        limit=limits.max_single_string_bytes,
    )
    if _SHA256.fullmatch(receptor_sha256) is None:
        raise UnsafeResultInputError("HDF5 receptor_sha256 is not a lower-case SHA-256")
    numbers = _read_numeric(
        datasets["receptors/antenna_number"],
        path="receptors/antenna_number",
    )
    if [int(value) for value in numbers] != list(antenna_numbers):
        raise UnsafeResultInputError(
            "HDF5 receptor antenna numbers disagree with the instrument"
        )
    names = tuple(
        _bounded_dataset_text(
            datasets["receptors/antenna_name"],
            path=f"receptors/antenna_name[{index}]",
            limit=limits.max_single_string_bytes,
            index=index,
        )[0]
        for index in range(len(antenna_numbers))
    )
    if list(names) != list(antenna_names):
        raise UnsafeResultInputError(
            "HDF5 receptor antenna names disagree with the instrument"
        )
    bases = tuple(
        _bounded_dataset_text(
            datasets["receptors/basis"],
            path=f"receptors/basis[{index}]",
            limit=limits.max_single_string_bytes,
            index=index,
        )[0]
        for index in range(len(antenna_numbers))
    )
    if any(value not in _RECEPTOR_BASES for value in bases):
        raise UnsafeResultInputError(
            f"HDF5 receptor basis values must be one of {_RECEPTOR_BASES!r}"
        )
    rotations = _read_numeric(
        datasets["receptors/feed_rotation_rad"],
        path="receptors/feed_rotation_rad",
    )
    angles = _read_numeric(
        datasets["receptors/feed_angle_rad"],
        path="receptors/feed_angle_rad",
    )
    if (
        not np.all(np.isfinite(rotations))
        or not np.all(np.isfinite(angles))
        or not np.all(np.abs(rotations) <= math.pi)
    ):
        raise UnsafeResultInputError("HDF5 receptor feed geometry is invalid")
    return {
        "output_basis": output_basis,
        "receptor_sha256": receptor_sha256,
        "receptors": [
            {
                "antenna_number": int(numbers[index]),
                "antenna_name": names[index],
                "basis": bases[index],
                "feed_rotation_rad": float(rotations[index]),
                "feed_angle_rad": [float(value) for value in angles[index]],
            }
            for index in range(len(antenna_numbers))
        ],
    }


def _bounded_component_sequence(
    snapshot: Mapping[str, object],
    key: str,
) -> list[object]:
    """Return a solver component sequence after its length is bounded.

    The length is checked from ``len`` alone, before any element is read or
    coerced, so a forged list cannot cost more than the JSON parse that already
    bounded it.
    """
    value = snapshot.get(key)
    if type(value) is not list:
        raise UnsafeResultInputError(f"HDF5 solver_json.{key} must be an array")
    items = cast(list[object], value)
    if len(items) > _MAX_SOLVED_COMPONENTS:
        raise UnsafeResultInputError(
            f"HDF5 solver_json.{key} exceeds the {_MAX_SOLVED_COMPONENTS} solved "
            "components a radiosim.visibility result can have"
        )
    return items


def _validate_component_provenance(snapshots: Mapping[str, object]) -> None:
    """Reject a forged component or timing record before any payload is read.

    Plan ``Tier6HybridRuntimePlan.md`` Section 19 makes the solved components,
    their element counts, and the two per-component timings part of schema
    ``4.0.0``.  The canonical model validates them again when the result is
    built, but that happens after the visibility cube has been allocated, so
    the checks are also made here, from the parsed and already byte-bounded
    JSON, while the reader has allocated nothing but metadata.

    The field sets are derived from the canonical dataclasses rather than
    restated, so a later field addition cannot leave this reader silently
    accepting the older shape.
    """
    solver = cast(dict[str, object], snapshots["solver"])
    performance = cast(dict[str, object], snapshots["performance"])
    resolved_config = cast(dict[str, object], snapshots["resolved_config"])

    if solver.get("solver") == "mmode":
        # ``docs/development/sci004_mmode_design.md`` Section 10: HDF5
        # preserves the *complete* tagged solver snapshot, and "a reader that
        # silently labels it ``rime`` fails acceptance".  The m-mode arm has its
        # own exact key set, checked here from the already byte-bounded JSON.
        if tuple(solver) != MMODE_SOLVER_SNAPSHOT_KEYS:
            raise UnsafeResultInputError("HDF5 solver_json has unexpected fields")
        _ = _bounded_component_sequence(solver, "components")
        _ = _bounded_component_sequence(solver, "component_element_counts")
        for key in ("tangent_polarization_frame", "stokes_v_basis_bridge"):
            if not isinstance(solver.get(key), str) or not solver[key]:
                raise UnsafeResultInputError("HDF5 solver_json is invalid")
        sky_representation = str(solver.get("sky_representation", ""))
    elif set(solver) != {field.name for field in fields(SolverResultProvenance)}:
        raise UnsafeResultInputError("HDF5 solver_json has unexpected fields")
    else:
        _ = _bounded_component_sequence(solver, "components")
        _ = _bounded_component_sequence(solver, "component_element_counts")
        try:
            identity = SolverResultProvenance(**cast(dict[str, Any], solver))
        except (TypeError, ValueError, InvalidResultError) as exc:
            raise UnsafeResultInputError("HDF5 solver_json is invalid") from exc
        sky_representation = identity.sky_representation

    visibility = _snapshot_mapping(
        resolved_config,
        "visibility",
        context="resolved_config",
    )
    if visibility.get("sky_representation") != sky_representation:
        raise UnsafeResultInputError(
            "HDF5 solver_json sky representation disagrees with the resolved "
            "configuration"
        )

    if set(performance) != {field.name for field in fields(ResultPerformance)}:
        raise UnsafeResultInputError("HDF5 performance_json has unexpected fields")
    try:
        _ = ResultPerformance(**cast(dict[str, Any], performance))
    except (TypeError, ValueError, InvalidResultError) as exc:
        raise UnsafeResultInputError("HDF5 performance_json is invalid") from exc


def _validate_structured_identity(
    datasets: Mapping[str, Any],
    snapshots: Mapping[str, object],
    limits: HDF5ReadLimits,
) -> tuple[
    ObservationTimeGrid,
    np.ndarray,
    np.ndarray,
    PhaseCenter,
    tuple[str, ...],
    dict[str, object],
]:
    _validate_component_provenance(snapshots)
    utc_jd1 = _read_numeric(
        datasets["coordinates/time/utc_jd1"],
        path="coordinates/time/utc_jd1",
    )
    utc_jd2 = _read_numeric(
        datasets["coordinates/time/utc_jd2"],
        path="coordinates/time/utc_jd2",
    )
    integration = _read_numeric(
        datasets["coordinates/time/integration_time_seconds"],
        path="coordinates/time/integration_time_seconds",
    )
    if (
        not np.all(np.isfinite(utc_jd1))
        or not np.all(np.isfinite(utc_jd2))
        or not np.all(np.isfinite(integration))
        or not np.all(integration > 0.0)
        or (utc_jd1.size > 1 and not np.all(np.diff(utc_jd1 + utc_jd2) > 0.0))
    ):
        raise UnsafeResultInputError("HDF5 time coordinates are invalid")
    resolved_config = cast(dict[str, object], snapshots["resolved_config"])
    time_grid = _build_time_grid(
        resolved_config,
        utc_jd1,
        utc_jd2,
        integration,
    )

    frequencies = _read_numeric(
        datasets["coordinates/frequency/center_hz"],
        path="coordinates/frequency/center_hz",
    )
    widths = _read_numeric(
        datasets["coordinates/frequency/channel_width_hz"],
        path="coordinates/frequency/channel_width_hz",
    )
    if (
        not np.all(np.isfinite(frequencies))
        or not np.all(frequencies > 0.0)
        or not np.all(np.diff(frequencies) > 0.0)
        or not np.all(np.isfinite(widths))
        or not np.all(widths > 0.0)
    ):
        raise UnsafeResultInputError("HDF5 frequency coordinates are invalid")
    frequency_snapshot = _snapshot_mapping(
        resolved_config,
        "frequency",
        context="resolved_config",
    )
    try:
        snapshot_frequencies = np.asarray(
            frequency_snapshot["channel_frequencies_hz"],
            dtype=np.float64,
        )
        snapshot_widths = np.asarray(
            frequency_snapshot["channel_widths_hz"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise UnsafeResultInputError(
            "HDF5 resolved frequency snapshot is invalid"
        ) from exc
    if not np.array_equal(
        snapshot_frequencies,
        frequencies,
    ) or not np.array_equal(snapshot_widths, widths):
        raise UnsafeResultInputError(
            "HDF5 structured frequencies disagree with frequency snapshot"
        )

    try:
        labels = tuple(
            bytes(item).decode("ascii", errors="strict")
            for item in datasets["coordinates/correlation/labels"][...]
        )
    except Exception as exc:
        raise UnsafeResultInputError("HDF5 correlation labels are invalid") from exc
    codes = _read_numeric(
        datasets["coordinates/correlation/aips_codes"],
        path="coordinates/correlation/aips_codes",
    )
    basis_text, _ = _bounded_dataset_text(
        datasets["coordinates/correlation/basis"],
        path="coordinates/correlation/basis",
        limit=limits.max_single_string_bytes,
    )
    if basis_text not in CORRELATION_LABELS:
        raise UnsafeResultInputError(
            "HDF5 correlation basis must be one of "
            f"{POLARIZATION_BASES!r}; got {basis_text!r}"
        )
    basis = cast(PolarizationBasis, basis_text)
    if (
        labels != CORRELATION_LABELS[basis]
        or tuple(int(value) for value in codes) != AIPS_CODES_CANONICAL[basis]
    ):
        raise UnsafeResultInputError(
            "HDF5 correlation coordinates are invalid: labels, AIPS codes, and "
            f"basis must be exactly one accepted row for {basis!r}"
        )

    instrument = cast(dict[str, object], snapshots["instrument"])
    instrument_name, _ = _bounded_dataset_text(
        datasets["instrument/name"],
        path="instrument/name",
        limit=limits.max_single_string_bytes,
    )
    numbers = _read_numeric(
        datasets["instrument/antenna/number"],
        path="instrument/antenna/number",
    )
    names = tuple(
        _bounded_dataset_text(
            datasets["instrument/antenna/name"],
            path=f"instrument/antenna/name[{index}]",
            limit=limits.max_single_string_bytes,
            index=index,
        )[0]
        for index in range(numbers.size)
    )
    positions = _read_numeric(
        datasets["instrument/antenna/position_enu_m"],
        path="instrument/antenna/position_enu_m",
    )
    diameters = _read_numeric(
        datasets["instrument/antenna/diameter_m"],
        path="instrument/antenna/diameter_m",
    )
    if (
        len({int(value) for value in numbers}) != numbers.size
        or len(set(names)) != len(names)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(diameters))
        or not np.all(diameters > 0.0)
    ):
        raise UnsafeResultInputError("HDF5 antenna coordinates are invalid")
    antenna_snapshots = _snapshot_list(
        instrument,
        "antennas",
        context="instrument",
    )
    expected_antennas: list[dict[str, object]] = []
    for antenna in antenna_snapshots:
        if type(antenna) is not dict:
            raise UnsafeResultInputError("HDF5 instrument antenna snapshot is invalid")
        expected_antennas.append(cast(dict[str, object], antenna))
    if (
        instrument.get("name") != instrument_name
        or [item.get("number") for item in expected_antennas]
        != [int(value) for value in numbers]
        or [item.get("name") for item in expected_antennas] != list(names)
        or [item.get("position_enu_m") for item in expected_antennas]
        != positions.tolist()
        or [item.get("diameter_m") for item in expected_antennas] != diameters.tolist()
    ):
        raise UnsafeResultInputError(
            "HDF5 structured antenna data disagree with instrument snapshot"
        )
    location = _snapshot_mapping(instrument, "location", context="instrument")
    itrs = _read_numeric(
        datasets["instrument/location/itrs_xyz_m"],
        path="instrument/location/itrs_xyz_m",
    )
    geodetic = _read_numeric(
        datasets["instrument/location/geodetic_lon_lat_height"],
        path="instrument/location/geodetic_lon_lat_height",
    )
    if (
        not np.all(np.isfinite(itrs))
        or not np.all(np.isfinite(geodetic))
        or not -180.0 <= float(geodetic[0]) < 180.0
        or not -90.0 <= float(geodetic[1]) <= 90.0
        or location.get("itrs_xyz_m") != itrs.tolist()
        or [
            location.get("longitude_deg"),
            location.get("latitude_deg"),
            location.get("height_m"),
        ]
        != geodetic.tolist()
    ):
        raise UnsafeResultInputError(
            "HDF5 structured location data disagree with instrument snapshot"
        )

    antenna1 = _read_numeric(
        datasets["coordinates/baseline/antenna1_number"],
        path="coordinates/baseline/antenna1_number",
    )
    antenna2 = _read_numeric(
        datasets["coordinates/baseline/antenna2_number"],
        path="coordinates/baseline/antenna2_number",
    )
    vectors = _read_numeric(
        datasets["coordinates/baseline/vector_enu_m"],
        path="coordinates/baseline/vector_enu_m",
    )
    pairs = [
        [int(first), int(second)]
        for first, second in zip(antenna1, antenna2, strict=True)
    ]
    selection = cast(dict[str, object], snapshots["selection"])
    if selection.get("selected_ids") != pairs or len(
        {tuple(pair) for pair in pairs}
    ) != len(pairs):
        raise UnsafeResultInputError(
            "HDF5 baseline identity disagrees with selection snapshot"
        )
    positions_by_number = {
        int(number): position
        for number, position in zip(numbers, positions, strict=True)
    }
    for index, pair in enumerate(pairs):
        if pair[0] not in positions_by_number or pair[1] not in positions_by_number:
            raise UnsafeResultInputError("HDF5 baseline references an unknown antenna")
        expected_vector = positions_by_number[pair[1]] - positions_by_number[pair[0]]
        if not np.all(np.isfinite(vectors[index])) or not np.allclose(
            vectors[index],
            expected_vector,
            rtol=0.0,
            atol=1e-9,
        ):
            raise UnsafeResultInputError("HDF5 baseline vector is inconsistent")

    phase_values: dict[str, object] = {}
    for name in ("kind", "frame", "w_reference"):
        phase_values[name] = _bounded_dataset_text(
            datasets[f"phase_center/{name}"],
            path=f"phase_center/{name}",
            limit=limits.max_single_string_bytes,
        )[0]
    for name in (
        "azimuth_rad",
        "altitude_rad",
        "time_dependent",
        "geometric_phase_sign",
    ):
        value = _read_numeric(
            datasets[f"phase_center/{name}"],
            path=f"phase_center/{name}",
        )[()]
        phase_values[name] = value.item() if isinstance(value, np.generic) else value
    try:
        phase_center = PhaseCenter(**cast(dict[str, Any], phase_values))
    except Exception as exc:
        raise UnsafeResultInputError("HDF5 phase-center contract is invalid") from exc

    beam = cast(dict[str, object], snapshots["beam"])
    resolved_beam = _snapshot_mapping(beam, "resolved", context="beam")
    assignments = _snapshot_list(
        resolved_beam,
        "assignments",
        context="beam.resolved",
    )
    assignment_identity: list[tuple[object, object]] = []
    for assignment in assignments:
        if type(assignment) is not dict:
            raise UnsafeResultInputError("HDF5 beam assignment is invalid")
        antenna_id = _snapshot_mapping(
            cast(dict[str, object], assignment),
            "antenna_id",
            context="beam assignment",
        )
        assignment_identity.append((antenna_id.get("number"), antenna_id.get("name")))
    if assignment_identity != list(zip(numbers.tolist(), names, strict=True)):
        raise UnsafeResultInputError(
            "HDF5 beam assignments disagree with instrument identity"
        )

    receptor_snapshot = _read_receptor_group(
        datasets,
        limits,
        basis=basis,
        antenna_numbers=[int(value) for value in numbers],
        antenna_names=names,
    )
    return time_grid, frequencies, widths, phase_center, labels, receptor_snapshot


def _load_open_file(
    handle: Any,
    h5py: Any,
    limits: HDF5ReadLimits,
) -> LoadedSimulationResult:
    root = _read_root_attributes(handle, h5py, limits)
    datasets = _inspect_tree(handle, h5py)
    counts = _checked_axis_counts(datasets)
    specs = _metadata_specs(
        time_count=counts[0],
        baseline_count=counts[1],
        frequency_count=counts[2],
        antenna_count=counts[3],
        visibility_dtype=counts[4],
    )
    for path, spec in specs.items():
        if path in _JONES_DATASETS and path not in datasets:
            # The one optional group.  ``_inspect_tree`` has already enforced
            # all-or-nothing, so a partially present group cannot arrive here.
            continue
        _validate_dataset_metadata(
            datasets[path],
            spec,
            h5py,
            path=path,
            limits=limits,
        )
    _enforce_axis_limits(counts, limits)
    _enforce_dataset_byte_limits(datasets, limits)
    _enforce_json_byte_limits(datasets, limits)
    snapshots = _read_json_snapshots(datasets, limits)
    (
        time_grid,
        frequencies,
        widths,
        phase_center,
        correlations,
        receptors_snapshot,
    ) = _validate_structured_identity(datasets, snapshots, limits)
    visibilities = _read_numeric(
        datasets["data/visibilities"],
        path="data/visibilities",
    )
    flags = _read_numeric(datasets["data/flags"], path="data/flags")
    weights = _read_numeric(datasets["data/weights"], path="data/weights")
    try:
        return build_loaded_simulation_result(
            visibilities=visibilities,
            flags=flags,
            weights=weights,
            time_grid=time_grid,
            frequencies_hz=frequencies,
            channel_widths_hz=widths,
            correlations=correlations,
            phase_center=phase_center,
            instrument_snapshot=cast(dict[str, object], snapshots["instrument"]),
            selection_snapshot=cast(dict[str, object], snapshots["selection"]),
            beam_snapshot=cast(dict[str, object], snapshots["beam"]),
            receptors_snapshot=receptors_snapshot,
            jones_snapshot=_read_jones_group(datasets, limits),
            backend_snapshot=cast(dict[str, object], snapshots["backend"]),
            solver_snapshot=cast(dict[str, object], snapshots["solver"]),
            resolved_config_snapshot=cast(
                dict[str, object],
                snapshots["resolved_config"],
            ),
            configuration_provenance_snapshot=cast(
                dict[str, object] | None,
                snapshots["configuration_source"],
            ),
            performance_snapshot=cast(dict[str, object], snapshots["performance"]),
            history=cast(list[str], snapshots["history"]),
            expected_scientific_sha256=root["scientific_sha256"],
            expected_provenance_sha256=root["provenance_sha256"],
        )
    except (TypeError, ValueError, InvalidResultError) as exc:
        raise UnsafeResultInputError(
            "HDF5 result failed canonical model or fingerprint validation"
        ) from exc


def _open_verified_binary(path: Path) -> Any:
    try:
        descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
    except OSError as exc:
        raise UnsafeResultInputError(
            f"could not safely open result input {path}"
        ) from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise UnsafeResultInputError(f"result input is not a regular file: {path}")
        signature = os.pread(descriptor, len(HDF5_SIGNATURE), 0)
        if signature != HDF5_SIGNATURE:
            raise UnsafeResultInputError(
                f"result input does not have an HDF5 signature: {path}"
            )
        return os.fdopen(descriptor, "rb", buffering=0)
    except Exception:
        os.close(descriptor)
        raise


def load_result_hdf5(
    path: str | Path,
    *,
    limits: HDF5ReadLimits = HDF5ReadLimits(),
) -> LoadedSimulationResult:
    """Load and fully validate a ``radiosim.visibility`` 4.0.0 result.

    No partial result is returned. Structural metadata and all allocation
    limits are validated before science payloads are read.
    """
    if type(limits) is not HDF5ReadLimits:
        raise TypeError("limits must be an exact HDF5ReadLimits")
    try:
        normalized = _validate_input_regular_file(path)
    except (OutputPathError, TypeError) as exc:
        raise UnsafeResultInputError(f"unsafe result input path: {path!s}") from exc
    stream = _open_verified_binary(normalized)
    handle: Any | None = None
    try:
        h5py = _import_h5py()
        try:
            handle = h5py.File(stream, "r")
        except Exception as exc:
            raise UnsafeResultInputError(
                f"result input is not a readable HDF5 file: {normalized}"
            ) from exc
        return _load_open_file(handle, h5py, limits)
    except (
        LegacyHDF5Error,
        OptionalResultDependencyError,
        UnsupportedSchemaVersionError,
    ):
        raise
    except UnsafeResultInputError:
        raise
    except Exception as exc:
        raise UnsafeResultInputError(
            f"result input failed bounded HDF5 validation: {normalized}"
        ) from exc
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        stream.close()


__all__ = [
    "HDF5ReadLimits",
    "load_result_hdf5",
    "write_result_hdf5",
]
