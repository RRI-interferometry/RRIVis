"""Tier 4D versioned HDF5 result and hostile-reader contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import radiosim.io.hdf5 as hdf5_module
from radiosim.api.simulator import Simulator
from radiosim.backends import get_backend
from radiosim.core.phase_center import PhaseCenter
from radiosim.core.result import (
    BackendResultProvenance,
    LoadedSimulationResult,
    ResultPerformance,
    SolverResultProvenance,
    build_simulation_result,
)
from radiosim.io.hdf5 import HDF5ReadLimits, load_result_hdf5, write_result_hdf5
from radiosim.io.result_errors import (
    FormatRepresentationError,
    LegacyHDF5Error,
    OptionalResultDependencyError,
    OutputPathError,
    UnsafeResultInputError,
    UnsupportedSchemaVersionError,
)
from tests.unit.test_core.test_result import _build, _mapping, _parts

ROOT_ATTRIBUTES = {
    "schema_name",
    "schema_version",
    "radiosim_version",
    "scientific_sha256",
    "provenance_sha256",
    "dimension_order",
    "visibility_unit",
}
DATASETS = {
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
GROUPS = {
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
LEGACY_GUIDANCE = (
    "Legacy unversioned RadioSim HDF5 is not accepted because baseline names "
    "were parsed unsafely and scientific fields were incomplete. Re-run the "
    "simulation with Tier 4 or convert a trusted file in an isolated pre-Tier-4 "
    "environment."
)


def _result(tmp_path: Path, dtype: str = "complex128", receptors=None):
    result, _ = _build(tmp_path, dtype=dtype, receptors=receptors)
    return result


def _independent_result(tmp_path: Path, dtype: str):
    simulator, backend, provenance, solver, performance, receptor = _parts(
        tmp_path,
        dtype=dtype,
    )
    receptor = (receptor * 0.25 + (37.0 - 11.0j)).astype(dtype)
    return build_simulation_result(
        receptor_visibilities=receptor,
        backend=backend,
        time_grid=simulator.config.observation.time_grid,
        frequencies_hz=simulator.config.frequency.channel_frequencies_hz,
        channel_widths_hz=simulator.config.frequency.channel_widths_hz,
        instrument=simulator.instrument,
        selection=simulator._instrument_state.selection,
        beam_state=simulator.beam_state,
        receptors=simulator.receptors,
        phase_center=PhaseCenter(),
        backend_provenance=provenance,
        solver_provenance=solver,
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=performance,
        history=("independent-h5py-probe",),
    )


def _unicode_antenna_result(tmp_path: Path):
    from radiosim.backends import get_backend
    from radiosim.core.result import (
        BackendResultProvenance,
        ResultPerformance,
        SolverResultProvenance,
    )

    mapping = _mapping(tmp_path)
    layout = Path(mapping["instrument"]["source"]["path"])
    layout.write_text(
        "Name Number BeamID E N U Diameter\n"
        "Antenna-é 0 0 0 0 0 14\n"
        "天線 1 0 10 0 0 14\n",
        encoding="utf-8",
    )
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    backend = get_backend("numpy")
    provenance = BackendResultProvenance(
        requested_backend="numpy",
        actual_backend=backend.name,
        requested_precision={"output": "complex128"},
        actual_precision={"output": "complex128"},
        result_dtype="complex128",
    )
    solver = SolverResultProvenance(
        solver="rime",
        sky_representation="point_sources",
        convention="radiosim.rime-zenith-drift.v1",
        execution_path="polarized",
        components=("point",),
        component_element_counts=(3,),
    )
    performance = ResultPerformance(
        setup_seconds=1.0,
        solver_seconds=2.0,
        solver_point_seconds=2.0,
        solver_healpix_seconds=0.0,
        result_construction_seconds=0.5,
        host_transfer_seconds=0.25,
        total_seconds=3.75,
    )
    receptor = np.arange(2 * 1 * 2 * 4, dtype=np.float64).reshape(2, 1, 2, 2, 2)
    receptor = receptor.astype(np.complex128)
    receptor += 1j * receptor
    return build_simulation_result(
        receptor_visibilities=receptor,
        backend=backend,
        time_grid=simulator.config.observation.time_grid,
        frequencies_hz=simulator.config.frequency.channel_frequencies_hz,
        channel_widths_hz=simulator.config.frequency.channel_widths_hz,
        instrument=simulator.instrument,
        selection=simulator._instrument_state.selection,
        beam_state=simulator.beam_state,
        receptors=simulator.receptors,
        phase_center=PhaseCenter(),
        backend_provenance=provenance,
        solver_provenance=solver,
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=performance,
        history=("unicode-antennas",),
    )


def _multi_baseline_result(tmp_path: Path):
    _unused, backend, provenance, solver, performance, _receptor = _parts(tmp_path)
    mapping = _mapping(tmp_path)
    layout = Path(mapping["instrument"]["source"]["path"])
    layout.write_text(
        "Name Number BeamID E N U Diameter\n"
        "A0 0 0 0 0 0 14\n"
        "A1 1 0 10 0 0 14\n"
        "A2 2 0 0 20 0 14\n",
        encoding="utf-8",
    )
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    baseline_count = len(simulator._instrument_state.selection.baselines)
    receptor = np.arange(
        2 * baseline_count * 2 * 4,
        dtype=np.float64,
    ).reshape(2, baseline_count, 2, 2, 2)
    receptor = receptor.astype(np.complex128)
    receptor += 1j * receptor
    return build_simulation_result(
        receptor_visibilities=receptor,
        backend=backend,
        time_grid=simulator.config.observation.time_grid,
        frequencies_hz=simulator.config.frequency.channel_frequencies_hz,
        channel_widths_hz=simulator.config.frequency.channel_widths_hz,
        instrument=simulator.instrument,
        selection=simulator._instrument_state.selection,
        beam_state=simulator.beam_state,
        receptors=simulator.receptors,
        phase_center=PhaseCenter(),
        backend_provenance=provenance,
        solver_provenance=solver,
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=performance,
        history=("multi-baseline",),
    )


def _object_paths(handle: h5py.File) -> tuple[set[str], set[str]]:
    groups: set[str] = set()
    datasets: set[str] = set()

    def record(name: str, value: h5py.Group | h5py.Dataset) -> None:
        if isinstance(value, h5py.Group):
            groups.add(name)
        else:
            datasets.add(name)

    handle.visititems(record)
    return groups, datasets


def _replace_dataset(
    path: Path,
    dataset_path: str,
    *,
    data: object | None = None,
    dtype: object | None = None,
    chunks: tuple[int, ...] | None = None,
    compression: str | None = None,
    compression_opts: int | None = None,
    shuffle: bool = False,
    fletcher32: bool = False,
) -> None:
    with h5py.File(path, "r+") as handle:
        original = handle[dataset_path]
        assert isinstance(original, h5py.Dataset)
        payload = original[()] if data is None else data
        attributes = dict(original.attrs)
        del handle[dataset_path]
        replacement = handle.create_dataset(
            dataset_path,
            data=payload,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            fletcher32=fletcher32,
        )
        for key, value in attributes.items():
            replacement.attrs[key] = value


def _write_text_probe(
    tmp_path: Path,
    payload: object,
    *,
    indexed: bool = False,
) -> Path:
    path = tmp_path / "bounded-text-probe.h5"
    with h5py.File(path, "w") as handle:
        data = np.asarray(payload, dtype=object) if indexed else payload
        handle.create_dataset(
            "value",
            data=data,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    return path


def _record_dataset_casts(monkeypatch) -> list[str]:
    calls: list[str] = []
    original_astype = h5py.Dataset.astype

    def recording_astype(dataset, *args, **kwargs):
        calls.append(dataset.name)
        return original_astype(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "astype", recording_astype)
    return calls


def _write_fixed_utf8_payload(dataset: h5py.Dataset, payload: object) -> None:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if isinstance(payload, bytes):
        encoded = np.asarray(payload, dtype=np.dtype(f"S{max(1, len(payload))}"))
    else:
        values = tuple(
            value.encode("utf-8") if isinstance(value, str) else value
            for value in payload
        )
        width = max(1, max((len(value) for value in values), default=0))
        encoded = np.asarray(values, dtype=np.dtype(f"S{width}"))
    width = int(encoded.dtype.itemsize)
    memory_type = h5py.h5t.C_S1.copy()
    file_space = dataset.id.get_space()
    memory_space = (
        h5py.h5s.create(h5py.h5s.SCALAR)
        if encoded.shape == ()
        else h5py.h5s.create_simple(encoded.shape)
    )
    try:
        memory_type.set_size(width)
        memory_type.set_cset(h5py.h5t.CSET_UTF8)
        memory_type.set_strpad(h5py.h5t.STR_NULLPAD)
        dataset.id.write(memory_space, file_space, encoded, mtype=memory_type)
    finally:
        memory_space.close()
        file_space.close()
        memory_type.close()


def _replace_fixed_dataset(
    path: Path,
    dataset_path: str,
    payload: object,
    *,
    width: int | None = None,
) -> None:
    with h5py.File(path, "r+") as handle:
        original = handle[dataset_path]
        assert isinstance(original, h5py.Dataset)
        attributes = dict(original.attrs)
        shape = tuple(int(value) for value in original.shape)
        if isinstance(payload, str):
            values = payload.encode("utf-8")
            inferred_width = len(values)
        elif isinstance(payload, bytes):
            values = payload
            inferred_width = len(payload)
        else:
            values = tuple(
                value.encode("utf-8") if isinstance(value, str) else value
                for value in payload
            )
            inferred_width = max((len(value) for value in values), default=0)
        itemsize = max(1, inferred_width) if width is None else width
        assert itemsize > 0
        del handle[dataset_path]
        replacement = handle.create_dataset(
            dataset_path,
            shape=shape,
            dtype=h5py.string_dtype(encoding="utf-8", length=itemsize),
        )
        _write_fixed_utf8_payload(replacement, values)
        for key, value in attributes.items():
            replacement.attrs[key] = value


def _write_fixed_text_probe(
    tmp_path: Path,
    payload: object,
    *,
    indexed: bool = False,
    width: int | None = None,
) -> Path:
    path = tmp_path / "fixed-text-probe.h5"
    if indexed:
        values = tuple(
            value.encode("utf-8") if isinstance(value, str) else value
            for value in payload
        )
        inferred_width = max((len(value) for value in values), default=0)
        shape = (len(values),)
    else:
        values = payload.encode("utf-8") if isinstance(payload, str) else payload
        inferred_width = len(values)
        shape = ()
    itemsize = max(1, inferred_width) if width is None else width
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "value",
            shape=shape,
            dtype=h5py.string_dtype(encoding="utf-8", length=itemsize),
        )
        _write_fixed_utf8_payload(dataset, values)
    return path


def _read_fixed_text_probe(
    tmp_path: Path,
    payload: object,
    *,
    limit: int,
    index: int | None = None,
    width: int | None = None,
) -> tuple[str, int]:
    path = _write_fixed_text_probe(
        tmp_path,
        payload,
        indexed=index is not None,
        width=width,
    )
    with h5py.File(path, "r") as handle:
        return hdf5_module._bounded_dataset_text(
            handle["value"],
            path="value",
            limit=limit,
            index=index,
        )


def test_hdf5_read_limits_are_exact_frozen_positive_integer_contract():
    limits = HDF5ReadLimits()

    assert repr(limits) == (
        "HDF5ReadLimits(max_time=10000000, max_baseline=10000000, "
        "max_frequency=1000000, max_antenna=1000000, "
        "max_visibility_elements=100000000, "
        "max_single_dataset_bytes=2147483648, "
        "max_total_json_bytes=67108864, max_single_string_bytes=1048576)"
    )
    assert limits == HDF5ReadLimits()
    with pytest.raises(FrozenInstanceError):
        limits.max_time = 1
    assert not hasattr(limits, "__dict__")

    for value in (True, False, 0, -1, 1.0, np.int64(1)):
        with pytest.raises((TypeError, ValueError)):
            HDF5ReadLimits(max_time=value)

    class HostileInt(int):
        pass

    with pytest.raises(TypeError):
        HDF5ReadLimits(max_time=HostileInt(1))


def test_radiosim_io_lazily_exports_exact_hdf5_objects_without_importing_h5py():
    script = """
import sys
import radiosim.io as io
assert "h5py" not in sys.modules
from radiosim.io import HDF5ReadLimits, load_result_hdf5, write_result_hdf5
assert "h5py" not in sys.modules
from radiosim.io.hdf5 import (
    HDF5ReadLimits as DirectLimits,
    load_result_hdf5 as direct_load,
    write_result_hdf5 as direct_write,
)
assert HDF5ReadLimits is DirectLimits
assert load_result_hdf5 is direct_load
assert write_result_hdf5 is direct_write
for name in ("pyuvdata", "casacore", "jax", "bokeh", "matplotlib", "webbrowser"):
    assert name not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_versioned_hdf5_round_trip_is_scientifically_exact(tmp_path, dtype):
    result = _result(tmp_path, dtype)
    output = tmp_path / f"{dtype}.h5"

    returned = write_result_hdf5(result, output)
    loaded = load_result_hdf5(output)

    assert returned == output.absolute()
    assert type(loaded) is LoadedSimulationResult
    assert loaded.scientifically_equal(result)
    assert loaded.visibilities.dtype == np.dtype(dtype)
    assert loaded.weights.dtype == (
        np.dtype("float32") if dtype == "complex64" else np.dtype("float64")
    )
    assert loaded.scientific_sha256 == result.scientific_sha256
    assert loaded.provenance_sha256 == result.provenance_sha256
    assert loaded.history == result.history


def test_writer_reader_round_trip_preserves_non_ascii_history(tmp_path):
    import radiosim.core.result as result_module

    result = _result(tmp_path)
    history = ("simulé 東京",)
    object.__setattr__(result, "history", history)
    object.__setattr__(
        result,
        "provenance_sha256",
        result_module._provenance_hash(
            scientific_sha256=result.scientific_sha256,
            backend_snapshot=result.backend.to_snapshot(),
            resolved_config=result.resolved_config,
            configuration_provenance=result.configuration_provenance,
            history=history,
        ),
    )

    output = write_result_hdf5(result, tmp_path / "non-ascii-history.h5")
    loaded = load_result_hdf5(output)

    assert loaded.history == history


def test_writer_uses_encoded_fixed_utf8_width_for_unicode_antenna_names(tmp_path):
    result = _unicode_antenna_result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "unicode-antennas.h5")
    names = tuple(antenna.id.name for antenna in result.instrument.antennas)
    expected_width = max(len(name.encode("utf-8")) for name in names)

    with h5py.File(output, "r") as handle:
        dataset = handle["instrument/antenna/name"]
        info = h5py.check_string_dtype(dataset.dtype)
        assert info is not None
        assert info.encoding == "utf-8"
        assert info.length == expected_width
        assert dataset.dtype.kind == "S"
        assert dataset.shape == (len(names),)
        assert dataset[:].dtype == np.dtype(f"S{expected_width}")
        assert dataset.asstr()[:].tolist() == list(names)

    loaded = load_result_hdf5(output)
    assert loaded.scientifically_equal(result)
    assert loaded.provenance_sha256 == result.provenance_sha256


@pytest.mark.parametrize(
    ("dtype", "visibility_dtype", "weight_dtype"),
    [
        ("complex64", "<c8", "<f4"),
        ("complex128", "<c16", "<f8"),
    ],
)
def test_independent_h5py_inspection_matches_exact_schema(
    tmp_path,
    dtype,
    visibility_dtype,
    weight_dtype,
):
    result = _independent_result(tmp_path, dtype)
    output = write_result_hdf5(result, tmp_path / f"inspect-{dtype}.h5")

    with h5py.File(output, "r") as handle:
        groups, datasets = _object_paths(handle)
        assert groups == GROUPS
        assert datasets == DATASETS
        assert set(handle.attrs) == ROOT_ATTRIBUTES
        root_values = {}
        for name in ROOT_ATTRIBUTES:
            attribute_id = handle.attrs.get_id(name)
            type_id = attribute_id.get_type()
            assert attribute_id.shape == ()
            assert type_id.get_class() == h5py.h5t.STRING
            assert type_id.is_variable_str() is False
            assert type_id.get_cset() == h5py.h5t.CSET_UTF8
            assert attribute_id.get_storage_size() == type_id.get_size()
            value = handle.attrs[name]
            assert type(value) is np.bytes_
            root_values[name] = bytes(value).decode("utf-8", errors="strict")
        assert root_values["schema_name"] == "radiosim.visibility"
        assert root_values["schema_version"] == "3.0.0"
        assert root_values["dimension_order"] == "time,baseline,frequency,correlation"
        assert root_values["visibility_unit"] == "Jy"
        assert root_values["scientific_sha256"] == result.scientific_sha256
        assert root_values["provenance_sha256"] == result.provenance_sha256
        assert "creation_time" not in handle.attrs

        vis = handle["data/visibilities"]
        flags = handle["data/flags"]
        weights = handle["data/weights"]
        assert vis.dtype.str == visibility_dtype
        assert flags.dtype == np.dtype("bool")
        assert weights.dtype.str == weight_dtype
        assert vis.shape == result.visibilities.shape
        np.testing.assert_array_equal(vis[:], result.visibilities)
        np.testing.assert_array_equal(flags[:], result.flags)
        np.testing.assert_array_equal(weights[:], result.weights)
        assert result.history == ("independent-h5py-probe",)
        assert vis.chunks == (2, 1, 2, 4)
        assert weights.chunks == vis.chunks
        assert flags.chunks == vis.chunks
        for dataset in (vis, weights):
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 4
            assert dataset.shuffle is True
            assert dataset.fletcher32 is True
        assert flags.compression == "gzip"
        assert flags.compression_opts == 4
        assert flags.shuffle is False
        assert flags.fletcher32 is True
        assert tuple(dimension.label for dimension in vis.dims) == (
            "time",
            "baseline",
            "frequency",
            "correlation",
        )

        labels = handle["coordinates/correlation/labels"]
        codes = handle["coordinates/correlation/aips_codes"]
        assert labels.dtype == np.dtype("S2")
        np.testing.assert_array_equal(labels[:], [b"XX", b"XY", b"YX", b"YY"])
        np.testing.assert_array_equal(codes[:], [-5, -7, -8, -6])
        assert labels.chunks is None
        assert labels.compression is None
        assert labels.fletcher32 is False

        np.testing.assert_array_equal(
            handle["coordinates/time/utc_jd1"][:],
            result.time_grid.utc_jd1,
        )
        np.testing.assert_array_equal(
            handle["coordinates/time/utc_jd2"][:],
            result.time_grid.utc_jd2,
        )
        np.testing.assert_array_equal(
            handle["coordinates/frequency/center_hz"][:],
            result.frequencies_hz,
        )
        np.testing.assert_array_equal(
            handle["coordinates/frequency/channel_width_hz"][:],
            result.channel_widths_hz,
        )
        np.testing.assert_array_equal(
            handle["coordinates/baseline/antenna1_number"][:],
            [baseline.ant1.number for baseline in result.selection.baselines],
        )
        np.testing.assert_array_equal(
            handle["coordinates/baseline/antenna2_number"][:],
            [baseline.ant2.number for baseline in result.selection.baselines],
        )
        np.testing.assert_array_equal(
            handle["instrument/antenna/number"][:],
            [antenna.id.number for antenna in result.instrument.antennas],
        )

        utf8_paths = {
            "instrument/name",
            "instrument/antenna/name",
            "phase_center/kind",
            "phase_center/frame",
            "phase_center/w_reference",
            *(path for path in DATASETS if path.startswith("provenance/")),
        }
        for dataset_path in utf8_paths:
            dataset = handle[dataset_path]
            string_info = h5py.check_string_dtype(dataset.dtype)
            assert string_info is not None
            assert string_info.encoding == "utf-8"
            assert type(string_info.length) is int
            assert string_info.length > 0
            assert dataset.dtype.kind == "S"
            type_id = dataset.id.get_type()
            try:
                assert type_id.is_variable_str() is False
                assert type_id.get_cset() == h5py.h5t.CSET_UTF8
                assert type_id.get_strpad() == h5py.h5t.STR_NULLPAD
                assert type_id.get_size() == string_info.length
            finally:
                type_id.close()
            assert dataset.chunks is None
            assert dataset.compression is None
            assert dataset.shuffle is False
            assert dataset.fletcher32 is False


def test_writer_rejects_complex256_before_dependency_or_filesystem_side_effect(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path, "complex128")
    object.__setattr__(
        result,
        "visibilities",
        SimpleNamespace(dtype=SimpleNamespace(kind="c", itemsize=32)),
    )
    output = tmp_path / "missing" / "unsupported.h5"
    monkeypatch.setattr(
        hdf5_module,
        "_import_h5py",
        lambda: (_ for _ in ()).throw(AssertionError("h5py imported")),
    )

    with pytest.raises(FormatRepresentationError, match="complex256"):
        write_result_hdf5(result, output)

    assert not output.parent.exists()


def test_writer_rejects_embedded_nul_before_dependency_or_filesystem_side_effect(
    tmp_path,
    monkeypatch,
):
    import radiosim.core.result as result_module

    result = _result(tmp_path)
    history = ("contains\x00nul",)
    object.__setattr__(result, "history", history)
    object.__setattr__(
        result,
        "provenance_sha256",
        result_module._provenance_hash(
            scientific_sha256=result.scientific_sha256,
            backend_snapshot=result.backend.to_snapshot(),
            resolved_config=result.resolved_config,
            configuration_provenance=result.configuration_provenance,
            history=history,
        ),
    )
    output = tmp_path / "missing" / "embedded-nul.h5"
    monkeypatch.setattr(
        hdf5_module,
        "_import_h5py",
        lambda: (_ for _ in ()).throw(AssertionError("h5py imported")),
    )

    with pytest.raises(FormatRepresentationError, match="NUL"):
        write_result_hdf5(result, output)

    assert not output.parent.exists()


def test_writer_reports_missing_h5py_before_parent_creation(tmp_path, monkeypatch):
    result = _result(tmp_path)
    output = tmp_path / "missing" / "result.h5"
    monkeypatch.setattr(
        hdf5_module,
        "_import_h5py",
        lambda: (_ for _ in ()).throw(
            OptionalResultDependencyError("HDF5 requires h5py")
        ),
    )

    with pytest.raises(OptionalResultDependencyError, match="h5py"):
        write_result_hdf5(result, output)

    assert not output.parent.exists()


def test_reader_closes_verified_stream_when_h5py_import_fails(tmp_path, monkeypatch):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "dependency.h5")
    opened = []
    original_open = hdf5_module._open_verified_binary

    def recording_open(path):
        stream = original_open(path)
        opened.append(stream)
        return stream

    monkeypatch.setattr(hdf5_module, "_open_verified_binary", recording_open)
    monkeypatch.setattr(
        hdf5_module,
        "_import_h5py",
        lambda: (_ for _ in ()).throw(
            OptionalResultDependencyError("HDF5 requires h5py")
        ),
    )

    with pytest.raises(OptionalResultDependencyError, match="h5py"):
        load_result_hdf5(output)

    assert len(opened) == 1
    assert opened[0].closed


@pytest.mark.parametrize(
    "name",
    ("result", "result.H5", "result.hdf5", "result.tar.h5"),
)
def test_writer_requires_one_exact_canonical_extension(tmp_path, name):
    result = _result(tmp_path)

    with pytest.raises(OutputPathError):
        write_result_hdf5(result, tmp_path / name)


def test_legacy_hdf5_is_rejected_without_evaluating_baseline_text(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "legacy.h5"
    hostile = (
        "baseline_(__import__('pathlib').Path("
        + repr(str(marker))
        + ").write_text('owned'), 1)"
    )
    with h5py.File(path, "w") as handle:
        handle.create_group(hostile)
        handle.create_dataset("frequencies", data=np.array([100e6]))

    with pytest.raises(LegacyHDF5Error) as caught:
        load_result_hdf5(path)

    assert str(caught.value) == LEGACY_GUIDANCE
    assert not marker.exists()


@pytest.mark.parametrize(
    "kind",
    (
        "missing",
        "directory",
        "symlink",
        "broken_symlink",
        "random",
        "truncated",
        "empty",
        "fifo",
    ),
)
def test_reader_rejects_non_regular_or_non_hdf5_input(tmp_path, kind):
    path = tmp_path / "hostile.h5"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.h5"
        target.write_bytes(b"not hdf5")
        path.symlink_to(target)
    elif kind == "broken_symlink":
        path.symlink_to(tmp_path / "absent.h5")
    elif kind == "random":
        path.write_bytes(os.urandom(64))
    elif kind == "truncated":
        path.write_bytes(hdf5_module.HDF5_SIGNATURE + b"\x00" * 8)
    elif kind == "empty":
        path.touch()
    elif kind == "fifo":
        os.mkfifo(path)

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(path)


def test_unknown_schema_version_fails_before_science_payload_read(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "unknown.h5")
    with h5py.File(output, "r+") as handle:
        del handle.attrs["schema_version"]
        version = b"999.0.0"
        handle.attrs.create(
            "schema_version",
            np.bytes_(version),
            shape=(),
            dtype=h5py.string_dtype(encoding="utf-8", length=len(version)),
        )

    reads: list[str] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(dataset, key, **kwargs):
        reads.append(dataset.name)
        return original_getitem(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    with pytest.raises(UnsupportedSchemaVersionError, match="999.0.0"):
        load_result_hdf5(output)
    assert not any(name.startswith("/data/") for name in reads)


def test_unknown_schema_version_precedes_other_root_attribute_reads(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "unknown-root-order.h5")
    with h5py.File(output, "r+") as handle:
        del handle.attrs["schema_version"]
        version = b"999.0.0"
        handle.attrs.create(
            "schema_version",
            np.bytes_(version),
            shape=(),
            dtype=h5py.string_dtype(encoding="utf-8", length=len(version)),
        )
        del handle.attrs["radiosim_version"]
        handle.attrs["radiosim_version"] = "hostile variable-length payload"

    reads: list[str] = []
    original_getitem = h5py.AttributeManager.__getitem__

    def recording_getitem(attributes, name):
        reads.append(name)
        return original_getitem(attributes, name)

    monkeypatch.setattr(h5py.AttributeManager, "__getitem__", recording_getitem)
    with pytest.raises(UnsupportedSchemaVersionError, match="999.0.0"):
        load_result_hdf5(output)
    assert reads == ["schema_name", "schema_version"]


@pytest.mark.parametrize("attribute_name", ["schema_name", "radiosim_version"])
@pytest.mark.parametrize("storage", ["fixed", "variable"])
def test_oversized_root_attribute_is_rejected_before_value_read(
    tmp_path,
    monkeypatch,
    attribute_name,
    storage,
):
    result = _result(tmp_path)
    output = write_result_hdf5(
        result,
        tmp_path / f"oversized-root-{attribute_name}-{storage}.h5",
    )
    payload = b"x" * 4096
    with h5py.File(output, "r+") as handle:
        del handle.attrs[attribute_name]
        if storage == "fixed":
            handle.attrs.create(
                attribute_name,
                np.bytes_(payload),
                shape=(),
                dtype=h5py.string_dtype(encoding="utf-8", length=len(payload)),
            )
        else:
            handle.attrs[attribute_name] = payload.decode("ascii")

    reads: list[str] = []
    original_getitem = h5py.AttributeManager.__getitem__

    def recording_getitem(attributes, name):
        reads.append(name)
        return original_getitem(attributes, name)

    monkeypatch.setattr(h5py.AttributeManager, "__getitem__", recording_getitem)
    with pytest.raises(
        UnsafeResultInputError,
        match=f"root attribute {attribute_name}",
    ):
        load_result_hdf5(
            output,
            limits=HDF5ReadLimits(max_single_string_bytes=64),
        )
    assert attribute_name not in reads


@pytest.mark.parametrize(
    ("mutation", "error_type"),
    [
        ("missing_schema_name", LegacyHDF5Error),
        ("missing_root_attribute", UnsafeResultInputError),
        ("wrong_schema_name", UnsafeResultInputError),
        ("wrong_root_type", UnsafeResultInputError),
        ("oversized_root_string", UnsafeResultInputError),
        ("extra_root_attribute", UnsafeResultInputError),
        ("missing_group", UnsafeResultInputError),
        ("extra_group", UnsafeResultInputError),
        ("missing_dataset", UnsafeResultInputError),
        ("extra_dataset", UnsafeResultInputError),
        ("soft_link", UnsafeResultInputError),
        ("external_link", UnsafeResultInputError),
        ("hard_alias", UnsafeResultInputError),
        ("object_reference", UnsafeResultInputError),
        ("wrong_dimension_label", UnsafeResultInputError),
        ("wrong_unit", UnsafeResultInputError),
        ("scientific_fingerprint", UnsafeResultInputError),
        ("provenance_fingerprint", UnsafeResultInputError),
    ],
)
def test_hostile_schema_and_fingerprint_matrix(tmp_path, mutation, error_type):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / f"{mutation}.h5")
    with h5py.File(output, "r+") as handle:
        if mutation == "missing_schema_name":
            del handle.attrs["schema_name"]
        elif mutation == "missing_root_attribute":
            del handle.attrs["visibility_unit"]
        elif mutation == "wrong_schema_name":
            handle.attrs.modify("schema_name", "other.schema")
        elif mutation == "wrong_root_type":
            del handle.attrs["visibility_unit"]
            handle.attrs["visibility_unit"] = np.int64(1)
        elif mutation == "oversized_root_string":
            del handle.attrs["radiosim_version"]
            payload = b"x" * 1_048_577
            handle.attrs.create(
                "radiosim_version",
                np.bytes_(payload),
                shape=(),
                dtype=h5py.string_dtype(encoding="utf-8", length=len(payload)),
            )
        elif mutation == "extra_root_attribute":
            handle.attrs["unknown"] = "x"
        elif mutation == "missing_group":
            del handle["phase_center"]
        elif mutation == "extra_group":
            handle.create_group("unexpected_group")
        elif mutation == "missing_dataset":
            del handle["coordinates/frequency/channel_width_hz"]
        elif mutation == "extra_dataset":
            handle.create_dataset("unexpected", data=np.int64(1))
        elif mutation == "soft_link":
            handle["soft"] = h5py.SoftLink("/data/visibilities")
        elif mutation == "external_link":
            handle["external"] = h5py.ExternalLink("elsewhere.h5", "/x")
        elif mutation == "hard_alias":
            handle["alias"] = handle["data/visibilities"]
        elif mutation == "object_reference":
            handle.create_dataset(
                "reference",
                data=np.asarray([handle["data/visibilities"].ref]),
                dtype=h5py.ref_dtype,
            )
        elif mutation == "wrong_dimension_label":
            handle["data/visibilities"].dims[0].label = "sample"
        elif mutation == "wrong_unit":
            handle["coordinates/frequency/center_hz"].attrs.modify("unit", "MHz")
        elif mutation == "scientific_fingerprint":
            handle.attrs.modify("scientific_sha256", "0" * 64)
        elif mutation == "provenance_fingerprint":
            handle.attrs.modify("provenance_sha256", "0" * 64)

    with pytest.raises(error_type):
        load_result_hdf5(output)


@pytest.mark.parametrize(
    ("mutation", "dataset_path"),
    [
        ("wrong_dtype", "data/visibilities"),
        ("wrong_rank", "data/visibilities"),
        ("zero_length_axis", "data/visibilities"),
        ("wrong_chunks", "data/visibilities"),
        ("wrong_compression", "data/visibilities"),
        ("wrong_gzip_level", "data/visibilities"),
        ("wrong_shuffle", "data/visibilities"),
        ("missing_fletcher32", "data/visibilities"),
    ],
)
def test_hostile_dataset_storage_matrix(tmp_path, mutation, dataset_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / f"{mutation}.h5")
    with h5py.File(output, "r") as handle:
        payload = handle[dataset_path][:]
    kwargs = {
        "dtype": np.dtype("<c16"),
        "chunks": (2, 1, 2, 4),
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
        "fletcher32": True,
    }
    if mutation == "wrong_dtype":
        kwargs["dtype"] = np.dtype("<f8")
        payload = payload.real
    elif mutation == "wrong_rank":
        payload = payload.reshape(-1)
        kwargs["chunks"] = (payload.size,)
    elif mutation == "zero_length_axis":
        payload = np.empty((0, 1, 2, 4), dtype="<c16")
        kwargs.update(
            chunks=None,
            compression=None,
            compression_opts=None,
            shuffle=False,
            fletcher32=False,
        )
    elif mutation == "wrong_chunks":
        kwargs["chunks"] = (1, 1, 1, 4)
    elif mutation == "wrong_compression":
        kwargs["compression"] = "lzf"
        kwargs["compression_opts"] = None
    elif mutation == "wrong_gzip_level":
        kwargs["compression_opts"] = 1
    elif mutation == "wrong_shuffle":
        kwargs["shuffle"] = False
    elif mutation == "missing_fletcher32":
        kwargs["fletcher32"] = False
    _replace_dataset(output, dataset_path, data=payload, **kwargs)

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output)


@pytest.mark.parametrize("dataset_path", ("data/flags", "data/weights"))
def test_science_dataset_shape_mismatch_is_rejected(tmp_path, dataset_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "shape-mismatch.h5")
    with h5py.File(output, "r") as handle:
        original = handle[dataset_path]
        payload = original[:1]
        dtype = original.dtype
        shuffle = original.shuffle
    _replace_dataset(
        output,
        dataset_path,
        data=payload,
        dtype=dtype,
        chunks=(1, 1, 2, 4),
        compression="gzip",
        compression_opts=4,
        shuffle=shuffle,
        fletcher32=True,
    )

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output)


@pytest.mark.parametrize(
    ("mutation", "dtype", "compression", "fletcher32"),
    [
        ("wrong_byte_order", ">f8", None, True),
        ("unexpected_filter", "<f8", "gzip", False),
    ],
)
def test_coordinate_storage_contract_rejects_hostile_metadata(
    tmp_path,
    mutation,
    dtype,
    compression,
    fletcher32,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / f"{mutation}.h5")
    _replace_dataset(
        output,
        "coordinates/frequency/center_hz",
        dtype=np.dtype(dtype),
        chunks=(2,),
        compression=compression,
        compression_opts=4 if compression == "gzip" else None,
        fletcher32=fletcher32,
    )

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output)


@pytest.mark.parametrize(
    ("dataset_path", "replacement"),
    [
        (
            "coordinates/time/utc_jd1",
            np.array([np.nan, np.nan], dtype="<f8"),
        ),
        ("coordinates/time/utc_jd2", np.array([0.5, 0.4], dtype="<f8")),
        (
            "coordinates/time/integration_time_seconds",
            np.array([1.0, 0.0], dtype="<f8"),
        ),
        (
            "coordinates/frequency/center_hz",
            np.array([101e6, 100e6], dtype="<f8"),
        ),
        (
            "coordinates/frequency/channel_width_hz",
            np.array([1e6, 0.0], dtype="<f8"),
        ),
        (
            "coordinates/correlation/labels",
            np.array([b"XX", b"YX", b"XY", b"YY"], dtype="S2"),
        ),
        (
            "coordinates/correlation/aips_codes",
            np.array([-5, -8, -7, -6], dtype="<i4"),
        ),
        (
            "coordinates/baseline/antenna2_number",
            np.array([999], dtype="<i8"),
        ),
        (
            "coordinates/baseline/vector_enu_m",
            np.array([[9.0, 9.0, 9.0]], dtype="<f8"),
        ),
        (
            "instrument/antenna/number",
            np.array([0, 0], dtype="<i8"),
        ),
    ],
)
def test_hostile_coordinate_and_identity_matrix(
    tmp_path,
    dataset_path,
    replacement,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "coordinate.h5")
    with h5py.File(output, "r") as handle:
        original = handle[dataset_path]
        chunks = original.chunks
        fletcher32 = original.fletcher32
    _replace_dataset(
        output,
        dataset_path,
        data=replacement,
        dtype=replacement.dtype,
        chunks=chunks,
        fletcher32=fletcher32,
    )

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output)


def test_reordered_baseline_identity_is_rejected(tmp_path):
    result = _multi_baseline_result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "reordered-baseline.h5")
    for path in (
        "coordinates/baseline/antenna1_number",
        "coordinates/baseline/antenna2_number",
        "coordinates/baseline/vector_enu_m",
    ):
        with h5py.File(output, "r") as handle:
            original = handle[path]
            replacement = original[...][::-1]
            dtype = original.dtype
            chunks = original.chunks
            fletcher32 = original.fletcher32
        _replace_dataset(
            output,
            path,
            data=replacement,
            dtype=dtype,
            chunks=chunks,
            fletcher32=fletcher32,
        )

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output)


def test_invalid_phase_center_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "invalid-phase.h5")
    _replace_dataset(
        output,
        "phase_center/geometric_phase_sign",
        data=np.int8(0),
        dtype=np.dtype("i1"),
    )

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output)


@pytest.mark.parametrize(
    ("dataset_path", "payload"),
    [
        ("provenance/history_json", "{"),
        ("provenance/history_json", "[NaN]"),
        ("provenance/history_json", "{}"),
        ("provenance/instrument_json", '{"bad":"\\u0000"}'),
    ],
)
def test_hostile_json_matrix(tmp_path, dataset_path, payload):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "json.h5")
    _replace_fixed_dataset(
        output,
        dataset_path,
        payload.encode("utf-8"),
    )

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output)


def test_malformed_utf8_string_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "utf8.h5")
    _replace_fixed_dataset(
        output,
        "provenance/history_json",
        b"\xff",
    )

    with pytest.raises(UnsafeResultInputError, match="UTF-8"):
        load_result_hdf5(output)


def test_embedded_nul_root_string_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "nul.h5")
    with h5py.File(output, "r+") as handle:
        del handle.attrs["visibility_unit"]
        payload = b"J\x00y"
        handle.attrs.create(
            "visibility_unit",
            np.bytes_(payload),
            shape=(),
            dtype=h5py.string_dtype(encoding="utf-8", length=len(payload)),
        )

    with pytest.raises(UnsafeResultInputError, match="bounded string"):
        load_result_hdf5(output)


def test_oversized_utf8_dataset_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "oversized-string.h5")
    _replace_fixed_dataset(
        output,
        "instrument/name",
        b"x" * 1_048_577,
    )

    with pytest.raises(UnsafeResultInputError, match="max_single_string_bytes"):
        load_result_hdf5(output)


def test_oversized_scalar_vlen_dataset_rejects_before_high_level_value_access(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "oversized-scalar-vlen.h5")
    _replace_dataset(
        output,
        "instrument/name",
        data="s" * (4 * 1024 * 1024),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    casts = _record_dataset_casts(monkeypatch)

    with pytest.raises(
        UnsafeResultInputError,
        match="variable-length UTF-8",
    ):
        load_result_hdf5(
            output,
            limits=HDF5ReadLimits(max_single_string_bytes=1_048_576),
        )

    assert "/instrument/name" not in casts


def test_oversized_indexed_vlen_dataset_rejects_before_high_level_value_access(
    tmp_path,
    monkeypatch,
):
    result = _multi_baseline_result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "oversized-indexed-vlen.h5")
    _replace_dataset(
        output,
        "instrument/antenna/name",
        data=["small-left", "i" * (4 * 1024 * 1024), "small-right"],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    casts = _record_dataset_casts(monkeypatch)

    with pytest.raises(
        UnsafeResultInputError,
        match="variable-length UTF-8",
    ):
        load_result_hdf5(
            output,
            limits=HDF5ReadLimits(max_single_string_bytes=1_048_576),
        )

    assert "/instrument/antenna/name" not in casts


def test_oversized_vlen_provenance_json_rejects_before_high_level_value_access(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "oversized-provenance-vlen.h5")
    _replace_dataset(
        output,
        "provenance/history_json",
        data="p" * (4 * 1024 * 1024),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    casts = _record_dataset_casts(monkeypatch)

    with pytest.raises(
        UnsafeResultInputError,
        match="variable-length UTF-8",
    ):
        load_result_hdf5(
            output,
            limits=HDF5ReadLimits(max_single_string_bytes=1_048_576),
        )

    assert "/provenance/history_json" not in casts


@pytest.mark.parametrize("text", ["café", "東京"])
def test_bounded_scalar_fixed_utf8_returns_strict_text_and_byte_count(
    tmp_path,
    text,
):
    expected_bytes = len(text.encode("utf-8"))

    assert _read_fixed_text_probe(
        tmp_path,
        text,
        limit=expected_bytes,
    ) == (text, expected_bytes)


def test_bounded_indexed_fixed_utf8_returns_only_selected_non_ascii_text(tmp_path):
    values = [b"left", "中央", b"right"]

    assert _read_fixed_text_probe(
        tmp_path,
        values,
        limit=len("中央".encode()),
        index=1,
    ) == ("中央", len("中央".encode()))


@pytest.mark.parametrize(
    ("payload", "limit", "expected"),
    [
        ("abcd", 4, ("abcd", 4)),
        ("abcde", 4, None),
        ("éé", 4, ("éé", 4)),
        ("ééé", 5, None),
    ],
)
def test_bounded_fixed_utf8_distinguishes_exact_and_one_over_byte_limits(
    tmp_path,
    payload,
    limit,
    expected,
):
    if expected is None:
        with pytest.raises(UnsafeResultInputError, match="max_single_string_bytes"):
            _read_fixed_text_probe(
                tmp_path,
                payload.encode("utf-8"),
                limit=limit,
                width=len(payload.encode("utf-8")),
            )
    else:
        assert (
            _read_fixed_text_probe(
                tmp_path,
                payload.encode("utf-8"),
                limit=limit,
                width=len(payload.encode("utf-8")),
            )
            == expected
        )


def test_bounded_fixed_invalid_utf8_is_rejected_strictly(tmp_path):
    with pytest.raises(UnsafeResultInputError, match="strict UTF-8"):
        _read_fixed_text_probe(tmp_path, b"\xff", limit=8, width=8)


def test_fixed_utf8_rejects_embedded_or_non_trailing_nul(tmp_path):
    with pytest.raises(UnsafeResultInputError, match="invalid NUL padding"):
        _read_fixed_text_probe(
            tmp_path,
            b"ab\x00cd",
            limit=5,
            width=5,
        )


def test_fixed_utf8_accepts_empty_value_with_one_byte_storage(tmp_path):
    assert _read_fixed_text_probe(tmp_path, b"", limit=1, width=1) == ("", 0)


def test_fixed_ascii_is_rejected_where_utf8_is_required(tmp_path):
    path = tmp_path / "ascii.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("value", data=np.asarray(b"text", dtype="S4"))

    with h5py.File(path, "r") as handle:
        with pytest.raises(UnsafeResultInputError, match="fixed UTF-8"):
            hdf5_module._bounded_dataset_text(
                handle["value"],
                path="value",
                limit=4,
            )


def test_vlen_text_rejects_before_any_value_access(tmp_path, monkeypatch):
    path = _write_text_probe(tmp_path, "hostile")
    with h5py.File(path, "r") as handle:
        dataset = handle["value"]
        read_calls: list[str] = []

        class SpyDatasetId:
            def get_type(self):
                return dataset.id.get_type()

            def get_space(self):
                raise AssertionError("dataspace inspection followed VLEN rejection")

            def read(self, *args, **kwargs):
                read_calls.append("read")
                raise AssertionError("payload read followed VLEN rejection")

        proxy = SimpleNamespace(
            dtype=dataset.dtype,
            shape=dataset.shape,
            id=SpyDatasetId(),
        )
        casts = _record_dataset_casts(monkeypatch)
        monkeypatch.setattr(
            h5py.Dataset,
            "__getitem__",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Dataset.__getitem__ called")
            ),
        )
        monkeypatch.setattr(
            h5py.Dataset,
            "asstr",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Dataset.asstr called")
            ),
        )

        with pytest.raises(UnsafeResultInputError, match="variable-length UTF-8"):
            hdf5_module._bounded_dataset_text(
                proxy,
                path="value",
                limit=64,
            )

        assert read_calls == []
        assert casts == []


@pytest.mark.parametrize(
    "dataset_path",
    (
        "instrument/name",
        "phase_center/kind",
        "phase_center/frame",
        "phase_center/w_reference",
        "provenance/instrument_json",
        "provenance/selection_json",
        "provenance/beam_json",
        "provenance/backend_json",
        "provenance/solver_json",
        "provenance/resolved_config_json",
        "provenance/configuration_source_json",
        "provenance/performance_json",
        "provenance/history_json",
    ),
)
def test_every_scalar_vlen_text_dataset_fails_closed_before_payload_access(
    tmp_path,
    monkeypatch,
    dataset_path,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "scalar-vlen.h5")
    _replace_dataset(
        output,
        dataset_path,
        data="hostile-vlen",
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    accesses: list[str] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(dataset, key, **kwargs):
        accesses.append(dataset.name)
        return original_getitem(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    monkeypatch.setattr(
        h5py.Dataset,
        "astype",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Dataset.astype called")
        ),
    )
    monkeypatch.setattr(
        h5py.Dataset,
        "asstr",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Dataset.asstr called")
        ),
    )
    monkeypatch.setattr(
        hdf5_module,
        "_bounded_dataset_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bounded text reader called")
        ),
    )
    monkeypatch.setattr(
        hdf5_module,
        "_parse_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("JSON parser called")
        ),
    )
    monkeypatch.setattr(
        hdf5_module,
        "_read_numeric",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("numeric payload read")
        ),
    )

    with pytest.raises(UnsafeResultInputError, match="variable-length UTF-8"):
        load_result_hdf5(output)

    assert accesses == []


def test_indexed_vlen_text_dataset_fails_closed_before_payload_access(
    tmp_path,
    monkeypatch,
):
    result = _multi_baseline_result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "indexed-vlen.h5")
    _replace_dataset(
        output,
        "instrument/antenna/name",
        data=["left", "hostile-vlen", "right"],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    bounded_calls: list[str] = []
    monkeypatch.setattr(
        hdf5_module,
        "_bounded_dataset_text",
        lambda *args, **kwargs: bounded_calls.append("called") or ("", 0),
    )

    with pytest.raises(UnsafeResultInputError, match="variable-length UTF-8"):
        load_result_hdf5(output)

    assert bounded_calls == []


def test_fixed_string_limit_is_enforced_before_low_level_value_read(tmp_path):
    path = _write_fixed_text_probe(tmp_path, b"x" * 65, width=65)
    with h5py.File(path, "r") as handle:
        dataset = handle["value"]
        reads: list[str] = []

        class SpyDatasetId:
            def get_type(self):
                return dataset.id.get_type()

            def get_space(self):
                raise AssertionError("fixed width was not preflighted")

            def read(self, *args, **kwargs):
                reads.append("read")
                raise AssertionError("fixed width payload was read")

        with pytest.raises(UnsafeResultInputError, match="max_single_string_bytes"):
            hdf5_module._bounded_dataset_text(
                SimpleNamespace(
                    dtype=dataset.dtype,
                    shape=dataset.shape,
                    id=SpyDatasetId(),
                ),
                path="value",
                limit=64,
            )

        assert reads == []


def test_fixed_string_array_total_bytes_are_enforced_before_value_read(tmp_path):
    path = _write_fixed_text_probe(
        tmp_path,
        (b"left", b"right"),
        indexed=True,
        width=8,
    )
    with h5py.File(path, "r") as handle:
        dataset = handle["value"]
        with pytest.raises(UnsafeResultInputError, match="max_single_dataset_bytes"):
            hdf5_module._enforce_dataset_byte_limits(
                {"value": dataset},
                HDF5ReadLimits(max_single_dataset_bytes=15),
            )


def test_hostile_vlen_rss_is_payload_size_independent_in_fresh_readers(tmp_path):
    child = r"""
import json
import resource
import sys
import tracemalloc
from types import SimpleNamespace

import h5py

from radiosim.io.hdf5 import HDF5ReadLimits, _bounded_dataset_text


def rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


path = sys.argv[1]
index = None if sys.argv[2] == "none" else int(sys.argv[2])
limits = HDF5ReadLimits(max_single_string_bytes=64)
with h5py.File(path, "r") as handle:
    dataset = handle["value"]
    _ = dataset.shape
    reads = []

    class SpyDatasetId:
        def get_type(self):
            return dataset.id.get_type()

        def get_space(self):
            raise AssertionError("VLEN rejection requested a dataspace")

        def read(self, *args, **kwargs):
            reads.append("read")
            raise AssertionError("VLEN rejection read a payload")

    proxy = SimpleNamespace(
        dtype=dataset.dtype,
        shape=dataset.shape,
        id=SpyDatasetId(),
    )
    getitem = h5py.Dataset.__getitem__
    astype = h5py.Dataset.astype
    asstr = h5py.Dataset.asstr
    h5py.Dataset.__getitem__ = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Dataset.__getitem__ called")
    )
    h5py.Dataset.astype = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Dataset.astype called")
    )
    h5py.Dataset.asstr = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Dataset.asstr called")
    )
    baseline = rss_bytes()
    tracemalloc.start()
    try:
        try:
            _bounded_dataset_text(
                proxy,
                path="value",
                limit=limits.max_single_string_bytes,
                index=index,
            )
        except Exception as exc:
            outcome = type(exc).__name__
            message = str(exc)
        else:
            outcome = "accepted"
            message = ""
        _current, python_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        h5py.Dataset.__getitem__ = getitem
        h5py.Dataset.astype = astype
        h5py.Dataset.asstr = asstr
    print(json.dumps({
        "outcome": outcome,
        "message": message,
        "python_peak": python_peak,
        "rss_delta": rss_bytes() - baseline,
        "id_read_calls": len(reads),
    }, sort_keys=True))
"""
    observations: dict[tuple[int, bool], dict[str, object]] = {}
    for payload_size in (1 << 20, 8 << 20):
        payload = b"h" * payload_size
        for indexed in (False, True):
            path = tmp_path / f"rss-{payload_size}-{indexed}.h5"
            with h5py.File(path, "w") as handle:
                if indexed:
                    dataset = handle.create_dataset(
                        "value",
                        shape=(1,),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                    dataset[0] = payload
                else:
                    dataset = handle.create_dataset(
                        "value",
                        shape=(),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                    dataset[()] = payload
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    child,
                    str(path),
                    "0" if indexed else "none",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, completed.stderr
            observations[(payload_size, indexed)] = json.loads(completed.stdout)

    for observation in observations.values():
        assert observation["outcome"] == "UnsafeResultInputError"
        assert "variable-length UTF-8" in observation["message"]
        assert observation["id_read_calls"] == 0
        assert observation["python_peak"] < 1 * 1024 * 1024

    # Repeated post-correction runs stayed below 2 MiB; this allowance covers
    # allocator noise while remaining far below the pre-correction growth.
    tolerance = 8 * 1024 * 1024
    for indexed in (False, True):
        small = observations[(1 << 20, indexed)]["rss_delta"]
        large = observations[(8 << 20, indexed)]["rss_delta"]
        assert large - small <= tolerance
        assert max(small, large) <= tolerance


def test_repeated_hostile_vlen_reads_close_hdf5_handles(tmp_path):
    path = _write_text_probe(tmp_path, "l" * (2 * 1024 * 1024))
    baseline = h5py.h5f.get_obj_count()

    for _ in range(20):
        with h5py.File(path, "r") as handle:
            with pytest.raises(
                UnsafeResultInputError,
                match="variable-length UTF-8",
            ):
                hdf5_module._bounded_dataset_text(
                    handle["value"],
                    path="value",
                    limit=64,
                )
        assert h5py.h5f.get_obj_count() == baseline


def test_aggregate_json_limit_is_enforced_before_any_json_payload_read(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "aggregate-json.h5")
    reads: list[str] = []
    original_bounded = hdf5_module._bounded_dataset_text

    def recording_bounded(dataset, *, path, limit, index=None):
        reads.append(path)
        return original_bounded(dataset, path=path, limit=limit, index=index)

    monkeypatch.setattr(hdf5_module, "_bounded_dataset_text", recording_bounded)

    with pytest.raises(UnsafeResultInputError, match="max_total_json_bytes"):
        load_result_hdf5(output, limits=HDF5ReadLimits(max_total_json_bytes=1))

    assert reads == []


def test_aggregate_json_limit_counts_non_ascii_utf8_bytes(monkeypatch):
    payloads = {
        "provenance/instrument_json": "{}",
        "provenance/selection_json": "{}",
        "provenance/beam_json": "{}",
        "provenance/backend_json": "{}",
        "provenance/solver_json": "{}",
        "provenance/resolved_config_json": "{}",
        "provenance/configuration_source_json": "null",
        "provenance/performance_json": "{}",
        "provenance/history_json": '["é"]',
    }
    returned = {
        path: (text, len(text.encode("utf-8"))) for path, text in payloads.items()
    }
    datasets = {path: object() for path in payloads}
    total = sum(byte_count for _text, byte_count in returned.values())

    monkeypatch.setattr(
        hdf5_module,
        "_bounded_dataset_text",
        lambda dataset, *, path, limit: returned[path],
    )

    values = hdf5_module._read_json_snapshots(
        datasets,
        HDF5ReadLimits(max_total_json_bytes=total),
    )
    assert values["history"] == ["é"]

    with pytest.raises(UnsafeResultInputError, match="max_total_json_bytes"):
        hdf5_module._read_json_snapshots(
            datasets,
            HDF5ReadLimits(max_total_json_bytes=total - 1),
        )


def test_corrupted_fletcher32_chunk_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "checksum.h5")
    with h5py.File(output, "r+") as handle:
        dataset = handle["data/visibilities"]
        filter_mask, raw = dataset.id.read_direct_chunk((0, 0, 0, 0))
        corrupted = bytearray(raw)
        corrupted[-1] ^= 0x01
        dataset.id.write_direct_chunk(
            (0, 0, 0, 0),
            bytes(corrupted),
            filter_mask=filter_mask,
        )

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output)


def test_reader_limits_reject_before_science_dataset_read(tmp_path, monkeypatch):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "limited.h5")
    reads: list[str] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(dataset, key, **kwargs):
        reads.append(dataset.name)
        return original_getitem(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    limits = HDF5ReadLimits(max_time=1)
    with pytest.raises(UnsafeResultInputError, match="max_time"):
        load_result_hdf5(output, limits=limits)
    assert not any(name.startswith("/data/") for name in reads)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "use_multiple_baselines"),
    [
        ("max_time", 1, False),
        ("max_baseline", 1, True),
        ("max_frequency", 1, False),
        ("max_antenna", 1, False),
        ("max_visibility_elements", 1, False),
        ("max_single_dataset_bytes", 1, False),
        ("max_total_json_bytes", 1, False),
        ("max_single_string_bytes", 1, False),
    ],
)
def test_each_hdf5_read_limit_is_enforced(
    tmp_path,
    limit_name,
    limit_value,
    use_multiple_baselines,
):
    result = (
        _multi_baseline_result(tmp_path)
        if use_multiple_baselines
        else _result(tmp_path)
    )
    output = write_result_hdf5(result, tmp_path / f"{limit_name}.h5")
    limits = HDF5ReadLimits(**{limit_name: limit_value})

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output, limits=limits)


def test_metadata_only_oversized_declared_shape_is_rejected_before_payload_read(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "oversized-shape.h5")
    oversized_time = 10_000_001
    with h5py.File(output, "r+") as handle:
        for path in ("data/visibilities", "data/flags", "data/weights"):
            original = handle[path]
            dtype = original.dtype
            attributes = dict(original.attrs)
            shuffle = original.shuffle
            del handle[path]
            replacement = handle.create_dataset(
                path,
                shape=(oversized_time, 1, 2, 4),
                dtype=dtype,
                chunks=(16, 1, 2, 4),
                compression="gzip",
                compression_opts=4,
                shuffle=shuffle,
                fletcher32=True,
            )
            for name, value in attributes.items():
                replacement.attrs[name] = value
        for path in (
            "coordinates/time/utc_jd1",
            "coordinates/time/utc_jd2",
            "coordinates/time/integration_time_seconds",
        ):
            original = handle[path]
            attributes = dict(original.attrs)
            del handle[path]
            replacement = handle.create_dataset(
                path,
                shape=(oversized_time,),
                dtype="<f8",
                chunks=(4096,),
                fletcher32=True,
            )
            for name, value in attributes.items():
                replacement.attrs[name] = value

    reads: list[str] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(dataset, key, **kwargs):
        reads.append(dataset.name)
        return original_getitem(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    with pytest.raises(UnsafeResultInputError, match="max_time"):
        load_result_hdf5(output)
    assert not any(name.startswith("/data/") for name in reads)


def test_reader_validates_all_dataset_metadata_before_enforcing_limits(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "metadata-before-limits.h5")
    with h5py.File(output, "r+") as handle:
        handle["data/weights"].dims[0].label = "sample"

    with pytest.raises(UnsafeResultInputError, match="dimension labels"):
        load_result_hdf5(output, limits=HDF5ReadLimits(max_time=1))


def test_reader_requires_exact_limit_model(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "limits.h5")

    with pytest.raises(TypeError):
        load_result_hdf5(output, limits={"max_time": 1})


def test_structured_snapshot_mismatch_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "snapshot.h5")
    with h5py.File(output, "r+") as handle:
        raw = handle["provenance/instrument_json"].asstr()[()]
        snapshot = json.loads(raw)
        snapshot["antennas"][0]["diameter_m"] += 1.0
    _replace_fixed_dataset(
        output,
        "provenance/instrument_json",
        json.dumps(
            snapshot,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output)


def test_frequency_snapshot_mismatch_is_rejected_before_science_read(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "frequency-snapshot.h5")
    with h5py.File(output, "r") as handle:
        raw = handle["provenance/resolved_config_json"].asstr()[()]
    snapshot = json.loads(raw)
    snapshot["frequency"]["channel_frequencies_hz"][0] = 99_000_000.0
    _replace_fixed_dataset(
        output,
        "provenance/resolved_config_json",
        json.dumps(
            snapshot,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )
    reads: list[str] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(dataset, key, **kwargs):
        reads.append(dataset.name)
        return original_getitem(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    with pytest.raises(UnsafeResultInputError, match="frequency snapshot"):
        load_result_hdf5(output)
    assert not any(name.startswith("/data/") for name in reads)


# ---------------------------------------------------------------------------
# Tier 5E: schema 2.0.0, polarization basis, and the receptor group
# ---------------------------------------------------------------------------

CIRCULAR = {"default": {"basis": "circular"}}


def test_schema_version_is_three_zero_zero() -> None:
    """Tier 6G, plan Section 19: the component-provenance bump."""
    assert hdf5_module.SCHEMA_VERSION == "3.0.0"
    assert not hasattr(hdf5_module, "CORRELATIONS")
    assert not hasattr(hdf5_module, "AIPS_CODES")


@pytest.mark.parametrize(
    ("receptors", "labels", "codes", "basis", "native"),
    [
        (None, [b"XX", b"XY", b"YX", b"YY"], [-5, -7, -8, -6], "linear_xy", "linear"),
        (
            CIRCULAR,
            [b"RR", b"RL", b"LR", b"LL"],
            [-1, -3, -4, -2],
            "circular_rl",
            "circular",
        ),
    ],
)
def test_written_file_records_the_true_basis_and_receptor_group(
    tmp_path,
    receptors,
    labels,
    codes,
    basis,
    native,
):
    result = _result(tmp_path, receptors=receptors)
    output = write_result_hdf5(result, tmp_path / f"{basis}.h5")

    with h5py.File(output, "r") as handle:
        groups, datasets = _object_paths(handle)
        assert groups == GROUPS
        assert datasets == DATASETS
        np.testing.assert_array_equal(
            handle["coordinates/correlation/labels"][:],
            labels,
        )
        np.testing.assert_array_equal(
            handle["coordinates/correlation/aips_codes"][:],
            codes,
        )
        assert bytes(handle["coordinates/correlation/basis"][()]).decode() == basis
        assert bytes(handle["receptors/output_basis"][()]).decode() == basis
        assert bytes(handle["receptors/receptor_sha256"][()]).decode() == (
            result.receptors.provenance.receptor_sha256
        )
        np.testing.assert_array_equal(
            handle["receptors/antenna_number"][:],
            [antenna.id.number for antenna in result.instrument.antennas],
        )
        assert [
            bytes(value).decode("utf-8")
            for value in handle["receptors/antenna_name"][:]
        ] == [antenna.id.name for antenna in result.instrument.antennas]
        assert [
            bytes(value).decode("utf-8") for value in handle["receptors/basis"][:]
        ] == [native, native]
        np.testing.assert_array_equal(
            handle["receptors/feed_rotation_rad"][:],
            [0.0, 0.0],
        )
        expected_angles = [
            list(result.receptors.receptor_by_antenna[antenna.id].feed_angle_rad)
            for antenna in result.instrument.antennas
        ]
        np.testing.assert_array_equal(
            handle["receptors/feed_angle_rad"][:],
            expected_angles,
        )
        assert handle["receptors/feed_rotation_rad"].attrs["unit"] == "radian"


@pytest.mark.parametrize("receptors", [None, CIRCULAR])
def test_both_bases_round_trip_through_hdf5(tmp_path, receptors):
    result = _result(tmp_path, receptors=receptors)
    output = write_result_hdf5(result, tmp_path / "round-trip.h5")

    loaded = load_result_hdf5(output)

    assert loaded.correlations == result.correlations
    assert loaded.polarization_basis == result.polarization_basis
    assert loaded.receptors["output_basis"] == result.receptors.output_basis
    assert loaded.receptors["receptor_sha256"] == (
        result.receptors.provenance.receptor_sha256
    )
    assert loaded.scientific_sha256 == result.scientific_sha256
    assert loaded.provenance_sha256 == result.provenance_sha256
    assert loaded.scientifically_equal(result)
    np.testing.assert_array_equal(loaded.stokes_i(), result.stokes_i())


def test_a_rotated_feed_survives_the_round_trip_exactly(tmp_path):
    result = _result(
        tmp_path,
        receptors={"default": {"basis": "circular", "feed_rotation_deg": 30.0}},
    )
    output = write_result_hdf5(result, tmp_path / "rotated.h5")

    loaded = load_result_hdf5(output)

    rows = loaded.receptors["receptors"]
    assert [row["basis"] for row in rows] == ["circular", "circular"]
    for row, antenna in zip(rows, result.instrument.antennas, strict=True):
        live = result.receptors.receptor_by_antenna[antenna.id]
        assert row["feed_rotation_rad"] == live.feed_rotation_rad
        assert tuple(row["feed_angle_rad"]) == live.feed_angle_rad


def _restamp_schema_version(output: Path, version: bytes) -> None:
    with h5py.File(output, "r+") as handle:
        del handle.attrs["schema_version"]
        handle.attrs.create(
            "schema_version",
            np.bytes_(version),
            shape=(),
            dtype=h5py.string_dtype(encoding="utf-8", length=len(version)),
        )


@pytest.mark.parametrize("version", [b"1.0.0", b"2.0.0"])
def test_every_superseded_schema_version_is_rejected_naming_tier_six(
    tmp_path,
    version,
):
    """Tier 6G, plan Section 32.7: no upgrade path, and the message says so."""
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "superseded-version.h5")
    _restamp_schema_version(output, version)

    with pytest.raises(UnsupportedSchemaVersionError) as caught:
        load_result_hdf5(output)

    message = str(caught.value)
    decoded = version.decode("ascii")
    assert decoded in message
    assert "Tier 6" in message
    assert "3.0.0" in message
    assert "re-run the simulation" in message
    assert caught.value.version == decoded


@pytest.mark.parametrize(
    ("dataset_path", "replacement", "dtype"),
    [
        ("coordinates/correlation/basis", b"circular_rl", None),
        ("coordinates/correlation/basis", b"linear_XY", None),
        ("coordinates/correlation/basis", b"stokes", None),
        (
            "coordinates/correlation/labels",
            np.array([b"RR", b"RL", b"LR", b"LL"], dtype="S2"),
            "S2",
        ),
        (
            "coordinates/correlation/aips_codes",
            np.array([-1, -3, -4, -2], dtype="<i4"),
            "<i4",
        ),
    ],
)
def test_a_hand_edited_basis_or_label_mismatch_is_rejected(
    tmp_path,
    dataset_path,
    replacement,
    dtype,
):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "mismatch.h5")
    if isinstance(replacement, bytes):
        _replace_fixed_dataset(output, dataset_path, replacement)
    else:
        with h5py.File(output, "r+") as handle:
            original = handle[dataset_path]
            chunks = original.chunks
            fletcher32 = original.fletcher32
        _replace_dataset(
            output,
            dataset_path,
            data=replacement,
            dtype=dtype,
            chunks=chunks,
            fletcher32=fletcher32,
        )

    with pytest.raises(UnsafeResultInputError, match="correlation"):
        load_result_hdf5(output)


def test_a_circular_file_with_a_reordered_axis_is_rejected(tmp_path):
    result = _result(tmp_path, receptors=CIRCULAR)
    output = write_result_hdf5(result, tmp_path / "reordered.h5")
    with h5py.File(output, "r+") as handle:
        original = handle["coordinates/correlation/labels"]
        chunks = original.chunks
        fletcher32 = original.fletcher32
    _replace_dataset(
        output,
        "coordinates/correlation/labels",
        data=np.array([b"RR", b"LL", b"RL", b"LR"], dtype="S2"),
        dtype="S2",
        chunks=chunks,
        fletcher32=fletcher32,
    )

    with pytest.raises(UnsafeResultInputError, match="correlation"):
        load_result_hdf5(output)


def test_a_hostile_receptor_group_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "hostile-receptors.h5")
    _replace_fixed_dataset(output, "receptors/output_basis", b"circular_rl")

    with pytest.raises(UnsafeResultInputError, match="receptor"):
        load_result_hdf5(output)


def test_a_receptor_row_outside_the_instrument_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "receptor-antenna.h5")
    with h5py.File(output, "r+") as handle:
        original = handle["receptors/antenna_number"]
        chunks = original.chunks
        fletcher32 = original.fletcher32
        payload = np.array(original[:], copy=True)
    payload[1] = 91
    _replace_dataset(
        output,
        "receptors/antenna_number",
        data=payload,
        dtype="<i8",
        chunks=chunks,
        fletcher32=fletcher32,
    )

    with pytest.raises(UnsafeResultInputError, match="receptor"):
        load_result_hdf5(output)


def test_a_hostile_receptor_basis_value_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "receptor-basis.h5")
    _replace_fixed_dataset(output, "receptors/basis", (b"linear", b"ellipt"))

    with pytest.raises(UnsafeResultInputError, match="receptor"):
        load_result_hdf5(output)


def test_a_nonfinite_feed_rotation_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "receptor-rotation.h5")
    with h5py.File(output, "r+") as handle:
        original = handle["receptors/feed_rotation_rad"]
        chunks = original.chunks
        fletcher32 = original.fletcher32
    _replace_dataset(
        output,
        "receptors/feed_rotation_rad",
        data=np.array([np.nan, 0.0], dtype="<f8"),
        dtype="<f8",
        chunks=chunks,
        fletcher32=fletcher32,
    )

    with pytest.raises(UnsafeResultInputError, match="receptor"):
        load_result_hdf5(output)


def test_the_receptor_group_cannot_be_silently_dropped(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "no-receptors.h5")
    with h5py.File(output, "r+") as handle:
        del handle["receptors/output_basis"]

    with pytest.raises(UnsafeResultInputError, match="allowlist"):
        load_result_hdf5(output)


# ---------------------------------------------------------------------------
# Tier 6G: schema 3.0.0 component provenance (plan Sections 19, 32.7; row H9)
# ---------------------------------------------------------------------------


def _component_result(
    tmp_path: Path,
    *,
    representation: str,
    components: tuple[str, ...],
    counts: tuple[int, ...],
    point_seconds: float = 0.0,
    healpix_seconds: float = 0.0,
):
    """Build a result whose solver provenance and resolved config agree.

    The reader cross-checks the two (plan Section 19 reader validation), so a
    fixture that declares ``hybrid`` in one place and ``point_sources`` in the
    other is not a legitimate result and must not be used as one.
    """
    directory = tmp_path / representation
    data = _mapping(directory)
    data["visibility"] = {"sky_representation": representation}
    simulator = Simulator.from_mapping(data, base_dir=directory)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    backend = get_backend("numpy")
    cube = np.arange(2 * 1 * 2 * 4, dtype=np.float64).reshape(2, 1, 2, 2, 2)
    receptor = cube.astype("complex128")
    receptor += 1j * receptor
    return build_simulation_result(
        receptor_visibilities=receptor,
        backend=backend,
        time_grid=simulator.config.observation.time_grid,
        frequencies_hz=simulator.config.frequency.channel_frequencies_hz,
        channel_widths_hz=simulator.config.frequency.channel_widths_hz,
        instrument=simulator.instrument,
        selection=simulator._instrument_state.selection,
        beam_state=simulator.beam_state,
        receptors=simulator.receptors,
        phase_center=PhaseCenter(),
        backend_provenance=BackendResultProvenance(
            requested_backend="numpy",
            actual_backend=backend.name,
            requested_precision={"output": "complex128"},
            actual_precision={"output": "complex128"},
            result_dtype="complex128",
        ),
        solver_provenance=SolverResultProvenance(
            solver="rime",
            sky_representation=representation,
            convention="radiosim.rime-zenith-drift.v1",
            execution_path="polarized",
            components=components,
            component_element_counts=counts,
        ),
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=ResultPerformance(
            setup_seconds=1.0,
            solver_seconds=2.0,
            solver_point_seconds=point_seconds,
            solver_healpix_seconds=healpix_seconds,
            result_construction_seconds=0.5,
            host_transfer_seconds=0.25,
            total_seconds=3.75,
        ),
        history=("simulated",),
    )


_COMPONENT_CASES = [
    ("point_sources", ("point",), (7,), 2.0, 0.0),
    ("healpix_map", ("healpix",), (3072,), 0.0, 2.0),
    ("hybrid", ("point", "healpix"), (7, 3072), 1.25, 0.5),
]


@pytest.mark.parametrize(
    ("representation", "components", "counts", "point_seconds", "healpix_seconds"),
    _COMPONENT_CASES,
    ids=[case[0] for case in _COMPONENT_CASES],
)
def test_every_representation_round_trips_component_provenance_and_timings(
    tmp_path,
    representation,
    components,
    counts,
    point_seconds,
    healpix_seconds,
):
    """H9: point-only, healpix-only, and hybrid all survive 3.0.0 exactly."""
    result = _component_result(
        tmp_path,
        representation=representation,
        components=components,
        counts=counts,
        point_seconds=point_seconds,
        healpix_seconds=healpix_seconds,
    )
    output = write_result_hdf5(result, tmp_path / f"{representation}.h5")

    with h5py.File(output, "r") as handle:
        groups, datasets = _object_paths(handle)
        assert groups == GROUPS
        assert datasets == DATASETS
        assert bytes(handle.attrs["schema_version"]).decode() == "3.0.0"
        solver_json = json.loads(
            bytes(handle["provenance/solver_json"][()]).rstrip(b"\x00").decode("utf-8")
        )
        performance_json = json.loads(
            bytes(handle["provenance/performance_json"][()])
            .rstrip(b"\x00")
            .decode("utf-8")
        )
    assert solver_json["sky_representation"] == representation
    assert tuple(solver_json["components"]) == components
    assert tuple(solver_json["component_element_counts"]) == counts
    assert performance_json["solver_point_seconds"] == point_seconds
    assert performance_json["solver_healpix_seconds"] == healpix_seconds

    loaded = load_result_hdf5(output)

    assert loaded.solver_snapshot["sky_representation"] == representation
    assert tuple(loaded.solver_snapshot["components"]) == components
    assert tuple(loaded.solver_snapshot["component_element_counts"]) == counts
    assert loaded.performance.solver_point_seconds == point_seconds
    assert loaded.performance.solver_healpix_seconds == healpix_seconds
    assert loaded.scientific_sha256 == result.scientific_sha256
    assert loaded.provenance_sha256 == result.provenance_sha256
    assert loaded.scientifically_equal(result)


def test_a_hybrid_file_is_not_scientifically_equal_to_a_point_only_file(tmp_path):
    """H8 at the serialization boundary: components are part of the identity."""
    hybrid = _component_result(
        tmp_path,
        representation="hybrid",
        components=("point", "healpix"),
        counts=(7, 3072),
        point_seconds=1.25,
        healpix_seconds=0.5,
    )
    point = _component_result(
        tmp_path,
        representation="point_sources",
        components=("point",),
        counts=(7,),
        point_seconds=2.0,
    )
    hybrid_loaded = load_result_hdf5(
        write_result_hdf5(hybrid, tmp_path / "hybrid-identity.h5")
    )
    point_loaded = load_result_hdf5(
        write_result_hdf5(point, tmp_path / "point-identity.h5")
    )

    assert np.array_equal(hybrid_loaded.visibilities, point_loaded.visibilities)
    assert not hybrid_loaded.scientifically_equal(point_loaded)
    assert hybrid_loaded.scientific_sha256 != point_loaded.scientific_sha256


def _rewrite_solver_json(output: Path, mutate) -> None:
    with h5py.File(output, "r") as handle:
        text = bytes(handle["provenance/solver_json"][()]).rstrip(b"\x00")
    record = json.loads(text.decode("utf-8"))
    mutate(record)
    _replace_fixed_dataset(
        output,
        "provenance/solver_json",
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    )


def _rewrite_performance_json(output: Path, mutate) -> None:
    with h5py.File(output, "r") as handle:
        text = bytes(handle["provenance/performance_json"][()]).rstrip(b"\x00")
    record = json.loads(text.decode("utf-8"))
    mutate(record)
    _replace_fixed_dataset(
        output,
        "provenance/performance_json",
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    )


def _relabel_representation(record: dict) -> None:
    record["sky_representation"] = "hybrid"
    record["components"] = ["point", "healpix"]
    record["component_element_counts"] = [7, 3072]


def _forge_components_only(record: dict) -> None:
    record["components"] = ["point", "healpix"]


def _forge_extra_field(record: dict) -> None:
    record["component_flux_jy"] = [1.0, 2.0]


def _forge_missing_field(record: dict) -> None:
    del record["component_element_counts"]


def _forge_negative_count(record: dict) -> None:
    record["component_element_counts"] = [-1]


def _forge_unbounded_component_list(record: dict) -> None:
    record["components"] = ["point"] * 4096
    record["component_element_counts"] = [1] * 4096


def _forge_component_name(record: dict) -> None:
    record["components"] = ["healpix"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_relabel_representation, "solver"),
        (_forge_components_only, "solver"),
        (_forge_extra_field, "solver"),
        (_forge_missing_field, "solver"),
        (_forge_negative_count, "solver"),
        (_forge_unbounded_component_list, "solver"),
        (_forge_component_name, "solver"),
    ],
    ids=[
        "relabelled_representation",
        "components_without_representation",
        "unexpected_field",
        "missing_field",
        "negative_count",
        "unbounded_component_list",
        "wrong_component_name",
    ],
)
def test_a_forged_solver_group_is_rejected_before_any_science_is_read(
    tmp_path,
    monkeypatch,
    mutate,
    match,
):
    """Tier 4/5 hostile discipline: reject component forgery pre-allocation."""
    result = _component_result(
        tmp_path,
        representation="point_sources",
        components=("point",),
        counts=(7,),
        point_seconds=2.0,
    )
    output = write_result_hdf5(result, tmp_path / "forged-solver.h5")
    _rewrite_solver_json(output, mutate)

    reads: list[str] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(dataset, key, **kwargs):
        reads.append(dataset.name)
        return original_getitem(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    with pytest.raises(UnsafeResultInputError, match=match):
        load_result_hdf5(output)
    assert not any(name.startswith("/data/") for name in reads)


def _forge_incoherent_component_times(record: dict) -> None:
    record["solver_point_seconds"] = 5.0
    record["solver_healpix_seconds"] = 5.0


def _forge_extra_performance_field(record: dict) -> None:
    record["solver_gpu_seconds"] = 1.0


def _forge_missing_performance_field(record: dict) -> None:
    del record["solver_healpix_seconds"]


@pytest.mark.parametrize(
    "mutate",
    [
        _forge_incoherent_component_times,
        _forge_extra_performance_field,
        _forge_missing_performance_field,
    ],
    ids=["incoherent_component_times", "unexpected_field", "missing_field"],
)
def test_a_forged_performance_group_is_rejected_before_any_science_is_read(
    tmp_path,
    monkeypatch,
    mutate,
):
    result = _component_result(
        tmp_path,
        representation="hybrid",
        components=("point", "healpix"),
        counts=(7, 3072),
        point_seconds=1.25,
        healpix_seconds=0.5,
    )
    output = write_result_hdf5(result, tmp_path / "forged-performance.h5")
    _rewrite_performance_json(output, mutate)

    reads: list[str] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(dataset, key, **kwargs):
        reads.append(dataset.name)
        return original_getitem(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    with pytest.raises(UnsafeResultInputError, match="performance"):
        load_result_hdf5(output)
    assert not any(name.startswith("/data/") for name in reads)


def test_an_oversized_solver_group_is_rejected_from_metadata_alone(
    tmp_path,
    monkeypatch,
):
    """A component list large enough to matter never reaches a value read."""
    result = _component_result(
        tmp_path,
        representation="point_sources",
        components=("point",),
        counts=(7,),
        point_seconds=2.0,
    )
    output = write_result_hdf5(result, tmp_path / "oversized-solver.h5")
    _replace_fixed_dataset(
        output,
        "provenance/solver_json",
        json.dumps({"components": ["point"] * 200_000}),
        width=2 * 1024 * 1024,
    )

    reads: list[str] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(dataset, key, **kwargs):
        reads.append(dataset.name)
        return original_getitem(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    with pytest.raises(UnsafeResultInputError, match="max_single_string_bytes"):
        load_result_hdf5(output)
    assert not any(name.startswith("/data/") for name in reads)
