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
from radiosim.core.phase_center import PhaseCenter
from radiosim.core.result import LoadedSimulationResult, build_simulation_result
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
}
LEGACY_GUIDANCE = (
    "Legacy unversioned RadioSim HDF5 is not accepted because baseline names "
    "were parsed unsafely and scientific fields were incomplete. Re-run the "
    "simulation with Tier 4 or convert a trusted file in an isolated pre-Tier-4 "
    "environment."
)


def _result(tmp_path: Path, dtype: str = "complex128"):
    result, _ = _build(tmp_path, dtype=dtype)
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
        phase_center=PhaseCenter(),
        backend_provenance=provenance,
        solver_provenance=solver,
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=performance,
        history=("independent-h5py-probe",),
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
        assert handle.attrs["schema_name"] == "radiosim.visibility"
        assert handle.attrs["schema_version"] == "1.0.0"
        assert handle.attrs["dimension_order"] == "time,baseline,frequency,correlation"
        assert handle.attrs["visibility_unit"] == "Jy"
        assert handle.attrs["scientific_sha256"] == result.scientific_sha256
        assert handle.attrs["provenance_sha256"] == result.provenance_sha256
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
            assert string_info.length is None
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
        handle.attrs.modify("schema_version", "999.0.0")

    reads: list[str] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(dataset, key, **kwargs):
        reads.append(dataset.name)
        return original_getitem(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    with pytest.raises(UnsupportedSchemaVersionError, match="999.0.0"):
        load_result_hdf5(output)
    assert not any(name.startswith("/data/") for name in reads)


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
            handle.attrs.modify("radiosim_version", "x" * 1_048_577)
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
    _replace_dataset(
        output,
        dataset_path,
        data=payload,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(output)


def test_malformed_utf8_string_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "utf8.h5")
    _replace_dataset(
        output,
        "provenance/history_json",
        data=b"\xff",
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    with pytest.raises(UnsafeResultInputError, match="UTF-8"):
        load_result_hdf5(output)


def test_embedded_nul_root_string_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "nul.h5")
    with h5py.File(output, "r+") as handle:
        del handle.attrs["visibility_unit"]
        handle.attrs["visibility_unit"] = np.bytes_(b"J\x00y")

    with pytest.raises(UnsafeResultInputError, match="bounded string"):
        load_result_hdf5(output)


def test_oversized_utf8_dataset_is_rejected(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "oversized-string.h5")
    _replace_dataset(
        output,
        "instrument/name",
        data="x" * 1_048_577,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    with pytest.raises(UnsafeResultInputError, match="size limit"):
        load_result_hdf5(output)


def test_aggregate_json_limit_is_enforced(tmp_path):
    result = _result(tmp_path)
    output = write_result_hdf5(result, tmp_path / "aggregate-json.h5")

    with pytest.raises(UnsafeResultInputError, match="max_total_json_bytes"):
        load_result_hdf5(output, limits=HDF5ReadLimits(max_total_json_bytes=1))


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
        del handle["provenance/instrument_json"]
        handle.create_dataset(
            "provenance/instrument_json",
            data=json.dumps(
                snapshot,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            dtype=h5py.string_dtype("utf-8"),
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
    _replace_dataset(
        output,
        "provenance/resolved_config_json",
        data=json.dumps(
            snapshot,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        dtype=h5py.string_dtype("utf-8"),
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
