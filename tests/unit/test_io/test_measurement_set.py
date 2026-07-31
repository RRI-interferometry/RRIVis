"""Canonical Measurement Set projection, reader, and publication contracts."""

from __future__ import annotations

import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from casacore.tables import table
from pyuvdata import UVData

import radiosim.io as io
import radiosim.io.measurement_set as measurement_set
from radiosim.core.result import LoadedSimulationResult, SimulationResult
from radiosim.io.measurement_set import (
    read_measurement_set,
    write_measurement_set,
)
from radiosim.io.result_errors import (
    AtomicWriteError,
    AtomicWriteUnsupportedError,
    FormatRepresentationError,
    OptionalResultDependencyError,
    OutputCollisionError,
    OutputPathError,
    OverwriteRefusedError,
    PartialCleanupError,
    UnsafeResultInputError,
)
from radiosim.io.standard_visibility import (
    PROJECTION_HISTORY_PREFIX,
    StandardReadLimits,
    StandardVisibilityData,
    project_simulation_result,
    projection_record_from_history,
)
from tests.unit.test_io.test_standard_visibility import build_standard_result

_PY312_MS_WARNING = (
    "'where' used without 'out', expect unitialized memory in output. "
    "If this is intentional, use out=None."
)


def _write_checked(*args: object, **kwargs: object) -> Path:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            return write_measurement_set(*args, **kwargs)
        finally:
            actual = [(item.category, str(item.message)) for item in caught]
            if sys.version_info >= (3, 12):
                assert all(
                    category is UserWarning and message == _PY312_MS_WARNING
                    for category, message in actual
                )
            else:
                assert actual == []


def test_measurement_set_public_signatures_and_removed_surface() -> None:
    assert str(inspect.signature(write_measurement_set)) == (
        "(result: 'SimulationResult', path: 'str | Path', *, "
        "overwrite: 'bool' = False) -> 'Path'"
    )
    assert str(inspect.signature(read_measurement_set)) == (
        "(path: 'str | Path', *, data_column: 'str' = 'DATA', "
        "limits: 'StandardReadLimits' = StandardReadLimits()) "
        "-> 'StandardVisibilityData'"
    )
    for name in (
        "write_ms",
        "read_ms",
        "read_ms_dask",
        "ms_info",
        "PYUVDATA_AVAILABLE",
        "CASACORE_AVAILABLE",
        "DASKMS_AVAILABLE",
        "MS_AVAILABLE",
    ):
        assert not hasattr(measurement_set, name)
        assert name not in io.__all__
        with pytest.raises(AttributeError):
            io.__getattr__(name)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_measurement_set_round_trip_and_raw_storage(
    tmp_path: Path,
    dtype: str,
) -> None:
    result = build_standard_result(
        tmp_path,
        dtype=dtype,
        channel_widths_hz=(1.25e6, 2.0e6),
    )
    target = tmp_path / f"canonical-{dtype}.ms"

    returned = _write_checked(result, target)
    loaded = read_measurement_set(target)
    expected = project_simulation_result(result, format="ms").data

    assert returned == target.absolute()
    assert type(loaded) is StandardVisibilityData
    assert not isinstance(loaded, (SimulationResult, LoadedSimulationResult))
    assert loaded.format == "ms"
    assert loaded.visibilities.dtype == np.dtype(np.complex64)
    assert loaded.weights.dtype == np.dtype(np.float32)
    assert loaded.correlations == ("XX", "XY", "YX", "YY")
    assert loaded.source_scientific_sha256 == result.scientific_sha256
    assert loaded.source_provenance_sha256 == result.provenance_sha256
    assert any(f"input_visibility_dtype={dtype}" in item for item in loaded.history)
    assert any("stored_visibility_dtype=complex64" in item for item in loaded.history)
    assert any(
        f"lossy_visibility_conversion={'true' if dtype == 'complex128' else 'false'}"
        in item
        for item in loaded.history
    )

    raw = UVData()
    raw.read_ms(str(target))
    assert raw.data_array.dtype == np.dtype(np.complex64)
    assert np.asarray(raw.polarization_array).tolist() == [-5, -6, -7, -8]
    assert raw.get_antpairs() == [
        (baseline.ant1.number, baseline.ant2.number)
        for baseline in result.selection.baselines
    ]
    np.testing.assert_allclose(
        loaded.visibilities,
        expected.visibilities,
        rtol=5e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(loaded.flags, expected.flags)
    np.testing.assert_allclose(
        loaded.weights,
        expected.weights,
        rtol=5e-6,
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Tier 5F: the Measurement Set carries the resolved basis (Sections 14.2, 22)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "receptors",
        "labels",
        "corr_type",
        "readback_codes",
        "feed_letters",
        "receptor_angle",
    ),
    [
        (
            None,
            ("XX", "XY", "YX", "YY"),
            [9, 10, 11, 12],
            [-5, -6, -7, -8],
            ["X", "Y"],
            math.pi / 2.0,
        ),
        (
            {"default": {"basis": "circular"}},
            ("RR", "RL", "LR", "LL"),
            [5, 6, 7, 8],
            [-1, -2, -3, -4],
            ["R", "L"],
            0.0,
        ),
    ],
    ids=["linear", "circular"],
)
def test_measurement_set_polarization_metadata_round_trips_both_bases(
    tmp_path: Path,
    receptors: dict[str, object] | None,
    labels: tuple[str, ...],
    corr_type: list[int],
    readback_codes: list[int],
    feed_letters: list[str],
    receptor_angle: float,
) -> None:
    """CORR_TYPE keeps the in-memory order; the reader canonicalizes (Q3)."""
    result = build_standard_result(
        tmp_path,
        dtype="complex64",
        receptors=receptors,
    )
    assert result.correlations == labels
    target = tmp_path / "basis.ms"

    _write_checked(result, target)
    loaded = read_measurement_set(target)
    assert loaded.correlations == labels

    # pyuvdata pads the ANTENNA and FEED subtables out to the highest antenna
    # number, so the row count is not the selected antenna count.
    feed_rows = max(antenna.id.number for antenna in result.instrument.antennas) + 1
    with table(str(target / "POLARIZATION"), ack=False) as handle:
        assert handle.getcol("NUM_CORR").tolist() == [4]
        assert handle.getcol("CORR_TYPE").tolist() == [corr_type]
    with table(str(target / "FEED"), ack=False) as handle:
        polarization_type = handle.getcol("POLARIZATION_TYPE")
        assert polarization_type["shape"] == [feed_rows, 2]
        assert polarization_type["array"] == feed_letters * feed_rows
        angles = np.asarray(handle.getcol("RECEPTOR_ANGLE"))
        assert angles.shape == (feed_rows, 2)
        selected = [antenna.id.number for antenna in result.instrument.antennas]
        np.testing.assert_allclose(
            angles[selected],
            np.tile([receptor_angle, 0.0], (len(selected), 1)),
            rtol=0.0,
            atol=1e-9,
        )

    raw = UVData()
    raw.read_ms(str(target))
    antenna_count = int(raw.telescope.Nants)
    assert np.asarray(raw.polarization_array).tolist() == readback_codes
    assert (
        raw.telescope.feed_array.tolist()
        == [[letter.lower() for letter in feed_letters]] * antenna_count
    )
    np.testing.assert_allclose(
        np.asarray(raw.telescope.feed_angle),
        np.tile([receptor_angle, 0.0], (antenna_count, 1)),
        rtol=0.0,
        atol=1e-9,
    )
    assert list(raw.telescope.mount_type) == ["fixed"] * antenna_count

    expected = project_simulation_result(result, format="ms").data
    np.testing.assert_allclose(
        loaded.visibilities,
        expected.visibilities,
        rtol=5e-6,
        atol=1e-7,
    )


def test_measurement_set_history_records_the_resolved_basis(tmp_path: Path) -> None:
    result = build_standard_result(
        tmp_path,
        dtype="complex64",
        receptors={"default": {"basis": "circular"}},
    )
    target = tmp_path / "circular-history.ms"
    _write_checked(result, target)

    loaded = read_measurement_set(target)
    record_lines = [
        item for item in loaded.history if item.startswith(PROJECTION_HISTORY_PREFIX)
    ]
    assert len(record_lines) == 1
    record = json.loads(record_lines[0][len(PROJECTION_HISTORY_PREFIX) :])
    assert record["polarization_basis"] == "circular_rl"
    assert record["receptor_sha256"] == result.receptors.provenance.receptor_sha256


def test_measurement_set_reader_maps_both_casacore_stokes_ranges() -> None:
    """Section 14.3: the casacore enumeration covers both accepted bases."""
    assert measurement_set._CASA_TO_AIPS == {
        5: -1,
        6: -3,
        7: -4,
        8: -2,
        9: -5,
        10: -7,
        11: -8,
        12: -6,
    }


def test_measurement_set_collision_replace_and_no_residue(tmp_path: Path) -> None:
    first = build_standard_result(tmp_path / "first", dtype="complex64")
    second = build_standard_result(tmp_path / "second", dtype="complex64")
    target = tmp_path / "replace.ms"

    _write_checked(first, target)
    original = read_measurement_set(target)
    with pytest.raises(OverwriteRefusedError):
        _write_checked(second, target)
    assert read_measurement_set(target) == original

    _write_checked(second, target, overwrite=True)
    assert (
        read_measurement_set(target).source_scientific_sha256
        == second.scientific_sha256
    )
    assert tuple(tmp_path.glob(f".{target.name}.*.tmp.ms")) == ()


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_measurement_set_reader_rejects_unsafe_path(
    tmp_path: Path,
    kind: str,
) -> None:
    target = tmp_path / "unsafe.ms"
    if kind == "file":
        target.write_bytes(b"not an MS")
    else:
        source = tmp_path / "source.ms"
        source.mkdir()
        target.symlink_to(source, target_is_directory=True)
    with pytest.raises((OutputPathError, UnsafeResultInputError)):
        read_measurement_set(target)


def test_measurement_set_metadata_limits_precede_data_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "limited.ms"
    _write_checked(result, target)
    data_reads = 0
    real_read = measurement_set._read_ms

    def recording_read(path, *, data_column, read_data, limits=None):
        nonlocal data_reads
        if read_data:
            data_reads += 1
        return real_read(
            path,
            data_column=data_column,
            read_data=read_data,
            limits=limits,
        )

    monkeypatch.setattr(measurement_set, "_read_ms", recording_read)
    with pytest.raises(UnsafeResultInputError, match="max_times"):
        read_measurement_set(
            target,
            limits=StandardReadLimits(max_times=1),
        )
    assert data_reads == 0


def test_measurement_set_declared_rows_reject_before_any_full_column_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "hostile.ms"
    target.mkdir()
    allocation_calls: list[tuple[str, str]] = []

    class HostileMain:
        def nrows(self) -> int:
            return 1_000_000_000

        def colnames(self) -> list[str]:
            return [
                "DATA",
                "ANTENNA1",
                "ANTENNA2",
                "TIME",
                "UVW",
                "EXPOSURE",
            ]

        def getcol(self, name: str, *_args: object, **_kwargs: object) -> object:
            allocation_calls.append(("getcol", name))
            raise AssertionError("attacker-sized full column allocation")

        def getcolshapestring(
            self,
            name: str,
            *_args: object,
            **_kwargs: object,
        ) -> object:
            allocation_calls.append(("getcolshapestring", name))
            raise AssertionError("attacker-sized shape-list allocation")

        def getcell(self, name: str, *_args: object, **_kwargs: object) -> object:
            allocation_calls.append(("getcell", name))
            raise AssertionError("attacker-sized cell allocation")

        def close(self) -> None:
            return None

    fake_tables = SimpleNamespace(
        table=lambda *_args, **_kwargs: HostileMain(),
    )
    real_import = measurement_set.import_module

    def import_spy(name: str) -> object:
        if name == "casacore.tables":
            return fake_tables
        return real_import(name)

    monkeypatch.setattr(measurement_set, "import_module", import_spy)
    monkeypatch.setattr(
        measurement_set,
        "_import_standard_dependencies",
        lambda: object,
    )

    with pytest.raises(UnsafeResultInputError):
        read_measurement_set(
            target,
            limits=StandardReadLimits(
                max_times=1,
                max_baselines=1,
                max_frequencies=1,
                max_antennas=1,
                max_visibility_elements=1,
                max_data_bytes=1,
            ),
        )
    assert allocation_calls == []


def test_measurement_set_rejects_wrong_main_column_descriptor_before_casting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "wrong-time-type.ms"
    _write_checked(result, target)

    class WrappedMain:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def getcoldesc(self, name: str) -> dict[str, object]:
            descriptor = dict(self._inner.getcoldesc(name))
            if name == "TIME":
                descriptor["valueType"] = "string"
            return descriptor

        def getcol(self, name: str, *args: object) -> object:
            values = self._inner.getcol(name, *args)
            if name == "TIME":
                return np.asarray(values).astype(str)
            return values

    real_import = measurement_set.import_module
    real_table = table

    class FakeTables:
        @staticmethod
        def table(location: str | Path, **kwargs: object) -> object:
            opened = real_table(str(location), **kwargs)
            if Path(location) == target:
                return WrappedMain(opened)
            return opened

    def import_spy(name: str) -> object:
        if name == "casacore.tables":
            return FakeTables
        return real_import(name)

    monkeypatch.setattr(measurement_set, "import_module", import_spy)
    with pytest.raises(UnsafeResultInputError, match="TIME"):
        measurement_set._read_ms_metadata(
            target,
            data_column="DATA",
            limits=StandardReadLimits(),
        )


def test_measurement_set_rejects_missing_feed_before_science_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "missing-feed.ms"
    _write_checked(result, target)
    shutil.rmtree(target / "FEED")

    data_reads = 0
    real_read = measurement_set._read_ms

    def recording_read(
        path: Path,
        *,
        data_column: str,
        read_data: bool,
        limits: StandardReadLimits | None = None,
    ) -> object:
        nonlocal data_reads
        data_reads += int(read_data)
        return real_read(
            path,
            data_column=data_column,
            read_data=read_data,
            limits=limits,
        )

    monkeypatch.setattr(measurement_set, "_read_ms", recording_read)
    with pytest.raises(UnsafeResultInputError):
        read_measurement_set(target)
    assert data_reads == 0


def _mutate_ms_projection_record(
    path: Path,
    mutation: str,
) -> None:
    history = table(str(path / "HISTORY"), readonly=False, ack=False)
    try:
        for row in range(int(history.nrows())):
            message = history.getcell("MESSAGE", row)
            if not message.startswith(PROJECTION_HISTORY_PREFIX):
                continue
            encoded = message[len(PROJECTION_HISTORY_PREFIX) :]
            record = json.loads(encoded)
            if mutation == "schema":
                record["schema"] = "attacker.projection.v1"
            elif mutation == "nonfinite":
                record["projected_phase"]["longitude_rad"] = math.nan
            elif mutation == "object_fingerprint":
                record["source_scientific_sha256"] = {"forged": True}
            elif mutation == "integer_fingerprint":
                record["source_provenance_sha256"] = 7
            else:
                raise AssertionError(f"unknown mutation {mutation}")
            history.putcell(
                "MESSAGE",
                row,
                PROJECTION_HISTORY_PREFIX
                + json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            return
    finally:
        history.close()
    raise AssertionError("projection record was not found")


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "nonfinite",
        "object_fingerprint",
        "integer_fingerprint",
    ),
)
def test_measurement_set_projection_history_rejects_before_science_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / f"history-{mutation}.ms"
    _write_checked(result, target)
    _mutate_ms_projection_record(target, mutation)
    science_reads = 0
    real_read = measurement_set._read_ms

    def recording_read(path, *, data_column, read_data, limits=None):
        nonlocal science_reads
        if read_data:
            science_reads += 1
        return real_read(
            path,
            data_column=data_column,
            read_data=read_data,
            limits=limits,
        )

    monkeypatch.setattr(measurement_set, "_read_ms", recording_read)
    with pytest.raises((UnsafeResultInputError, FormatRepresentationError)):
        read_measurement_set(target)
    assert science_reads == 0


def test_measurement_set_oversized_history_storage_rejects_before_json_or_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = _write_checked(result, tmp_path / "oversized-history.ms")
    (target / "HISTORY" / "hostile-padding").write_bytes(b"X" * (1024 * 1024))
    json_reads = 0
    science_reads = 0
    real_projection_record = measurement_set.projection_record_from_history
    real_read = measurement_set._read_ms

    def recording_projection_record(history):
        nonlocal json_reads
        json_reads += 1
        return real_projection_record(history)

    def recording_read(path, *, data_column, read_data, limits=None):
        nonlocal science_reads
        if read_data:
            science_reads += 1
        return real_read(
            path,
            data_column=data_column,
            read_data=read_data,
            limits=limits,
        )

    monkeypatch.setattr(
        measurement_set,
        "projection_record_from_history",
        recording_projection_record,
    )
    monkeypatch.setattr(measurement_set, "_read_ms", recording_read)
    with pytest.raises(UnsafeResultInputError, match="HISTORY storage"):
        read_measurement_set(target)
    assert json_reads == 0
    assert science_reads == 0


def test_measurement_set_hostile_history_subprocess_allocation_is_bounded(
    tmp_path: Path,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = _write_checked(result, tmp_path / "subprocess-history.ms")
    (target / "HISTORY" / "hostile-padding").write_bytes(b"X" * (1024 * 1024))
    script = """
import json
import resource
import sys
import tracemalloc
import radiosim.io.measurement_set as module

science_reads = 0
json_reads = 0
real_read = module._read_ms
real_projection = module.projection_record_from_history

def recording_read(path, *, data_column, read_data, limits=None):
    global science_reads
    science_reads += int(read_data)
    return real_read(
        path,
        data_column=data_column,
        read_data=read_data,
        limits=limits,
    )

def recording_projection(history):
    global json_reads
    json_reads += 1
    return real_projection(history)

module._read_ms = recording_read
module.projection_record_from_history = recording_projection
module._import_standard_dependencies()
tracemalloc.start()
try:
    module.read_measurement_set(sys.argv[1])
except Exception as exc:
    rejection = type(exc).__name__
    message = str(exc)
else:
    rejection = None
    message = ""
_current, peak = tracemalloc.get_traced_memory()
print(json.dumps({
    "rejection": rejection,
    "message": message,
    "python_peak": peak,
    "native_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "science_reads": science_reads,
    "json_reads": json_reads,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout.splitlines()[-1])
    assert observed["rejection"] == "UnsafeResultInputError"
    assert "HISTORY storage" in observed["message"]
    assert observed["python_peak"] < 4 * 1024 * 1024
    assert observed["native_rss"] > 0
    assert observed["science_reads"] == 0
    assert observed["json_reads"] == 0


def test_measurement_set_optional_dependency_failure_has_no_path_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "missing" / "dependency.ms"

    monkeypatch.setattr(
        measurement_set,
        "_import_standard_dependencies",
        lambda: (_ for _ in ()).throw(
            OptionalResultDependencyError(
                "format=ms missing_package=python-casacore "
                "pyuvdata_version=3.2.1 install_extra=radiosim[ms]"
            )
        ),
    )
    with pytest.raises(OptionalResultDependencyError, match="radiosim\\[ms\\]"):
        _write_checked(result, target)
    assert not target.parent.exists()


def test_measurement_set_read_closes_handles_before_return(tmp_path: Path) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "closed.ms"
    _write_checked(result, target)

    _ = read_measurement_set(target)
    shutil.rmtree(target)
    assert not target.exists()


def test_measurement_set_lazy_import_surface() -> None:
    code = (
        "import sys, radiosim, radiosim.api, radiosim.io; "
        "forbidden={'pyuvdata','casacore','daskms'}; "
        "print(sorted(name for name in forbidden if name in sys.modules))"
    )
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "[]"


def test_measurement_set_writer_passes_explicit_safe_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    calls: list[dict[str, object]] = []
    real_write = measurement_set._write_ms

    def recording_write(
        uvdata: object,
        path: Path,
        **kwargs: object,
    ) -> None:
        calls.append(kwargs)
        real_write(uvdata, path, **kwargs)

    monkeypatch.setattr(measurement_set, "_write_ms", recording_write)
    _ = _write_checked(result, tmp_path / "arguments.ms")
    assert calls == [{"clobber": False, "force_phase": False}]


@pytest.mark.parametrize("kind", ["file", "symlink", "fifo"])
def test_measurement_set_writer_rejects_wrong_target_kind(
    tmp_path: Path,
    kind: str,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "wrong.ms"
    if kind == "file":
        target.write_bytes(b"old")
    elif kind == "symlink":
        destination = tmp_path / "destination.ms"
        destination.mkdir()
        target.symlink_to(destination, target_is_directory=True)
    else:
        os.mkfifo(target)

    with pytest.raises(OutputPathError):
        _write_checked(result, target, overwrite=True)


def test_measurement_set_racing_target_is_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "race.ms"
    real_publish = measurement_set.publish_directory_no_clobber

    def racing_publish(
        temporary: Path,
        final: Path,
        parent_fd: int,
        *,
        source_parent_fd: int | None = None,
    ) -> None:
        final.mkdir()
        (final / "owner").write_text("racer", encoding="utf-8")
        real_publish(
            temporary,
            final,
            parent_fd,
            source_parent_fd=source_parent_fd,
        )

    monkeypatch.setattr(
        measurement_set,
        "publish_directory_no_clobber",
        racing_publish,
    )
    with pytest.raises(OutputCollisionError):
        _write_checked(result, target)
    assert (target / "owner").read_text(encoding="utf-8") == "racer"
    assert tuple(tmp_path.glob(f".{target.name}.*.tmp.ms")) == ()


def test_measurement_set_exchange_failure_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build_standard_result(tmp_path / "first", dtype="complex64")
    second = build_standard_result(tmp_path / "second", dtype="complex64")
    target = tmp_path / "exchange.ms"
    _ = _write_checked(first, target)
    original = read_measurement_set(target)
    monkeypatch.setattr(
        measurement_set,
        "exchange_directories",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AtomicWriteError("exchange")),
    )

    with pytest.raises(AtomicWriteError, match="exchange"):
        _write_checked(second, target, overwrite=True)
    assert read_measurement_set(target) == original
    assert tuple(tmp_path.glob(f".{target.name}.*.tmp.ms")) == ()


def test_measurement_set_old_directory_cleanup_failure_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build_standard_result(tmp_path / "first", dtype="complex64")
    second = build_standard_result(tmp_path / "second", dtype="complex64")
    target = tmp_path / "cleanup.ms"
    _ = _write_checked(first, target)
    real_remove = measurement_set.remove_temporary_directory
    monkeypatch.setattr(
        measurement_set,
        "remove_temporary_directory",
        lambda _path: (_ for _ in ()).throw(OSError("cleanup")),
    )

    with pytest.raises(PartialCleanupError) as caught:
        _write_checked(second, target, overwrite=True)
    assert (
        read_measurement_set(target).source_scientific_sha256
        == second.scientific_sha256
    )
    assert caught.value.residual_path.name == "payload.ms"
    assert caught.value.residual_path.parent.name.endswith(".tmp.ms")
    assert caught.value.residual_path.is_dir()
    assert str(caught.value.residual_path) in str(caught.value)
    real_remove(caught.value.residual_path.parent)


def test_measurement_set_parent_fsync_failure_leaves_verified_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "fsync.ms"
    monkeypatch.setattr(
        measurement_set,
        "fsync_directory",
        lambda _descriptor: (_ for _ in ()).throw(OSError("fsync")),
    )

    with pytest.raises(AtomicWriteError, match="fsync"):
        _write_checked(result, target)
    assert (
        read_measurement_set(target).source_scientific_sha256
        == result.scientific_sha256
    )
    assert tuple(tmp_path.glob(f".{target.name}.*.tmp.ms")) == ()


def test_measurement_set_unsupported_platform_precedes_dependency_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    imported = False

    def record_import() -> object:
        nonlocal imported
        imported = True
        raise AssertionError("dependency import must not run")

    monkeypatch.setattr(
        measurement_set,
        "require_atomic_directory_support",
        lambda: (_ for _ in ()).throw(AtomicWriteUnsupportedError("unsupported")),
    )
    monkeypatch.setattr(
        measurement_set,
        "_import_standard_dependencies",
        record_import,
    )
    with pytest.raises(AtomicWriteUnsupportedError):
        _write_checked(result, tmp_path / "unsupported.ms")
    assert not imported


def test_measurement_set_history_names_every_solved_component(
    tmp_path: Path,
) -> None:
    """Tier 6G, plan Section 19 / row H10: MS HISTORY records the components."""
    result = build_standard_result(
        tmp_path,
        sky_representation="hybrid",
        components=("point", "healpix"),
        component_element_counts=(3, 3072),
    )
    target = _write_checked(result, tmp_path / "hybrid-components.ms")

    loaded = read_measurement_set(target)

    assert any("sky_representation=hybrid" in item for item in loaded.history)
    assert any("solver_components=point,healpix" in item for item in loaded.history)
    assert any(
        "solver_component_element_counts=3,3072" in item for item in loaded.history
    )
    record = projection_record_from_history("\n".join(loaded.history))[0]
    assert record["solver"]["components"] == ["point", "healpix"]
    assert record["solver"]["component_element_counts"] == [3, 3072]
