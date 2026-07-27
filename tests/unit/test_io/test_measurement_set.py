"""Canonical Measurement Set projection, reader, and publication contracts."""

from __future__ import annotations

import inspect
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
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
    OptionalResultDependencyError,
    OutputCollisionError,
    OutputPathError,
    OverwriteRefusedError,
    PartialCleanupError,
    UnsafeResultInputError,
)
from radiosim.io.standard_visibility import (
    StandardReadLimits,
    StandardVisibilityData,
    project_simulation_result,
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

    def recording_read(path, *, data_column, read_data):
        nonlocal data_reads
        if read_data:
            data_reads += 1
        return real_read(path, data_column=data_column, read_data=read_data)

    monkeypatch.setattr(measurement_set, "_read_ms", recording_read)
    with pytest.raises(UnsafeResultInputError, match="max_times"):
        read_measurement_set(
            target,
            limits=StandardReadLimits(max_times=1),
        )
    assert data_reads == 0


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
    assert caught.value.residual_path.is_dir()
    assert str(caught.value.residual_path) in str(caught.value)
    real_remove(caught.value.residual_path)


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
