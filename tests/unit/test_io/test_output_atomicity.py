"""Tier 4D regular-file atomic publication and cleanup contracts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import radiosim.io.atomic_paths as atomic_paths
import radiosim.io.hdf5 as hdf5_module
import radiosim.io.measurement_set as measurement_set_module
from radiosim.io.hdf5 import load_result_hdf5, write_result_hdf5
from radiosim.io.measurement_set import read_measurement_set
from radiosim.io.result_errors import (
    AtomicWriteError,
    AtomicWriteUnsupportedError,
    OutputCollisionError,
    OutputPathError,
    OverwriteRefusedError,
    PartialCleanupError,
    ResultIOError,
    UnsafeOutputDirectoryError,
)
from tests.unit.test_core.test_result import _build
from tests.unit.test_io.test_measurement_set import _write_checked
from tests.unit.test_io.test_standard_visibility import build_standard_result


def _result(tmp_path: Path):
    result, _ = _build(tmp_path)
    return result


def _temporary_artifacts(target: Path) -> tuple[Path, ...]:
    return tuple(target.parent.glob(f".{target.name}.*.tmp"))


def test_absent_target_is_atomically_published_and_immediately_readable(tmp_path):
    result = _result(tmp_path)
    target = tmp_path / "result.h5"

    returned = write_result_hdf5(result, target)

    assert returned == target.absolute()
    assert load_result_hdf5(target).scientifically_equal(result)
    assert _temporary_artifacts(target) == ()


def test_no_overwrite_refuses_existing_file_without_changing_it(tmp_path):
    result = _result(tmp_path)
    target = tmp_path / "result.h5"
    target.write_bytes(b"old bytes")

    with pytest.raises(OverwriteRefusedError):
        write_result_hdf5(result, target)

    assert target.read_bytes() == b"old bytes"
    assert _temporary_artifacts(target) == ()


def test_overwrite_replaces_only_after_verified_temporary_file(tmp_path):
    result = _result(tmp_path)
    target = tmp_path / "result.h5"
    target.write_bytes(b"old bytes")

    write_result_hdf5(result, target, overwrite=True)

    assert load_result_hdf5(target).scientifically_equal(result)
    assert target.read_bytes() != b"old bytes"
    assert _temporary_artifacts(target) == ()


def test_temporary_descriptor_is_closed_before_reader_reopens_file(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    target = tmp_path / "closed-before-readback.h5"
    descriptors: list[int] = []
    real_create = hdf5_module._create_sibling_temporary
    real_verify = hdf5_module._verify_temporary_result

    def recording_create(final, parent_fd):
        descriptor, temporary = real_create(final, parent_fd)
        descriptors.append(descriptor)
        return descriptor, temporary

    def verify_closed(canonical_result, temporary):
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
        return real_verify(canonical_result, temporary)

    monkeypatch.setattr(
        hdf5_module,
        "_create_sibling_temporary",
        recording_create,
    )
    monkeypatch.setattr(
        hdf5_module,
        "_verify_temporary_result",
        verify_closed,
    )

    write_result_hdf5(result, target)

    assert load_result_hdf5(target).scientifically_equal(result)


def test_prepublication_verification_failure_preserves_old_target(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    target = tmp_path / "result.h5"
    target.write_bytes(b"old bytes")
    monkeypatch.setattr(
        hdf5_module,
        "_verify_temporary_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("verification")),
    )

    with pytest.raises(AtomicWriteError) as caught:
        write_result_hdf5(result, target, overwrite=True)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert target.read_bytes() == b"old bytes"
    assert _temporary_artifacts(target) == ()


@pytest.mark.parametrize(
    "failure_name",
    (
        "temporary_creation",
        "group_creation",
        "dataset_creation",
        "attribute_write",
        "payload_write",
        "hdf5_close",
        "reader_reopen",
        "reader_validation",
        "fingerprint_verification",
        "hard_link_publication",
    ),
)
def test_transaction_failure_injection_leaves_no_half_written_output(
    tmp_path,
    monkeypatch,
    failure_name,
):
    result = _result(tmp_path)
    target = tmp_path / f"{failure_name}.h5"

    def fail(*_args, **_kwargs):
        raise RuntimeError(failure_name)

    if failure_name == "temporary_creation":
        monkeypatch.setattr(
            hdf5_module,
            "_create_sibling_temporary",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AtomicWriteError(failure_name)
            ),
        )
    elif failure_name == "group_creation":
        monkeypatch.setattr(hdf5_module, "_create_groups", fail)
    elif failure_name == "dataset_creation":
        monkeypatch.setattr(hdf5_module, "_create_dataset", fail)
    elif failure_name == "attribute_write":
        monkeypatch.setattr(hdf5_module, "_set_attribute", fail)
    elif failure_name == "payload_write":
        monkeypatch.setattr(hdf5_module, "_write_dataset_payload", fail)
    elif failure_name == "hdf5_close":
        real_close = hdf5_module._close_hdf5
        close_calls = 0

        def fail_first_close(handle):
            nonlocal close_calls
            close_calls += 1
            if close_calls == 1:
                raise RuntimeError(failure_name)
            return real_close(handle)

        monkeypatch.setattr(hdf5_module, "_close_hdf5", fail_first_close)
    elif failure_name == "reader_reopen":
        monkeypatch.setattr(hdf5_module, "_reopen_temporary_result", fail)
    elif failure_name == "reader_validation":
        monkeypatch.setattr(hdf5_module, "_load_open_file", fail)
    elif failure_name == "fingerprint_verification":
        monkeypatch.setattr(hdf5_module, "_verify_temporary_result", fail)
    else:
        monkeypatch.setattr(hdf5_module, "_publish_no_clobber", fail)

    with pytest.raises(ResultIOError) as caught:
        write_result_hdf5(result, target)

    if failure_name != "temporary_creation":
        assert isinstance(caught.value.__cause__, RuntimeError)
    assert not target.exists()
    assert _temporary_artifacts(target) == ()


@pytest.mark.parametrize(
    "failure_name",
    ("write", "flush", "file_fsync", "reader", "fingerprint"),
)
def test_prepublication_failure_matrix_leaves_no_final_or_temporary(
    tmp_path,
    monkeypatch,
    failure_name,
):
    result = _result(tmp_path)
    target = tmp_path / f"{failure_name}.h5"

    if failure_name == "write":
        monkeypatch.setattr(
            hdf5_module,
            "_write_hdf5_content",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("write")),
        )
    elif failure_name == "flush":
        monkeypatch.setattr(
            hdf5_module,
            "_flush_hdf5",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("flush")),
        )
    elif failure_name == "file_fsync":
        monkeypatch.setattr(
            hdf5_module,
            "_fsync_file",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fsync")),
        )
    else:
        monkeypatch.setattr(
            hdf5_module,
            "_verify_temporary_result",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(failure_name)),
        )

    with pytest.raises(AtomicWriteError):
        write_result_hdf5(result, target)

    assert not target.exists()
    assert _temporary_artifacts(target) == ()


def test_cleanup_failure_reports_exact_residual_temporary_path(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    target = tmp_path / "cleanup.h5"
    monkeypatch.setattr(
        hdf5_module,
        "_write_hdf5_content",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("write")),
    )
    real_unlink = hdf5_module._unlink_temporary

    def fail_unlink(*_args, **_kwargs):
        raise OSError("cleanup")

    monkeypatch.setattr(hdf5_module, "_unlink_temporary", fail_unlink)
    with pytest.raises(PartialCleanupError) as caught:
        write_result_hdf5(result, target)

    artifacts = _temporary_artifacts(target)
    assert len(artifacts) == 1
    assert str(artifacts[0].absolute()) in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
    real_unlink(artifacts[0])


def test_postpublication_temporary_unlink_failure_reports_residual(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    target = tmp_path / "published-cleanup.h5"
    real_unlink = hdf5_module._unlink_temporary

    def fail_unlink(*_args, **_kwargs):
        raise OSError("cleanup")

    monkeypatch.setattr(hdf5_module, "_unlink_temporary", fail_unlink)
    with pytest.raises(PartialCleanupError) as caught:
        write_result_hdf5(result, target)

    artifacts = _temporary_artifacts(target)
    assert len(artifacts) == 1
    assert caught.value.residual_path == artifacts[0]
    assert load_result_hdf5(target).scientifically_equal(result)
    real_unlink(artifacts[0])


def test_replace_failure_preserves_old_target_and_removes_temporary(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    target = tmp_path / "replace.h5"
    target.write_bytes(b"old bytes")
    monkeypatch.setattr(
        hdf5_module,
        "_publish_replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace")),
    )

    with pytest.raises(AtomicWriteError) as caught:
        write_result_hdf5(result, target, overwrite=True)

    assert isinstance(caught.value.__cause__, OSError)
    assert target.read_bytes() == b"old bytes"
    assert _temporary_artifacts(target) == ()


def test_ms_overwrite_cleanup_failure_during_old_payload_removal_is_truthful(
    tmp_path,
    monkeypatch,
):
    first = build_standard_result(tmp_path / "first", dtype="complex64")
    second = build_standard_result(tmp_path / "second", dtype="complex64")
    target = tmp_path / "during-cleanup.ms"
    _write_checked(first, target)
    real_remove = measurement_set_module.remove_temporary_directory

    def remove_part_then_fail(path):
        path = Path(path)
        if path.name == "payload.ms":
            victim = next(item for item in path.rglob("*") if item.is_file())
            victim.unlink()
            raise OSError("cleanup failed after partial removal")
        real_remove(path)

    monkeypatch.setattr(
        measurement_set_module,
        "remove_temporary_directory",
        remove_part_then_fail,
    )
    with pytest.raises(PartialCleanupError) as caught:
        _write_checked(second, target, overwrite=True)

    residual = caught.value.residual_path
    assert residual.name == "payload.ms"
    assert residual.is_dir()
    assert "only partially intact" in "\n".join(caught.value.__notes__)
    assert (
        read_measurement_set(target).source_scientific_sha256
        == second.scientific_sha256
    )
    real_remove(residual.parent)


def test_ms_overwrite_outer_container_cleanup_failure_reports_container(
    tmp_path,
    monkeypatch,
):
    first = build_standard_result(tmp_path / "first", dtype="complex64")
    second = build_standard_result(tmp_path / "second", dtype="complex64")
    target = tmp_path / "outer-cleanup.ms"
    _write_checked(first, target)
    real_remove = measurement_set_module.remove_temporary_directory

    def remove_payload_then_fail(path):
        path = Path(path)
        if path.name == "payload.ms":
            real_remove(path)
            return
        raise OSError("outer container cleanup failed")

    monkeypatch.setattr(
        measurement_set_module,
        "remove_temporary_directory",
        remove_payload_then_fail,
    )
    with pytest.raises(PartialCleanupError) as caught:
        _write_checked(second, target, overwrite=True)

    residual = caught.value.residual_path
    assert residual.name.endswith(".tmp.ms")
    assert residual.is_dir()
    assert not (residual / "payload.ms").exists()
    assert (
        read_measurement_set(target).source_scientific_sha256
        == second.scientific_sha256
    )
    real_remove(residual)


def test_ms_overwrite_postexchange_fsync_retains_readable_old_payload(
    tmp_path,
    monkeypatch,
):
    first = build_standard_result(tmp_path / "first", dtype="complex64")
    second = build_standard_result(tmp_path / "second", dtype="complex64")
    target = tmp_path / "exchange-fsync.ms"
    _write_checked(first, target)
    real_remove = measurement_set_module.remove_temporary_directory
    calls = 0

    def fail_first_fsync(_descriptor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("postexchange fsync failed")

    monkeypatch.setattr(
        measurement_set_module,
        "fsync_directory",
        fail_first_fsync,
    )
    with pytest.raises(PartialCleanupError) as caught:
        _write_checked(second, target, overwrite=True)

    residual = caught.value.residual_path
    assert residual.name == "payload.ms"
    assert (
        read_measurement_set(residual).source_scientific_sha256
        == first.scientific_sha256
    )
    assert (
        read_measurement_set(target).source_scientific_sha256
        == second.scientific_sha256
    )
    assert "old Measurement Set was retained" in "\n".join(
        caught.value.__notes__,
    )
    real_remove(residual.parent)


def test_no_clobber_publish_rejects_a_concurrent_target_without_changing_it(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    target = tmp_path / "race.h5"
    real_publish = hdf5_module._publish_no_clobber

    def racing_publish(temporary, final, parent_fd):
        final.write_bytes(b"racing writer")
        return real_publish(temporary, final, parent_fd)

    monkeypatch.setattr(hdf5_module, "_publish_no_clobber", racing_publish)
    with pytest.raises(OutputCollisionError):
        write_result_hdf5(result, target)

    assert target.read_bytes() == b"racing writer"
    assert _temporary_artifacts(target) == ()


def test_exclusive_temporary_name_collision_is_retried(tmp_path, monkeypatch):
    result = _result(tmp_path)
    target = tmp_path / "collision.h5"
    values = iter(("0" * 32, "1" * 32))
    first = tmp_path / f".{target.name}.{'0' * 32}.tmp"
    first.write_bytes(b"attacker")
    monkeypatch.setattr(atomic_paths.secrets, "token_hex", lambda _size: next(values))

    write_result_hdf5(result, target)

    assert first.read_bytes() == b"attacker"
    assert load_result_hdf5(target).scientifically_equal(result)


@pytest.mark.parametrize("kind", ("directory", "symlink", "fifo"))
def test_special_or_link_target_is_rejected_without_mutation(tmp_path, kind):
    result = _result(tmp_path)
    target = tmp_path / "unsafe.h5"
    if kind == "directory":
        target.mkdir()
    elif kind == "symlink":
        destination = tmp_path / "destination"
        destination.write_bytes(b"safe")
        target.symlink_to(destination)
    else:
        os.mkfifo(target)

    with pytest.raises((OutputPathError, UnsafeOutputDirectoryError)):
        write_result_hdf5(result, target, overwrite=True)


def test_symlink_parent_is_rejected_before_creating_output(tmp_path):
    result = _result(tmp_path)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(UnsafeOutputDirectoryError):
        write_result_hdf5(result, linked_parent / "result.h5")

    assert not (real_parent / "result.h5").exists()


def test_parent_directory_fsync_failure_is_typed_after_publication(
    tmp_path,
    monkeypatch,
):
    result = _result(tmp_path)
    target = tmp_path / "durability.h5"
    monkeypatch.setattr(
        hdf5_module,
        "_fsync_directory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("dir fsync")),
    )

    with pytest.raises(AtomicWriteError, match="directory"):
        write_result_hdf5(result, target)

    assert load_result_hdf5(target).scientifically_equal(result)


def test_directory_no_replace_and_exchange_primitives(tmp_path):
    first = tmp_path / "first.ms"
    second = tmp_path / "second.ms"
    final = tmp_path / "final.ms"
    first.mkdir()
    second.mkdir()
    (first / "value").write_text("first", encoding="utf-8")
    (second / "value").write_text("second", encoding="utf-8")
    parent_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        atomic_paths.publish_directory_no_clobber(first, final, parent_fd)
        assert (final / "value").read_text(encoding="utf-8") == "first"
        with pytest.raises(OutputCollisionError):
            atomic_paths.publish_directory_no_clobber(second, final, parent_fd)
        atomic_paths.exchange_directories(second, final, parent_fd)
    finally:
        os.close(parent_fd)
    assert (final / "value").read_text(encoding="utf-8") == "second"
    assert (second / "value").read_text(encoding="utf-8") == "first"


def test_directory_platform_support_fails_closed(monkeypatch):
    monkeypatch.setattr(atomic_paths.sys, "platform", "unsupported")
    with pytest.raises(AtomicWriteUnsupportedError):
        atomic_paths.require_atomic_directory_support()


def test_exclusive_sibling_temporary_directory_has_ms_suffix(tmp_path):
    target = tmp_path / "result.ms"
    parent_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        temporary = atomic_paths.create_sibling_temporary_directory(target, parent_fd)
    finally:
        os.close(parent_fd)
    assert temporary.is_dir()
    assert temporary.name.startswith(".result.ms.")
    assert temporary.name.endswith(".tmp.ms")
