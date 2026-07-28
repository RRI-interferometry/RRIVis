"""Private, bounded artifacts for one owned CLI workflow run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import yaml

from radiosim.io.atomic_paths import fsync_file, normalize_path
from radiosim.io.result_errors import (
    AtomicWriteError,
    OutputCollisionError,
    UnsafeOutputDirectoryError,
)

_MANIFEST_SCHEMA = "radiosim.workflow-manifest.v1"
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_MANIFEST_ARTIFACTS = 10_000
_MAX_NESTING = 32
_SHA256_LENGTH = 64
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True, slots=True)
class OwnedRunManifest:
    """Validated manifest identity for an exact run directory."""

    run_directory: Path
    artifacts: tuple[tuple[str, str, str], ...]


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError("duplicate JSON object key")
        mapping[key] = value
    return mapping


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _validate_json_tree(value: object, *, depth: int = 0) -> None:
    if depth > _MAX_NESTING:
        raise ValueError("JSON nesting exceeds the workflow manifest limit")
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str) and "\x00" in value:
            raise ValueError("JSON text contains a NUL")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON number is non-finite")
        return
    if type(value) is list:
        for item in cast(list[object], value):
            _validate_json_tree(item, depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str or "\x00" in key:
                raise ValueError("JSON object key is invalid")
            _validate_json_tree(item, depth=depth + 1)
        return
    raise ValueError(f"JSON value has unsupported type {type(value).__name__}")


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
            0o600,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AtomicWriteError(
                    f"workflow artifact write made no progress: {path}"
                )
            view = view[written:]
        fsync_file(descriptor)
    except FileExistsError as exc:
        raise OutputCollisionError(f"workflow artifact already exists: {path}") from exc
    except AtomicWriteError:
        raise
    except OSError as exc:
        raise AtomicWriteError(f"could not write workflow artifact: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_resolved_config_artifact(
    value: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Write and verify one deterministic resolved-configuration YAML artifact."""
    target = normalize_path(path, field_name="path")
    if target.name != "resolved-config.yaml":
        raise AtomicWriteError(
            "resolved configuration artifact must be named resolved-config.yaml"
        )
    try:
        canonical_json = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
        )
        canonical = json.loads(canonical_json)
        encoded = yaml.safe_dump(
            canonical,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError, yaml.YAMLError) as exc:
        raise AtomicWriteError(
            "resolved configuration artifact is not finite JSON-safe data"
        ) from exc
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise AtomicWriteError(
            "resolved configuration artifact exceeds the 16 MiB limit"
        )
    _exclusive_write(target, encoded)
    try:
        observed = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise AtomicWriteError(
            "resolved configuration artifact failed safe read-back"
        ) from exc
    if observed != canonical:
        raise AtomicWriteError(
            "resolved configuration artifact failed exact read-back verification"
        )
    return target


def _regular_file_digest(path: Path) -> str:
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise UnsafeOutputDirectoryError(
            f"owned artifact is not a regular file: {path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_digest(path: Path) -> str:
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise UnsafeOutputDirectoryError(f"owned artifact is not a directory: {path}")
    digest = hashlib.sha256()
    saw_file = False
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(path).as_posix()
        candidate_status = candidate.lstat()
        if stat.S_ISLNK(candidate_status.st_mode):
            raise UnsafeOutputDirectoryError(
                f"owned directory artifact contains a symbolic link: {candidate}"
            )
        if stat.S_ISDIR(candidate_status.st_mode):
            digest.update(b"D\0" + relative.encode("utf-8") + b"\0")
            continue
        if not stat.S_ISREG(candidate_status.st_mode):
            raise UnsafeOutputDirectoryError(
                f"owned directory artifact contains a special file: {candidate}"
            )
        saw_file = True
        digest.update(b"F\0" + relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(_regular_file_digest(candidate)))
    if not saw_file:
        raise UnsafeOutputDirectoryError(
            f"owned directory artifact is unexpectedly empty: {path}"
        )
    return digest.hexdigest()


def _artifact_entry(run_directory: Path, artifact: Path) -> dict[str, str]:
    normalized = normalize_path(artifact, field_name="artifact")
    try:
        relative = normalized.relative_to(run_directory)
    except ValueError as exc:
        raise UnsafeOutputDirectoryError(
            f"workflow artifact is outside the exact run directory: {normalized}"
        ) from exc
    relative_text = relative.as_posix()
    pure = PurePosixPath(relative_text)
    if (
        not relative_text
        or pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or relative_text == "manifest.json"
    ):
        raise UnsafeOutputDirectoryError(
            f"workflow artifact path is not a safe owned relative path: {relative_text}"
        )
    status = normalized.lstat()
    if stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode):
        kind = "file"
        fingerprint = _regular_file_digest(normalized)
    elif stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
        kind = "directory"
        fingerprint = _directory_digest(normalized)
    else:
        raise UnsafeOutputDirectoryError(
            f"workflow artifact has an unsafe target kind: {normalized}"
        )
    return {"kind": kind, "path": relative_text, "sha256": fingerprint}


def write_workflow_manifest(
    run_directory: str | Path,
    artifact_paths: object,
) -> Path:
    """Write and verify the exact sorted ownership manifest in staging."""
    run = normalize_path(run_directory, field_name="run_directory")
    if not run.is_dir() or run.is_symlink():
        raise UnsafeOutputDirectoryError(
            f"workflow staging path is not a safe directory: {run}"
        )
    if isinstance(artifact_paths, (str, bytes)) or not isinstance(
        artifact_paths, Sequence
    ):
        raise TypeError("artifact_paths must be a sequence of paths")
    paths = cast(Sequence[str | Path], artifact_paths)
    entries = [_artifact_entry(run, Path(path)) for path in paths]
    entries.sort(key=lambda entry: entry["path"])
    names = [entry["path"] for entry in entries]
    if (
        not entries
        or len(entries) > _MAX_MANIFEST_ARTIFACTS
        or len(names) != len(set(names))
    ):
        raise UnsafeOutputDirectoryError(
            "workflow manifest artifact paths must be nonempty and unique"
        )
    document = {"artifacts": entries, "schema": _MANIFEST_SCHEMA}
    encoded = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8", errors="strict")
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise UnsafeOutputDirectoryError("workflow manifest exceeds the 16 MiB limit")
    target = run / "manifest.json"
    _exclusive_write(target, encoded)
    _ = validate_owned_run_directory(run)
    return target


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        status = path.lstat()
    except FileNotFoundError as exc:
        raise UnsafeOutputDirectoryError(
            f"workflow manifest is missing: {path}"
        ) from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise UnsafeOutputDirectoryError(
            f"workflow manifest is not a regular file: {path}"
        )
    if status.st_size <= 0 or status.st_size > _MAX_ARTIFACT_BYTES:
        raise UnsafeOutputDirectoryError(
            "workflow manifest violates the bounded-size contract"
        )
    try:
        raw = path.read_bytes()
        document = cast(
            object,
            json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            ),
        )
        _validate_json_tree(document)
    except (
        OSError,
        RecursionError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise UnsafeOutputDirectoryError(
            "workflow manifest is not strict bounded JSON"
        ) from exc
    if type(document) is not dict:
        raise UnsafeOutputDirectoryError("workflow manifest root must be an object")
    return cast(dict[str, object], document)


def validate_owned_run_directory(
    run_directory: str | Path,
) -> OwnedRunManifest:
    """Validate exact run ownership without following or reconstructing content."""
    run = normalize_path(run_directory, field_name="run_directory")
    try:
        run_status = run.lstat()
    except FileNotFoundError as exc:
        raise UnsafeOutputDirectoryError(
            f"workflow run directory does not exist: {run}"
        ) from exc
    if stat.S_ISLNK(run_status.st_mode) or not stat.S_ISDIR(run_status.st_mode):
        raise UnsafeOutputDirectoryError(
            f"workflow run target is not a safe directory: {run}"
        )
    document = _load_manifest(run / "manifest.json")
    if set(document) != {"artifacts", "schema"}:
        raise UnsafeOutputDirectoryError(
            "workflow manifest contains unexpected top-level keys"
        )
    if document["schema"] != _MANIFEST_SCHEMA:
        raise UnsafeOutputDirectoryError("workflow manifest schema is unsupported")
    artifact_values = document["artifacts"]
    if type(artifact_values) is not list:
        raise UnsafeOutputDirectoryError(
            "workflow manifest artifacts must be a bounded nonempty array"
        )
    artifacts = cast(list[object], artifact_values)
    if not artifacts or len(artifacts) > _MAX_MANIFEST_ARTIFACTS:
        raise UnsafeOutputDirectoryError(
            "workflow manifest artifacts must be a bounded nonempty array"
        )
    observed: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    top_level_owned = {"manifest.json"}
    for item in artifacts:
        if type(item) is not dict:
            raise UnsafeOutputDirectoryError(
                "workflow manifest artifact entry has an invalid shape"
            )
        entry_mapping = cast(dict[str, object], item)
        if set(entry_mapping) != {"kind", "path", "sha256"}:
            raise UnsafeOutputDirectoryError(
                "workflow manifest artifact entry has an invalid shape"
            )
        path_text = entry_mapping["path"]
        kind = entry_mapping["kind"]
        fingerprint = entry_mapping["sha256"]
        if (
            type(path_text) is not str
            or type(kind) is not str
            or type(fingerprint) is not str
            or len(fingerprint) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise UnsafeOutputDirectoryError(
                "workflow manifest artifact entry has an invalid type"
            )
        pure = PurePosixPath(path_text)
        if (
            not path_text
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
            or "\\" in path_text
            or "\x00" in path_text
            or path_text == "manifest.json"
            or path_text in seen
            or len(pure.parts) != 1
        ):
            raise UnsafeOutputDirectoryError(
                "workflow manifest artifact path is unsafe or duplicated"
            )
        seen.add(path_text)
        top_level_owned.add(path_text)
        candidate = run / path_text
        entry = _artifact_entry(run, candidate)
        if entry != entry_mapping:
            raise UnsafeOutputDirectoryError(
                f"workflow artifact failed ownership verification: {path_text}"
            )
        observed.append((path_text, kind, fingerprint))
    if [path for path, _kind, _digest in observed] != sorted(seen):
        raise UnsafeOutputDirectoryError(
            "workflow manifest artifact entries are not sorted"
        )
    actual_top_level = {candidate.name for candidate in run.iterdir()}
    if actual_top_level != top_level_owned:
        raise UnsafeOutputDirectoryError(
            "workflow run contains unlisted or missing top-level content"
        )
    return OwnedRunManifest(run, tuple(observed))


__all__: list[str] = []
