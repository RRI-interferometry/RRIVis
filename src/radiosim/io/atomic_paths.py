"""Race-safe atomic primitives for regular result files."""

from __future__ import annotations

import errno
import os
import secrets
import stat
from pathlib import Path

from radiosim.io.result_errors import (
    AtomicWriteError,
    OutputCollisionError,
    OutputPathError,
    OverwriteRefusedError,
    UnsafeOutputDirectoryError,
)

_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_TEMPORARY_ATTEMPTS = 128


def normalize_path(value: str | Path, *, field_name: str) -> Path:
    """Return an absolute lexical path without resolving symbolic links."""
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a string or pathlib.Path") from exc
    if type(raw) is not str or not raw:
        raise TypeError(f"{field_name} must be a nonempty filesystem path")
    if "\x00" in raw:
        raise OutputPathError(f"{field_name} contains a NUL byte")
    return Path(os.path.abspath(os.path.normpath(raw)))


def _path_kind(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OutputPathError(
            f"could not inspect normalized output path {path}"
        ) from exc


def _validate_existing_ancestors(parent: Path) -> None:
    current = Path(parent.anchor)
    for component in parent.parts[1:]:
        current = current / component
        status = _path_kind(current)
        if status is None:
            continue
        if stat.S_ISLNK(status.st_mode):
            raise UnsafeOutputDirectoryError(
                f"output ancestor is a symbolic link: {current}"
            )
        if not stat.S_ISDIR(status.st_mode):
            raise UnsafeOutputDirectoryError(
                f"output ancestor is not a directory: {current}"
            )


def validate_output_target(
    value: str | Path,
    *,
    extension: str,
    overwrite: object,
) -> Path:
    """Validate one final regular-file target without filesystem mutation."""
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean")
    path = normalize_path(value, field_name="path")
    if path.name in {"", ".", ".."}:
        raise OutputPathError(f"output path has no filename: {path}")
    if path.suffixes != [extension]:
        raise OutputPathError(
            f"output path must use one exact {extension} extension: {path}"
        )
    _validate_existing_ancestors(path.parent)
    status = _path_kind(path)
    if status is None:
        return path
    if stat.S_ISLNK(status.st_mode):
        raise OutputPathError(f"output target is a symbolic link: {path}")
    if not stat.S_ISREG(status.st_mode):
        raise OutputPathError(f"output target is not a regular file: {path}")
    if not overwrite:
        raise OverwriteRefusedError(f"output target already exists: {path}")
    return path


def validate_input_regular_file(value: str | Path) -> Path:
    """Validate one existing non-symlink regular input path."""
    path = normalize_path(value, field_name="path")
    status = _path_kind(path)
    if status is None:
        raise OutputPathError(f"result input does not exist: {path}")
    if stat.S_ISLNK(status.st_mode):
        raise OutputPathError(f"result input is a symbolic link: {path}")
    if not stat.S_ISREG(status.st_mode):
        raise OutputPathError(f"result input is not a regular file: {path}")
    return path


def open_parent_directory(parent: Path, *, create: bool) -> int:
    """Open a parent directory by walking from root without following links."""
    normalized = normalize_path(parent, field_name="parent")
    descriptor = os.open(normalized.anchor, _DIRECTORY_FLAGS)
    try:
        for component in normalized.parts[1:]:
            while True:
                try:
                    child = os.open(
                        component,
                        _DIRECTORY_FLAGS | _NOFOLLOW,
                        dir_fd=descriptor,
                    )
                    break
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(component, mode=0o755, dir_fd=descriptor)
                    except FileExistsError:
                        continue
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise UnsafeOutputDirectoryError(
                            f"output ancestor is unsafe: {normalized}"
                        ) from exc
                    raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def create_sibling_temporary(target: Path, parent_fd: int) -> tuple[int, Path]:
    """Exclusively create a cryptographically named sibling regular file."""
    for _attempt in range(_TEMPORARY_ATTEMPTS):
        name = f".{target.name}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        except OSError as exc:
            raise AtomicWriteError(
                f"could not create exclusive temporary file beside {target}"
            ) from exc
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            os.close(descriptor)
            raise AtomicWriteError(
                f"exclusive temporary path is not a regular file beside {target}"
            )
        return descriptor, target.parent / name
    raise AtomicWriteError(
        f"could not allocate a unique temporary file beside {target}"
    )


def unlink_temporary(path: Path, parent_fd: int | None = None) -> None:
    """Remove one exact temporary file without following a link."""
    if parent_fd is None:
        os.unlink(path)
    else:
        os.unlink(path.name, dir_fd=parent_fd)


def publish_no_clobber(temporary: Path, final: Path, parent_fd: int) -> None:
    """Atomically publish a sibling file without replacing a racing target."""
    try:
        os.link(
            temporary.name,
            final.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise OutputCollisionError(
            f"output target was created concurrently: {final}"
        ) from exc
    except OSError as exc:
        raise AtomicWriteError(
            f"atomic no-clobber publication failed: {final}"
        ) from exc


def publish_replace(temporary: Path, final: Path, parent_fd: int) -> None:
    """Atomically replace one regular non-symlink sibling target."""
    try:
        status = os.stat(final.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        status = None
    if status is not None and not stat.S_ISREG(status.st_mode):
        raise OutputCollisionError(
            f"overwrite target is no longer a regular file: {final}"
        )
    try:
        os.replace(
            temporary.name,
            final.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    except OSError as exc:
        raise AtomicWriteError(f"atomic replacement failed: {final}") from exc


def fsync_file(descriptor: int) -> None:
    """Durably flush one open temporary regular file."""
    os.fsync(descriptor)


def fsync_directory(descriptor: int) -> None:
    """Durably flush one open publication directory."""
    os.fsync(descriptor)


__all__: list[str] = []
