"""Race-safe atomic primitives for canonical result files and directories."""

from __future__ import annotations

import ctypes
import errno
import os
import secrets
import shutil
import stat
import sys
from pathlib import Path

from radiosim.io.result_errors import (
    AtomicWriteError,
    AtomicWriteUnsupportedError,
    OutputCollisionError,
    OutputPathError,
    OverwriteRefusedError,
    PartialCleanupError,
    UnsafeOutputDirectoryError,
)

_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_TEMPORARY_ATTEMPTS = 128
_LINUX_RENAME_NOREPLACE = 1
_LINUX_RENAME_EXCHANGE = 2
_DARWIN_RENAME_SWAP = 0x00000002
_DARWIN_RENAME_EXCL = 0x00000004


def _host_platform() -> str:
    return sys.platform


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


def validate_output_directory_target(
    value: str | Path,
    *,
    extension: str,
    overwrite: object,
) -> Path:
    """Validate one final directory target without filesystem mutation."""
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean")
    path = normalize_path(value, field_name="path")
    if path.name in {"", ".", ".."}:
        raise OutputPathError(f"output path has no directory name: {path}")
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
    if not stat.S_ISDIR(status.st_mode):
        raise OutputPathError(f"output target is not a directory: {path}")
    if not overwrite:
        raise OverwriteRefusedError(f"output target already exists: {path}")
    return path


def validate_input_directory(value: str | Path) -> Path:
    """Validate one existing non-symlink directory input path."""
    path = normalize_path(value, field_name="path")
    status = _path_kind(path)
    if status is None:
        raise OutputPathError(f"result input does not exist: {path}")
    if stat.S_ISLNK(status.st_mode):
        raise OutputPathError(f"result input is a symbolic link: {path}")
    if not stat.S_ISDIR(status.st_mode):
        raise OutputPathError(f"result input is not a directory: {path}")
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


def create_sibling_temporary_directory(target: Path, parent_fd: int) -> Path:
    """Exclusively create a cryptographically named sibling MS directory."""
    for _attempt in range(_TEMPORARY_ATTEMPTS):
        name = f".{target.name}.{secrets.token_hex(16)}.tmp.ms"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        except OSError as exc:
            raise AtomicWriteError(
                f"could not create exclusive temporary directory beside {target}"
            ) from exc
        temporary = target.parent / name
        try:
            status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError as cleanup_error:
                error = PartialCleanupError(temporary)
                error.add_note(f"cleanup failure: {cleanup_error!r}")
                raise error from exc
            raise AtomicWriteError(
                f"could not inspect temporary directory beside {target}"
            ) from exc
        if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError as cleanup_error:
                error = PartialCleanupError(temporary)
                error.add_note(f"cleanup failure: {cleanup_error!r}")
                raise error from cleanup_error
            raise AtomicWriteError(
                f"exclusive temporary path is not a directory beside {target}"
            )
        return temporary
    raise AtomicWriteError(
        f"could not allocate a unique temporary directory beside {target}"
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


def _directory_rename(
    source: Path,
    destination: Path,
    parent_fd: int,
    *,
    no_replace: bool,
    source_parent_fd: int | None = None,
) -> None:
    """Invoke the host's fail-closed atomic directory rename primitive."""
    library = ctypes.CDLL(None, use_errno=True)
    source_name = os.fsencode(source.name)
    destination_name = os.fsencode(destination.name)
    platform_name = _host_platform()
    if platform_name == "darwin":
        try:
            operation = library.renameatx_np
        except AttributeError as exc:
            raise AtomicWriteUnsupportedError(
                "atomic directory publication requires renameatx_np"
            ) from exc
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        flag = _DARWIN_RENAME_EXCL if no_replace else _DARWIN_RENAME_SWAP
    elif platform_name.startswith("linux"):
        try:
            operation = library.renameat2
        except AttributeError as exc:
            raise AtomicWriteUnsupportedError(
                "atomic directory publication requires renameat2"
            ) from exc
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        flag = _LINUX_RENAME_NOREPLACE if no_replace else _LINUX_RENAME_EXCHANGE
    else:
        raise AtomicWriteUnsupportedError(
            f"atomic directory publication is unsupported on {platform_name}"
        )
    if (
        operation(
            parent_fd if source_parent_fd is None else source_parent_fd,
            source_name,
            parent_fd,
            destination_name,
            flag,
        )
        == 0
    ):
        return
    error_number = ctypes.get_errno()
    if no_replace and error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise OutputCollisionError(
            f"output target was created concurrently: {destination}"
        )
    if error_number in {errno.ENOSYS, errno.ENOTSUP}:
        raise AtomicWriteUnsupportedError(
            "host filesystem lacks the required atomic directory rename"
        )
    raise AtomicWriteError(
        f"atomic directory {'publication' if no_replace else 'exchange'} "
        f"failed: {destination}"
    ) from OSError(error_number, os.strerror(error_number))


def require_atomic_directory_support() -> None:
    """Fail unless the host exposes the required directory rename primitive."""
    library = ctypes.CDLL(None, use_errno=True)
    platform_name = _host_platform()
    if platform_name == "darwin":
        name = "renameatx_np"
    elif platform_name.startswith("linux"):
        name = "renameat2"
    else:
        raise AtomicWriteUnsupportedError(
            f"atomic directory publication is unsupported on {platform_name}"
        )
    if not hasattr(library, name):
        raise AtomicWriteUnsupportedError(
            f"atomic directory publication requires {name}"
        )


def publish_directory_no_clobber(
    temporary: Path,
    final: Path,
    parent_fd: int,
    *,
    source_parent_fd: int | None = None,
) -> None:
    """Atomically rename a sibling directory only when the target is absent."""
    _directory_rename(
        temporary,
        final,
        parent_fd,
        no_replace=True,
        source_parent_fd=source_parent_fd,
    )


def exchange_directories(
    temporary: Path,
    final: Path,
    parent_fd: int,
    *,
    source_parent_fd: int | None = None,
) -> None:
    """Atomically exchange two existing sibling directories."""
    _directory_rename(
        temporary,
        final,
        parent_fd,
        no_replace=False,
        source_parent_fd=source_parent_fd,
    )


def remove_temporary_directory(path: Path) -> None:
    """Remove one exact temporary directory without following a symlink."""
    status = _path_kind(path)
    if status is None:
        return
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise AtomicWriteError(
            f"temporary cleanup target is not a safe directory: {path}"
        )
    shutil.rmtree(path)


def fsync_file(descriptor: int) -> None:
    """Durably flush one open temporary regular file."""
    os.fsync(descriptor)


def fsync_directory(descriptor: int) -> None:
    """Durably flush one open publication directory."""
    os.fsync(descriptor)


__all__: list[str] = []
