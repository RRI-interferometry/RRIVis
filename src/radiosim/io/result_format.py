"""Typed canonical result formats and exact final-path normalization."""

from __future__ import annotations

import stat
from enum import Enum
from importlib.util import find_spec
from pathlib import Path

from radiosim.io.atomic_paths import normalize_path
from radiosim.io.result_errors import (
    OptionalResultDependencyError,
    OutputPathError,
    OverwriteRefusedError,
    UnsafeOutputDirectoryError,
)


class ResultFormat(str, Enum):
    """The four exact canonical result output formats."""

    HDF5 = "hdf5"
    SUMMARY_JSON = "summary_json"
    MS = "ms"
    UVFITS = "uvfits"

    @property
    def extension(self) -> str:
        """Return the one canonical extension for this format."""
        return {
            ResultFormat.HDF5: ".h5",
            ResultFormat.SUMMARY_JSON: ".summary.json",
            ResultFormat.MS: ".ms",
            ResultFormat.UVFITS: ".uvfits",
        }[self]


def require_result_format(value: object) -> ResultFormat:
    """Require an exact enum value at Python API boundaries."""
    if type(value) is not ResultFormat:
        raise TypeError("format must be a ResultFormat")
    return value


def normalize_result_path(value: str | Path, format: ResultFormat) -> Path:
    """Normalize a final artifact path and append only a missing extension."""
    typed_format = require_result_format(format)
    path = normalize_path(value, field_name="path")
    if path.name in {"", ".", ".."}:
        raise OutputPathError(f"output path has no artifact name: {path}")
    extension = typed_format.extension
    if path.name.endswith(extension):
        return path
    if path.suffixes:
        raise OutputPathError(
            f"output path extension conflicts with format "
            f"{typed_format.value!r}; expected {extension}: {path}"
        )
    return path.with_name(path.name + extension)


def require_result_dependencies(format: ResultFormat) -> None:
    """Fail before path mutation when one requested dependency is unavailable."""
    typed_format = require_result_format(format)
    dependencies = {
        ResultFormat.HDF5: (("h5py", "h5py"),),
        ResultFormat.MS: (("pyuvdata", "pyuvdata"), ("casacore", "python-casacore")),
        ResultFormat.UVFITS: (
            ("pyuvdata", "pyuvdata"),
            ("astropy.io.fits", "astropy"),
        ),
    }.get(typed_format, ())
    for module_name, package_name in dependencies:
        if find_spec(module_name) is None:
            raise OptionalResultDependencyError(
                f"{typed_format.value} output requires missing package "
                f"{package_name}; install the documented result-output dependencies"
            )


def preflight_result_target(
    value: str | Path,
    format: ResultFormat,
    *,
    overwrite: object,
) -> Path:
    """Validate one exact final target without importing a writer or mutating."""
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean")
    typed_format = require_result_format(format)
    path = normalize_result_path(value, typed_format)
    current = Path(path.anchor)
    for component in path.parent.parts[1:]:
        current = current / component
        try:
            status = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise UnsafeOutputDirectoryError(
                f"result output ancestor is unsafe: {current}"
            )
    try:
        status = path.lstat()
    except FileNotFoundError:
        return path
    if stat.S_ISLNK(status.st_mode):
        raise OutputPathError(f"output target is a symbolic link: {path}")
    expected_directory = typed_format is ResultFormat.MS
    if expected_directory and not stat.S_ISDIR(status.st_mode):
        raise OutputPathError(f"MS output target is not a directory: {path}")
    if not expected_directory and not stat.S_ISREG(status.st_mode):
        raise OutputPathError(f"file output target is not a regular file: {path}")
    if not overwrite:
        raise OverwriteRefusedError(f"output target already exists: {path}")
    return path


__all__ = ["ResultFormat"]
