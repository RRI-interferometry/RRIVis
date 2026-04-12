# rrivis/core/sky/_allocation.py
"""Memory-mapped HEALPix cube allocation helpers.

This is a LEAF module — imports only stdlib (`os`, `shutil`, `tempfile`,
`atexit`) plus `numpy`.  No imports from other sky-module files.

Provides three primitives used by loaders and converters to stream
`(n_freq, npix)` HEALPix cubes directly to disk, keeping peak memory
bounded to one frequency slice instead of the full cube.

Usage pattern (in a loader)::

    from ._allocation import ensure_scratch_dir, allocate_cube, finalize_cube

    scratch = ensure_scratch_dir(memmap_path) if memmap_path is not None else None
    i_arr = allocate_cube((n_freq, npix), np.float32, scratch, "i_maps")
    for fi, freq in enumerate(frequencies):
        i_arr[fi] = compute_slice(freq)  # row-at-a-time write
    i_arr = finalize_cube(i_arr, scratch, "i_maps")  # re-open read-only
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

import numpy as np

# Track scratch directories we created so we can clean them up on exit.
_SCRATCH_DIRS: list[str] = []


def ensure_scratch_dir(path: str | None) -> str:
    """Return a directory to store memmap files in.

    If ``path`` is given, it is used (created if needed) and the caller
    owns its lifecycle.  If ``path`` is ``None``, a fresh directory is
    created via ``tempfile.mkdtemp(prefix="rrivis_sky_")`` and registered
    for ``atexit`` cleanup.

    Parameters
    ----------
    path : str or None
        User-supplied directory, or ``None`` to create a temporary one.

    Returns
    -------
    str
        Absolute directory path.
    """
    if path is None:
        d = tempfile.mkdtemp(prefix="rrivis_sky_")
        _SCRATCH_DIRS.append(d)
        return d
    os.makedirs(path, exist_ok=True)
    return path


@atexit.register
def _cleanup_scratch_dirs() -> None:
    """Best-effort cleanup of temp dirs created by ``ensure_scratch_dir(None)``.

    Directories explicitly supplied by the user are NOT in ``_SCRATCH_DIRS``
    and are never removed here.
    """
    for d in _SCRATCH_DIRS:
        shutil.rmtree(d, ignore_errors=True)


def allocate_cube(
    shape: tuple[int, int],
    dtype: np.dtype | type,
    memmap_dir: str | None,
    name: str,
) -> np.ndarray:
    """Allocate a zero-initialised (n_freq, npix) cube in RAM or on disk.

    Parameters
    ----------
    shape : (int, int)
        ``(n_freq, npix)`` shape.
    dtype : np.dtype or type
        Element dtype.
    memmap_dir : str or None
        If ``None``, returns ``np.zeros(shape, dtype=dtype)`` (RAM-backed).
        If a directory path, returns ``np.memmap`` at
        ``<memmap_dir>/<name>.dat``, mode ``w+``, zero-filled.
    name : str
        Logical map name (``"i_maps"``, ``"q_maps"``, ``"u_maps"``,
        ``"v_maps"``).  Used as the filename stem under ``memmap_dir``.

    Returns
    -------
    np.ndarray
        ``np.ndarray`` for in-memory allocation, ``np.memmap`` (which is
        an ``ndarray`` subclass) for disk-backed allocation.  In both
        cases the array is zero-filled.
    """
    if memmap_dir is None:
        return np.zeros(shape, dtype=dtype)

    fpath = os.path.join(memmap_dir, f"{name}.dat")
    mm = np.memmap(fpath, dtype=dtype, mode="w+", shape=shape)
    # ``np.memmap`` with mode="w+" allocates the file but zero-fill is not
    # guaranteed on all platforms when the file is grown.  Explicitly zero
    # the memory to match ``np.zeros`` semantics.
    mm[:] = 0
    return mm


def finalize_cube(
    arr: np.ndarray,
    memmap_dir: str | None,
    name: str,
) -> np.ndarray:
    """Flush and re-open a memmap-backed cube read-only.

    For in-memory arrays (``memmap_dir is None`` or ``arr`` is a plain
    ndarray), returns ``arr`` unchanged.

    For memmap-backed arrays, calls ``arr.flush()`` and returns a new
    ``np.memmap`` opened in ``mode="r"`` so callers cannot accidentally
    mutate persisted cubes.

    Parameters
    ----------
    arr : np.ndarray
        Array returned by ``allocate_cube`` and subsequently filled.
    memmap_dir : str or None
        Same value passed to ``allocate_cube``.
    name : str
        Same logical name passed to ``allocate_cube``.

    Returns
    -------
    np.ndarray
    """
    if memmap_dir is None or not isinstance(arr, np.memmap):
        return arr
    arr.flush()
    fpath = os.path.join(memmap_dir, f"{name}.dat")
    return np.memmap(fpath, dtype=arr.dtype, mode="r", shape=arr.shape)
