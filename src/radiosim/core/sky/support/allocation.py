"""Memory-mapped HEALPix cube allocation helpers.

This is a LEAF module — imports only stdlib (`os`, `shutil`, `tempfile`,
`atexit`) plus `numpy`.  No imports from other sky-module files.

Provides three primitives used by loaders and converters to stream
`(n_freq, npix)` HEALPix cubes directly to disk, keeping peak memory
bounded to one frequency slice instead of the full cube.

Usage pattern (in a loader)::

    from .allocation import ensure_scratch_dir, allocate_cube, finalize_cube

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
    created via ``tempfile.mkdtemp(prefix="radiosim_sky_")`` and registered
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
        d = tempfile.mkdtemp(prefix="radiosim_sky_")
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


# Platforms on which a freshly-grown ``np.memmap("w+")`` file is guaranteed to
# read back as zeros without an explicit write.  POSIX ``ftruncate`` (used by
# numpy to size the backing file) zero-fills the grown region, so the eager
# ``mm[:] = 0`` pass is pure redundant disk IO there.  Windows offers no such
# guarantee for sparse/grown files, so we keep the eager fill on by default.
_ZERO_FILL_GUARANTEED_BY_PLATFORM = os.name == "posix"


def allocate_cube(
    shape: tuple[int, int],
    dtype: np.dtype | type,
    memmap_path: str | None,
    name: str,
    *,
    zero_fill: bool | None = None,
) -> np.ndarray:
    """Allocate a zero-initialised (n_freq, npix) cube in RAM or on disk.

    Parameters
    ----------
    shape : (int, int)
        ``(n_freq, npix)`` shape.
    dtype : np.dtype or type
        Element dtype.
    memmap_path : str or None
        If ``None``, returns ``np.zeros(shape, dtype=dtype)`` (RAM-backed).
        If a directory path, returns ``np.memmap`` at
        ``<memmap_path>/<name>.dat``, mode ``w+``.
    name : str
        Logical map name (``"i_maps"``, ``"q_maps"``, ``"u_maps"``,
        ``"v_maps"``).  Used as the filename stem under ``memmap_path``.
    zero_fill : bool or None, default None
        Controls the up-front zero-fill of the memmap backing file:

        * ``None`` (default) — lazy/platform-aware.  The eager ``mm[:] = 0``
          pass is **skipped** when the platform guarantees a grown ``w+``
          memmap reads back as zeros (POSIX ``ftruncate``), and performed
          otherwise (e.g. Windows).  This avoids a full-cube disk write on
          the common path while preserving the zero guarantee everywhere.
        * ``True`` — always perform the eager zero-fill (use when the caller
          requires the cross-platform zero guarantee regardless of platform).
        * ``False`` — never perform the eager zero-fill (opt-out; the caller
          promises to fully populate the cube before reading it).

        Ignored for RAM-backed allocations (``memmap_path is None``), which are
        always zero via ``np.zeros``.

    Returns
    -------
    np.ndarray
        ``np.ndarray`` for in-memory allocation, ``np.memmap`` (which is
        an ``ndarray`` subclass) for disk-backed allocation.  In-memory
        allocations are always zero-filled; memmap allocations read as zero
        wherever the platform or ``zero_fill`` guarantees it.
    """
    if memmap_path is None:
        return np.zeros(shape, dtype=dtype)

    fpath = os.path.join(memmap_path, f"{name}.dat")
    mm = np.memmap(fpath, dtype=dtype, mode="w+", shape=shape)
    # ``np.memmap`` with mode="w+" allocates the file but zero-fill is not
    # guaranteed on all platforms when the file is grown.  Only pay for the
    # explicit zero-fill when it is actually needed (see ``zero_fill``).
    if zero_fill is None:
        zero_fill = not _ZERO_FILL_GUARANTEED_BY_PLATFORM
    if zero_fill:
        mm[:] = 0
    return mm


def finalize_cube(
    arr: np.ndarray,
    memmap_path: str | None,
    name: str,
) -> np.ndarray:
    """Flush and re-open a memmap-backed cube read-only.

    For in-memory arrays (``memmap_path is None`` or ``arr`` is a plain
    ndarray), returns ``arr`` unchanged.

    For memmap-backed arrays, calls ``arr.flush()`` and returns a new
    ``np.memmap`` opened in ``mode="r"`` so callers cannot accidentally
    mutate persisted cubes.

    Parameters
    ----------
    arr : np.ndarray
        Array returned by ``allocate_cube`` and subsequently filled.
    memmap_path : str or None
        Same value passed to ``allocate_cube``.
    name : str
        Same logical name passed to ``allocate_cube``.

    Returns
    -------
    np.ndarray
    """
    if memmap_path is None or not isinstance(arr, np.memmap):
        return arr
    arr.flush()
    fpath = os.path.join(memmap_path, f"{name}.dat")
    return np.memmap(fpath, dtype=arr.dtype, mode="r", shape=arr.shape)
