"""Internal helpers shared by every container dataclass.

Kept private (underscore-prefixed module name) — external code should
import the dataclasses from :mod:`radiosim.core.sky.containers` instead.
"""

from __future__ import annotations

import numpy as np
from pydantic import ConfigDict

_FROZEN_NDARRAY_CONFIG = ConfigDict(arbitrary_types_allowed=True)


def _freeze(arr: np.ndarray | None) -> np.ndarray | None:
    """Mark a numpy array read-only in place and return it.

    The container dataclasses are ``frozen=True`` against *attribute*
    rebinding, but numpy buffers stored inside them remain writeable unless
    explicitly locked.  Every container calls this on its stored arrays in
    its ``model_validator(mode="after")`` so the copy-on-write contract the
    sky module relies on is actually enforced: an in-place
    ``model.point.flux[0] = …`` raises ``ValueError`` instead of silently
    corrupting a model that advertises bit-exact equality.

    ``None`` passes through unchanged.  Locking a buffer read-only is always
    permitted (unlike *un*-locking), so this is a zero-copy backstop even
    when the array is shared by another (also read-only) container.
    """
    if arr is not None:
        arr.setflags(write=False)
    return arr


def _validate_mask(mask: object, n: int, *, label: str = "mask") -> np.ndarray:
    """Validate a boolean selection mask of length ``n``.

    Returns the mask as a boolean ndarray.  Rejects non-boolean dtypes
    (e.g. an integer array, which would silently fancy-index and
    re-order/duplicate rows) and wrong-length masks, both of which would
    otherwise corrupt a masked payload or raise a cryptic numpy error.
    """
    mask_arr = np.asarray(mask)
    if mask_arr.dtype != np.bool_:
        raise ValueError(
            f"{label} must be a boolean array, got dtype {mask_arr.dtype}. "
            "Integer masks are rejected because they silently fancy-index."
        )
    if mask_arr.shape != (n,):
        raise ValueError(
            f"{label} must have shape ({n},) to match the source count, "
            f"got {mask_arr.shape}."
        )
    return mask_arr


def _require_floating_array(arr: np.ndarray | None, *, label: str) -> None:
    """Reject non-floating container arrays without silent coercion."""
    if arr is None:
        return
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(
            f"{label} must have a floating dtype; got {arr.dtype}. "
            "Integer, complex, and object arrays are rejected by the raw "
            "sky container."
        )


def validate_frequency_axis(
    value: object, *, label: str = "frequencies", ascending: bool = True
) -> np.ndarray:
    """Coerce and validate a 1-D frequency axis (Hz), returned as float64.

    Enforces non-empty, 1-D, finite, and strictly-positive.  When
    ``ascending`` is True the axis must also be strictly increasing.  Shared
    by every spectral/HEALPix container and by loaders so the same contract
    is applied everywhere instead of being re-implemented per call site.
    """
    freqs = np.asarray(value, dtype=np.float64)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError(f"{label} must be a non-empty 1-D array.")
    if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0):
        raise ValueError(f"{label} must be finite and positive.")
    if ascending and freqs.size > 1 and not np.all(np.diff(freqs) > 0):
        raise ValueError(f"{label} must be strictly ascending.")
    return freqs


def _arrays_equal(
    a: np.ndarray | None,
    b: np.ndarray | None,
    *,
    close: bool = False,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> bool:
    """Array-aware equality: ``np.array_equal`` (strict) or ``np.allclose``.

    Used by container ``__eq__`` / ``is_close`` methods to compare optional
    numpy fields without falling into the "broadcasts to ndarray" trap.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if close:
        return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(a, b))
