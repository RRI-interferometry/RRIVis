"""Backend-cast helpers shared across the sky package.

Consolidates the ``None if x is None else backend.asarray(x)`` ternary that
was repeated ~10× across ``operations/convert.py`` and ``combine/healpix.py``
(spec item B1). Routing through a single helper keeps the optional-array,
optional-backend, optional-dtype handling consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from radiosim.backends import ArrayBackend


def maybe_asarray(
    backend: ArrayBackend | None,
    x: Any,
    dtype: Any | None = None,
) -> Any:
    """Cast ``x`` to a backend (or numpy) array, preserving ``None``.

    Parameters
    ----------
    backend : ArrayBackend or None
        Array backend whose ``asarray`` is used when provided. When
        ``None``, :func:`numpy.asarray` is used instead.
    x : array-like or None
        Value to cast. Returned unchanged when ``None`` so optional
        payload arrays (Q/U/V maps, optional point columns) flow through
        untouched.
    dtype : dtype or None, optional
        Target dtype. ``None`` leaves the dtype to the caster.

    Returns
    -------
    array or None
        ``None`` when ``x is None``; otherwise ``backend.asarray(x,
        dtype=dtype)`` when ``backend`` is not None, else
        ``np.asarray(x, dtype=dtype)``.
    """
    if x is None:
        return None
    if backend is not None:
        return backend.asarray(x, dtype=dtype)
    return np.asarray(x, dtype=dtype)
