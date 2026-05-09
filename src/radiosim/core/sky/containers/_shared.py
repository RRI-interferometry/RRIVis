"""Internal helpers shared by every container dataclass.

Kept private (underscore-prefixed module name) — external code should
import the dataclasses from :mod:`radiosim.core.sky.containers` instead.
"""

from __future__ import annotations

import numpy as np
from pydantic import ConfigDict

_FROZEN_NDARRAY_CONFIG = ConfigDict(arbitrary_types_allowed=True)


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
