"""Precision helpers shared across sky-model modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig


def get_sky_storage_dtype(
    precision: PrecisionConfig | None,
    category: str,
    default: object = np.float32,
) -> np.dtype:
    """Resolve the storage dtype for a sky-model precision category."""
    if precision is None:
        return np.dtype(default)
    return np.dtype(precision.sky_model.get_dtype(category))
