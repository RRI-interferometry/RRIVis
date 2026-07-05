"""Single source of truth for sky-model storage dtypes.

Every sky payload array (point-source columns, HEALPix cubes) is stored at a
dtype derived from the active :class:`~radiosim.core.precision.PrecisionConfig`
via :func:`get_sky_storage_dtype`.  Consolidating the lookup here means there is
exactly one place that maps a precision *category* to a concrete NumPy dtype, so
loaders, converters, and the combine path cannot drift apart.

Categories currently in use across the sky package::

    "source_positions"   ra/dec radians
    "flux"               Stokes I/Q/U/V and HEALPix brightness maps
    "spectral_index"     power-law / log-polynomial spectral coefficients
    "healpix_maps"       HEALPix cube storage

The category strings are passed through to
``PrecisionConfig.sky_model.get_dtype(...)``; this module deliberately does not
hard-code the category-to-dtype mapping (that lives in ``PrecisionConfig``), it
only provides the ``precision is None`` fallback and a uniform return type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

__all__ = [
    "get_sky_storage_dtype",
    "require_precision",
    "resolve_combine_precision",
]


def require_precision(precision: PrecisionConfig | None) -> PrecisionConfig:
    """Require an explicit :class:`~radiosim.core.precision.PrecisionConfig`.

    Call at loader, combine, and ops boundaries so ``None`` is not propagated
    into builders or dtype resolution helpers.
    """
    if precision is None:
        raise ValueError(
            "Sky model construction requires an explicit PrecisionConfig. "
            "Pass precision=... at the loader or constructor boundary."
        )
    return precision


def resolve_combine_precision(
    precision: PrecisionConfig | None,
    models: list[Any],
) -> PrecisionConfig:
    """Resolve combine-path precision from an explicit value or input models."""
    if precision is not None:
        return precision
    for model in models:
        model_precision = getattr(model, "precision", None)
        if model_precision is not None:
            return model_precision
    return require_precision(None)


def get_sky_storage_dtype(
    precision: PrecisionConfig | None,
    category: str,
    default: np.dtype | type = np.float32,
) -> np.dtype:
    """Resolve the storage dtype for a sky-model precision *category*.

    Parameters
    ----------
    precision : PrecisionConfig or None
        Active precision configuration.  When ``None``, ``default`` is used so
        that callers without a precision in scope still get a deterministic
        dtype.
    category : str
        Sky-model precision category (see module docstring), forwarded to
        ``precision.sky_model.get_dtype(category)``.
    default : np.dtype or type, default ``np.float32``
        Dtype used when ``precision is None``.

    Returns
    -------
    np.dtype
        The resolved NumPy dtype.
    """
    if precision is None:
        return np.dtype(default)
    return np.dtype(precision.sky_model.get_dtype(category))
