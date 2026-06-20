"""Astropy ``Quantity`` unwrapping helper.

Consolidates the ``q.to_value(unit) if hasattr(q, "to_value") else
np.asarray(q)`` idiom that was repeated ~10× across
``loaders/pyradiosky.py`` and ``loaders/skyh5_multifile.py`` (spec item
B7).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_value(q: Any, unit: Any) -> np.ndarray:
    """Return the bare numeric value of ``q`` expressed in ``unit``.

    Accepts either an astropy :class:`~astropy.units.Quantity` (in which
    case ``q.to_value(unit)`` is used) or a plain array/scalar (returned
    via :func:`numpy.asarray`).

    Parameters
    ----------
    q : Quantity or array-like or scalar
        Value to unwrap. A ``Quantity`` is converted to ``unit``; a plain
        array/scalar passes through unchanged.
    unit : astropy unit
        Target unit applied only when ``q`` is a ``Quantity``.

    Returns
    -------
    np.ndarray
        The bare value as a numpy array.
    """
    if hasattr(q, "to_value"):
        return np.asarray(q.to_value(unit))
    return np.asarray(q)
