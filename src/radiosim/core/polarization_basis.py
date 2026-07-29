"""The single canonical correlation-coordinate table for both output bases.

RadioSim reports visibilities in exactly one polarization basis per simulation,
named by a ``PolarizationBasis`` token:

``linear_xy``
    Two orthogonal linear receptors; correlations ``XX, XY, YX, YY``.
``circular_rl``
    Right- and left-hand circular receptors, IAU sense; correlations
    ``RR, RL, LR, LL``.

The in-memory correlation axis is always the row-major flattening of the 2x2
visibility matrix, so it is basis independent:

===========  ===============  ============  ==============
Flat index   Matrix element   ``linear_xy``  ``circular_rl``
===========  ===============  ============  ==============
0            ``[0, 0]``       ``XX``         ``RR``
1            ``[0, 1]``       ``XY``         ``RL``
2            ``[1, 0]``       ``YX``         ``LR``
3            ``[1, 1]``       ``YY``         ``LL``
===========  ===============  ============  ==============

Indices ``0`` and ``3`` are therefore the parallel hands and ``1`` and ``2`` the
cross hands in **both** bases.  Consumers must derive those indices through
:func:`parallel_hand_indices` rather than hard-coding them.

Two distinct AIPS code orders exist and must not be confused:

``AIPS_CODES_CANONICAL``
    The in-memory order, matching ``CORRELATION_LABELS`` element for element.
    This is also what a Measurement Set's ``POLARIZATION`` table records in
    ``CORR_TYPE``, because pyuvdata preserves the order it was handed.
``AIPS_CODES_FILE_ORDER``
    The descending order a UVFITS file stores on its polarization axis, and the
    order ``UVData.read_ms()`` canonicalizes ``polarization_array`` into on
    read-back.  It is *not* what a freshly written Measurement Set contains.

The linear rows reproduce the pre-existing production constants exactly, so no
linear behavior changes when a consumer migrates onto this table.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, get_args

PolarizationBasis = Literal["linear_xy", "circular_rl"]

POLARIZATION_BASES: Final[tuple[PolarizationBasis, ...]] = get_args(PolarizationBasis)

CORRELATION_LABELS: Final[Mapping[PolarizationBasis, tuple[str, str, str, str]]] = (
    MappingProxyType(
        {
            "linear_xy": ("XX", "XY", "YX", "YY"),
            "circular_rl": ("RR", "RL", "LR", "LL"),
        }
    )
)

AIPS_CODES_CANONICAL: Final[Mapping[PolarizationBasis, tuple[int, int, int, int]]] = (
    MappingProxyType(
        {
            "linear_xy": (-5, -7, -8, -6),
            "circular_rl": (-1, -3, -4, -2),
        }
    )
)

AIPS_CODES_FILE_ORDER: Final[Mapping[PolarizationBasis, tuple[int, int, int, int]]] = (
    MappingProxyType(
        {
            "linear_xy": (-5, -6, -7, -8),
            "circular_rl": (-1, -2, -3, -4),
        }
    )
)

PYUVDATA_FEEDS: Final[Mapping[PolarizationBasis, tuple[str, str]]] = MappingProxyType(
    {
        "linear_xy": ("x", "y"),
        "circular_rl": ("r", "l"),
    }
)

PYUVDATA_POLARIZATIONS: Final[Mapping[PolarizationBasis, tuple[str, ...]]] = (
    MappingProxyType(
        {
            "linear_xy": ("xx", "xy", "yx", "yy"),
            "circular_rl": ("rr", "rl", "lr", "ll"),
        }
    )
)

_PARALLEL_HAND_LABELS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "linear_xy": ("XX", "YY"),
        "circular_rl": ("RR", "LL"),
    }
)

_BASIS_BY_LABELS: Final[Mapping[tuple[str, ...], PolarizationBasis]] = MappingProxyType(
    {labels: basis for basis, labels in CORRELATION_LABELS.items()}
)


def _accepted_tuples_message() -> str:
    """Return the shared rejection text naming both accepted label tuples."""
    return " or ".join(repr(CORRELATION_LABELS[basis]) for basis in POLARIZATION_BASES)


def basis_for_correlations(correlations: tuple[str, ...]) -> PolarizationBasis:
    """Return the polarization basis named by an exact correlation label tuple.

    Parameters
    ----------
    correlations
        A four-element tuple of correlation labels in the canonical in-memory
        (row-major) order.

    Returns
    -------
    PolarizationBasis
        ``"linear_xy"`` or ``"circular_rl"``.

    Raises
    ------
    TypeError
        ``correlations`` is not a tuple of strings.
    ValueError
        The tuple is not exactly one of the two accepted orders.  A reordering
        of an accepted tuple is rejected: the correlation axis order is part of
        the contract, not a presentation choice.
    """
    if type(correlations) is not tuple or any(
        type(label) is not str for label in correlations
    ):
        raise TypeError("correlations must be a tuple of strings")
    try:
        return _BASIS_BY_LABELS[correlations]
    except KeyError:
        raise ValueError(
            f"correlations={correlations!r} is not an accepted correlation "
            f"coordinate set; expected exactly {_accepted_tuples_message()} "
            "in that order."
        ) from None


def parallel_hand_indices(correlations: tuple[str, ...]) -> tuple[int, int]:
    """Return the two parallel-hand indices on the correlation axis.

    The result is ``(0, 3)`` for both accepted bases, but it is *derived* from
    the labels rather than assumed, so a future basis cannot silently inherit
    the linear indices.

    Parameters
    ----------
    correlations
        A four-element tuple of correlation labels in canonical order.

    Returns
    -------
    tuple[int, int]
        The indices of the two parallel-hand correlations, ascending.

    Raises
    ------
    TypeError, ValueError
        As :func:`basis_for_correlations`.
    """
    basis = basis_for_correlations(correlations)
    indices = tuple(correlations.index(label) for label in _PARALLEL_HAND_LABELS[basis])
    if len(set(indices)) != 2:
        raise ValueError(
            f"correlations={correlations!r} does not contain two distinct "
            "parallel-hand correlations"
        )
    first, second = sorted(indices)
    return (first, second)


__all__ = [
    "AIPS_CODES_CANONICAL",
    "AIPS_CODES_FILE_ORDER",
    "CORRELATION_LABELS",
    "POLARIZATION_BASES",
    "PYUVDATA_FEEDS",
    "PYUVDATA_POLARIZATIONS",
    "PolarizationBasis",
    "basis_for_correlations",
    "parallel_hand_indices",
]
