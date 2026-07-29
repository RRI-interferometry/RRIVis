"""Receptor configuration (``C``) and basis transform (``H``) Jones terms.

Both terms are direction-, time-, and frequency-independent unitary factors in
the sky-linear basis of ``Tier5ReceptorFeedPlan.md`` Section 10.  Row index is
the receptor feed, column index the sky component (``jones[feed, sky_basis]``).

Building blocks (Section 18.1)
------------------------------
Rotation of the receptor pair by ``chi`` within the sky-linear plane::

    R(chi) = [[ cos chi,  sin chi],
              [-sin chi,  cos chi]]

Linear-to-circular basis matrix, rows ordered right/left, columns ``(x, y)``::

    S = (1/sqrt 2) * [[1,  i],
                      [1, -i]]

``S`` is unitary: ``S S^H = S^H S = I2``.

``ReceptorConfigJones`` -- what the receptor physically is (Section 18.2)
------------------------------------------------------------------------
::

    C_p = M(basis_p) @ R(chi_p)

    M(linear)   = I2
    M(circular) = S

``BasisTransformJones`` -- what basis the result is reported in (Section 18.3)
-----------------------------------------------------------------------------
::

    H_p = T(basis_p -> output_basis)

    T(linear   -> linear_xy)   = I2
    T(circular -> circular_rl) = I2
    T(linear   -> circular_rl) = S
    T(circular -> linear_xy)   = S^H

``H_p @ C_p`` collapses to ``S R(chi)`` for a circular output basis regardless
of the native basis.  The two terms are nonetheless kept separate because they
answer different questions -- what the receptor physically is, and what basis
the result is reported in -- and because Tier 7's leakage term ``D`` must be
inserted *between* them.

Both terms are constructed from a
:class:`~radiosim.core.receptor.ResolvedReceptorSet` and a
:class:`~radiosim.core.instrument_adapters.SolverInstrumentView`.  Neither is
wired into a Jones chain yet; that is Tier 5D.

Modelling assumption
--------------------
Expressing a circular-native antenna in a linear output basis (or the reverse)
is exact **only** when both feeds are ideal, orthogonal, and share a common
complex gain.  That holds while the leakage (``D``) and gain (``G``) terms are
disabled identity stubs.  When Tier 7 implements ``D``, the conversion becomes
approximate and this assumption must be re-examined.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import InstrumentAdapterInvariantError
from radiosim.core.jones.base import JonesTerm
from radiosim.core.receptor import (
    ReceptorAssignmentError,
    ResolvedReceptor,
    ResolvedReceptorSet,
    UnsupportedBasisTransformError,
    UnsupportedReceptorBasisError,
)

if TYPE_CHECKING:
    from radiosim.core.instrument_adapters import SolverInstrumentView

_IDENTITY_2: Final[npt.NDArray[np.complex128]] = np.eye(2, dtype=np.complex128)

#: The Section 18.1 linear-to-circular basis matrix ``S``.
LINEAR_TO_CIRCULAR: Final[npt.NDArray[np.complex128]] = (
    1.0 / math.sqrt(2.0)
) * np.array([[1.0, 1.0j], [1.0, -1.0j]], dtype=np.complex128)

_IDENTITY_2.setflags(write=False)
LINEAR_TO_CIRCULAR.setflags(write=False)

#: Section 18.2 ``M(basis)``: the leading factor of the receptor matrix.
_LEADING_BY_BASIS: Final[Mapping[str, npt.NDArray[np.complex128]]] = {
    "linear": _IDENTITY_2,
    "circular": LINEAR_TO_CIRCULAR,
}

#: Section 18.3 ``T(native_basis -> output_basis)``.
_TRANSFORM_BY_PAIR: Final[Mapping[tuple[str, str], npt.NDArray[np.complex128]]] = {
    ("linear", "linear_xy"): _IDENTITY_2,
    ("circular", "circular_rl"): _IDENTITY_2,
    ("linear", "circular_rl"): LINEAR_TO_CIRCULAR,
    ("circular", "linear_xy"): LINEAR_TO_CIRCULAR.conj().T,
}

#: The output basis a native receptor basis already reports in.
_NATIVE_OUTPUT_BASIS: Final[Mapping[str, str]] = {
    "linear": "linear_xy",
    "circular": "circular_rl",
}

_REMOVED_KEYWORDS: Final = ("feed_type", "from_basis", "to_basis")
_REQUIRED_KEYWORDS: Final = ("receptors", "instrument")


def basis_rotation_matrix(chi_rad: float) -> npt.NDArray[np.complex128]:
    """Return the Section 18.1 receptor rotation ``R(chi)``.

    Parameters
    ----------
    chi_rad
        Receptor rotation within the sky-linear plane, in radians.

    Returns
    -------
    ndarray
        A fresh, writable ``(2, 2)`` complex rotation matrix.
    """
    cos_chi = math.cos(chi_rad)
    sin_chi = math.sin(chi_rad)
    return np.array(
        [[cos_chi, sin_chi], [-sin_chi, cos_chi]],
        dtype=np.complex128,
    )


def receptor_matrix(basis: str, chi_rad: float) -> npt.NDArray[np.complex128]:
    """Return the Section 18.2 receptor matrix ``C = M(basis) @ R(chi)``.

    Parameters
    ----------
    basis
        ``"linear"`` or ``"circular"``.
    chi_rad
        Resolved feed rotation in radians.

    Returns
    -------
    ndarray
        A fresh, writable ``(2, 2)`` unitary complex matrix.

    Raises
    ------
    UnsupportedReceptorBasisError
        ``basis`` is outside the two bases Tier 5 implements.
    """
    try:
        leading = _LEADING_BY_BASIS[basis]
    except (KeyError, TypeError):
        raise UnsupportedReceptorBasisError(
            f"basis={basis!r} is not a supported receptor basis; Tier 5 "
            "supports exactly 'linear' and 'circular'."
        ) from None
    return np.asarray(leading @ basis_rotation_matrix(chi_rad), dtype=np.complex128)


def basis_transform_matrix(
    native_basis: str,
    output_basis: str,
) -> npt.NDArray[np.complex128]:
    """Return the Section 18.3 transform ``T(native_basis -> output_basis)``.

    Parameters
    ----------
    native_basis
        The antenna's physical receptor basis, ``"linear"`` or ``"circular"``.
    output_basis
        The array-wide reporting basis, ``"linear_xy"`` or ``"circular_rl"``.

    Returns
    -------
    ndarray
        A fresh, writable ``(2, 2)`` unitary complex matrix.

    Raises
    ------
    UnsupportedBasisTransformError
        The requested pair is not one of the four Tier 5 implements.
    """
    try:
        transform = _TRANSFORM_BY_PAIR[(native_basis, output_basis)]
    except (KeyError, TypeError):
        raise UnsupportedBasisTransformError(
            f"no basis transform from receptor basis {native_basis!r} to output "
            f"basis {output_basis!r} is implemented; the receptor basis must be "
            "'linear' or 'circular' and the output basis must be 'linear_xy' or "
            "'circular_rl'."
        ) from None
    return np.array(transform, dtype=np.complex128, copy=True)


def _validate_construction(
    class_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[ResolvedReceptorSet, Any]:
    """Enforce the Section 24 keyword-only signature and the stub removal."""
    if args:
        raise TypeError(
            f"{class_name} takes no positional arguments; pass "
            "receptors=ResolvedReceptorSet and instrument=SolverInstrumentView "
            "by keyword."
        )
    removed = [name for name in _REMOVED_KEYWORDS if name in kwargs]
    if removed:
        raise TypeError(
            f"{class_name} no longer accepts {', '.join(removed)}. The receptor "
            "basis, feed rotation, and output basis are resolved once from the "
            "`receptors:` configuration section; pass "
            "receptors=ResolvedReceptorSet and instrument=SolverInstrumentView."
        )
    unexpected = sorted(set(kwargs) - set(_REQUIRED_KEYWORDS))
    if unexpected:
        raise TypeError(
            f"{class_name} got unexpected keyword arguments: "
            f"{', '.join(unexpected)}; it accepts only receptors= and instrument=."
        )
    missing = [name for name in _REQUIRED_KEYWORDS if name not in kwargs]
    if missing:
        raise TypeError(
            f"{class_name} requires the keyword arguments {', '.join(missing)}; "
            "receptors= comes from resolve_receptors() and instrument= from "
            "SolverInstrumentView.from_state()."
        )

    from radiosim.core.instrument_adapters import SolverInstrumentView as _View

    receptors = kwargs["receptors"]
    instrument = kwargs["instrument"]
    if type(receptors) is not ResolvedReceptorSet:
        raise TypeError(f"{class_name} receptors must be a ResolvedReceptorSet")
    if type(instrument) is not _View:
        raise TypeError(f"{class_name} instrument must be a SolverInstrumentView")
    return receptors, instrument


class _ReceptorTermBase(JonesTerm):
    """Shared antenna resolution and evaluation for the ``C`` and ``H`` terms."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        receptors, instrument = _validate_construction(
            type(self).__name__,
            args,
            kwargs,
        )
        self._receptors: ResolvedReceptorSet = receptors
        self._instrument: SolverInstrumentView = instrument
        self._resolved: tuple[ResolvedReceptor, ...] = (
            self._receptors_in_instrument_order()
        )
        self._matrices: tuple[npt.NDArray[np.complex128], ...] = tuple(
            self._matrix_for(receptor) for receptor in self._resolved
        )

    def _receptors_in_instrument_order(self) -> tuple[ResolvedReceptor, ...]:
        """Return one resolved receptor per solver antenna row, in row order."""
        resolved: list[ResolvedReceptor] = []
        assignments = self._receptors.receptor_by_antenna
        for number, name in zip(
            self._instrument.antenna_numbers,
            self._instrument.antenna_names,
            strict=True,
        ):
            receptor = assignments.get(AntennaId(number, name))
            if receptor is None:
                raise ReceptorAssignmentError(
                    f"antenna number {number} named {name!r} is present in the "
                    "solver instrument view but absent from the resolved "
                    "receptor set; resolve_receptors() must run against the "
                    "same canonical instrument."
                )
            resolved.append(receptor)
        return tuple(resolved)

    def _matrix_for(
        self,
        receptor: ResolvedReceptor,
    ) -> npt.NDArray[np.complex128]:  # pragma: no cover - abstract hook
        raise NotImplementedError

    @property
    def is_direction_dependent(self) -> bool:
        return False

    @property
    def is_time_dependent(self) -> bool:
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        return False

    def is_unitary(self) -> bool:
        """Always ``True``: every accepted matrix is a product of unitaries."""
        return True

    def _antenna_id(self, antenna_idx: int) -> AntennaId:
        if type(antenna_idx) is not int:
            raise InstrumentAdapterInvariantError("antenna_idx must be an integer")
        if antenna_idx < 0:
            raise InstrumentAdapterInvariantError(
                "antenna_idx must be a nonnegative antenna row"
            )
        try:
            number = self._instrument.antenna_numbers[antenna_idx]
            name = self._instrument.antenna_names[antenna_idx]
        except IndexError as exc:
            raise InstrumentAdapterInvariantError(
                f"antenna row {antenna_idx} is absent from the solver instrument view"
            ) from exc
        return AntennaId(number, name)

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs: Any,
    ) -> Any:
        """Return this antenna's 2x2 matrix on the backend device.

        The term is direction, time, and frequency independent, so
        ``source_idx``, ``freq_idx``, and ``time_idx`` are accepted and ignored.
        """
        self._antenna_id(antenna_idx)
        return backend.xp.array(self._matrices[antenna_idx], dtype=np.complex128)


class ReceptorConfigJones(_ReceptorTermBase):
    """Receptor configuration Jones term ``C`` (Section 18.2).

    ``C_p = M(basis_p) @ R(chi_p)``, with ``M(linear) = I2`` and
    ``M(circular) = S``.

    Parameters
    ----------
    receptors : ResolvedReceptorSet
        The resolved array-wide receptor inventory, keyed by ``AntennaId``.
    instrument : SolverInstrumentView
        The solver view whose antenna rows index this term.

    Raises
    ------
    TypeError
        Positional arguments, the removed ``feed_type`` keyword, or a wrong
        argument type.
    ReceptorAssignmentError
        A solver antenna row has no resolved receptor.
    UnsupportedReceptorBasisError
        A resolved receptor carries a basis Tier 5 does not implement.
    """

    @property
    def name(self) -> str:
        return "C"

    def _matrix_for(self, receptor: ResolvedReceptor) -> npt.NDArray[np.complex128]:
        return receptor_matrix(receptor.basis, receptor.feed_rotation_rad)

    def is_diagonal(self) -> bool:
        """``True`` only when every receptor is linear with zero rotation.

        Section 18.2 states the condition as ``basis == "linear"`` and
        ``chi == 0``, under which ``C`` is exactly ``I2``.  Reporting ``False``
        elsewhere is conservative: this is an optimization hint, never a
        correctness claim.
        """
        return all(
            receptor.basis == "linear" and receptor.feed_rotation_rad == 0.0
            for receptor in self._resolved
        )

    def is_scalar(self) -> bool:
        """``True`` under the same condition that makes every ``C`` exactly ``I2``."""
        return self.is_diagonal()


class BasisTransformJones(_ReceptorTermBase):
    """Polarization basis transform Jones term ``H`` (Section 18.3).

    ``H_p = T(basis_p -> output_basis)``, taking each antenna's native receptor
    basis into the one array-wide output basis resolved by
    :func:`~radiosim.core.receptor.resolve_receptors`.

    Parameters
    ----------
    receptors : ResolvedReceptorSet
        The resolved receptor inventory; supplies both the per-antenna native
        basis and the common ``output_basis``.
    instrument : SolverInstrumentView
        The solver view whose antenna rows index this term.

    Raises
    ------
    TypeError
        Positional arguments, the removed ``from_basis``/``to_basis`` keywords,
        or a wrong argument type.
    ReceptorAssignmentError
        A solver antenna row has no resolved receptor.
    UnsupportedBasisTransformError
        A native/output basis pair outside the Section 18.3 table.
    """

    @property
    def name(self) -> str:
        return "H"

    def _matrix_for(self, receptor: ResolvedReceptor) -> npt.NDArray[np.complex128]:
        return basis_transform_matrix(receptor.basis, self._receptors.output_basis)

    def is_diagonal(self) -> bool:
        """``True`` only when every antenna's native basis is the output basis.

        Those are exactly the two Section 18.3 rows where ``T`` is ``I2``.
        """
        output_basis = self._receptors.output_basis
        return all(
            _NATIVE_OUTPUT_BASIS[receptor.basis] == output_basis
            for receptor in self._resolved
        )

    def is_scalar(self) -> bool:
        """``True`` under the same condition that makes every ``H`` exactly ``I2``."""
        return self.is_diagonal()


__all__ = [
    "LINEAR_TO_CIRCULAR",
    "BasisTransformJones",
    "ReceptorConfigJones",
    "basis_rotation_matrix",
    "basis_transform_matrix",
    "receptor_matrix",
]
