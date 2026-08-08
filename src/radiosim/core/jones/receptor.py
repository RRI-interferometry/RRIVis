"""Receptor configuration (``C``) and basis transform (``H``) Jones terms.

Both terms are direction-, time-, and frequency-independent unitary factors in
the sky-linear basis of ``Tier5ReceptorFeedPlan.md`` Section 10.  Row index is
the receptor feed, column index the sky component (``jones[feed, sky_basis]``).

Building blocks
---------------
Rotation of the receptor pair by ``chi`` within the sky-linear plane::

    R(chi) = [[ cos chi,  sin chi],
              [-sin chi,  cos chi]]

Canonical sky brightness columns are ordered ``(North, East)``.  The immutable
sky-to-receptor matrices are owned by :mod:`radiosim.core.polarization_basis`::

    P = [[0, 1],
         [1, 0]]

    S = (1/sqrt 2) * [[1,  i],
                      [1, -i]]

``P`` reports ``(X=east, Y=north)`` and ``S`` reports IAU ``(R, L)``.

``ReceptorConfigJones`` -- what the receptor physically is (Section 18.2)
-------------------------------------------------------------------------
::

    C_p = M(basis_p) @ R(chi_p)

    M(linear)   = P
    M(circular) = S

``BasisTransformJones`` -- what basis the result is reported in (Section 18.3)
------------------------------------------------------------------------------
::

    H_p = T(basis_p -> output_basis)

    T(linear   -> linear_xy)   = I2
    T(circular -> circular_rl) = I2
    T(linear   -> circular_rl) = S P
    T(circular -> linear_xy)   = P S^H

``H_p @ C_p`` collapses to ``S R(chi)`` for a circular output basis regardless
of the native basis.  The two terms are nonetheless kept separate because they
answer different questions -- what the receptor physically is, and what basis
the result is reported in -- and because Tier 7's leakage term ``D`` must be
inserted *between* them.

Both terms are constructed from a
:class:`~radiosim.core.receptor.ResolvedReceptorSet` and a
:class:`~radiosim.core.instrument_adapters.SolverInstrumentView`.  The
maintained Jones chain applies ``C`` on the sky side of native-feed terms and
``H`` on their output side.  Consequently leakage, cross-coupling, and
feed-asymmetric gain-like terms remain attached to the physical native feeds;
they cannot in general be commuted through either basis boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import InstrumentAdapterInvariantError
from radiosim.core.jones.base import JonesTerm
from radiosim.core.polarization_basis import (
    SKY_NORTH_EAST_TO_CIRCULAR_RL,
    SKY_TO_NATIVE_RECEPTOR,
    SKY_TO_OUTPUT_RECEPTOR,
)
from radiosim.core.receptor import (
    ReceptorAssignmentError,
    ResolvedReceptor,
    ResolvedReceptorSet,
    UnsupportedBasisTransformError,
    UnsupportedReceptorBasisError,
)

if TYPE_CHECKING:
    from radiosim.core.instrument_adapters import SolverInstrumentView
    from radiosim.core.jones.directions import DirectionBatch

_IDENTITY_2: Final[npt.NDArray[np.complex128]] = np.eye(2, dtype=np.complex128)

#: The linear ``(X, Y)`` to circular ``(R, L)`` transform, ``S P``.
LINEAR_TO_CIRCULAR: Final[npt.NDArray[np.complex128]] = np.asarray(
    SKY_NORTH_EAST_TO_CIRCULAR_RL @ SKY_TO_OUTPUT_RECEPTOR["linear_xy"].conj().T,
    dtype=np.complex128,
)

_IDENTITY_2.setflags(write=False)
LINEAR_TO_CIRCULAR.setflags(write=False)

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
    cos_chi = np.cos(chi_rad)
    sin_chi = np.sin(chi_rad)
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
        leading = SKY_TO_NATIVE_RECEPTOR[basis]
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
    """Return ``H = M_output M_native^H`` for the requested bases.

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
        native = SKY_TO_NATIVE_RECEPTOR[native_basis]
        output = SKY_TO_OUTPUT_RECEPTOR[output_basis]
    except (KeyError, TypeError):
        raise UnsupportedBasisTransformError(
            f"no basis transform from receptor basis {native_basis!r} to output "
            f"basis {output_basis!r} is implemented; the receptor basis must be "
            "'linear' or 'circular' and the output basis must be 'linear_xy' or "
            "'circular_rl'."
        ) from None
    # Canonicalize the algebraic identity cases exactly.  In particular,
    # evaluating S @ S^H numerically would otherwise return diagonal entries a
    # few ULP below one even though the contractual transform is I2.
    if _NATIVE_OUTPUT_BASIS[native_basis] == output_basis:
        return np.array(_IDENTITY_2, dtype=np.complex128, copy=True)
    return np.asarray(output @ native.conj().T, dtype=np.complex128)


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
    def term_status(self) -> str:
        """``"implemented"``: ``C`` and ``H`` are Tier 5 physics, not scaffolds.

        Both carry the exact Section 18 matrices, both change the visibilities
        for a receptor they are not trivial for, and both are swept numerically
        by invariant I2.  They are the two terms that were already real when
        Tier 7 began.
        """
        return "implemented"

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

    def compute_jones_batch(
        self,
        *,
        antenna_idx: int,
        directions: DirectionBatch,
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        """Return this antenna's ``(1, 2, 2)`` matrix on the backend device.

        The term is direction, time, and frequency independent, so ``directions``
        and the frequency and time arguments are accepted and ignored, and the
        return is the mandated ``(1, 2, 2)`` direction-independent form: one
        matrix that broadcasts against the direction batch, never ``n_dir``
        copies of a constant (invariant I3).

        ``dtype`` comes from the caller.  It used to be a literal
        ``np.complex128`` here, which silently overrode ``PrecisionConfig`` for
        the two terms that are always in the chain (defect D9).  Under every
        preset whose accumulation precision is ``float64`` -- which includes
        both ``standard`` and ``fast`` -- the resolved dtype *is* ``complex128``,
        so this is bit-identical for every shipped configuration; a preset that
        resolves something else is where the fix becomes observable, which is
        what invariant I17 tests.
        """
        self._antenna_id(antenna_idx)
        return backend.xp.array(self._matrices[antenna_idx][None, :, :], dtype=dtype)


class ReceptorConfigJones(_ReceptorTermBase):
    """Receptor configuration Jones term ``C`` (Section 18.2).

    ``C_p = M(basis_p) @ R(chi_p)``, with ``M(linear) = P`` and
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
        """Whether every resolved ``C`` matrix is exactly diagonal."""
        return all(
            np.array_equal(matrix, np.diag(np.diag(matrix)))
            for matrix in self._matrices
        )

    def is_scalar(self) -> bool:
        """Whether every resolved ``C`` is exactly a scalar multiple of ``I2``."""
        return all(
            np.array_equal(matrix, matrix[0, 0] * _IDENTITY_2)
            for matrix in self._matrices
        )


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
