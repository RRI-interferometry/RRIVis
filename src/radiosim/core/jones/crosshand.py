"""The cross-hand term (X).

``X`` is the relative phase between an antenna's two feed paths, constant in
frequency or linear in it::

    X_p(nu) = diag( 1, exp( i (phi_x + 2 pi nu tau_x) ) )

Cross-hand phase and cross-hand delay are the same diagonal matrix -- one
frequency-constant term and one frequency-linear one -- so they are one term
with two parameters rather than two classes (``Tier7JonesSciencePlan.md``
Sections 9.1 and 20.4).

Why the first entry is exactly ``1``
------------------------------------
Only the *relative* phase between the two feeds is physical.  A second free
parameter on feed 0 would be exactly degenerate with ``G``'s per-feed phase
error: any pair ``(a, b)`` factorizes as a common phase, which ``G`` owns and
which cancels on every baseline, times a relative phase, which is this term.
Writing ``1`` rather than a second parameter is what keeps the two terms
independent, and it is why the parameter has no feed index in configuration.

Sign, and where the phase lands
-------------------------------
The phase is written with a **positive** exponent, following the CASA
``crosshand phase`` (``Xf``) and ``KCROSS`` conventions that the parameter is
quoted in.  This is not in tension with Section 20.0's ``exp(-i phi)`` rule for
a *delay*: ``phi_x`` and ``tau_x`` are a calibration-frame relative phase and
its frequency slope, not an excess path length on a signal, and the tier's delay
terms (``Kd``, ``Rc``, ``T``, ``Z``) are the ones invariant I4 governs.  The
observable consequence is fixed and tested either way: with linear receptors the
reported cross hand ``V_01`` is multiplied by ``exp(-i phi_x)``, which is the
classic X-Y phase signature.

``X`` sits correlator-side of ``C`` (Section 12.2), so its feed indices are the
antenna's own 0/1 and not ``x``/``y``: on a circular receptor the phased feed is
``L``, and the affected correlations are the ``(RL, LR)`` pair.

References
----------
CASA ``crosshand phase`` (``Xf``) and ``KCROSS`` calibration conventions.
Sault, Hamaker & Bregman (1996), A&AS 117, 149.
Smirnov (2011), A&A 527, A106 (Paper I), Section 6.
Thompson, Moran & Swenson (2017), 3rd ed., Chapter 7.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones_errors import require_finite_jones_block

if TYPE_CHECKING:
    from radiosim.core.jones.directions import DirectionBatch

__all__ = ["CrosshandJones"]


class CrosshandJones(JonesTerm):
    """Cross-hand phase and delay ``X`` (Section 20.4).

    Constructed only by
    :func:`~radiosim.core.jones_terms.resolve_jones_terms`, from a validated
    ``jones.X`` block.

    Tier 7C renamed this term and folded the former separate cross-hand delay
    class into it, because the two are the same diagonal matrix; see
    ``docs/migration_guide.md`` for both old names.

    Parameters
    ----------
    phases_rad : ndarray
        ``(n_antenna_rows,)`` relative phase ``phi_x`` in radians, in solver
        antenna-row order.  One value per antenna, not per feed: the quantity is
        the phase *between* the two feeds.
    delays_s : ndarray
        ``(n_antenna_rows,)`` cross-hand delay ``tau_x`` in seconds.
    """

    def __init__(
        self,
        *,
        phases_rad: npt.NDArray[np.float64],
        delays_s: npt.NDArray[np.float64],
    ) -> None:
        phases = np.array(phases_rad, dtype=np.float64, copy=True, order="C")
        delays = np.array(delays_s, dtype=np.float64, copy=True, order="C")
        if phases.ndim != 1 or phases.size < 1:
            raise ValueError(
                "CrosshandJones phases_rad must have shape (n_antenna_rows,), got "
                f"{phases.shape}"
            )
        if delays.shape != phases.shape:
            raise ValueError(
                "CrosshandJones delays_s must have the same shape as phases_rad; "
                f"got {delays.shape} and {phases.shape}"
            )
        if not bool(np.isfinite(phases).all()) or not bool(np.isfinite(delays).all()):
            raise ValueError("CrosshandJones phases_rad and delays_s must be finite")
        phases.setflags(write=False)
        delays.setflags(write=False)
        self._phases_rad = phases
        self._delays_s = delays

    @property
    def name(self) -> str:
        return "X"

    @property
    def term_status(self) -> str:
        """``"implemented"``: ``X`` carries the exact Section 20.4 mathematics."""
        return "implemented"

    @property
    def is_direction_dependent(self) -> bool:
        """``False``: a relative phase between two cables has no direction."""
        return False

    @property
    def is_time_dependent(self) -> bool:
        """``False``: drift over the observation is ``G``'s job, not ``X``'s."""
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        """``True`` only when some antenna carries a non-zero cross-hand delay.

        A pure cross-hand *phase* is frequency-flat, and claiming chromaticity
        for it would be a false ``True`` -- the vacuous-flag failure mode
        invariant I2 exists to prevent.
        """
        return bool(np.any(self._delays_s != 0.0))

    def is_diagonal(self) -> bool:
        """``True`` always: ``X`` is ``diag(1, e^{i theta})`` by construction."""
        return True

    def is_scalar(self) -> bool:
        """``True`` only when the relative phase vanishes -- that is, for ``I2``.

        Computed from the resolved numbers rather than hard-coded ``False``,
        because a hard-coded flag is a claim nothing checks.  R7 makes the
        ``True`` case unreachable from a document, so in practice this is
        ``False``.
        """
        return self.is_identity()

    def is_unitary(self) -> bool:
        """``True`` always: every diagonal entry has unit modulus.

        This is the one flag that is a genuine constant for this term, and it is
        the reason ``X`` can be inserted anywhere among the diagonal
        correlator-side factors without changing the chain's power budget.
        """
        return True

    def is_identity(self) -> bool:
        """``True`` when this term is exactly ``I2`` at every frequency.

        Exactness matters: a configured phase of ``6.283185307179586`` produces
        ``1 - 2.4e-16 i`` rather than ``1``, so it is *not* the identity and R7
        does not reject it.  That is the honest answer -- the term does change
        the visibilities, by an amount the user presumably did not intend, and
        pretending otherwise would mean silently discarding a configured value.
        """
        return bool(np.all(self._phases_rad == 0.0) and np.all(self._delays_s == 0.0))

    def phasors_at_frequency(self, frequency_hz: float) -> npt.NDArray[np.complex128]:
        """Return the ``(n_antenna_rows,)`` feed-1 phasors at one frequency.

        Public because it is the closed form the invariant tests compare
        against.
        """
        angles = self._phases_rad + 2.0 * math.pi * float(frequency_hz) * self._delays_s
        return np.asarray(np.exp(1j * angles), dtype=np.complex128)

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
        """Return this antenna's ``(1, 2, 2)`` cross-hand matrix on the device.

        Direction- and time-independent, so ``directions``, ``time_mjd`` and
        ``time_idx`` are accepted and ignored, and the return is the mandated
        ``(1, 2, 2)`` broadcast form (invariant I3).  The two numbers are
        computed on the host from Python floats -- no array value is branched on
        and no device value is read back (Section 17.2).
        """
        row = int(antenna_idx)
        if row < 0 or row >= self._phases_rad.size:
            raise IndexError(
                f"CrosshandJones has phases for {self._phases_rad.size} antenna "
                f"rows; row {row} is out of range."
            )
        angle = float(self._phases_rad[row]) + (
            2.0 * math.pi * float(frequency_hz) * float(self._delays_s[row])
        )
        block = np.zeros((1, 2, 2), dtype=np.complex128)
        block[0, 0, 0] = 1.0
        block[0, 1, 1] = complex(math.cos(angle), math.sin(angle))
        require_finite_jones_block(self.name, block)
        return backend.xp.array(block, dtype=dtype)
