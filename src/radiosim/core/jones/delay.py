"""Delay-like instrumental terms: the electronic delay (Kd) and cable
reflection (Rc).

``Kd`` is a per-antenna, per-feed instrumental delay offset::

    Kd_p(nu) = diag( exp(-2 pi i nu tau_p0), exp(-2 pi i nu tau_p1) )

whose negative exponent matches the geometric phase's own ``exp(-2 pi i b.s)``
(``visibility.py``), so that a positive delay produces ``exp(-i * positive)`` on
both paths.  That is invariant **I4**, and it is the one sign convention the
whole tier shares (``Tier7JonesSciencePlan.md`` Section 20.0).

``Rc`` is the standing-wave ripple from a reflection in the RF cable::

    Rc_p(nu) = diag( r_p0(nu), r_p1(nu) )
    r_pf(nu) = 1 + A_pf exp( -2 pi i nu tau_cable,pf + i phi_pf )

with ``0 < |A| < 1`` enforced -- a reflection cannot return more power than it
receives.  This is the **first-order, single-bounce** reflection: multiple
bounces would add terms in ``A^2 exp(-4 pi i nu tau_c)`` and further, and they
are out of scope.

Why ``Rc`` is a term and not a bandpass shape
---------------------------------------------
``B`` and ``Rc`` are both diagonal and both frequency-dependent, so nothing
stops a user expressing one as the other.  They are separate terms because their
*delay-domain* signatures differ in kind: a smooth bandpass is compact around
zero delay, while a cable reflection puts a discrete secondary peak at exactly
``tau_cable`` with relative amplitude ``A``.  That peak is the observable a
21 cm power-spectrum analysis actually cares about (Kern et al. 2020), and it is
what ``tests/unit/test_jones/test_cable_reflection.py`` asserts numerically.

Both terms sit correlator-side of ``C`` (Section 12.2), so their feed indices
are the antenna's own 0/1 and not ``x``/``y``.  Their order relative to ``G``,
``B`` and ``X`` is a **convention**: all five are diagonal in the same basis and
therefore commute (Section 12.3).

There is no fringe-fitting term: fringe fitting is a *calibration solution*, and
its forward-model content is exactly ``G`` times ``Kd`` times a phase rate
(Section 9.1).

References
----------
Thompson, Moran & Swenson (2017), *Interferometry and Synthesis in Radio
Astronomy*, 3rd ed., Chapter 7 -- instrumental delay; CASA ``K`` Jones.
Kern et al. (2020), ApJ 888, 70 -- HERA cable reflections as a delay-domain
ripple.
Beardsley et al. (2016), ApJ 833, 102.
Ewall-Wice et al. (2016), MNRAS 460, 4320 -- cable reflections in HERA/PAPER.
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

__all__ = ["CableReflectionJones", "DelayJones"]


def _per_feed_table(name: str, field: str, values: Any) -> npt.NDArray[np.float64]:
    """Return a validated, read-only ``(n_antenna_rows, 2)`` float table."""
    table = np.array(values, dtype=np.float64, copy=True, order="C")
    if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 1:
        raise ValueError(
            f"{name} {field} must have shape (n_antenna_rows, 2), got {table.shape}"
        )
    if not bool(np.isfinite(table).all()):
        raise ValueError(f"{name} {field} must be finite")
    table.setflags(write=False)
    return table


class DelayJones(JonesTerm):
    """Instrumental delay ``Kd`` (Section 20.5).

    Constructed only by
    :func:`~radiosim.core.jones_terms.resolve_jones_terms`, from a validated
    ``jones.Kd`` block.

    Parameters
    ----------
    delays_s : ndarray
        ``(n_antenna_rows, 2)`` delay ``tau_pf`` in seconds, in solver
        antenna-row order.
    """

    def __init__(self, *, delays_s: npt.NDArray[np.float64]) -> None:
        self._delays_s = _per_feed_table("DelayJones", "delays_s", delays_s)

    @property
    def name(self) -> str:
        return "Kd"

    @property
    def term_status(self) -> str:
        """``"implemented"``: ``Kd`` carries the exact Section 20.5 mathematics."""
        return "implemented"

    @property
    def is_direction_dependent(self) -> bool:
        """``False``: a cable length is the same for every direction."""
        return False

    @property
    def is_time_dependent(self) -> bool:
        """``False``: RadioSim models the instrumental delay as static."""
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        """``True`` for any non-zero delay: the whole content is a phase slope."""
        return bool(np.any(self._delays_s != 0.0))

    def is_diagonal(self) -> bool:
        """``True`` always: ``Kd`` is ``diag(e^{i a}, e^{i b})`` by construction."""
        return True

    def is_scalar(self) -> bool:
        """``True`` when both feeds of every antenna share one delay.

        Unlike ``X``, this term's scalar case is a real, non-identity
        configuration -- a whole-antenna delay is a scalar phase on that antenna
        -- so the flag has a witness on both sides of invariant I2's sweep.
        """
        return bool(np.array_equal(self._delays_s[:, 0], self._delays_s[:, 1]))

    def is_unitary(self) -> bool:
        """``True`` always: a pure delay is a pure phase, so it preserves power."""
        return True

    def is_identity(self) -> bool:
        """``True`` when every delay is exactly zero.

        Asked of the resolved numbers, so an omitted field, an explicit ``0.0``
        and an override that restores zero are all caught by R7 as the same
        no-op.
        """
        return bool(np.all(self._delays_s == 0.0))

    def phasors_at_frequency(self, frequency_hz: float) -> npt.NDArray[np.complex128]:
        """Return the ``(n_antenna_rows, 2)`` delay phasors at one frequency.

        Public because it is the closed form the invariant tests compare
        against, and because a caller inspecting a resolved delay should not have
        to build a Jones block to read it.
        """
        angles = -2.0 * math.pi * float(frequency_hz) * self._delays_s
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
        """Return this antenna's ``(1, 2, 2)`` delay matrix on the device.

        Direction- and time-independent, so ``directions``, ``time_mjd`` and
        ``time_idx`` are accepted and ignored, and the return is the mandated
        ``(1, 2, 2)`` broadcast form (invariant I3).  The phase is computed on
        the host from Python floats -- no array value is branched on and no
        device value is read back (Section 17.2).
        """
        row = int(antenna_idx)
        if row < 0 or row >= self._delays_s.shape[0]:
            raise IndexError(
                f"DelayJones has delays for {self._delays_s.shape[0]} antenna rows; "
                f"row {row} is out of range."
            )
        block = np.zeros((1, 2, 2), dtype=np.complex128)
        for feed in (0, 1):
            angle = (
                -2.0 * math.pi * float(frequency_hz) * float(self._delays_s[row, feed])
            )
            block[0, feed, feed] = complex(math.cos(angle), math.sin(angle))
        require_finite_jones_block(self.name, block)
        return backend.xp.array(block, dtype=dtype)


class CableReflectionJones(JonesTerm):
    """Cable-reflection ripple ``Rc`` (Section 20.6).

    Constructed only by
    :func:`~radiosim.core.jones_terms.resolve_jones_terms`, from a validated
    ``jones.Rc`` block.

    Parameters
    ----------
    amplitudes : ndarray
        ``(n_antenna_rows, 2)`` dimensionless reflection amplitude ``A_pf``, with
        ``|A| < 1``.  Enforced here as well as at resolution (R8), because the
        constructor is reachable from library code that never sees a document.
    cable_delays_s : ndarray
        ``(n_antenna_rows, 2)`` round-trip cable delay ``tau_cable,pf`` in
        seconds.
    phases_rad : ndarray
        ``(n_antenna_rows, 2)`` phase offset ``phi_pf`` in radians.
    """

    def __init__(
        self,
        *,
        amplitudes: npt.NDArray[np.float64],
        cable_delays_s: npt.NDArray[np.float64],
        phases_rad: npt.NDArray[np.float64],
    ) -> None:
        name = "CableReflectionJones"
        self._amplitudes = _per_feed_table(name, "amplitudes", amplitudes)
        self._cable_delays_s = _per_feed_table(name, "cable_delays_s", cable_delays_s)
        self._phases_rad = _per_feed_table(name, "phases_rad", phases_rad)
        if (
            self._cable_delays_s.shape != self._amplitudes.shape
            or self._phases_rad.shape != self._amplitudes.shape
        ):
            raise ValueError(
                "CableReflectionJones amplitudes, cable_delays_s and phases_rad "
                "must all have the same shape"
            )
        if not bool(np.all(np.abs(self._amplitudes) < 1.0)):
            raise ValueError(
                "CableReflectionJones amplitudes must satisfy |A| < 1; a reflection "
                "cannot return more power than it receives"
            )

    @property
    def name(self) -> str:
        return "Rc"

    @property
    def term_status(self) -> str:
        """``"implemented"``: ``Rc`` carries the exact Section 20.6 mathematics."""
        return "implemented"

    @property
    def is_direction_dependent(self) -> bool:
        """``False``: a reflection in a cable has no direction."""
        return False

    @property
    def is_time_dependent(self) -> bool:
        """``False``: RadioSim models the reflection as a static cable property."""
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        """``True`` when some feed has both a reflection and a cable delay.

        A reflection with zero cable delay is a constant complex offset rather
        than a ripple.  Such a configuration is legal -- it is not the identity,
        so R7 does not reject it -- and it really is frequency-flat, so claiming
        chromaticity for it would be a false ``True``.
        """
        return bool(np.any((self._amplitudes != 0.0) & (self._cable_delays_s != 0.0)))

    def is_diagonal(self) -> bool:
        """``True`` always: ``Rc`` is ``diag(r_0, r_1)`` by construction."""
        return True

    def is_scalar(self) -> bool:
        """``True`` when both feeds of every antenna share one cable.

        Asked of the three parameter tables together, because two cables agree
        only if their amplitude, delay and phase all do.
        """
        return bool(
            np.array_equal(self._amplitudes[:, 0], self._amplitudes[:, 1])
            and np.array_equal(self._cable_delays_s[:, 0], self._cable_delays_s[:, 1])
            and np.array_equal(self._phases_rad[:, 0], self._phases_rad[:, 1])
        )

    def is_unitary(self) -> bool:
        """``True`` only for a zero reflection.

        Section 20.6: ``Rc`` is non-unitary.  ``|r|`` swings between ``1 - A``
        and ``1 + A``, so a real reflection both attenuates and amplifies across
        the band and cannot preserve power at every channel.  R8 makes the
        ``True`` case unreachable from a document.
        """
        return bool(np.all(self._amplitudes == 0.0))

    def is_identity(self) -> bool:
        """``True`` only for a zero reflection amplitude.

        R8 rejects ``A = 0`` outright with a message that names the physics, so
        this never fires from configuration; it is computed from the resolved
        numbers anyway, because a flag that cannot be wrong is a flag nothing
        checks.
        """
        return bool(np.all(self._amplitudes == 0.0))

    def responses_at_frequency(self, frequency_hz: float) -> npt.NDArray[np.complex128]:
        """Return the ``(n_antenna_rows, 2)`` complex responses at one frequency.

        Public because it is the closed form the invariant tests compare
        against.
        """
        angles = (
            -2.0 * math.pi * float(frequency_hz) * self._cable_delays_s
            + self._phases_rad
        )
        return np.asarray(
            1.0 + self._amplitudes * np.exp(1j * angles), dtype=np.complex128
        )

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
        """Return this antenna's ``(1, 2, 2)`` reflection matrix on the device.

        Direction- and time-independent, so ``directions``, ``time_mjd`` and
        ``time_idx`` are accepted and ignored, and the return is the mandated
        ``(1, 2, 2)`` broadcast form (invariant I3).
        """
        row = int(antenna_idx)
        if row < 0 or row >= self._amplitudes.shape[0]:
            raise IndexError(
                f"CableReflectionJones has cables for {self._amplitudes.shape[0]} "
                f"antenna rows; row {row} is out of range."
            )
        block = np.zeros((1, 2, 2), dtype=np.complex128)
        for feed in (0, 1):
            angle = -2.0 * math.pi * float(frequency_hz) * float(
                self._cable_delays_s[row, feed]
            ) + float(self._phases_rad[row, feed])
            block[0, feed, feed] = 1.0 + float(self._amplitudes[row, feed]) * complex(
                math.cos(angle), math.sin(angle)
            )
        require_finite_jones_block(self.name, block)
        return backend.xp.array(block, dtype=dtype)
