"""The complex electronic gain term (G).

``G_p`` is the per-antenna, per-feed complex voltage gain of the receiving
chain downstream of the feed::

    G_p(t) = g_el(el_ref(t)) * diag( g_p0(t), g_p1(t) )

    g_pf(t) = (1 + a_pf) * exp(i phi_pf) * s_pf(t)

with ``a_pf`` a fractional amplitude error, ``phi_pf`` a phase error in radians,
``f in {0, 1}`` the feed index **in the antenna's own receptor basis** (``G``
sits correlator-side of ``C``, so it is defined per feed index and not per
``x``/``y``), and ``s(t)`` an exactly reproducible time model.

Tier 7D implements it, absorbing what used to be three separate classes: a time
model is a ``time_model`` field and an elevation gain curve is an
``elevation_curve`` field, because both multiply the same diagonal matrix
(``Tier7JonesSciencePlan.md`` Sections 9.1 and 20.1).

The time model
--------------
``s(t)`` is one of three closed forms, none of which draws a random number, so
two runs of the same configuration produce the same gains bit for bit::

    constant      s = 1
    linear_drift  s(t) = 1 + rate * dt          (dt in hours from the first sample)
    sinusoidal    s(t) = 1 + depth * sin(2 pi dt / period + phase)

``t0`` is the **first sample of the resolved time grid**, fixed once at
resolution, so ``s`` does not depend on which time block a worker thread happens
to evaluate.

The elevation gain curve, and what it currently means
-----------------------------------------------------
``g_el(el) = sum_k c_k el^k`` is a polynomial in the elevation of the *pointing
centre* -- a direction-independent quantity, which is why enabling it does not
make ``G`` a DDE.

RadioSim's one phase convention is zenith drift
(:class:`~radiosim.core.phase_center.PhaseCenter`, ``altitude_rad = pi/2``
exactly, ``kind = "zenith_drift"``), so the pointing elevation is **90 degrees
at every time sample** and the curve therefore evaluates to a single constant
for the whole run.  It is a real, non-identity gain -- it scales every
visibility by ``g_el(90)`` -- but it does not yet vary, and it will not until
RadioSim gains a steerable phase centre.  This is said here, in
:meth:`GainJones.is_time_dependent`, and in the user guide, rather than left for
a user to discover from a flat plot: an elevation *curve* that never moves is
the kind of thing that reads as working when it is merely well defined.

References
----------
Hamaker, Bregman & Sault (1996), A&AS 117, 137 -- the ``G`` factorization of the
measurement equation.
Smirnov (2011), A&A 527, A106 (Paper I), Section 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones_errors import require_finite_jones_block

if TYPE_CHECKING:
    from radiosim.core.jones.directions import DirectionBatch

__all__ = ["GainJones", "ResolvedGainTimeModel"]

#: Hours per day, for the MJD-to-hours conversion the time model is written in.
_HOURS_PER_DAY = 24.0


@dataclass(frozen=True, slots=True)
class ResolvedGainTimeModel:
    """One resolved, closed-form gain time model ``s(t)`` (Section 20.1).

    A single frozen record rather than three classes: the three kinds share one
    signature and one evaluation site, and a discriminated ``kind`` keeps the
    provenance snapshot flat.

    Parameters
    ----------
    kind
        ``"constant"``, ``"linear_drift"`` or ``"sinusoidal"``.
    rate_per_hour
        Fractional drift per hour; used by ``linear_drift`` only.
    depth, period_hours, phase_rad
        Sinusoid amplitude, period in hours, and phase at ``t0``; used by
        ``sinusoidal`` only.
    """

    kind: Literal["constant", "linear_drift", "sinusoidal"]
    rate_per_hour: float = 0.0
    depth: float = 0.0
    period_hours: float = 1.0
    phase_rad: float = 0.0

    @property
    def is_constant(self) -> bool:
        """``True`` when ``s(t) == 1`` for every ``t``."""
        if self.kind == "constant":
            return True
        if self.kind == "linear_drift":
            return self.rate_per_hour == 0.0
        return self.depth == 0.0

    def factor(self, hours_since_reference: float) -> float:
        """Return ``s(t)`` for a time offset in hours from ``t0``."""
        if self.kind == "constant":
            return 1.0
        if self.kind == "linear_drift":
            return 1.0 + self.rate_per_hour * hours_since_reference
        return 1.0 + self.depth * math.sin(
            2.0 * math.pi * hours_since_reference / self.period_hours + self.phase_rad
        )


class GainJones(JonesTerm):
    """Complex electronic gains ``G`` (Section 20.1).

    Constructed only by
    :func:`~radiosim.core.jones_terms.resolve_jones_terms`, from a validated
    ``jones.G`` block: the term never parses configuration and never chooses a
    default, so there is exactly one place a gain value can come from
    (Section 22 rule 2).

    Parameters
    ----------
    base_gains : ndarray
        ``(n_antenna_rows, 2)`` complex ``(1 + a) exp(i phi)`` per feed, in
        solver antenna-row order.
    time_model : ResolvedGainTimeModel
        The resolved ``s(t)``.
    reference_time_mjd : float
        ``t0``: the first sample of the resolved time grid.
    elevation_gain : float
        The resolved ``g_el(el_ref)``.  ``1.0`` when no elevation curve is
        configured.  See the module docstring for why this is one number and not
        a function of time under the zenith-drift phase convention.
    """

    def __init__(
        self,
        *,
        base_gains: npt.NDArray[np.complex128],
        time_model: ResolvedGainTimeModel,
        reference_time_mjd: float,
        elevation_gain: float = 1.0,
    ) -> None:
        gains = np.array(base_gains, dtype=np.complex128, copy=True, order="C")
        if gains.ndim != 2 or gains.shape[1] != 2 or gains.shape[0] < 1:
            raise ValueError(
                "GainJones base_gains must have shape (n_antenna_rows, 2), got "
                f"{gains.shape}"
            )
        if not bool(np.isfinite(gains).all()):
            raise ValueError("GainJones base_gains must be finite")
        if type(time_model) is not ResolvedGainTimeModel:
            raise TypeError("GainJones time_model must be a ResolvedGainTimeModel")
        gains.setflags(write=False)
        self._base_gains = gains
        self._time_model = time_model
        self._reference_time_mjd = float(reference_time_mjd)
        self._elevation_gain = float(elevation_gain)
        if not math.isfinite(self._elevation_gain):
            raise ValueError("GainJones elevation_gain must be finite")

    @property
    def name(self) -> str:
        return "G"

    @property
    def term_status(self) -> str:
        """``"implemented"``: ``G`` carries the exact Section 20.1 mathematics."""
        return "implemented"

    @property
    def is_direction_dependent(self) -> bool:
        """``False``.

        Every factor is per antenna, per feed, per time -- including the
        elevation curve, which is a polynomial in the *pointing* elevation and
        therefore one number for the whole sky at a given time.
        """
        return False

    @property
    def is_time_dependent(self) -> bool:
        """``True`` only when the resolved time model actually varies.

        The elevation gain does not enter this answer.  Under the zenith-drift
        phase convention the pointing elevation is constant, so a configured
        elevation curve is a constant factor and claiming time dependence for it
        would be a claim the numbers do not support.
        """
        return not self._time_model.is_constant

    @property
    def is_frequency_dependent(self) -> bool:
        """``False``: structure across the band is ``B``'s job, not ``G``'s."""
        return False

    def is_diagonal(self) -> bool:
        """``True`` always: ``G`` is ``diag(g_0, g_1)`` by construction."""
        return True

    def is_scalar(self) -> bool:
        """``True`` only when both feeds of every antenna carry the same gain.

        The time and elevation factors are common to both feeds, so they cannot
        break scalarity and cannot create it either: the condition is exactly
        equality of the two base gains, antenna by antenna.
        """
        return bool(np.array_equal(self._base_gains[:, 0], self._base_gains[:, 1]))

    def is_unitary(self) -> bool:
        """``True`` only when every resolved gain has unit modulus at every time.

        A pure phase error is unitary; any amplitude error, any non-constant
        time model, and any elevation gain other than exactly ``1`` breaks it.
        ``G`` is therefore **not** unitary in general, and Section 20.1 says so:
        a term that attenuates cannot preserve power.  The condition is written
        conservatively so that invariant I2's numerical sweep can confirm it.
        """
        if self._elevation_gain != 1.0 or not self._time_model.is_constant:
            return False
        return bool(np.all(np.abs(self._base_gains) == 1.0))

    def is_identity(self) -> bool:
        """``True`` when this term is exactly ``I2`` for every antenna and time.

        The rejection R7 asks this of the *resolved numbers* and never of the
        configuration text, so that two differently written blocks that both
        resolve to unity are both caught.  A non-constant time model is enough
        to make the answer ``False`` even when the base gains are unity: such a
        term is unity only at ``t0``.
        """
        if self._elevation_gain != 1.0 or not self._time_model.is_constant:
            return False
        return bool(np.all(self._base_gains == 1.0 + 0.0j))

    def gains_at_time(self, time_mjd: float) -> npt.NDArray[np.complex128]:
        """Return the ``(n_antenna_rows, 2)`` complex gains at one time.

        Public because it is the closed form the invariant tests compare
        against, and because a caller that wants to plot the gain track should
        not have to build a Jones block to get it.
        """
        hours = (float(time_mjd) - self._reference_time_mjd) * _HOURS_PER_DAY
        scale = self._time_model.factor(hours) * self._elevation_gain
        return np.asarray(self._base_gains * scale, dtype=np.complex128)

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
        """Return this antenna's ``(1, 2, 2)`` gain matrix on the backend device.

        Direction- and frequency-independent, so ``directions``,
        ``frequency_hz`` and ``freq_idx`` are accepted and ignored, and the
        return is the mandated ``(1, 2, 2)`` broadcast form (invariant I3).

        The two complex numbers are computed on the host from Python floats --
        no array value is branched on and no device value is read back
        (Section 17.2) -- and the block is finiteness-checked before it crosses
        to the backend, so a ``nan`` is attributed to ``G`` rather than found
        later in the cube (Section 26).
        """
        row = int(antenna_idx)
        if row < 0 or row >= self._base_gains.shape[0]:
            raise IndexError(
                f"GainJones has gains for {self._base_gains.shape[0]} antenna rows; "
                f"row {row} is out of range."
            )
        gain_0, gain_1 = self.gains_at_time(time_mjd)[row]
        block = np.zeros((1, 2, 2), dtype=np.complex128)
        block[0, 0, 0] = gain_0
        block[0, 1, 1] = gain_1
        require_finite_jones_block(self.name, block)
        return backend.xp.array(block, dtype=dtype)
