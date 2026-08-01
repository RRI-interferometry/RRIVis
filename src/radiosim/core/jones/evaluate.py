"""The one place a Jones chain is evaluated for a solver.

``Tier7JonesSciencePlan.md`` Section 14 (defect D4): before Tier 7B the point
solver composed a :class:`~radiosim.core.jones.chain.JonesChain` while the
HEALPix solver built its own constant ``H_p @ C_p`` product and left-multiplied
it onto the beam, touching no chain at all.  A term added to the chain therefore
applied to point sources and silently did **not** apply to diffuse sky -- the
failure mode that makes a forward model quietly wrong rather than loudly broken.

:func:`evaluate_antenna_jones` closes that permanently.  Both solvers call this
function and nothing else to obtain per-antenna Jones matrices, so there is
exactly one composition site and no term can reach one sky representation
without reaching the other.

The chain evaluation stays host-orchestrated and is deliberately *not* pushed
into the compiled kernel: ``Tier6HybridRuntimePlan.md`` Section 13.6 authorizes
exactly one compiled kernel (``core/contraction.py``), and the chain calls into
astropy-derived quantities and, for ``E``, pyuvdata interpolation.  Tier 7 does
not widen that boundary, add a second ``backend.compile`` call site, or change
the kernel's signature.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from radiosim.core.jones.directions import DirectionBatch

if TYPE_CHECKING:
    from radiosim.core.jones.chain import JonesChain

__all__ = ["evaluate_antenna_jones"]


def evaluate_antenna_jones(
    *,
    chain: JonesChain,
    antenna_rows: Sequence[int],
    directions: DirectionBatch,
    frequency_hz: float,
    freq_idx: int,
    time_mjd: float,
    time_idx: int,
    backend: Any,
    dtype: Any,
) -> dict[int, Any]:
    """Evaluate one chain for several antennas over one direction batch.

    Parameters
    ----------
    chain : JonesChain
        The composed per-antenna chain, correlator-side term first.
    antenna_rows : sequence of int
        Solver instrument view rows to evaluate, in the caller's order.  Rows,
        not antenna numbers: every chain term indexes the instrument by row, and
        keying the result by that same index is what makes a row/number mix-up
        structurally impossible instead of something a runtime cross-check has
        to catch.  A repeated row is evaluated once.
    directions : DirectionBatch
        The directions for this ``(time, frequency)`` step.
    frequency_hz, time_mjd : float
        Physical frequency and time, passed as values and not only as indices
        (Section 13.2), so no term has to be pre-loaded with the observation
        grids at construction just to know what it is being evaluated at.
    freq_idx, time_idx : int
        The corresponding grid indices.
    backend : ArrayBackend
        The array backend every term computes through.  It must be the chain's
        own backend: two backends in one chain would silently mix array domains.
    dtype : dtype
        The resolved complex dtype for this chain.  The solver resolves it once
        from ``PrecisionConfig`` and hands it down, so no term chooses its own
        (Section 17.1, defects D8 and D9).

    Returns
    -------
    dict of int to array
        ``{antenna_row: J}``, where ``J`` is ``(n_dir, 2, 2)`` for a chain that
        carries at least one direction-dependent term and ``(1, 2, 2)`` for a
        chain that is entirely direction-independent, in the backend's own array
        domain and in ``dtype``.
    """
    if type(directions) is not DirectionBatch:
        raise TypeError("directions must be an exact DirectionBatch")
    if backend is not chain.backend:
        raise ValueError(
            "evaluate_antenna_jones must be called with the chain's own backend"
        )

    evaluated: dict[int, Any] = {}
    for antenna_row in antenna_rows:
        if antenna_row in evaluated:
            continue
        evaluated[antenna_row] = chain.compute_antenna_jones_batch(
            antenna_idx=antenna_row,
            directions=directions,
            frequency_hz=frequency_hz,
            freq_idx=freq_idx,
            time_mjd=time_mjd,
            time_idx=time_idx,
            dtype=dtype,
        )
    return evaluated
