"""The bandpass term (B).

``B_p(nu)`` is the per-antenna, per-feed complex frequency response of the
signal chain::

    B_p(nu) = diag( b_p0(nu), b_p1(nu) )

It is the frequency-dependent counterpart of ``G``: the same diagonal matrix,
with structure across the band rather than across time, and like ``G`` it sits
correlator-side of ``C`` and is therefore indexed by feed 0/1 in the antenna's
own receptor basis (``Tier7JonesSciencePlan.md`` Sections 20.0 and 20.2).

Tier 7D implements it with two models, which are the two former subclasses:

``polynomial``
    ``b(nu) = sum_k c_k x^k``, ``x = (nu - nu_ref) / nu_scale``, with ``nu_ref``
    defaulting to the band centre and ``nu_scale`` to the half-bandwidth so that
    ``x`` spans ``[-1, 1]`` across the observed band.  Complex coefficients are
    accepted.
``tabulated``
    Complex gains at explicit node frequencies, cubic-spline interpolated in the
    real and imaginary parts separately.  Frequencies outside the node range are
    **rejected** at resolution (R11), never extrapolated: a bandpass continued
    past its own measurement is a fabricated number.

The two kinds replace the two former subclasses one for one (Section 20.2), as
model *variants* rather than as types: a bandpass written as a polynomial and a
bandpass written as a table are the same term evaluated differently, and making
them one class with one ``kind`` is what lets the chain, the provenance record
and the invariant sweep treat them identically.

There is no RFI-flagging variant: flagging is a data-quality product, not a
voltage-domain Jones factor, and RadioSim's result contract has no flag array
for it to write (Section 9.1).

Why the values are precomputed
------------------------------
Every observation channel's response is evaluated once, at construction, into a
``(n_antenna_rows, 2, n_freq)`` table.  Evaluation is then a lookup, which makes
the per-``(time, frequency)`` cost independent of the model kind, keeps the
spline off the hot path, and -- because the table is built once and never
mutated -- makes the term safe to share across the solver's worker threads
without a lock.  A frequency outside the observation grid is still evaluated
correctly, on demand, so a direct unit test is not forced through the grid.

Relation to ``G``
-----------------
A real, frequency-flat bandpass is exactly a ``G`` amplitude error; the two are
different terms because one is *defined* to carry frequency structure and the
other is not.  The equivalence is pinned as a cross-term consistency check
rather than used to collapse the two (Section 20.2).

References
----------
Hamaker, Bregman & Sault (1996), A&AS 117, 137.
Smirnov (2011), A&A 527, A106 (Paper I), Section 6.
CASA ``bandpass`` conventions (van Moorsel et al., CASA Reference Manual,
``B`` Jones).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones_errors import require_finite_jones_block

if TYPE_CHECKING:
    from radiosim.core.jones.directions import DirectionBatch

__all__ = [
    "BandpassJones",
    "BandpassResponse",
    "PolynomialBandpassResponse",
    "TabulatedBandpassResponse",
]


class BandpassResponse:
    """One feed's resolved complex frequency response ``b(nu)``."""

    def evaluate(self, frequencies_hz: Any) -> npt.NDArray[np.complex128]:
        """Return ``b(nu)`` for an array of frequencies, in ``complex128``."""
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        """Return a fresh JSON-safe record of this response, for provenance."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PolynomialBandpassResponse(BandpassResponse):
    """``b(nu) = sum_k c_k x^k`` with ``x = (nu - nu_ref) / nu_scale``.

    Parameters
    ----------
    coefficients
        Complex coefficients, lowest order first.
    reference_frequency_hz, scale_frequency_hz
        The resolved ``nu_ref`` and ``nu_scale``.  Both are resolved values, not
        optional inputs: the band-centre and half-bandwidth defaults are applied
        once, at resolution, so this class has no notion of "unspecified".
    """

    coefficients: tuple[complex, ...]
    reference_frequency_hz: float
    scale_frequency_hz: float

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("PolynomialBandpassResponse needs a coefficient")
        if not self.scale_frequency_hz > 0.0:
            raise ValueError(
                "PolynomialBandpassResponse scale_frequency_hz must be positive"
            )

    def evaluate(self, frequencies_hz: Any) -> npt.NDArray[np.complex128]:
        normalized = (
            np.asarray(frequencies_hz, dtype=np.float64) - self.reference_frequency_hz
        ) / self.scale_frequency_hz
        # Horner from the highest order down: one multiply-add per coefficient,
        # and the same association order every run, so the result is
        # reproducible bit for bit.
        value = np.full(normalized.shape, self.coefficients[-1], dtype=np.complex128)
        for coefficient in reversed(self.coefficients[:-1]):
            value = value * normalized + coefficient
        return value

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "polynomial",
            "coefficients": [
                [float(value.real), float(value.imag)] for value in self.coefficients
            ],
            "reference_frequency_hz": float(self.reference_frequency_hz),
            "scale_frequency_hz": float(self.scale_frequency_hz),
        }


@dataclass(frozen=True, slots=True)
class TabulatedBandpassResponse(BandpassResponse):
    """Complex node gains, cubic-spline interpolated, never extrapolated.

    Parameters
    ----------
    node_frequencies_hz
        Strictly increasing node frequencies.
    gains
        One complex gain per node.
    """

    node_frequencies_hz: tuple[float, ...]
    gains: tuple[complex, ...]
    _splines: tuple[Any, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from scipy.interpolate import CubicSpline

        nodes = np.asarray(self.node_frequencies_hz, dtype=np.float64)
        values = np.asarray(self.gains, dtype=np.complex128)
        if nodes.size < 4 or values.size != nodes.size:
            raise ValueError(
                "TabulatedBandpassResponse needs at least four nodes and exactly "
                "one gain per node"
            )
        if not bool(np.all(np.diff(nodes) > 0.0)):
            raise ValueError(
                "TabulatedBandpassResponse node_frequencies_hz must be strictly "
                "increasing"
            )
        # Real and imaginary parts are splined separately, which is the
        # convention Section 20.2 names: a complex spline through the modulus
        # and phase would wrap, and a spline through the complex value is not
        # what any calibration table means.
        object.__setattr__(
            self,
            "_splines",
            (
                CubicSpline(nodes, values.real, extrapolate=False),
                CubicSpline(nodes, values.imag, extrapolate=False),
            ),
        )

    @property
    def frequency_span_hz(self) -> tuple[float, float]:
        """Return the closed node range this response is defined over."""
        return (float(self.node_frequencies_hz[0]), float(self.node_frequencies_hz[-1]))

    def evaluate(self, frequencies_hz: Any) -> npt.NDArray[np.complex128]:
        requested = np.asarray(frequencies_hz, dtype=np.float64)
        real_spline, imaginary_spline = self._splines
        return np.asarray(
            real_spline(requested) + 1j * imaginary_spline(requested),
            dtype=np.complex128,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "tabulated",
            "node_frequencies_hz": [float(node) for node in self.node_frequencies_hz],
            "gains": [[float(value.real), float(value.imag)] for value in self.gains],
        }


class BandpassJones(JonesTerm):
    """Frequency-dependent bandpass ``B`` (Section 20.2).

    Constructed only by
    :func:`~radiosim.core.jones_terms.resolve_jones_terms`, from a validated
    ``jones.B`` block.

    Parameters
    ----------
    responses : tuple of tuple of BandpassResponse
        One ``(feed 0, feed 1)`` pair per solver antenna row, in row order.
    frequencies_hz : ndarray
        The observation channel centres, used to precompute the response table.
    """

    def __init__(
        self,
        *,
        responses: tuple[tuple[BandpassResponse, BandpassResponse], ...],
        frequencies_hz: Any,
    ) -> None:
        if not responses:
            raise ValueError("BandpassJones needs at least one antenna row")
        for pair in responses:
            if len(pair) != 2 or any(
                not isinstance(item, BandpassResponse) for item in pair
            ):
                raise TypeError(
                    "BandpassJones responses must be one (feed 0, feed 1) pair of "
                    "BandpassResponse per antenna row"
                )
        frequencies = np.array(frequencies_hz, dtype=np.float64, copy=True)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError(
                "BandpassJones frequencies_hz must be a nonempty one-dimensional array"
            )
        self._responses = tuple(tuple(pair) for pair in responses)
        self._frequencies_hz = frequencies
        self._frequencies_hz.setflags(write=False)
        table = np.empty(
            (len(self._responses), 2, frequencies.size), dtype=np.complex128
        )
        for row, pair in enumerate(self._responses):
            for feed, response in enumerate(pair):
                table[row, feed, :] = response.evaluate(frequencies)
        if not bool(np.isfinite(table).all()):
            raise ValueError(
                "BandpassJones resolved a non-finite response on the observation "
                "grid; a tabulated model must cover every observed channel"
            )
        table.setflags(write=False)
        self._table = table
        self._index_by_frequency = {
            float(value): index for index, value in enumerate(frequencies)
        }

    @property
    def name(self) -> str:
        return "B"

    @property
    def term_status(self) -> str:
        """``"implemented"``: ``B`` carries the exact Section 20.2 mathematics."""
        return "implemented"

    @property
    def is_direction_dependent(self) -> bool:
        """``False``: a signal-chain response is the same for every direction."""
        return False

    @property
    def is_time_dependent(self) -> bool:
        """``False``: structure across time is ``G``'s job, not ``B``'s."""
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        """``True``: this is the term whose whole content is frequency structure."""
        return True

    def is_diagonal(self) -> bool:
        """``True`` always: ``B`` is ``diag(b_0, b_1)`` by construction."""
        return True

    def is_scalar(self) -> bool:
        """``True`` when both feeds match within every antenna and channel.

        Checked on the resolved table over the observation grid, not on the
        configuration, so two differently written models that happen to produce
        the same numbers are correctly reported as scalar.  Different antennas
        may carry different scalar responses; scalarity is a property of each
        antenna's 2x2 Jones matrix, not equality across the array.
        """
        return bool(np.array_equal(self._table[:, 0, :], self._table[:, 1, :]))

    def is_unitary(self) -> bool:
        """``True`` only when every resolved response has unit modulus.

        A bandpass is an attenuation profile, so this is ``False`` for every
        realistic configuration; it can be ``True`` for a pure phase bandpass,
        which is why the check is on the numbers and not on the model kind.
        """
        return bool(np.all(np.abs(self._table) == 1.0))

    def is_identity(self) -> bool:
        """``True`` when this term is exactly ``I2`` at every observed channel.

        Asked of the resolved table rather than of the configuration, because a
        constant polynomial of value 1 and a tabulated model of all-unity nodes
        are the same no-op written two ways, and R7 must reject both.  The
        observation grid is the right domain: it is the only place the run ever
        evaluates the bandpass.
        """
        return bool(np.all(self._table == 1.0 + 0.0j))

    def responses_at_frequency(self, frequency_hz: float) -> npt.NDArray[np.complex128]:
        """Return the ``(n_antenna_rows, 2)`` complex responses at one frequency.

        A frequency on the observation grid is a table lookup; any other
        frequency is evaluated from the resolved models on demand, so this is
        also the closed form the invariant tests compare against.
        """
        index = self._index_by_frequency.get(float(frequency_hz))
        if index is not None:
            return np.asarray(self._table[:, :, index], dtype=np.complex128)
        requested = np.asarray([float(frequency_hz)], dtype=np.float64)
        values = np.empty((len(self._responses), 2), dtype=np.complex128)
        for row, pair in enumerate(self._responses):
            for feed, response in enumerate(pair):
                values[row, feed] = response.evaluate(requested)[0]
        return values

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
        """Return this antenna's ``(1, 2, 2)`` bandpass matrix on the device.

        Direction- and time-independent, so ``directions``, ``time_mjd`` and
        ``time_idx`` are accepted and ignored, and the return is the mandated
        ``(1, 2, 2)`` broadcast form (invariant I3).  The lookup is keyed on
        ``frequency_hz``, the physical value, rather than on ``freq_idx``: a term
        that trusted the index would silently return the wrong channel if it were
        ever evaluated against a different grid.
        """
        row = int(antenna_idx)
        if row < 0 or row >= len(self._responses):
            raise IndexError(
                f"BandpassJones has responses for {len(self._responses)} antenna "
                f"rows; row {row} is out of range."
            )
        response_0, response_1 = self.responses_at_frequency(frequency_hz)[row]
        block = np.zeros((1, 2, 2), dtype=np.complex128)
        block[0, 0, 0] = response_0
        block[0, 1, 1] = response_1
        require_finite_jones_block(self.name, block)
        return backend.xp.array(block, dtype=dtype)
