"""The polarization leakage term (D).

``D_p`` carries the first-order cross-coupling between an antenna's two feed
chains::

    D_p(nu) = [[ 1,            d_p0(nu) ],
               [ -d_p1(nu)^*,  1        ]]

``d_p0`` is the leakage of feed 1's signal into feed 0's chain and ``d_p1`` the
converse.  Both are dimensionless and complex; ``|d| ~ 0.01-0.05`` is typical of
a well built receiver.  The conjugate-and-negate on the lower left is the
Hamaker, Bregman & Sault convention, and it is what makes ``D`` reduce to a
(scaled) rotation for real, equal leakages -- the property
``tests/unit/test_jones/test_leakage.py`` pins, because a sign or conjugate
error there is invisible in every other check.

``D`` sits **correlator-side of ``C``** (``Tier7JonesSciencePlan.md``
Section 12.2), so it is defined per feed *index* 0/1 in the antenna's own
receptor basis and not per ``x``/``y``.  That placement is a physical claim, not
a convention: leakage is a property of the receiving hardware, which lives in
the receptor's own frame.

Tier 7E implements it with one ``d_terms`` field whose three kinds subsume what
used to be three classes:

``explicit``
    complex ``d0`` and ``d1`` per antenna;
``ixr``
    an intrinsic cross-polarization ratio in dB, converted by
    ``|d| = 1 / sqrt(IXR_lin)`` with ``IXR_lin = 10^(IXR_dB/10)``;
``frequency_polynomial``
    ``d(nu)`` as a complex polynomial in normalized frequency, which is what the
    deleted frequency-dependent leakage class was; ``docs/migration_guide.md``
    names it.

There is no separate Mueller class -- a Mueller matrix is a derived 4x4 view of
this same 2x2 Jones -- and no separate frequency-dependent class, because ``D``
is frequency-capable by construction.  Beam squint is a *beam* property and is
routed to the beam subsystem rather than modelled as a direction-dependent
D-term, which would create the second beam pathway Section 4 forbids.

The IXR conversion, and why it is written this way
--------------------------------------------------
Carozzi & Woan (2011) define ``IXR_J = ((kappa + 1)/(kappa - 1))^2`` for the
condition number ``kappa`` of the Jones matrix.  The convention above,
``[[1, d], [-d^*, 1]]``, has equal singular values --
``D^dagger D = (1 + |d|^2) I2`` -- so by itself it has unit condition number.
The ``1 +- |d|`` singular-value pair the derivation uses belongs instead to
the Hermitian matrix ``[[1, d], [d^*, 1]]``, whose condition number is
``kappa = (1 + |d|)/(1 - |d|)``; writing
``s = sqrt(IXR_lin)`` gives ``kappa = (s + 1)/(s - 1)`` and therefore
``|d| = (kappa - 1)/(kappa + 1) = 1/s``.  A larger IXR is a *smaller* leakage,
``|d| -> 0`` as ``IXR_dB -> infinity``, and ``|d| = 1`` at ``IXR_dB = 0``.

The beam subsystem's future-work note and, until this slice,
``Tier7JonesSciencePlan.md`` Section 20.3 both carried the inverted form
``(sqrt(IXR_lin) - 1)/(sqrt(IXR_lin) + 1)``, which is what
``(kappa - 1)/(kappa + 1)`` becomes if ``kappa`` and ``s`` are interchanged.
The plan was corrected in this slice; the beam note belongs to Tier 7I, which
owns that file and rewrites it.

What ``D`` does to a visibility
-------------------------------
``V_pq -> D_p V_pq D_q^H`` in native-feed coordinates before ``H``.  In the
matched scalar-beam oracle the homogeneous default-linear receptors use
``C=P`` on both antennas and matching ``linear_xy`` output makes ``H=I2``.  An
unpolarized source then gives a leakage-free cell ``c I2``, and the reported
corrupted cross hand is exactly

    V_01 = c (d_p0 - d_q1)

where ``c=I/2`` in the normalized unit-response case.  ``D_q^H`` contributes
``-d_q1`` at ``[0, 1]``, not ``+d_q1^*``.  Section 20.3 gives this as a
first-order prediction; under this matched setup it is exact, and the tests read
``c`` from the leakage-free run and assert the relation at machine precision.

References
----------
Hamaker, Bregman & Sault (1996), A&AS 117, 137, Section 4 -- the ``D``
factorization and the first-order form used here.
Sault, Hamaker & Bregman (1996), A&AS 117, 149.
Smirnov (2011), A&A 527, A106 (Paper I), Section 6.4.
Carozzi & Woan (2011), IEEE Trans. Antennas Propag. 59, 2058 -- the IXR measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones_errors import require_finite_jones_block

if TYPE_CHECKING:
    from radiosim.core.jones.directions import DirectionBatch

__all__ = ["LeakageCoefficient", "PolarizationLeakageJones"]


@dataclass(frozen=True, slots=True)
class LeakageCoefficient:
    """One feed's resolved leakage ``d(nu)``, as a polynomial in normalized ``nu``.

    All three configured kinds resolve to this one shape.  ``explicit`` and
    ``ixr`` produce a single coefficient -- a frequency-flat ``d`` -- and
    ``frequency_polynomial`` produces as many as were written.  Collapsing the
    three to one evaluation path is what lets the term, the flags, the
    provenance snapshot and the invariant sweep treat them identically instead
    of branching on a kind at every call.

    Parameters
    ----------
    coefficients
        Complex coefficients ``c_k``, lowest order first, of
        ``d(nu) = sum_k c_k x^k`` with ``x = (nu - nu_ref) / nu_scale``.
    reference_frequency_hz, scale_frequency_hz
        The resolved ``nu_ref`` and ``nu_scale``.  Both are resolved values, not
        optional inputs: the band-centre and half-bandwidth defaults are applied
        once, at resolution, so this class has no notion of "unspecified".  They
        are irrelevant for a single coefficient and default to a normalization
        that is exactly the identity on ``nu``.
    """

    coefficients: tuple[complex, ...]
    reference_frequency_hz: float = 0.0
    scale_frequency_hz: float = 1.0

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("LeakageCoefficient needs at least one coefficient")
        if not all(np.isfinite(complex(value)) for value in self.coefficients):
            raise ValueError("LeakageCoefficient coefficients must be finite")
        if not self.scale_frequency_hz > 0.0:
            raise ValueError("LeakageCoefficient scale_frequency_hz must be positive")

    @property
    def is_zero(self) -> bool:
        """``True`` when ``d(nu) = 0`` at every frequency."""
        return all(value == 0.0 for value in self.coefficients)

    @property
    def is_frequency_dependent(self) -> bool:
        """``True`` when any coefficient beyond the constant term is non-zero."""
        return any(value != 0.0 for value in self.coefficients[1:])

    def evaluate(self, frequencies_hz: Any) -> npt.NDArray[np.complex128]:
        """Return ``d(nu)`` for an array of frequencies, in ``complex128``."""
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

    def at(self, frequency_hz: float) -> complex:
        """Return ``d(nu)`` at one frequency, as a Python ``complex``."""
        if len(self.coefficients) == 1:
            return self.coefficients[0]
        return complex(self.evaluate(np.asarray([float(frequency_hz)]))[0])

    def snapshot(self) -> dict[str, Any]:
        """Return a fresh JSON-safe record of this coefficient, for provenance."""
        return {
            "coefficients": [
                [float(value.real), float(value.imag)] for value in self.coefficients
            ],
            "reference_frequency_hz": float(self.reference_frequency_hz),
            "scale_frequency_hz": float(self.scale_frequency_hz),
        }


def leakage_from_ixr_db(ixr_db: float, phase_rad: float = 0.0) -> complex:
    """Return ``d = |d| exp(i phi)`` for an intrinsic cross-polarization ratio.

    ``|d| = 1 / sqrt(10^(IXR_dB/10))``, equivalently ``IXR_dB = -20 log10 |d|``;
    see the module docstring for the derivation from Carozzi & Woan (2011).  A
    module-level function rather than a method, because it is the one conversion
    and both the resolution step and the documentation quote it.
    """
    magnitude = 10.0 ** (-float(ixr_db) / 20.0)
    return complex(
        magnitude * np.cos(float(phase_rad)), magnitude * np.sin(float(phase_rad))
    )


class PolarizationLeakageJones(JonesTerm):
    """Polarization leakage D-terms ``D`` (Section 20.3).

    Constructed only by
    :func:`~radiosim.core.jones_terms.resolve_jones_terms`, from a validated
    ``jones.D`` block: the term never parses configuration and never chooses a
    default, so there is exactly one place a leakage value can come from
    (Section 22 rule 2).

    Parameters
    ----------
    d_terms : tuple of tuple of LeakageCoefficient
        One ``(feed 0, feed 1)`` pair per solver antenna row, in row order.
    """

    def __init__(
        self,
        *,
        d_terms: tuple[tuple[LeakageCoefficient, LeakageCoefficient], ...],
    ) -> None:
        if not d_terms:
            raise ValueError("PolarizationLeakageJones needs at least one antenna row")
        for pair in d_terms:
            if len(pair) != 2 or any(
                type(item) is not LeakageCoefficient for item in pair
            ):
                raise TypeError(
                    "PolarizationLeakageJones d_terms must be one (feed 0, feed 1) "
                    "pair of LeakageCoefficient per antenna row"
                )
        self._d_terms = tuple((pair[0], pair[1]) for pair in d_terms)
        self._all_coefficients = tuple(
            coefficient for pair in self._d_terms for coefficient in pair
        )

    @property
    def name(self) -> str:
        return "D"

    @property
    def term_status(self) -> str:
        """``"implemented"``: ``D`` carries the exact Section 20.3 mathematics."""
        return "implemented"

    @property
    def is_direction_dependent(self) -> bool:
        """``False``: receiver cross-coupling is the same for every direction.

        A leakage that varied across the beam is *beam squint*, which belongs to
        the beam subsystem (Section 9.1); modelling it here would create a second
        beam pathway.
        """
        return False

    @property
    def is_time_dependent(self) -> bool:
        """``False``: RadioSim models leakage as a static receiver property."""
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        """``True`` only when some resolved ``d(nu)`` actually varies.

        Asked of the resolved coefficients rather than of the configured kind, so
        a ``frequency_polynomial`` written with a single constant coefficient is
        correctly reported as flat.
        """
        return any(
            coefficient.is_frequency_dependent for coefficient in self._all_coefficients
        )

    def _is_zero(self) -> bool:
        return all(coefficient.is_zero for coefficient in self._all_coefficients)

    def is_diagonal(self) -> bool:
        """``True`` only for a zero leakage, which is exactly ``I2``.

        Computed from the resolved numbers rather than hard-coded ``False``: a
        hard-coded flag is the vacuous claim invariant I2 exists to prevent, and
        the zero case really is diagonal.  R7 makes it unreachable from a
        document, so in practice this is ``False``.
        """
        return self._is_zero()

    def is_scalar(self) -> bool:
        """``True`` only for a zero leakage; see :meth:`is_diagonal`."""
        return self._is_zero()

    def is_unitary(self) -> bool:
        """``True`` only for a zero leakage.

        Section 20.3: ``D`` is non-unitary for any non-zero ``d``.
        ``D D^H = [[1 + |d_0|^2, ...], [..., 1 + |d_1|^2]]``, which is ``I2``
        only when both vanish -- a leaking receptor moves power between the two
        chains, and a matrix that did that while preserving ``J J^H = I`` would
        be a rotation rather than a leakage.
        """
        return self._is_zero()

    def is_identity(self) -> bool:
        """``True`` when this term is exactly ``I2`` at every frequency.

        The rejection R7 asks this of the *resolved numbers* and never of the
        configuration text, so that an explicit zero, an all-zero polynomial and
        an omitted field are all caught as the same no-op.
        """
        return self._is_zero()

    def d_terms_at_frequency(self, frequency_hz: float) -> npt.NDArray[np.complex128]:
        """Return the ``(n_antenna_rows, 2)`` complex leakages at one frequency.

        Public because it is the closed form the invariant tests compare against,
        and because a caller inspecting a resolved leakage should not have to
        build a Jones block to read it.
        """
        values = np.empty((len(self._d_terms), 2), dtype=np.complex128)
        for row, pair in enumerate(self._d_terms):
            for feed, coefficient in enumerate(pair):
                values[row, feed] = coefficient.at(frequency_hz)
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
        """Return this antenna's ``(1, 2, 2)`` leakage matrix on the device.

        Direction- and time-independent, so ``directions``, ``time_mjd`` and
        ``time_idx`` are accepted and ignored, and the return is the mandated
        ``(1, 2, 2)`` broadcast form (invariant I3).  The lookup is keyed on
        ``frequency_hz``, the physical value, rather than on ``freq_idx``: a term
        that trusted the index would silently return the wrong channel if it were
        ever evaluated against a different grid.
        """
        row = int(antenna_idx)
        if row < 0 or row >= len(self._d_terms):
            raise IndexError(
                f"PolarizationLeakageJones has leakages for {len(self._d_terms)} "
                f"antenna rows; row {row} is out of range."
            )
        feed_0, feed_1 = self._d_terms[row]
        block = np.zeros((1, 2, 2), dtype=np.complex128)
        block[0, 0, 0] = 1.0
        block[0, 1, 1] = 1.0
        block[0, 0, 1] = feed_0.at(frequency_hz)
        block[0, 1, 0] = -feed_1.at(frequency_hz).conjugate()
        require_finite_jones_block(self.name, block)
        return backend.xp.array(block, dtype=dtype)
