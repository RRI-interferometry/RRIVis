"""Baseline-dependent Jones terms (closure errors, smearing).

Some RIME terms are baseline-dependent (not per-antenna) and cannot be
represented as per-antenna Jones matrices. These are:

- ``M_pq``: per-baseline multiplicative closure errors
- ``Q_spq``: time and bandwidth smearing decorrelation factors

They apply to visibilities by Hadamard (element-wise) multiplication rather than
through the matrix chain, which is why they descend from
:class:`JonesBaselineTerm` and not from
:class:`~radiosim.core.jones.base.JonesTerm`, and why
:meth:`~radiosim.core.jones.chain.JonesChain.add_term` rejects them by type.

The two attachment points
-------------------------
``Tier7JonesSciencePlan.md`` Section 15.1.
Both already exist in the compiled kernel's signature, which is why this is a
small slice rather than a kernel redesign (invariant I16):

===========  ==============  ===================================================
Term         Factor shape    Where it attaches
===========  ==============  ===================================================
``Q``        ``(B, n_dir)``  multiplied into the kernel's ``envelope`` argument,
                             beside the Gaussian morphology envelope
``M``        ``(B, 2, 2)``   multiplied elementwise into the kernel's
                             ``(B, 2, 2)`` return, before the output cast
===========  ==============  ===================================================

Which of the two a term uses is *declared*, by :attr:`hadamard_target`, rather
than inferred: a solver has to dispatch on something, and inferring it from
``is_direction_dependent`` would turn a statement about physics into the wiring.
:func:`evaluate_baseline_factors` is the one place either factor is computed, so
that a baseline term cannot reach the point solver and silently not the diffuse
one -- the same reason Section 14 gives for ``evaluate_antenna_jones``.

References
----------
Bridle & Schwab (1999), in *Synthesis Imaging in Radio Astronomy II*, ASP Conf.
Ser. 180, 371 -- the time and bandwidth smearing expressions ``Q`` implements.
Thompson, Moran & Swenson (2017), 3rd ed., Sections 6.4 and 10.3-10.4 -- the
smearing envelopes and the closure relations, which ``M`` breaks and every
per-antenna term preserves (invariant I11).
Smirnov (2011), A&A 527, A106, Sections 1.6 and 7 -- baseline-dependent errors
as the residual that closure quantities cannot absorb.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from radiosim.core.jones_errors import JonesEvaluationError

if TYPE_CHECKING:
    from radiosim.core.jones.directions import DirectionBatch

__all__ = [
    "BASELINE_FACTOR_TARGETS",
    "BaselineFactors",
    "BaselineMultiplicativeJones",
    "EARTH_ROTATION_RAD_PER_S",
    "JonesBaselineTerm",
    "SmearingFactorJones",
    "evaluate_baseline_factors",
]

#: Earth's sidereal rotation rate, rad/s (IERS Conventions 2010, Table 1.1;
#: ``omega_E`` in ``Tier7JonesSciencePlan.md`` Section 20.11).  The *sidereal*
#: rate and not ``2 pi / 86400``: the sky, not the Sun, is what rotates through
#: a fringe.
EARTH_ROTATION_RAD_PER_S: float = 7.2921150e-5

#: The two points a baseline factor may attach to, and the only two values
#: :attr:`JonesBaselineTerm.hadamard_target` may take.
BASELINE_FACTOR_TARGETS: tuple[str, ...] = ("envelope", "correlation")


class JonesBaselineTerm(ABC):
    """Abstract base for per-BASELINE (not per-antenna) RIME terms.

    These terms cannot be added to JonesChain (which expects per-antenna terms,
    and rejects these by type). They apply to visibilities directly via
    element-wise multiplication:

        V_pq_corrected = M_pq ⊙ V_pq_original

    where ⊙ denotes Hadamard (element-wise) product.

    This is a separate abstraction from JonesTerm because baseline-dependent
    effects fundamentally differ from antenna effects.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name identifier (e.g., 'M', 'Q')."""

    @property
    @abstractmethod
    def is_direction_dependent(self) -> bool:
        """True if effect varies across the sky (DDE)."""

    @property
    @abstractmethod
    def hadamard_target(self) -> Literal["envelope", "correlation"]:
        """Which of the two attachment points this term's factor belongs to.

        ``"envelope"`` for a real ``(B, n_dir)`` scalar attenuation, which the
        solver multiplies into the compiled kernel's ``envelope`` argument;
        ``"correlation"`` for a complex ``(B, 2, 2)`` factor, which it multiplies
        elementwise into the kernel's output.

        Abstract, so a new baseline term cannot omit it and be quietly applied
        in the wrong place.
        """

    @property
    def term_status(self) -> str:
        """``"implemented"`` if this term's physics exists, else ``"planned"``.

        The baseline-path counterpart of
        :attr:`~radiosim.core.jones.base.JonesTerm.term_status`, with the same
        contract and the same honest default; see that docstring for why the
        default is ``"planned"``.  Both ``M`` and ``Q`` override it since
        Tier 7H, and no term in this package declares ``"planned"`` any more.
        """
        return "planned"

    @abstractmethod
    def compute_baseline_factor(
        self,
        *,
        baseline_pairs: Sequence[tuple[int, int]],
        baseline_uvw_wavelengths: Any,
        directions: DirectionBatch,
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        """Return this term's Hadamard factor for every selected baseline.

        The baseline-path counterpart of
        :meth:`~radiosim.core.jones.base.JonesTerm.compute_jones_batch`, batched
        over both baselines and directions for the same reason
        (``Tier7JonesSciencePlan.md`` Sections 13.2 and 15.1): ``Q``'s factor is
        direction-dependent, and one Python call per direction cannot carry a
        HEALPix pixel batch.

        Parameters
        ----------
        baseline_pairs : sequence of (int, int)
            The selected baselines as ordered antenna **numbers**, in
            selected-baseline order -- the order of the returned leading axis.
            Numbers rather than rows, because ``M`` is configured by antenna
            number and matching the configuration to the wrong pair is exactly
            the defect this argument exists to prevent.
        baseline_uvw_wavelengths : array
            ``(B, 3)`` baseline vectors in wavelengths at ``frequency_hz``,
            already in the backend's array domain -- the same array the
            geometric phase is computed from.
        directions : DirectionBatch
            Every direction for this ``(time, frequency)`` step.
        frequency_hz, time_mjd : float
            Physical frequency and time.
        freq_idx, time_idx : int
            The corresponding grid indices.  A term that reads a per-channel or
            per-sample resolved quantity indexes it with these, and checks the
            physical value against the grid it was resolved with.
        backend : ArrayBackend
            The backend to compute through.
        dtype : dtype
            The dtype the factor must be returned in: the resolved **real**
            dtype for an ``"envelope"`` term and the resolved **complex** dtype
            for a ``"correlation"`` one.  Passed in and never chosen by the term
            (Section 17.1).

        Returns
        -------
        array
            ``(B, n_dir)`` real for an ``"envelope"`` term, or ``(B, 2, 2)``
            complex for a ``"correlation"`` one, to be applied elementwise --
            never composed into the matrix chain.

        Notes
        -----
        ``@abstractmethod`` since Tier 7H.  It was concrete-and-raising while
        ``M`` and ``Q`` were ``term_status: planned``, because an abstract
        declaration would have made them impossible to instantiate; both now
        implement it, so the contract is enforced at construction instead --
        strictly earlier, and strictly harder to get wrong.
        """


@dataclass(frozen=True, slots=True)
class BaselineFactors:
    """The two Hadamard factors for one ``(time, frequency)`` step.

    Each is ``None`` when no configured term declares that target, which is what
    lets a solver keep its pre-Tier-7H arithmetic exactly when the run
    configured no baseline term at all (invariant I1).
    """

    envelope: Any = None
    correlation: Any = None

    @property
    def is_empty(self) -> bool:
        return self.envelope is None and self.correlation is None


def _require_envelope_factor(
    name: str,
    factor: Any,
    *,
    n_baselines: int,
    n_dir: int,
) -> Any:
    """Return ``factor`` after checking it is the kernel's ``envelope`` shape.

    Shape only, deliberately, and not the full ``isfinite().all()`` that
    :func:`~radiosim.core.jones_errors.require_finite_jones_block` runs on a
    Jones block.  The difference is where the array lives: every ``JonesTerm``
    builds its matrices on the host and hands them across once, so that
    reduction is free, while a baseline factor is computed *through the backend*
    over device-resident baseline coordinates, and reducing it here would force
    a device synchronization inside the frequency loop for every step of every
    run.  Both terms validate their inputs at construction instead, where the
    check runs once and on the host, and neither can produce a non-finite value
    from finite inputs (Section 17.2, Section 26).
    """
    shape = getattr(factor, "shape", None)
    if shape != (n_baselines, n_dir):
        raise JonesEvaluationError(
            f"baseline term {name!r} returned an envelope factor of shape "
            f"{shape}; an 'envelope' target must return (n_baselines, n_dir) = "
            f"{(n_baselines, n_dir)}."
        )
    return factor


def _require_correlation_factor(name: str, factor: Any, *, n_baselines: int) -> Any:
    """Return ``factor`` after checking it is the kernel's output shape."""
    shape = getattr(factor, "shape", None)
    if shape != (n_baselines, 2, 2):
        raise JonesEvaluationError(
            f"baseline term {name!r} returned a correlation factor of shape "
            f"{shape}; a 'correlation' target must return (n_baselines, 2, 2) = "
            f"{(n_baselines, 2, 2)}."
        )
    return factor


def evaluate_baseline_factors(
    terms: Sequence[JonesBaselineTerm],
    *,
    baseline_pairs: Sequence[tuple[int, int]],
    baseline_uvw_wavelengths: Any,
    directions: DirectionBatch,
    frequency_hz: float,
    freq_idx: int,
    time_mjd: float,
    time_idx: int,
    backend: Any,
    real_dtype: Any,
    complex_dtype: Any,
) -> BaselineFactors:
    """Evaluate every baseline term for one ``(time, frequency)`` step.

    The one place either Hadamard factor is computed, for the reason Section 14
    gives for :func:`~radiosim.core.jones.evaluate.evaluate_antenna_jones`: two
    solvers with two copies of this would be defect D4 one axis over -- a term
    that corrupts point sources and silently leaves the diffuse sky clean.

    Parameters
    ----------
    terms : sequence of JonesBaselineTerm
        The resolved inventory's ``baseline_terms``, in canonical order.  An
        empty sequence returns an empty :class:`BaselineFactors` without
        touching the backend.
    real_dtype, complex_dtype : dtype
        The resolved dtypes handed to an ``"envelope"`` and a ``"correlation"``
        term respectively.

    Returns
    -------
    BaselineFactors
        The product of all envelope factors and the product of all correlation
        factors, each ``None`` when no term declares that target.  Products are
        formed in term order, so the floating-point result is reproducible.
    """
    envelope: Any = None
    correlation: Any = None
    n_baselines = len(baseline_pairs)
    for term in terms:
        target = term.hadamard_target
        if target not in BASELINE_FACTOR_TARGETS:
            raise JonesEvaluationError(
                f"baseline term {term.name!r} declares hadamard_target "
                f"{target!r}, which is not one of {BASELINE_FACTOR_TARGETS}."
            )
        factor = term.compute_baseline_factor(
            baseline_pairs=tuple(baseline_pairs),
            baseline_uvw_wavelengths=baseline_uvw_wavelengths,
            directions=directions,
            frequency_hz=frequency_hz,
            freq_idx=freq_idx,
            time_mjd=time_mjd,
            time_idx=time_idx,
            backend=backend,
            dtype=real_dtype if target == "envelope" else complex_dtype,
        )
        if target == "envelope":
            factor = _require_envelope_factor(
                term.name,
                factor,
                n_baselines=n_baselines,
                n_dir=directions.n_dir,
            )
            envelope = factor if envelope is None else envelope * factor
        else:
            factor = _require_correlation_factor(
                term.name, factor, n_baselines=n_baselines
            )
            correlation = factor if correlation is None else correlation * factor
    return BaselineFactors(envelope=envelope, correlation=correlation)


def _read_only_array(name: str, values: Any, dtype: Any) -> npt.NDArray[Any]:
    """Return one owned, finite, read-only array of ``dtype``."""
    array = np.array(values, dtype=dtype, copy=True, order="C")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


class BaselineMultiplicativeJones(JonesBaselineTerm):
    """Per-baseline multiplicative closure error ``M_pq`` (Section 20.10).

    ``V_pq -> M_pq (*) V_pq``: one complex ``2x2`` of multiplicative errors per
    baseline, applied by Hadamard product to the contracted correlation matrix.

    ``M`` is the one term in this package that is **not** expressible as any
    product of per-antenna Jones matrices, and that is its defining property
    rather than a limitation.  Every antenna-based term -- a gain, a bandpass, a
    leakage -- cancels identically in the closure phase of a triangle, because
    each antenna appears once conjugated and once not; a per-baseline factor does
    not, so it changes the closure phase by ``arg(M_01) + arg(M_12) - arg(M_02)``
    and no set of antenna gains can reproduce it.  That is invariant I11, and it
    is what ``Fix.md`` Section 16's Workstream D means by "enforce the
    distinction between matrix-chain terms and baseline-dependent Hadamard
    terms".

    Constructed only by
    :func:`~radiosim.core.jones_terms.resolve_jones_terms`, from a validated
    ``jones.M`` block resolved against the run's own baseline selection: a
    ``per_baseline`` entry naming a pair the selection does not contain is
    rejected (R14) rather than ignored.

    Parameters
    ----------
    baseline_pairs : tuple of (int, int)
        The selected baselines as ordered antenna numbers, in selected-baseline
        order.  Retained so that evaluation can refuse a solver whose baselines
        are not the ones this term was resolved against.
    matrices : ndarray
        ``(B, 2, 2)`` complex errors in that same order.  Every entry is a
        multiplicative factor on the correlation of the same name, so the
        neutral matrix is **all ones**: a baseline named by neither an override
        nor an array-wide default carries ``[[1, 1], [1, 1]]``, and a zero
        anywhere nulls that correlation rather than leaving it alone.

    Raises
    ------
    ValueError
        A shape that does not match ``baseline_pairs``, a matrix that is not
        ``2x2``, or a non-finite entry.

    References
    ----------
    Smirnov (2011), A&A 527, A106, Sections 1.6 and 7; Thompson, Moran &
    Swenson (2017), 3rd ed., Section 10.3.
    """

    def __init__(
        self,
        *,
        baseline_pairs: Sequence[tuple[int, int]],
        matrices: Any,
    ) -> None:
        pairs = tuple((int(first), int(second)) for first, second in baseline_pairs)
        if not pairs:
            raise ValueError("baseline_pairs must be nonempty")
        if len(set(pairs)) != len(pairs):
            raise ValueError("baseline_pairs must be unique")
        array = _read_only_array("matrices", matrices, np.complex128)
        if array.shape != (len(pairs), 2, 2):
            raise ValueError(
                f"matrices must have shape {(len(pairs), 2, 2)}, got {array.shape}"
            )
        self._baseline_pairs = pairs
        self._matrices = array

    @property
    def name(self) -> str:
        return "M"

    @property
    def is_direction_dependent(self) -> bool:
        return False

    @property
    def hadamard_target(self) -> Literal["envelope", "correlation"]:
        return "correlation"

    @property
    def term_status(self) -> str:
        return "implemented"

    @property
    def baseline_pairs(self) -> tuple[tuple[int, int], ...]:
        """The ordered antenna-number pairs this term was resolved against."""
        return self._baseline_pairs

    @property
    def matrices(self) -> npt.NDArray[np.complex128]:
        """The resolved ``(B, 2, 2)`` errors, read-only, in selection order."""
        return self._matrices

    def is_identity(self) -> bool:
        """``True`` when every resolved matrix is exactly **ones** (rejection R7).

        Ones, not ``I2``.  The neutral element of a Hadamard product is the
        all-ones matrix; ``I2`` is the neutral element of a *matrix* product, and
        under ``(*)`` it would null both cross-hand correlations rather than
        leave the visibility alone.  Writing the identity here would therefore
        have made the one configuration a user is most likely to copy -- a
        diagonal-looking ``[[1.02, 0], [0, 0.98]]`` -- into a silent
        cross-hand killer that R7 then called "exactly the identity".
        """
        return bool(np.all(self._matrices == 1.0))

    def compute_baseline_factor(
        self,
        *,
        baseline_pairs: Sequence[tuple[int, int]],
        baseline_uvw_wavelengths: Any,
        directions: DirectionBatch,
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        """Return the ``(B, 2, 2)`` closure errors, in the caller's dtype.

        Direction-independent: a correlator's per-baseline error does not depend
        on where the array is looking, so the same matrix multiplies the whole
        contracted block rather than each direction's contribution.  That is
        also why it can be applied *after* the sum over directions at all.
        """
        pairs = tuple((int(first), int(second)) for first, second in baseline_pairs)
        if pairs != self._baseline_pairs:
            raise JonesEvaluationError(
                "Jones term 'M' was resolved against baselines "
                f"{self._baseline_pairs} but was evaluated for {pairs}; a "
                "closure error must not be applied to a baseline it was not "
                "configured for."
            )
        return backend.asarray(self._matrices, dtype=dtype)

    def get_config(self) -> dict[str, Any]:
        """Return this term's record, including the resolved errors."""
        return {
            "name": self.name,
            "term_status": self.term_status,
            "is_direction_dependent": self.is_direction_dependent,
            "hadamard_target": self.hadamard_target,
            "baseline_pairs": [list(pair) for pair in self._baseline_pairs],
            "matrices": [
                [[[float(value.real), float(value.imag)] for value in row] for row in m]
                for m in self._matrices
            ],
        }


class SmearingFactorJones(JonesBaselineTerm):
    """Time and bandwidth smearing decorrelation ``Q_spq`` (Section 20.11).

    Two real envelopes, both ``sinc``, multiplied into the kernel's ``envelope``
    argument::

        Q_pqs = sinc(pi dnu tau_res) * sinc(pi dt nu_f)

    with ``sinc(x) = sin(x)/x``, evaluated as ``numpy.sinc``, whose argument is
    already divided by ``pi`` and whose value at zero is exactly ``1``.

    The residual delay
    ------------------
    ``tau_res`` is the delay the correlator has **not** removed, measured from
    the phase centre rather than from the array's reference antenna::

        tau_res = ( u l + v m + w (n - 1) ) / nu

    which is exactly the argument of the kernel's own phase
    (``core/jones/geometric.py``) divided by ``nu``.  Written as ``b.s/c`` -- the
    form the plan's prose uses -- it would not vanish at the phase centre, and
    the bandwidth envelope would decorrelate a source the correlator is perfectly
    phased to.

    The fringe rate
    ---------------
    RadioSim's baselines are constant ENU vectors and its phase centre is the
    fixed zenith, so the whole time dependence of the phase is the sky rotating
    through both.  With ``p = (0, cos(lat), sin(lat))`` the celestial pole in
    ENU and ``ds/dt = -omega_E (p x s)``::

        dl/dt = -omega_E ( n cos(lat) - m sin(lat) )
        dm/dt = -omega_E l sin(lat)
        dn/dt = +omega_E l cos(lat)

    and therefore, in cycles per second::

        nu_f = omega_E [ u ( n cos(lat) - m sin(lat) ) + l ( v sin(lat) - w cos(lat) ) ]

    The first component is ``-dl/dt`` and equals ``omega_E cos(dec) cos(H)``,
    which is the textbook fringe rate of an East-West baseline; the other two
    are what a non-coplanar array adds.

    Consequences worth stating, because they are asserted rather than assumed
    (invariant I12): both envelopes are ``<= 1``; the bandwidth envelope is
    exactly ``1`` at the phase centre and on a zero-length baseline; the time
    envelope is ``1`` at the phase centre only for a baseline with no East-West
    component, because a drift-scan array's zenith source really does move
    through its own phase centre during an integration; and beyond the first
    sinc zero the exact top-hat average changes sign, which is the real
    behaviour of an average and is not clamped away.

    ``dnu`` and ``dt`` are not parameters of the term
    -------------------------------------------------
    They come from the resolved observation configuration -- the per-channel
    ``ResolvedFrequencyConfig.channel_widths_hz`` and the per-sample
    ``ObservationTimeGrid.integration_time_seconds`` -- because a smearing
    bandwidth that disagreed with the one the same run publishes in its own
    result would be a fabrication (Section 41, Q6).  The term therefore carries
    both grids, indexes them with ``freq_idx``/``time_idx``, and refuses a call
    whose physical frequency or time does not match the grid entry that index
    names.

    Parameters
    ----------
    bandwidth_smearing, time_smearing : bool
        Which envelopes are active.  At least one must be (rejection R16).
    channel_frequencies_hz, channel_widths_hz : ndarray
        ``(F,)`` resolved channel centres and their declared widths.
    integration_time_s, sample_times_mjd : ndarray
        ``(T,)`` resolved integration times and sample centres.
    latitude_rad : float
        The array's geodetic latitude, for the fringe rate.

    Raises
    ------
    ValueError
        Neither envelope enabled, a mismatched or empty grid, a non-positive
        width or integration time, or a non-finite value.

    References
    ----------
    Bridle & Schwab (1999), ASP Conf. Ser. 180, 371; Thompson, Moran & Swenson
    (2017), 3rd ed., Section 6.4.
    """

    def __init__(
        self,
        *,
        bandwidth_smearing: bool,
        time_smearing: bool,
        channel_frequencies_hz: Any,
        channel_widths_hz: Any,
        integration_time_s: Any,
        sample_times_mjd: Any,
        latitude_rad: float,
    ) -> None:
        if type(bandwidth_smearing) is not bool or type(time_smearing) is not bool:
            raise TypeError("bandwidth_smearing and time_smearing must be booleans")
        if not (bandwidth_smearing or time_smearing):
            raise ValueError(
                "SmearingFactorJones requires at least one smearing kind; an "
                "envelope of ones is indistinguishable from no term at all"
            )
        frequencies = _read_only_array(
            "channel_frequencies_hz", channel_frequencies_hz, np.float64
        )
        widths = _read_only_array("channel_widths_hz", channel_widths_hz, np.float64)
        integrations = _read_only_array(
            "integration_time_s", integration_time_s, np.float64
        )
        times = _read_only_array("sample_times_mjd", sample_times_mjd, np.float64)
        for name, array in (
            ("channel_frequencies_hz", frequencies),
            ("channel_widths_hz", widths),
            ("integration_time_s", integrations),
            ("sample_times_mjd", times),
        ):
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"{name} must be a nonempty one-dimensional array")
        if widths.size != frequencies.size:
            raise ValueError(
                "channel_widths_hz must have one entry per channel frequency"
            )
        if integrations.size != times.size:
            raise ValueError("integration_time_s must have one entry per time sample")
        if not bool(np.all(frequencies > 0.0)):
            raise ValueError("channel_frequencies_hz must be positive")
        if not bool(np.all(widths > 0.0)):
            raise ValueError("channel_widths_hz must be positive")
        if not bool(np.all(integrations > 0.0)):
            raise ValueError("integration_time_s must be positive")
        latitude = float(latitude_rad)
        if not math.isfinite(latitude):
            raise ValueError("latitude_rad must be finite")

        self._bandwidth_smearing = bandwidth_smearing
        self._time_smearing = time_smearing
        self._channel_frequencies_hz = frequencies
        self._channel_widths_hz = widths
        self._integration_time_s = integrations
        self._sample_times_mjd = times
        self._latitude_rad = latitude

    @property
    def name(self) -> str:
        return "Q"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    @property
    def hadamard_target(self) -> Literal["envelope", "correlation"]:
        return "envelope"

    @property
    def term_status(self) -> str:
        return "implemented"

    @property
    def bandwidth_smearing(self) -> bool:
        return self._bandwidth_smearing

    @property
    def time_smearing(self) -> bool:
        return self._time_smearing

    @property
    def channel_frequencies_hz(self) -> npt.NDArray[np.float64]:
        return self._channel_frequencies_hz

    @property
    def channel_widths_hz(self) -> npt.NDArray[np.float64]:
        return self._channel_widths_hz

    @property
    def integration_time_s(self) -> npt.NDArray[np.float64]:
        return self._integration_time_s

    @property
    def latitude_rad(self) -> float:
        return self._latitude_rad

    def _channel_width(self, freq_idx: int, frequency_hz: float) -> float:
        index = int(freq_idx)
        if index < 0 or index >= self._channel_frequencies_hz.size:
            raise JonesEvaluationError(
                f"Jones term 'Q' was resolved against "
                f"{self._channel_frequencies_hz.size} channels; channel index "
                f"{index} is out of range."
            )
        expected = float(self._channel_frequencies_hz[index])
        if abs(float(frequency_hz) - expected) > 1e-6 * expected:
            raise JonesEvaluationError(
                f"Jones term 'Q' was resolved with channel {index} at "
                f"{expected} Hz but was evaluated at {float(frequency_hz)} Hz; "
                "the smearing bandwidth must be the one the run declares for "
                "the channel being computed."
            )
        return float(self._channel_widths_hz[index])

    def _integration(self, time_idx: int, time_mjd: float) -> float:
        index = int(time_idx)
        if index < 0 or index >= self._sample_times_mjd.size:
            raise JonesEvaluationError(
                f"Jones term 'Q' was resolved against "
                f"{self._sample_times_mjd.size} time samples; time index "
                f"{index} is out of range."
            )
        expected = float(self._sample_times_mjd[index])
        # One millisecond, in days: far tighter than any cadence RadioSim
        # supports and far looser than the float64 round trip through Astropy.
        if abs(float(time_mjd) - expected) > 1.2e-8:
            raise JonesEvaluationError(
                f"Jones term 'Q' was resolved with sample {index} at MJD "
                f"{expected} but was evaluated at MJD {float(time_mjd)}; the "
                "smearing integration time must be the one the run declares for "
                "the sample being computed."
            )
        return float(self._integration_time_s[index])

    def compute_baseline_factor(
        self,
        *,
        baseline_pairs: Sequence[tuple[int, int]],
        baseline_uvw_wavelengths: Any,
        directions: DirectionBatch,
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        """Return the ``(B, n_dir)`` real smearing envelope.

        Everything here runs through the backend over the same baseline
        coordinates the geometric phase uses, so the envelope never crosses the
        host boundary and no array value is branched on (Section 17.2).  The two
        host-side quantities -- this channel's width and this sample's
        integration time -- are Python floats read from the resolved grids
        before any array work begins.
        """
        xp = backend.xp
        uvw = baseline_uvw_wavelengths
        bl_u = uvw[:, 0:1]
        bl_v = uvw[:, 1:2]
        bl_w = uvw[:, 2:3]
        dir_l = backend.asarray(directions.dir_l, dtype=dtype)
        dir_m = backend.asarray(directions.dir_m, dtype=dtype)
        dir_n = backend.asarray(directions.dir_n, dtype=dtype)

        factor: Any = None
        if self._bandwidth_smearing:
            width_hz = self._channel_width(freq_idx, frequency_hz)
            residual_delay_s = (
                bl_u * dir_l + bl_v * dir_m + bl_w * (dir_n - 1.0)
            ) / float(frequency_hz)
            factor = xp.sinc(width_hz * residual_delay_s)
        if self._time_smearing:
            integration_s = self._integration(time_idx, time_mjd)
            sin_latitude = math.sin(self._latitude_rad)
            cos_latitude = math.cos(self._latitude_rad)
            fringe_rate_hz = EARTH_ROTATION_RAD_PER_S * (
                bl_u * (dir_n * cos_latitude - dir_m * sin_latitude)
                + dir_l * (bl_v * sin_latitude - bl_w * cos_latitude)
            )
            time_factor = xp.sinc(integration_s * fringe_rate_hz)
            factor = time_factor if factor is None else factor * time_factor
        return xp.asarray(factor, dtype=dtype)

    def get_config(self) -> dict[str, Any]:
        """Return this term's record, including the grids it smears over."""
        return {
            "name": self.name,
            "term_status": self.term_status,
            "is_direction_dependent": self.is_direction_dependent,
            "hadamard_target": self.hadamard_target,
            "bandwidth_smearing": self._bandwidth_smearing,
            "time_smearing": self._time_smearing,
            "channel_widths_hz": [float(value) for value in self._channel_widths_hz],
            "integration_time_s": [float(value) for value in self._integration_time_s],
            "latitude_rad": self._latitude_rad,
        }
