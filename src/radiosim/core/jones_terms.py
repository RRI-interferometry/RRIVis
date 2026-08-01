"""The one canonical, frozen Jones-term inventory for a run.

``Tier7JonesSciencePlan.md`` Section 22.  :func:`resolve_jones_terms` runs
**once**, in ``Simulator.setup()``, before any beam load, any sky load, any
network access, and any solver work, and it is the only place a configured
:class:`~radiosim.core.jones.base.JonesTerm` is constructed.  Solvers receive
:class:`ResolvedJonesTerms` and never see raw configuration, which is what makes
"where did this gain come from?" a question with exactly one answer
(``Fix.md`` Section 4.3).

Precedence
----------
Within a term, an explicit ``per_antenna`` entry beats the array-wide default.
There is no third source: no environment variable, no CLI flag, and no
per-antenna file override for a Jones parameter (Section 22 rule 5).

Failure ordering
----------------
The order failures are raised in is part of the contract, because a user fixing
a configuration must not be sent around a loop (Section 26.1).  This module
implements stages 3-6 of that order, in this sequence and across *all* terms at
each stage before moving to the next:

3. structural validation -- ``per_antenna`` antenna existence (R4), duplicate
   ``(antenna, feed)`` pairs (R5), and feed index range (R6);
4. physical-range validation -- R8 for ``Rc``, R9 for ``Z``, R10 for ``T``
   (R16 belongs to ``Q``, which Tier 7H adds);
5. cross-object consistency -- bandpass frequency coverage (R11) and mount types
   (R12, R15);
6. the identity check (R7), **last**, because it needs fully resolved values.

R13, the low-elevation guard on ``T`` and ``Z``, is deliberately *not* here.
Its condition is "a direction survives the horizon mask below
``minimum_elevation_deg``", and no direction exists until a solver resolves one
for a ``(time, frequency)`` step -- so the two terms raise it themselves, from
``compute_jones_batch``.  What *is* checked here is everything about those two
blocks that is decidable without a sky.

Stages 1 and 2 -- the strict Pydantic parse and the removed-field guidance --
have already happened by the time this function is called.

What is here and what is not
----------------------------
``chain_terms`` carries the terms this run's ``jones:`` section configured, in
canonical chain order.  It does **not** carry ``H``, ``C`` and ``E``: those
three are always present and are built by the solver, ``E`` necessarily so,
because the beam adapter closes over the direction, frequency and time of the
step it is evaluated at and cannot exist before the time loop.  The full
composed order, including all three, is recorded in
``provenance.chain_order``, which is what a reader of the record needs; the
solver splices ``chain_terms`` into their canonical slots around them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from radiosim.core.jones.bandpass import (
    BandpassJones,
    BandpassResponse,
    PolynomialBandpassResponse,
    TabulatedBandpassResponse,
)
from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones.baseline_errors import JonesBaselineTerm
from radiosim.core.jones.crosshand import CrosshandJones
from radiosim.core.jones.delay import CableReflectionJones, DelayJones
from radiosim.core.jones.gain import GainJones, ResolvedGainTimeModel
from radiosim.core.jones.ionosphere import IonosphereJones, ResolvedTecModel
from radiosim.core.jones.parallactic import (
    ROTATING_MOUNT_TYPES,
    SUPPORTED_MOUNT_TYPES,
    ParallacticAngleJones,
)
from radiosim.core.jones.polarization_leakage import (
    LeakageCoefficient,
    PolarizationLeakageJones,
    leakage_from_ixr_db,
)
from radiosim.core.jones.troposphere import (
    TroposphereJones,
    saastamoinen_zenith_hydrostatic_delay_m,
)
from radiosim.core.jones_errors import (
    IdentityJonesTermError,
    InvalidJonesConfigError,
    JonesAssignmentError,
    UnsupportedMountTypeError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from radiosim.core.instrument import ResolvedInstrument
    from radiosim.core.precision import PrecisionConfig
    from radiosim.core.time_grid import ObservationTimeGrid
    from radiosim.io.jones_config import (
        BandpassTermConfig,
        CableReflectionTermConfig,
        CrosshandTermConfig,
        DelayTermConfig,
        GainTermConfig,
        IonosphereTermConfig,
        JonesConfig,
        LeakageTermConfig,
        TroposphereTermConfig,
    )

__all__ = [
    "CANONICAL_CHAIN_ORDER",
    "EMPTY_JONES_TERMS",
    "JONES_SCHEMA_VERSION",
    "JonesProvenance",
    "ResolvedJonesDtypes",
    "ResolvedJonesTerms",
    "SOLVER_OWNED_TERMS",
    "resolve_jones_terms",
]

JONES_SCHEMA_VERSION = "1.0.0"

#: The canonical chain order, correlator-side first (Section 12.2, 20.12)::
#:
#:     J = H @ G @ B @ Rc @ Kd @ X @ D @ C @ E @ P @ T @ Z
#:
#: The full designed order is written out here even though not every letter is
#: implemented yet, so that the ordering lives in exactly one place and each
#: later slice adds physics rather than re-deciding where its term goes.
#:
#: ``P`` sits **sky-side** of ``C`` (Tier 7F, defect D12).  Tier 5 Section 19.1
#: placed it correlator-side, which is wrong for a circular receptor: the
#: physical composite is ``M(circular) R(chi + psi) = C R(psi)``, so the field
#: rotation must be the right-hand factor.  The two orders agree only for a
#: linear receptor, where ``M = I2`` and rotations commute -- which is why the
#: error was unobservable while ``P`` did not exist.
CANONICAL_CHAIN_ORDER: tuple[str, ...] = (
    "H",
    "G",
    "B",
    "Rc",
    "Kd",
    "X",
    "D",
    "C",
    "E",
    "P",
    "T",
    "Z",
)

#: The three terms the solver always builds itself and this module never
#: constructs: the reporting-basis transform, the receptor configuration, and
#: the beam adapter.
SOLVER_OWNED_TERMS: tuple[str, ...] = ("H", "C", "E")

#: Term letter -> the ``JonesPrecision`` field that declares its precision.
#: Extended by Tier 7D so that every term in ``CANONICAL_CHAIN_ORDER`` and both
#: baseline terms have one, closing defect D15's "eight terms only".
PRECISION_FIELD_BY_TERM: Mapping[str, str] = MappingProxyType(
    {
        "K": "geometric_phase",
        "E": "beam",
        "Z": "ionosphere",
        "T": "troposphere",
        "P": "parallactic",
        "G": "gain",
        "B": "bandpass",
        "D": "polarization_leakage",
        "C": "receptor_config",
        "H": "basis_transform",
        "X": "crosshand",
        "Kd": "delay",
        "Rc": "cable_reflection",
        "M": "baseline_multiplicative",
        "Q": "smearing",
    }
)


# --------------------------------------------------------------- resolved model


@dataclass(frozen=True, slots=True)
class ResolvedJonesDtypes:
    """The resolved per-term and accumulation precisions for one run.

    Parameters
    ----------
    by_term
        Term letter to ``(complex dtype, real dtype)``, resolved from
        ``PrecisionConfig.jones``.  Every letter in
        :data:`CANONICAL_CHAIN_ORDER` plus ``K``, ``M`` and ``Q`` has an entry:
        before Tier 7D only eight terms had a precision field at all, so ``C``,
        ``H`` and every extended term silently inherited someone else's
        (defect D15).
    accumulation_complex
        The dtype the chain product is actually composed in, and the one handed
        to every term's ``compute_jones_batch``.

    Notes
    -----
    ``by_term`` is a *record*, not yet a dispatch table.  Tier 7B's accepted
    contract composes the whole chain in one dtype -- the accumulation dtype --
    because a chain whose factors carried different precisions would have to
    pick one at every matrix product anyway, and picking it once is the honest
    version.  Recording the per-term resolution is what makes a later change to
    that decision a visible one rather than a silent one.
    """

    by_term: Mapping[str, tuple[Any, Any]]
    accumulation_complex: Any

    def __post_init__(self) -> None:
        if not isinstance(self.by_term, Mapping):
            raise TypeError("by_term must be a mapping")
        object.__setattr__(self, "by_term", MappingProxyType(dict(self.by_term)))


@dataclass(frozen=True, slots=True)
class JonesProvenance:
    """Versioned explanation and fingerprint of one resolved Jones inventory.

    Parameters
    ----------
    enabled_terms
        Every term in the composed chain, in canonical order, including the
        three the solver always adds (``H``, ``C``, ``E``).  This is the answer
        to "what was actually applied", so leaving the always-on terms out would
        make the record read as though a run with no ``jones:`` section applied
        nothing at all.
    chain_order
        The same tuple; kept as a separate field because Section 22 names both,
        and because a later tier that lets a term appear twice would make them
        differ.
    term_snapshots
        Term letter to its frozen, JSON-safe configuration snapshot.  Pure
        configuration: nothing filesystem-path-derived enters it, which is the
        ``RUN-005``/``RUN-006`` lesson applied to a new section.
    mount_types
        Antenna number to resolved mount type, recorded because ``P``'s validity
        depends on it and because a record that explains a field rotation must
        say what the feeds were mounted on.
    jones_sha256
        SHA-256 over the canonical payload of the three fields above.
    """

    schema_version: str
    enabled_terms: tuple[str, ...]
    chain_order: tuple[str, ...]
    term_snapshots: Mapping[str, Any]
    mount_types: Mapping[int, str | None]
    jones_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != JONES_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {JONES_SCHEMA_VERSION!r}")
        if not isinstance(self.jones_sha256, str) or len(self.jones_sha256) != 64:
            raise ValueError("jones_sha256 must be a 64-character hexadecimal SHA-256")
        object.__setattr__(
            self, "term_snapshots", MappingProxyType(dict(self.term_snapshots))
        )
        object.__setattr__(
            self, "mount_types", MappingProxyType(dict(self.mount_types))
        )
        expected = _compute_jones_sha256(
            enabled_terms=self.enabled_terms,
            chain_order=self.chain_order,
            term_snapshots=self.term_snapshots,
            mount_types=self.mount_types,
        )
        if self.jones_sha256 != expected:
            raise ValueError(
                "jones_sha256 does not match canonical Jones configuration content"
            )


@dataclass(frozen=True, slots=True)
class ResolvedJonesTerms:
    """The one canonical, frozen Jones-term inventory for a run.

    Parameters
    ----------
    chain_terms
        The configured per-antenna terms, in canonical chain order.  The three
        solver-owned terms (``H``, ``C``, ``E``) are deliberately not here; see
        the module docstring.
    baseline_terms
        The configured baseline-dependent Hadamard terms (``M``, ``Q``).  Empty
        until Tier 7H.
    dtypes
        The resolved precisions.
    provenance
        The record and the fingerprint.
    """

    chain_terms: tuple[JonesTerm, ...] = ()
    baseline_terms: tuple[JonesBaselineTerm, ...] = ()
    dtypes: ResolvedJonesDtypes = field(
        default_factory=lambda: ResolvedJonesDtypes(
            by_term={}, accumulation_complex=np.complex128
        )
    )
    provenance: JonesProvenance = field(
        default_factory=lambda: _empty_jones_provenance()
    )

    def __post_init__(self) -> None:
        chain = tuple(self.chain_terms)
        if any(not isinstance(term, JonesTerm) for term in chain):
            raise TypeError("chain_terms must contain only JonesTerm values")
        if any(isinstance(term, JonesBaselineTerm) for term in chain):
            raise TypeError(
                "a JonesBaselineTerm cannot be a chain term; baseline-dependent "
                "effects multiply finished visibilities elementwise"
            )
        names = [term.name for term in chain]
        if len(set(names)) != len(names):
            raise ValueError("chain_terms must not repeat a term letter")
        expected = [letter for letter in CANONICAL_CHAIN_ORDER if letter in set(names)]
        if names != expected:
            raise ValueError(
                f"chain_terms must be in canonical order {tuple(expected)}, got "
                f"{tuple(names)}"
            )
        if type(self.dtypes) is not ResolvedJonesDtypes:
            raise TypeError("dtypes must be a ResolvedJonesDtypes")
        if type(self.provenance) is not JonesProvenance:
            raise TypeError("provenance must be a JonesProvenance")
        object.__setattr__(self, "chain_terms", chain)
        object.__setattr__(self, "baseline_terms", tuple(self.baseline_terms))

    @property
    def is_empty(self) -> bool:
        """``True`` when this run configured no Jones term at all."""
        return not self.chain_terms and not self.baseline_terms

    @property
    def configured_letters(self) -> tuple[str, ...]:
        """Return the configured term letters, in canonical chain order."""
        return tuple(term.name for term in self.chain_terms)

    def term(self, letter: str) -> JonesTerm | None:
        """Return the configured term with this letter, or ``None``."""
        for candidate in self.chain_terms:
            if candidate.name == letter:
                return candidate
        return None

    def to_snapshot(self) -> dict[str, Any]:
        """Return a fresh deterministic JSON-safe snapshot, or ``{}`` if empty.

        The empty return is load-bearing.  A run with no ``jones:`` section
        contributes **nothing** to ``scientific_sha256`` -- not an empty object,
        not a null, nothing -- so its digest is bit-identical to the same
        configuration before this section existed (invariant I1).  A section
        that is present but configures no term never reaches here: it is
        rejected as R2.
        """
        if not self.provenance.enabled_terms:
            return {}
        return {
            "schema_version": self.provenance.schema_version,
            "enabled_terms": list(self.provenance.enabled_terms),
            "chain_order": list(self.provenance.chain_order),
            "term_snapshots": json.loads(
                json.dumps(dict(self.provenance.term_snapshots))
            ),
            "mount_types": {
                str(number): value
                for number, value in sorted(self.provenance.mount_types.items())
            },
            "jones_sha256": self.provenance.jones_sha256,
        }


# ----------------------------------------------------------------- fingerprint


def _canonical_jones_fingerprint_payload(
    *,
    enabled_terms: Sequence[str],
    chain_order: Sequence[str],
    term_snapshots: Mapping[str, Any],
    mount_types: Mapping[int, str | None],
) -> dict[str, Any]:
    """Return the exact canonical Jones hash payload."""
    return {
        "schema_version": JONES_SCHEMA_VERSION,
        "enabled_terms": list(enabled_terms),
        "chain_order": list(chain_order),
        "term_snapshots": dict(term_snapshots),
        "mount_types": {
            str(number): value for number, value in sorted(mount_types.items())
        },
    }


def _compute_jones_sha256(
    *,
    enabled_terms: Sequence[str],
    chain_order: Sequence[str],
    term_snapshots: Mapping[str, Any],
    mount_types: Mapping[int, str | None],
) -> str:
    """Compute SHA-256 over the canonical UTF-8 JSON Jones payload.

    The same canonical-JSON convention Tier 5 uses for ``receptor_sha256``:
    sorted keys, no whitespace, no NaN, UTF-8.  Two runs differing only in a
    Jones parameter produce different digests; two runs with ``jones:`` absent
    never reach a digest at all, because the snapshot is empty and nothing is
    hashed.
    """
    encoded = json.dumps(
        _canonical_jones_fingerprint_payload(
            enabled_terms=enabled_terms,
            chain_order=chain_order,
            term_snapshots=term_snapshots,
            mount_types=mount_types,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empty_jones_provenance() -> JonesProvenance:
    """Return the provenance of a run that configured no Jones term."""
    return JonesProvenance(
        schema_version=JONES_SCHEMA_VERSION,
        enabled_terms=(),
        chain_order=(),
        term_snapshots={},
        mount_types={},
        jones_sha256=_compute_jones_sha256(
            enabled_terms=(),
            chain_order=(),
            term_snapshots={},
            mount_types={},
        ),
    )


#: The inventory of a run with no ``jones:`` section.  Solvers default to it, so
#: a direct solver call is exactly the pre-Tier-7D forward model.
EMPTY_JONES_TERMS: ResolvedJonesTerms = ResolvedJonesTerms()


# ---------------------------------------------------------- structural checking


def _validate_overrides(
    letter: str,
    entries: Sequence[Any],
    known_numbers: Sequence[int],
) -> dict[tuple[int, int], Any]:
    """Return the ``(antenna, feed)`` override map after stage-3 validation.

    Implements R4, R5 and R6 in that order for one term, so that a user with
    three mistakes in one block is told about the first one in reading order
    rather than in whatever order a dictionary happened to iterate.
    """
    known = set(known_numbers)
    shown = ", ".join(str(number) for number in sorted(known))
    resolved: dict[tuple[int, int], Any] = {}
    for entry in entries:
        number = int(entry.antenna)
        if number not in known:
            raise JonesAssignmentError(
                f"jones.{letter}.per_antenna references antenna number {number}, "
                f"which is not in the resolved instrument; known numbers are "
                f"{shown}."
            )
        feed = int(entry.feed)
        if feed not in (0, 1):
            raise InvalidJonesConfigError(
                f"jones.{letter}.per_antenna feed={feed} is invalid; feeds are "
                "indexed 0 and 1 in the antenna's own receptor basis."
            )
        if (number, feed) in resolved:
            raise InvalidJonesConfigError(
                f"jones.{letter}.per_antenna contains a duplicate entry for "
                f"antenna {number} feed {feed}; each (antenna, feed) may appear "
                "once."
            )
        resolved[(number, feed)] = entry
    return resolved


def _validate_antenna_overrides(
    letter: str,
    entries: Sequence[Any],
    known_numbers: Sequence[int],
) -> dict[int, Any]:
    """Return the per-antenna override map for a term with no feed index.

    ``X`` is the only such term: its parameter is the *relative* phase between
    an antenna's two feeds, so there is one entry per antenna and no feed to
    key on (Section 20.4).  R4 is raised verbatim; the duplicate message is R5's
    sentence with the pair reduced to the key that exists, because naming a feed
    the configuration never wrote would be worse than adapting the wording.
    """
    known = set(known_numbers)
    shown = ", ".join(str(number) for number in sorted(known))
    resolved: dict[int, Any] = {}
    for entry in entries:
        number = int(entry.antenna)
        if number not in known:
            raise JonesAssignmentError(
                f"jones.{letter}.per_antenna references antenna number {number}, "
                f"which is not in the resolved instrument; known numbers are "
                f"{shown}."
            )
        if number in resolved:
            raise InvalidJonesConfigError(
                f"jones.{letter}.per_antenna contains a duplicate entry for "
                f"antenna {number}; each antenna may appear once."
            )
        resolved[number] = entry
    return resolved


def _reject_unsupported_mounts(
    mount_types: Mapping[int, str | None],
    *,
    parallactic_enabled: bool,
) -> None:
    """Raise R12 and R15 in that order (Section 24, 7F correction).

    Two rules, one pass, in canonical antenna order so the first mistake a
    reader sees is the first one in the instrument:

    * **R12** -- an antenna's ``mount_type`` is outside the five ``P`` models.
      Rejected whether or not ``jones.P`` is configured, because gating it on
      the term would mean a ``phased`` mount, which Tier 5 rejected outright,
      became a silent ``fixed`` in every run that did not enable ``P``.  An
      unspecified mount (``None``) is the ``fixed`` case and is never rejected;
      that is what invariant I1 rests on.
    * **R15** -- an antenna's feeds rotate relative to the sky
      (:data:`~radiosim.core.jones.parallactic.ROTATING_MOUNT_TYPES`) and
      ``jones.P`` is not enabled.  ``equatorial`` is deliberately outside that
      set: its feeds track the sky, ``P`` is exactly ``I2`` for it, and
      demanding the term would collide with R7's identity rejection and leave
      such an array with no accepted configuration at all.

    This is the replacement for ``core/receptor.py``'s Tier 5 blanket rejection,
    and it is a strictly better contract: it names the fix rather than the tier.
    """
    for number in sorted(mount_types):
        mount = mount_types[number]
        if mount is None:
            continue
        if mount not in SUPPORTED_MOUNT_TYPES:
            raise UnsupportedMountTypeError(
                f"antenna {number} has mount_type={mount}, which the "
                "parallactic-angle term does not model; supported mounts are "
                f"{', '.join(SUPPORTED_MOUNT_TYPES)}."
            )
    if parallactic_enabled:
        return
    for number in sorted(mount_types):
        mount = mount_types[number]
        if mount in ROTATING_MOUNT_TYPES:
            raise UnsupportedMountTypeError(
                f"antenna {number} has mount_type={mount}, whose feeds rotate "
                "with the sky; enable 'jones.P' or the simulation would "
                "silently treat it as a fixed mount."
            )


def _reject_identity(letter: str) -> None:
    """Raise R7 for a term whose resolved parameters make it exactly ``I2``."""
    raise IdentityJonesTermError(
        f"jones.{letter} is configured with parameters that make it exactly the "
        "identity; a term that cannot change the visibilities must be removed "
        "rather than configured."
    )


# --------------------------------------------------------------------- G and B


def _resolve_gain_time_model(config: Any) -> ResolvedGainTimeModel:
    """Return the resolved ``s(t)`` for one ``jones.G`` block."""
    kind = config.kind
    if kind == "constant":
        return ResolvedGainTimeModel(kind="constant")
    if kind == "linear_drift":
        return ResolvedGainTimeModel(
            kind="linear_drift", rate_per_hour=float(config.rate_per_hour)
        )
    return ResolvedGainTimeModel(
        kind="sinusoidal",
        depth=float(config.depth),
        period_hours=float(config.period_hours),
        phase_rad=float(config.phase_rad),
    )


def _resolve_gain_term(
    config: GainTermConfig,
    *,
    antenna_numbers: Sequence[int],
    overrides: Mapping[tuple[int, int], Any],
    reference_time_mjd: float,
    pointing_elevation_deg: float,
) -> GainJones:
    """Build the ``G`` term from a validated ``jones.G`` block (Section 20.1)."""
    base = np.empty((len(antenna_numbers), 2), dtype=np.complex128)
    for row, number in enumerate(antenna_numbers):
        for feed in (0, 1):
            override = overrides.get((number, feed))
            amplitude = config.amplitude_error
            phase = config.phase_error_rad
            if override is not None:
                if override.amplitude_error is not None:
                    amplitude = override.amplitude_error
                if override.phase_error_rad is not None:
                    phase = override.phase_error_rad
            base[row, feed] = (1.0 + float(amplitude)) * np.exp(1j * float(phase))

    elevation_gain = 1.0
    if config.elevation_curve is not None:
        elevation_gain = 0.0
        for order, coefficient in enumerate(config.elevation_curve):
            elevation_gain += float(coefficient) * pointing_elevation_deg**order
        if not math.isfinite(elevation_gain):
            raise InvalidJonesConfigError(
                "jones.G.elevation_curve evaluates to a non-finite gain at the "
                f"pointing elevation {pointing_elevation_deg} deg; reduce the "
                "polynomial order or rescale the coefficients."
            )

    return GainJones(
        base_gains=base,
        time_model=_resolve_gain_time_model(config.time_model),
        reference_time_mjd=reference_time_mjd,
        elevation_gain=elevation_gain,
    )


def _resolve_bandpass_response(
    model: Any,
    *,
    frequencies_hz: np.ndarray,
) -> BandpassResponse:
    """Return one resolved feed response, applying the derived defaults."""
    from radiosim.io.jones_config import as_complex

    if model.kind == "polynomial":
        low = float(frequencies_hz[0])
        high = float(frequencies_hz[-1])
        reference = model.reference_frequency_hz
        if reference is None:
            reference = 0.5 * (low + high)
        scale = model.scale_frequency_hz
        if scale is None:
            scale = 0.5 * (high - low)
            if scale <= 0.0:
                raise InvalidJonesConfigError(
                    "jones.B polynomial scale_frequency_hz defaults to the "
                    "half-bandwidth, which is zero for a single-channel "
                    "observation; set jones.B.model.scale_frequency_hz "
                    "explicitly."
                )
        return PolynomialBandpassResponse(
            coefficients=tuple(as_complex(value) for value in model.coefficients),
            reference_frequency_hz=float(reference),
            scale_frequency_hz=float(scale),
        )
    return TabulatedBandpassResponse(
        node_frequencies_hz=tuple(float(node) for node in model.node_frequencies_hz),
        gains=tuple(as_complex(value) for value in model.gains),
    )


def _reject_uncovered_bandpass(model: Any, frequencies_hz: np.ndarray) -> None:
    """Raise R11 when a tabulated model does not span every observed channel."""
    if model.kind != "tabulated":
        return
    low = float(model.node_frequencies_hz[0])
    high = float(model.node_frequencies_hz[-1])
    observed_low = float(frequencies_hz[0])
    observed_high = float(frequencies_hz[-1])
    if low <= observed_low and observed_high <= high:
        return
    raise InvalidJonesConfigError(
        f"jones.B tabulated nodes span {low}-{high} Hz but the observation "
        f"covers {observed_low}-{observed_high} Hz; RadioSim does not "
        "extrapolate a bandpass."
    )


def _resolve_bandpass_term(
    config: BandpassTermConfig,
    *,
    antenna_numbers: Sequence[int],
    overrides: Mapping[tuple[int, int], Any],
    frequencies_hz: np.ndarray,
) -> BandpassJones:
    """Build the ``B`` term from a validated ``jones.B`` block (Section 20.2)."""
    default_response = _resolve_bandpass_response(
        config.model, frequencies_hz=frequencies_hz
    )
    responses: list[tuple[BandpassResponse, BandpassResponse]] = []
    for number in antenna_numbers:
        pair: list[BandpassResponse] = []
        for feed in (0, 1):
            override = overrides.get((number, feed))
            if override is None:
                pair.append(default_response)
            else:
                pair.append(
                    _resolve_bandpass_response(
                        override.model, frequencies_hz=frequencies_hz
                    )
                )
        responses.append((pair[0], pair[1]))
    return BandpassJones(responses=tuple(responses), frequencies_hz=frequencies_hz)


# --------------------------------------------------------------- Rc, Kd, X, D


def _reject_reflection_amplitude(amplitude: float) -> None:
    """Raise R8 for a reflection amplitude outside ``0 < |A| < 1``.

    Zero is rejected here rather than left to the identity check, because
    Section 24 states the physical range as ``0 < |A| < 1``: a reflection of zero
    amplitude is not a reflection, and a message that says so is more useful
    than the generic identity one.
    """
    if 0.0 < abs(amplitude) < 1.0:
        return
    raise InvalidJonesConfigError(
        f"jones.Rc.amplitude={amplitude} must satisfy 0 < |A| < 1; a reflection "
        "cannot return more power than it receives."
    )


def _resolve_cable_reflection_term(
    config: CableReflectionTermConfig,
    *,
    antenna_numbers: Sequence[int],
    overrides: Mapping[tuple[int, int], Any],
) -> CableReflectionJones:
    """Build the ``Rc`` term from a validated ``jones.Rc`` block (Section 20.6)."""
    rows = len(antenna_numbers)
    amplitudes = np.empty((rows, 2), dtype=np.float64)
    delays = np.empty((rows, 2), dtype=np.float64)
    phases = np.empty((rows, 2), dtype=np.float64)
    for row, number in enumerate(antenna_numbers):
        for feed in (0, 1):
            override = overrides.get((number, feed))
            amplitude = config.amplitude
            delay = config.cable_delay_s
            phase = config.phase_rad
            if override is not None:
                if override.amplitude is not None:
                    amplitude = override.amplitude
                if override.cable_delay_s is not None:
                    delay = override.cable_delay_s
                if override.phase_rad is not None:
                    phase = override.phase_rad
            amplitudes[row, feed] = float(amplitude)
            delays[row, feed] = float(delay)
            phases[row, feed] = float(phase)
    return CableReflectionJones(
        amplitudes=amplitudes, cable_delays_s=delays, phases_rad=phases
    )


def _resolve_delay_term(
    config: DelayTermConfig,
    *,
    antenna_numbers: Sequence[int],
    overrides: Mapping[tuple[int, int], Any],
) -> DelayJones:
    """Build the ``Kd`` term from a validated ``jones.Kd`` block (Section 20.5)."""
    delays = np.empty((len(antenna_numbers), 2), dtype=np.float64)
    for row, number in enumerate(antenna_numbers):
        for feed in (0, 1):
            override = overrides.get((number, feed))
            delay = config.delay_s if override is None else override.delay_s
            delays[row, feed] = float(delay)
    return DelayJones(delays_s=delays)


def _resolve_crosshand_term(
    config: CrosshandTermConfig,
    *,
    antenna_numbers: Sequence[int],
    overrides: Mapping[int, Any],
) -> CrosshandJones:
    """Build the ``X`` term from a validated ``jones.X`` block (Section 20.4)."""
    rows = len(antenna_numbers)
    phases = np.empty(rows, dtype=np.float64)
    delays = np.empty(rows, dtype=np.float64)
    for row, number in enumerate(antenna_numbers):
        override = overrides.get(number)
        phase = config.phase_rad
        delay = config.delay_s
        if override is not None:
            if override.phase_rad is not None:
                phase = override.phase_rad
            if override.delay_s is not None:
                delay = override.delay_s
        phases[row] = float(phase)
        delays[row] = float(delay)
    return CrosshandJones(phases_rad=phases, delays_s=delays)


def _leakage_normalization(
    model: Any,
    *,
    frequencies_hz: np.ndarray,
) -> tuple[float, float]:
    """Return the resolved ``(nu_ref, nu_scale)`` for a polynomial leakage.

    The same band-centre and half-bandwidth defaults ``B`` uses, and the same
    refusal to invent a scale a single-channel observation does not have: a
    normalization silently substituted would make one document mean different
    things at different bandwidths.
    """
    low = float(frequencies_hz[0])
    high = float(frequencies_hz[-1])
    reference = model.reference_frequency_hz
    if reference is None:
        reference = 0.5 * (low + high)
    scale = model.scale_frequency_hz
    if scale is None:
        scale = 0.5 * (high - low)
        if scale <= 0.0:
            raise InvalidJonesConfigError(
                "jones.D frequency_polynomial scale_frequency_hz defaults to the "
                "half-bandwidth, which is zero for a single-channel observation; "
                "set jones.D.d_terms.scale_frequency_hz explicitly."
            )
    return float(reference), float(scale)


def _resolve_leakage_coefficient(
    model: Any,
    *,
    feed: int,
    frequencies_hz: np.ndarray,
) -> LeakageCoefficient:
    """Return one feed's resolved ``d(nu)`` from either leakage model shape.

    ``model`` is either the array-wide two-feed block -- which names ``d0``/``d1``
    or ``coefficients0``/``coefficients1`` -- or a per-feed override block, which
    names ``d`` or ``coefficients``.  The two shapes differ only in whether the
    feed is chosen here or was chosen by the override's key, so one function
    resolves both and there is exactly one place a leakage number is made.
    """
    from radiosim.io.jones_config import as_complex

    if model.kind == "explicit":
        if hasattr(model, "d"):
            value = as_complex(model.d)
        else:
            value = as_complex(model.d0 if feed == 0 else model.d1)
        return LeakageCoefficient(coefficients=(value,))
    if model.kind == "ixr":
        return LeakageCoefficient(
            coefficients=(leakage_from_ixr_db(model.ixr_db, model.phase_rad),)
        )
    if hasattr(model, "coefficients"):
        written = model.coefficients
    else:
        written = model.coefficients0 if feed == 0 else model.coefficients1
    reference, scale = _leakage_normalization(model, frequencies_hz=frequencies_hz)
    return LeakageCoefficient(
        coefficients=tuple(as_complex(value) for value in written),
        reference_frequency_hz=reference,
        scale_frequency_hz=scale,
    )


def _resolve_leakage_term(
    config: LeakageTermConfig,
    *,
    antenna_numbers: Sequence[int],
    overrides: Mapping[tuple[int, int], Any],
    frequencies_hz: np.ndarray,
) -> PolarizationLeakageJones:
    """Build the ``D`` term from a validated ``jones.D`` block (Section 20.3)."""
    defaults = tuple(
        _resolve_leakage_coefficient(
            config.d_terms, feed=feed, frequencies_hz=frequencies_hz
        )
        for feed in (0, 1)
    )
    resolved: list[tuple[LeakageCoefficient, LeakageCoefficient]] = []
    for number in antenna_numbers:
        pair: list[LeakageCoefficient] = []
        for feed in (0, 1):
            override = overrides.get((number, feed))
            if override is None:
                pair.append(defaults[feed])
            else:
                pair.append(
                    _resolve_leakage_coefficient(
                        override.d_term, feed=feed, frequencies_hz=frequencies_hz
                    )
                )
        resolved.append((pair[0], pair[1]))
    return PolarizationLeakageJones(d_terms=tuple(resolved))


# ------------------------------------------------------------------- T and Z


def _reject_negative_opacity(opacity: float) -> None:
    """Raise R10 for a negative zenith opacity.

    Zero is accepted: a transparent atmosphere with a real excess path is a
    perfectly good ``T``, and it is the case in which the term is unitary.  What
    is refused is the value that would *amplify*, and the message says so.
    """
    if opacity >= 0.0:
        return
    raise InvalidJonesConfigError(
        f"jones.T.opacity.zenith_opacity={opacity} must be non-negative; a "
        "negative opacity would amplify."
    )


def _reject_negative_tec(vertical_tec_tecu: float) -> None:
    """Raise R9 for a negative vertical electron column."""
    if vertical_tec_tecu >= 0.0:
        return
    raise InvalidJonesConfigError(
        f"jones.Z.tec.vertical_tec_tecu={vertical_tec_tecu} must be non-negative."
    )


def _resolve_troposphere_term(
    config: TroposphereTermConfig,
    *,
    instrument: ResolvedInstrument,
) -> TroposphereJones:
    """Build the ``T`` term from a validated ``jones.T`` block (Section 20.9).

    The zenith hydrostatic delay is per antenna even when it comes from one
    configured pressure, because the Saastamoinen formula reads the antenna's own
    height above sea level -- the site height plus its Up offset -- and an array
    on a slope really does have different dry delays.
    """
    latitude_deg = float(instrument.location.latitude_deg)
    heights_m = np.array(
        [
            float(instrument.location.height_m) + float(antenna.position_enu_m[2])
            for antenna in instrument.antennas
        ],
        dtype=np.float64,
    )
    delay_config = config.zenith_delay
    if delay_config.kind == "explicit":
        hydrostatic = np.full(
            heights_m.shape,
            float(delay_config.zenith_hydrostatic_delay_m),
            dtype=np.float64,
        )
    else:
        hydrostatic = np.array(
            [
                saastamoinen_zenith_hydrostatic_delay_m(
                    surface_pressure_hpa=float(delay_config.surface_pressure_hpa),
                    latitude_deg=latitude_deg,
                    height_m=float(height),
                )
                for height in heights_m
            ],
            dtype=np.float64,
        )
    wet = np.full(
        heights_m.shape, float(delay_config.zenith_wet_delay_m), dtype=np.float64
    )
    opacity = None if config.opacity is None else float(config.opacity.zenith_opacity)
    return TroposphereJones(
        zenith_hydrostatic_delay_m=hydrostatic,
        zenith_wet_delay_m=wet,
        mapping_function=config.mapping_function,
        latitude_deg=latitude_deg,
        heights_m=heights_m,
        zenith_opacity=opacity,
        minimum_elevation_deg=float(config.minimum_elevation_deg),
    )


def _resolve_ionosphere_term(
    config: IonosphereTermConfig,
    *,
    instrument: ResolvedInstrument,
    antenna_numbers: Sequence[int],
    overrides: Mapping[int, Any],
) -> IonosphereJones:
    """Build the ``Z`` term from a validated ``jones.Z`` block (Section 20.8)."""
    tec_config = config.tec
    tec_model = ResolvedTecModel(
        vertical_tec_tecu=float(tec_config.vertical_tec_tecu),
        gradient_east_tecu_per_km=float(
            getattr(tec_config, "gradient_east_tecu_per_km", 0.0)
        ),
        gradient_north_tecu_per_km=float(
            getattr(tec_config, "gradient_north_tecu_per_km", 0.0)
        ),
    )
    default_rotation_measure = (
        0.0 if config.faraday is None else float(config.faraday.rotation_measure_rad_m2)
    )
    rotation_measures = np.empty(len(antenna_numbers), dtype=np.float64)
    for row, number in enumerate(antenna_numbers):
        override = overrides.get(number)
        rotation_measures[row] = (
            default_rotation_measure
            if override is None
            else float(override.rotation_measure_rad_m2)
        )
    positions = np.array(
        [antenna.position_enu_m for antenna in instrument.antennas], dtype=np.float64
    )
    return IonosphereJones(
        tec_model=tec_model,
        antenna_positions_enu_m=positions,
        shell_height_m=1000.0 * float(config.shell_height_km),
        rotation_measures_rad_m2=rotation_measures,
        minimum_elevation_deg=float(config.minimum_elevation_deg),
    )


# ------------------------------------------------------------------ resolution


def _resolved_dtypes(precision: PrecisionConfig | None) -> ResolvedJonesDtypes:
    """Return the per-term and accumulation dtypes for one run."""
    from radiosim.core.precision import PrecisionConfig as _PrecisionConfig
    from radiosim.core.precision import get_complex_dtype, get_real_dtype

    resolved = precision if precision is not None else _PrecisionConfig.standard()
    by_term: dict[str, tuple[Any, Any]] = {}
    for letter, field_name in PRECISION_FIELD_BY_TERM.items():
        level = getattr(resolved.jones, field_name)
        by_term[letter] = (get_complex_dtype(level), get_real_dtype(level))
    return ResolvedJonesDtypes(
        by_term=by_term,
        accumulation_complex=get_complex_dtype(resolved.accumulation),
    )


def resolve_jones_terms(
    config: JonesConfig | None,
    instrument: ResolvedInstrument,
    *,
    frequencies_hz: Any,
    time_grid: ObservationTimeGrid,
    precision: PrecisionConfig | None = None,
    pointing_elevation_deg: float | None = None,
) -> ResolvedJonesTerms:
    """Resolve one canonical Jones inventory from configuration and a run.

    Parameters
    ----------
    config
        The strict ``jones:`` input section, or ``None`` when the document has
        none.  ``None`` returns :data:`EMPTY_JONES_TERMS` without touching
        anything else, so a configuration that does not mention ``jones:``
        cannot be perturbed by this function's existence (invariant I1).
    instrument
        The already-resolved canonical instrument.  Supplies the antenna numbers
        every ``per_antenna`` entry is validated against, the antenna *order*
        every term is indexed in, and the mount types the record carries.
    frequencies_hz
        The observation channel centres, strictly increasing.  Used for the
        bandpass's derived reference and scale frequencies and for R11.
    time_grid
        The resolved observation time grid.  Its first sample is ``t0`` for
        every gain time model.
    precision
        The run's precision configuration; ``None`` resolves the standard
        preset.
    pointing_elevation_deg
        The elevation of the pointing centre, in degrees.  ``None`` reads it
        from :class:`~radiosim.core.phase_center.PhaseCenter`, which under
        RadioSim's zenith-drift convention is exactly 90.

    Returns
    -------
    ResolvedJonesTerms
        The complete inventory, its resolved precisions, and its fingerprint.

    Raises
    ------
    InvalidJonesConfigError
        R2, R5, R6, R11, or a derived default that cannot be computed.
    JonesAssignmentError
        R4: a ``per_antenna`` entry names an antenna the instrument does not
        have.
    IdentityJonesTermError
        R7: a configured term resolves to exactly the identity.
    """
    from radiosim.core.instrument import ResolvedInstrument as _ResolvedInstrument
    from radiosim.core.time_grid import ObservationTimeGrid as _ObservationTimeGrid
    from radiosim.io.jones_config import JonesConfig as _JonesConfig

    if config is not None and type(config) is not _JonesConfig:
        raise TypeError("config must be a JonesConfig or None")
    if type(instrument) is not _ResolvedInstrument:
        raise TypeError("instrument must be a ResolvedInstrument")

    mount_types = {
        antenna.id.number: antenna.mount_type for antenna in instrument.antennas
    }
    parallactic_enabled = (
        config is not None and config.P is not None and config.P.enabled
    )

    if config is None:
        # R12 and R15 run even here.  Tier 5's blanket mount rejection ran on
        # *every* document, and moving the rule into this function must not turn
        # it into one that applies only to documents which happen to configure
        # something.  With a section present the same check runs at stage 5,
        # where Section 26.1 puts it.
        _reject_unsupported_mounts(mount_types, parallactic_enabled=False)
        return EMPTY_JONES_TERMS
    if type(time_grid) is not _ObservationTimeGrid:
        raise TypeError("time_grid must be an ObservationTimeGrid")

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a nonempty one-dimensional array")

    configured = config.configured_terms
    if not configured:
        # R2.  An empty section is a statement of intent the file does not carry
        # out; accepting it silently would hide a deleted term or a mis-indented
        # key, which is exactly the class of quiet nothing this tier removes.
        raise InvalidJonesConfigError(
            "jones: is present but configures no term; remove the section or "
            "configure at least one term."
        )

    antenna_numbers = tuple(antenna.id.number for antenna in instrument.antennas)

    # Stage 3 -- structural validation, every term before any physical check.
    # ``X`` is the one term whose overrides are keyed by antenna alone, because
    # its parameter is the relative phase *between* an antenna's two feeds.
    overrides_by_term: dict[str, dict[tuple[int, int], Any]] = {}
    crosshand_overrides: dict[int, Any] = {}
    ionosphere_overrides: dict[int, Any] = {}
    for letter in configured:
        block = getattr(config, letter)
        if letter in ("P", "T"):
            # Neither has a per-antenna block.  ``P``'s only per-antenna
            # quantity is the mount type and ``T``'s is the antenna height, and
            # both come from the resolved instrument rather than from the
            # document -- which is why neither can be overridden here.
            continue
        if letter == "X":
            crosshand_overrides = _validate_antenna_overrides(
                letter, block.per_antenna, antenna_numbers
            )
            continue
        if letter == "Z":
            # ``Z``'s only per-antenna quantity is the line-of-sight rotation
            # measure, which is one number per antenna and carries no feed
            # index, so its overrides validate through the antenna-keyed path.
            if block.faraday is not None:
                ionosphere_overrides = _validate_antenna_overrides(
                    letter, block.faraday.per_antenna, antenna_numbers
                )
            continue
        overrides_by_term[letter] = _validate_overrides(
            letter, block.per_antenna, antenna_numbers
        )

    # Stage 4 -- physical-range validation.  No range rejection belongs to G,
    # B, Kd, X or D: their parameters are unbounded by physics (a gain may be
    # any complex number and a delay any real one), which is why R8-R10 name
    # Rc, Z and T instead.
    if "Rc" in configured:
        reflection_config = config.Rc
        assert reflection_config is not None
        _reject_reflection_amplitude(float(reflection_config.amplitude))
        for entry in reflection_config.per_antenna:
            if entry.amplitude is not None:
                _reject_reflection_amplitude(float(entry.amplitude))
    if "T" in configured:
        troposphere_config = config.T
        assert troposphere_config is not None
        if troposphere_config.opacity is not None:
            _reject_negative_opacity(float(troposphere_config.opacity.zenith_opacity))
    if "Z" in configured:
        ionosphere_config = config.Z
        assert ionosphere_config is not None
        _reject_negative_tec(float(ionosphere_config.tec.vertical_tec_tecu))

    # Stage 5 -- cross-object consistency.
    if "B" in configured:
        bandpass_config = config.B
        assert bandpass_config is not None
        _reject_uncovered_bandpass(bandpass_config.model, frequencies)
        for entry in bandpass_config.per_antenna:
            _reject_uncovered_bandpass(entry.model, frequencies)
    _reject_unsupported_mounts(mount_types, parallactic_enabled=parallactic_enabled)

    if pointing_elevation_deg is None:
        from radiosim.core.phase_center import PhaseCenter

        pointing_elevation_deg = math.degrees(PhaseCenter().altitude_rad)
    reference_time_mjd = float(np.asarray(time_grid.as_astropy().mjd).reshape(-1)[0])

    built: dict[str, JonesTerm] = {}
    if "G" in configured:
        gain_config = config.G
        assert gain_config is not None
        built["G"] = _resolve_gain_term(
            gain_config,
            antenna_numbers=antenna_numbers,
            overrides=overrides_by_term["G"],
            reference_time_mjd=reference_time_mjd,
            pointing_elevation_deg=float(pointing_elevation_deg),
        )
    if "B" in configured:
        bandpass_config = config.B
        assert bandpass_config is not None
        built["B"] = _resolve_bandpass_term(
            bandpass_config,
            antenna_numbers=antenna_numbers,
            overrides=overrides_by_term["B"],
            frequencies_hz=frequencies,
        )
    if "Rc" in configured:
        reflection_config = config.Rc
        assert reflection_config is not None
        built["Rc"] = _resolve_cable_reflection_term(
            reflection_config,
            antenna_numbers=antenna_numbers,
            overrides=overrides_by_term["Rc"],
        )
    if "Kd" in configured:
        delay_config = config.Kd
        assert delay_config is not None
        built["Kd"] = _resolve_delay_term(
            delay_config,
            antenna_numbers=antenna_numbers,
            overrides=overrides_by_term["Kd"],
        )
    if "X" in configured:
        crosshand_config = config.X
        assert crosshand_config is not None
        built["X"] = _resolve_crosshand_term(
            crosshand_config,
            antenna_numbers=antenna_numbers,
            overrides=crosshand_overrides,
        )
    if "D" in configured:
        leakage_config = config.D
        assert leakage_config is not None
        built["D"] = _resolve_leakage_term(
            leakage_config,
            antenna_numbers=antenna_numbers,
            overrides=overrides_by_term["D"],
            frequencies_hz=frequencies,
        )
    if "P" in configured:
        # ``P``'s parameters are the instrument's, not the document's: the site
        # latitude the direction batch was built with, and one mount type per
        # antenna row in canonical instrument order.
        built["P"] = ParallacticAngleJones(
            latitude_rad=math.radians(instrument.location.latitude_deg),
            mount_types=tuple(mount_types[number] for number in antenna_numbers),
        )
    if "T" in configured:
        troposphere_config = config.T
        assert troposphere_config is not None
        built["T"] = _resolve_troposphere_term(
            troposphere_config, instrument=instrument
        )
    if "Z" in configured:
        ionosphere_config = config.Z
        assert ionosphere_config is not None
        built["Z"] = _resolve_ionosphere_term(
            ionosphere_config,
            instrument=instrument,
            antenna_numbers=antenna_numbers,
            overrides=ionosphere_overrides,
        )

    # Stage 6 -- the identity check, last, because it needs resolved values.
    if "P" in configured and not parallactic_enabled:
        # ``enabled: false`` is a disabled term, and Section 21 has no such
        # thing: the fix is to remove the block, which is exactly what R7 says.
        # Raised here rather than at the schema so the reader gets that sentence
        # instead of a type error.
        _reject_identity("P")
    for letter in configured:
        term = built[letter]
        is_identity = getattr(term, "is_identity", None)
        if is_identity is not None and is_identity():
            _reject_identity(letter)

    chain_terms = tuple(
        built[letter] for letter in CANONICAL_CHAIN_ORDER if letter in built
    )
    chain_order = tuple(
        letter
        for letter in CANONICAL_CHAIN_ORDER
        if letter in built or letter in SOLVER_OWNED_TERMS
    )
    term_snapshots = {
        letter: getattr(config, letter).model_dump(mode="json") for letter in configured
    }
    provenance = JonesProvenance(
        schema_version=JONES_SCHEMA_VERSION,
        enabled_terms=chain_order,
        chain_order=chain_order,
        term_snapshots=term_snapshots,
        mount_types=mount_types,
        jones_sha256=_compute_jones_sha256(
            enabled_terms=chain_order,
            chain_order=chain_order,
            term_snapshots=term_snapshots,
            mount_types=mount_types,
        ),
    )
    return ResolvedJonesTerms(
        chain_terms=chain_terms,
        baseline_terms=(),
        dtypes=_resolved_dtypes(precision),
        provenance=provenance,
    )
