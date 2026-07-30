"""Canonical resolved receptor state and its single precedence authority.

Receptor state is a **sibling** of the resolved instrument, not a part of it:
receptor orientation is not an instrument geometry property, so it must not
enter ``instrument_sha256``.  The models here are keyed by
:class:`~radiosim.core.instrument.AntennaId`, exactly as the beam subsystem
keys loaded beams.

:func:`resolve_receptors` is the only place receptor precedence is decided.  It
is pure: it performs no filesystem, network, backend, or device work, and it
runs before any beam file is opened.

Modelling assumption
--------------------
Expressing a circular-native antenna in a linear output basis (or the reverse)
is exact **only** when both feeds are ideal, orthogonal, and share a common
complex gain.  That holds in Tier 5 because the leakage (``D``) and gain
(``G``) terms are disabled identity stubs.  When Tier 7 implements ``D``, the
conversion becomes approximate and this assumption must be re-examined.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from radiosim.core.instrument import (
    AntennaFieldSource,
    AntennaId,
    ResolvedInstrument,
)
from radiosim.core.polarization_basis import PolarizationBasis

if TYPE_CHECKING:
    from radiosim.io.receptor_config import ReceptorsConfig

ReceptorBasis = Literal["linear", "circular"]
OutputBasisRule = Literal[
    "auto_homogeneous_linear",
    "auto_homogeneous_circular",
    "explicit_linear",
    "explicit_circular",
]

_RECEPTOR_SCHEMA_VERSION = "1.0.0"
_NOMINAL_FEED_ARRAY: dict[str, tuple[str, str]] = {
    "linear": ("x", "y"),
    "circular": ("r", "l"),
}
_SUPPORTED_BASES = frozenset(_NOMINAL_FEED_ARRAY)
_SUPPORTED_OUTPUT_REQUESTS = frozenset({"auto", "linear", "circular"})
_OUTPUT_BASIS_BY_NATIVE: dict[str, PolarizationBasis] = {
    "linear": "linear_xy",
    "circular": "circular_rl",
}
_SUPPORTED_MOUNT_TYPE = "fixed"


class ReceptorError(RuntimeError):
    """Base class for every typed receptor failure."""


class InvalidReceptorConfigError(ReceptorError):
    """The receptor configuration cannot produce a resolved receptor set."""


class UnsupportedReceptorBasisError(InvalidReceptorConfigError):
    """A requested receptor basis is outside the two Tier 5 bases."""


class UnsupportedFeedGeometryError(InvalidReceptorConfigError):
    """A feed geometry Tier 5 explicitly defers to a later tier."""


class AmbiguousOutputBasisError(InvalidReceptorConfigError):
    """A mixed array cannot resolve a common output basis under ``auto``."""


class ReceptorAssignmentError(InvalidReceptorConfigError):
    """An override does not name exactly one unclaimed canonical antenna."""


class UnsupportedBasisTransformError(ReceptorError):
    """A requested basis transform is not implemented."""


def _normalize_rotation_deg(value: object, *, field_name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{field_name} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    wrapped = math.remainder(value, 360.0)
    if wrapped == -180.0:
        wrapped = 180.0
    return wrapped + 0.0


def _require_basis(value: object, *, field_name: str) -> ReceptorBasis:
    if not isinstance(value, str) or value not in _SUPPORTED_BASES:
        raise UnsupportedReceptorBasisError(
            f"{field_name}={value!r} is not a supported receptor basis; Tier 5 "
            "supports exactly 'linear' and 'circular'."
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ResolvedReceptor:
    """One antenna's canonical receptor pair in the topocentric frame.

    Parameters
    ----------
    basis
        ``linear`` or ``circular``.
    feed_rotation_rad
        The resolved rotation offset, normalized into ``(-pi, pi]``.
    feed_array
        The pyuvdata feed identifiers, ``("x", "y")`` or ``("r", "l")``.
    feed_angle_rad
        The absolute pyuvdata feed angles, measured from North toward East.
    source
        Whether the values came from the default block or an override.
    """

    basis: ReceptorBasis
    feed_rotation_rad: float
    feed_array: tuple[str, str]
    feed_angle_rad: tuple[float, float]
    source: AntennaFieldSource

    def __post_init__(self) -> None:
        basis = _require_basis(self.basis, field_name="basis")
        rotation = self.feed_rotation_rad
        if type(rotation) is not float or not math.isfinite(rotation):
            raise ValueError("feed_rotation_rad must be a finite float")
        if not -math.pi < rotation <= math.pi:
            raise ValueError("feed_rotation_rad must lie in (-pi, pi]")
        feed_array = tuple(self.feed_array)
        if feed_array != _NOMINAL_FEED_ARRAY[basis]:
            raise ValueError(
                f"feed_array must be exactly {_NOMINAL_FEED_ARRAY[basis]!r} for a "
                f"{basis!r} receptor"
            )
        angles = tuple(self.feed_angle_rad)
        if len(angles) != 2 or any(
            type(angle) is not float or not math.isfinite(angle) for angle in angles
        ):
            raise ValueError("feed_angle_rad must be two finite floats")
        if not isinstance(self.source, AntennaFieldSource):
            raise TypeError("source must be an AntennaFieldSource")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "feed_array", feed_array)
        object.__setattr__(self, "feed_angle_rad", angles)


@dataclass(frozen=True, slots=True)
class ReceptorOverrideApplication:
    """One recorded application of a configured override, in declared order."""

    index: int
    antenna: AntennaId
    basis_applied: bool
    feed_rotation_applied: bool

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be a nonnegative integer")
        if type(self.antenna) is not AntennaId:
            raise TypeError("antenna must be an AntennaId")
        if type(self.basis_applied) is not bool:
            raise TypeError("basis_applied must be a bool")
        if type(self.feed_rotation_applied) is not bool:
            raise TypeError("feed_rotation_applied must be a bool")


@dataclass(frozen=True, slots=True)
class ReceptorProvenance:
    """Versioned explanation and fingerprint of one resolved receptor set."""

    schema_version: str
    requested_output_basis: Literal["auto", "linear", "circular"]
    output_basis_rule: OutputBasisRule
    override_applications: tuple[ReceptorOverrideApplication, ...]
    receptor_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _RECEPTOR_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_RECEPTOR_SCHEMA_VERSION!r}")
        if self.requested_output_basis not in _SUPPORTED_OUTPUT_REQUESTS:
            raise InvalidReceptorConfigError(
                f"receptors.output_basis={self.requested_output_basis!r} is not a "
                "supported request; use 'auto', 'linear', or 'circular'."
            )
        applications = tuple(self.override_applications)
        if any(type(item) is not ReceptorOverrideApplication for item in applications):
            raise TypeError(
                "override_applications must contain only "
                "ReceptorOverrideApplication values"
            )
        if not isinstance(self.receptor_sha256, str) or len(self.receptor_sha256) != 64:
            raise ValueError(
                "receptor_sha256 must be a 64-character hexadecimal SHA-256"
            )
        object.__setattr__(self, "override_applications", applications)


def _canonical_receptor_fingerprint_payload(
    output_basis: PolarizationBasis,
    receptor_by_antenna: Mapping[AntennaId, ResolvedReceptor],
) -> dict[str, Any]:
    """Return the exact canonical receptor hash payload in antenna order."""
    return {
        "schema_version": _RECEPTOR_SCHEMA_VERSION,
        "output_basis": output_basis,
        "receptors": [
            {
                "antenna_number": antenna_id.number,
                "antenna_name": antenna_id.name,
                "basis": receptor.basis,
                "feed_rotation_rad": receptor.feed_rotation_rad,
                "feed_array": list(receptor.feed_array),
                "feed_angle_rad": list(receptor.feed_angle_rad),
                "source": receptor.source.value,
            }
            for antenna_id, receptor in receptor_by_antenna.items()
        ],
    }


def _compute_receptor_sha256(
    output_basis: PolarizationBasis,
    receptor_by_antenna: Mapping[AntennaId, ResolvedReceptor],
) -> str:
    """Compute SHA-256 over the canonical UTF-8 JSON receptor payload."""
    encoded = json.dumps(
        _canonical_receptor_fingerprint_payload(output_basis, receptor_by_antenna),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolvedReceptorSet:
    """One array-wide receptor inventory and its single common output basis.

    Every antenna carries exactly two ideal orthogonal feeds sharing one basis.
    The whole array is reported in exactly one ``output_basis``.

    Modelling assumption
    --------------------
    Converting a circular-native antenna into a linear output basis (or the
    reverse) is exact **only** when both feeds are ideal, orthogonal, and share
    a common complex gain.  That holds in Tier 5 because the leakage (``D``)
    and gain (``G``) terms are disabled identity stubs.  When Tier 7 implements
    ``D``, the conversion becomes approximate and must be re-examined.

    Parameters
    ----------
    output_basis
        ``linear_xy`` or ``circular_rl``, resolved once before any solver work.
    receptor_by_antenna
        Read-only mapping in canonical instrument antenna order.
    provenance
        Requested basis, resolution rule, override applications, and fingerprint.
    """

    output_basis: PolarizationBasis
    receptor_by_antenna: Mapping[AntennaId, ResolvedReceptor]
    provenance: ReceptorProvenance

    def __post_init__(self) -> None:
        if self.output_basis not in _OUTPUT_BASIS_BY_NATIVE.values():
            raise InvalidReceptorConfigError(
                f"output_basis={self.output_basis!r} must be 'linear_xy' or "
                "'circular_rl'"
            )
        if not isinstance(self.receptor_by_antenna, Mapping):
            raise TypeError("receptor_by_antenna must be a mapping")
        canonical: dict[AntennaId, ResolvedReceptor] = {}
        for antenna_id, receptor in self.receptor_by_antenna.items():
            if type(antenna_id) is not AntennaId:
                raise TypeError("receptor_by_antenna keys must be AntennaId values")
            if type(receptor) is not ResolvedReceptor:
                raise TypeError(
                    "receptor_by_antenna values must be ResolvedReceptor values"
                )
            canonical[antenna_id] = receptor
        if not canonical:
            raise ValueError("receptor_by_antenna must contain at least one antenna")
        if type(self.provenance) is not ReceptorProvenance:
            raise TypeError("provenance must be a ReceptorProvenance")

        expected = _compute_receptor_sha256(self.output_basis, canonical)
        if self.provenance.receptor_sha256 != expected:
            raise ValueError(
                "provenance.receptor_sha256 does not match canonical receptor content"
            )
        object.__setattr__(
            self,
            "receptor_by_antenna",
            MappingProxyType(canonical),
        )

    @property
    def native_basis_counts(self) -> dict[str, int]:
        """Return a fresh count of native receptor bases present in the array."""
        counts = {"linear": 0, "circular": 0}
        for receptor in self.receptor_by_antenna.values():
            counts[receptor.basis] += 1
        return counts

    def to_snapshot(self) -> dict[str, Any]:
        """Return a fresh deterministic JSON-safe receptor snapshot."""
        payload = _canonical_receptor_fingerprint_payload(
            self.output_basis,
            self.receptor_by_antenna,
        )
        payload["receptor_sha256"] = self.provenance.receptor_sha256
        payload["requested_output_basis"] = self.provenance.requested_output_basis
        payload["output_basis_rule"] = self.provenance.output_basis_rule
        payload["native_basis_counts"] = self.native_basis_counts
        payload["override_applications"] = [
            {
                "index": item.index,
                "antenna_number": item.antenna.number,
                "antenna_name": item.antenna.name,
                "basis_applied": item.basis_applied,
                "feed_rotation_applied": item.feed_rotation_applied,
            }
            for item in self.provenance.override_applications
        ]
        return payload


def _feed_angles(basis: ReceptorBasis, rotation_rad: float) -> tuple[float, float]:
    """Return the absolute pyuvdata feed angles for one resolved receptor."""
    if basis == "linear":
        return (math.pi / 2.0 + rotation_rad, rotation_rad)
    return (rotation_rad, rotation_rad)


def _shown_reference(reference: object) -> str:
    from radiosim.io.instrument_config import (
        AntennaNameReference,
        AntennaNumberReference,
    )

    if type(reference) is AntennaNumberReference:
        return f"number {reference.number}"
    if type(reference) is AntennaNameReference:
        return f"name {reference.name!r}"
    raise TypeError("override antenna must be an exact Tier 2 AntennaReference")


def resolve_receptors(
    config: ReceptorsConfig,
    instrument: ResolvedInstrument,
) -> ResolvedReceptorSet:
    """Resolve one canonical receptor set from configuration and an instrument.

    Precedence, in order: the default definition for every antenna in canonical
    instrument order; then each override in declared order, replacing only the
    fields it declares; then the derived feed identifiers and angles; then the
    common output basis; then the deferred-geometry rejections; then the
    fingerprint.

    Parameters
    ----------
    config
        The strict ``receptors:`` input section.
    instrument
        The already-resolved canonical instrument.

    Returns
    -------
    ResolvedReceptorSet
        The complete receptor inventory and its provenance.

    Raises
    ------
    UnsupportedReceptorBasisError
        A basis outside ``linear``/``circular`` reached resolution.
    UnsupportedFeedGeometryError
        An antenna mount type Tier 5 defers to Tier 7.
    AmbiguousOutputBasisError
        A mixed array was requested under ``output_basis: auto``.
    ReceptorAssignmentError
        An override names an absent or already-overridden antenna.
    """
    from radiosim.io.receptor_config import ReceptorsConfig as _ReceptorsConfig

    if type(config) is not _ReceptorsConfig:
        raise TypeError("config must be a ReceptorsConfig")
    if type(instrument) is not ResolvedInstrument:
        raise TypeError("instrument must be a ResolvedInstrument")

    requested_output = config.output_basis
    if requested_output not in _SUPPORTED_OUTPUT_REQUESTS:
        raise InvalidReceptorConfigError(
            f"receptors.output_basis={requested_output!r} is not a supported "
            "request; use 'auto', 'linear', or 'circular'."
        )

    for antenna in instrument.antennas:
        mount_type = antenna.mount_type
        if mount_type is not None and mount_type != _SUPPORTED_MOUNT_TYPE:
            raise UnsupportedFeedGeometryError(
                f"mount_type={mount_type!r} is unsupported by Tier 5 receptors; "
                "time-dependent feed orientation requires the parallactic-angle "
                "term (Tier 7)."
            )

    default_basis = _require_basis(
        config.default.basis,
        field_name="receptors.default.basis",
    )
    default_rotation_deg = _normalize_rotation_deg(
        config.default.feed_rotation_deg,
        field_name="receptors.default.feed_rotation_deg",
    )

    staged: dict[AntennaId, tuple[ReceptorBasis, float, AntennaFieldSource]] = {}
    for antenna in instrument.antennas:
        antenna_id = AntennaId(antenna.id.number, antenna.id.name)
        staged[antenna_id] = (
            default_basis,
            default_rotation_deg,
            AntennaFieldSource.CONFIG_DEFAULT,
        )

    by_number = {antenna.id.number: antenna.id for antenna in instrument.antennas}
    by_name = {antenna.id.name: antenna.id for antenna in instrument.antennas}
    claimed_by: dict[AntennaId, int] = {}
    applications: list[ReceptorOverrideApplication] = []

    for index, override in enumerate(config.overrides):
        from radiosim.io.instrument_config import AntennaNumberReference

        reference = override.antenna
        if type(reference) is AntennaNumberReference:
            matched = by_number.get(reference.number)
        else:
            matched = by_name.get(getattr(reference, "name", None))
        if matched is None:
            raise ReceptorAssignmentError(
                f"receptors.overrides[{index}] references antenna "
                f"{_shown_reference(reference)}, which is absent from the "
                "resolved instrument."
            )
        antenna_id = AntennaId(matched.number, matched.name)
        previous = claimed_by.get(antenna_id)
        if previous is not None:
            raise ReceptorAssignmentError(
                f"receptors.overrides[{index}] duplicates antenna "
                f"{antenna_id.name!r}, already set by "
                f"receptors.overrides[{previous}]."
            )
        claimed_by[antenna_id] = index

        basis, rotation_deg, _source = staged[antenna_id]
        if override.basis is not None:
            basis = _require_basis(
                override.basis,
                field_name=f"receptors.overrides[{index}].basis",
            )
        if override.feed_rotation_deg is not None:
            rotation_deg = _normalize_rotation_deg(
                override.feed_rotation_deg,
                field_name=f"receptors.overrides[{index}].feed_rotation_deg",
            )
        staged[antenna_id] = (
            basis,
            rotation_deg,
            AntennaFieldSource.EXPLICIT_OVERRIDE,
        )
        applications.append(
            ReceptorOverrideApplication(
                index=index,
                antenna=antenna_id,
                basis_applied=override.basis is not None,
                feed_rotation_applied=override.feed_rotation_deg is not None,
            )
        )

    receptor_by_antenna: dict[AntennaId, ResolvedReceptor] = {}
    for antenna_id, (basis, rotation_deg, source) in staged.items():
        rotation_rad = math.radians(rotation_deg)
        receptor_by_antenna[antenna_id] = ResolvedReceptor(
            basis=basis,
            feed_rotation_rad=rotation_rad,
            feed_array=_NOMINAL_FEED_ARRAY[basis],
            feed_angle_rad=_feed_angles(basis, rotation_rad),
            source=source,
        )

    counts = {"linear": 0, "circular": 0}
    for receptor in receptor_by_antenna.values():
        counts[receptor.basis] += 1

    if requested_output == "auto":
        if counts["linear"] and counts["circular"]:
            raise AmbiguousOutputBasisError(
                "receptors.output_basis='auto' cannot resolve a mixed array "
                f"(linear antennas: {counts['linear']}, circular antennas: "
                f"{counts['circular']}); set receptors.output_basis to 'linear' "
                "or 'circular'."
            )
        native = "linear" if counts["linear"] else "circular"
        output_basis = _OUTPUT_BASIS_BY_NATIVE[native]
        rule: OutputBasisRule = f"auto_homogeneous_{native}"  # type: ignore[assignment]
    else:
        output_basis = _OUTPUT_BASIS_BY_NATIVE[requested_output]
        rule = f"explicit_{requested_output}"  # type: ignore[assignment]

    provenance = ReceptorProvenance(
        schema_version=_RECEPTOR_SCHEMA_VERSION,
        requested_output_basis=requested_output,
        output_basis_rule=rule,
        override_applications=tuple(applications),
        receptor_sha256=_compute_receptor_sha256(output_basis, receptor_by_antenna),
    )
    return ResolvedReceptorSet(
        output_basis=output_basis,
        receptor_by_antenna=receptor_by_antenna,
        provenance=provenance,
    )


__all__ = [
    "AmbiguousOutputBasisError",
    "InvalidReceptorConfigError",
    "OutputBasisRule",
    "ReceptorAssignmentError",
    "ReceptorBasis",
    "ReceptorError",
    "ReceptorOverrideApplication",
    "ReceptorProvenance",
    "ResolvedReceptor",
    "ResolvedReceptorSet",
    "UnsupportedBasisTransformError",
    "UnsupportedFeedGeometryError",
    "UnsupportedReceptorBasisError",
    "resolve_receptors",
]
