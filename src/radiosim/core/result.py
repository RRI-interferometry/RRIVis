"""Immutable canonical simulation-result models and fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import override

from radiosim.core.polarization_basis import (
    CORRELATION_LABELS,
    POLARIZATION_BASES,
    PolarizationBasis,
    basis_for_correlations,
    parallel_hand_indices,
)
from radiosim.core.runtime_config import FrozenMapping, json_safe_mapping

if TYPE_CHECKING:
    from radiosim.backends.base import ArrayBackend
    from radiosim.core.beam.models import LoadedBeamState
    from radiosim.core.instrument import (
        ResolvedBaselineSelection,
        ResolvedInstrument,
    )
    from radiosim.core.jones_terms import ResolvedJonesTerms
    from radiosim.core.phase_center import PhaseCenter
    from radiosim.core.receptor import ResolvedReceptorSet
    from radiosim.core.time_grid import ObservationTimeGrid

from radiosim.core.jones_terms import EMPTY_JONES_TERMS  # noqa: E402

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# Must equal ``radiosim.core.receptor``'s resolved receptor schema version; the
# equality is asserted by ``tests/unit/test_core/test_result.py`` so the two
# never drift silently.
_RECEPTOR_SCHEMA_VERSION = "1.0.0"
_RECEPTOR_ROW_KEYS = (
    "antenna_number",
    "antenna_name",
    "basis",
    "feed_rotation_rad",
    "feed_angle_rad",
)
_RECEPTOR_BASES = ("linear", "circular")


def _accepted_correlations_text() -> str:
    """Return the rejection text naming both accepted correlation tuples."""
    return " or ".join(repr(CORRELATION_LABELS[basis]) for basis in POLARIZATION_BASES)


class ResultError(RuntimeError):
    """Base class for canonical result failures."""


class ResultUnavailableError(ResultError):
    """A requested canonical result does not exist."""


class InvalidResultError(ResultError):
    """A result violates the canonical model contract."""


class ResultShapeError(InvalidResultError):
    """A result array shape is incoherent."""


class ResultCoordinateError(InvalidResultError):
    """A result coordinate is invalid."""


class InvalidPhaseCenterError(InvalidResultError):
    """A phase center is invalid or scientifically unsupported."""


class InvalidTimeGridError(InvalidResultError):
    """An observation time grid is invalid."""


class TimeGridLimitError(InvalidTimeGridError):
    """The requested time grid exceeds its allocation limit."""

    def __init__(self, *, requested_count: int, limit: int) -> None:
        self.requested_count = int(requested_count)
        self.limit = int(limit)
        super().__init__(
            f"requested {self.requested_count} time samples; limit is {self.limit}"
        )


def _reject_subclass(name: str) -> None:
    raise TypeError(f"{name} cannot be subclassed")


def _nonblank(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be nonblank")
    try:
        _ = normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must contain valid UTF-8 text") from exc
    return normalized


@dataclass(frozen=True, slots=True)
class BackendResultProvenance:
    """Backend request, realization, precision, and output dtype identity."""

    requested_backend: str
    actual_backend: str
    requested_precision: FrozenMapping | Mapping[str, object]
    actual_precision: FrozenMapping | Mapping[str, object]
    result_dtype: str
    #: Device the run actually executed on: ``"cpu"``, ``"gpu"``, or ``"tpu"``.
    #: An execution fact, not a capability claim
    #: (``Tier6HybridRuntimePlan.md`` Section 14.3).
    device_kind: str = "cpu"
    #: Whether the one compiled solver kernel was compiled for this run
    #: (``Tier6HybridRuntimePlan.md`` Sections 13.6, 14.3).
    compilation_used: bool = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("BackendResultProvenance")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_backend",
            _nonblank(self.requested_backend, field_name="requested_backend"),
        )
        object.__setattr__(
            self,
            "actual_backend",
            _nonblank(self.actual_backend, field_name="actual_backend"),
        )
        object.__setattr__(
            self,
            "requested_precision",
            json_safe_mapping(self.requested_precision),
        )
        object.__setattr__(
            self,
            "actual_precision",
            json_safe_mapping(self.actual_precision),
        )
        try:
            dtype = np.dtype(self.result_dtype)
        except TypeError as exc:
            raise InvalidResultError("result_dtype is not a NumPy dtype") from exc
        if dtype.kind != "c" or dtype.itemsize not in {8, 16, 32}:
            raise InvalidResultError(
                "result_dtype must be complex64, complex128, or complex256"
            )
        object.__setattr__(self, "result_dtype", dtype.name)
        if self.device_kind not in {"cpu", "gpu", "tpu"}:
            raise InvalidResultError("device_kind must be 'cpu', 'gpu', or 'tpu'")
        if type(self.compilation_used) is not bool:
            raise InvalidResultError("compilation_used must be a bool")

    def to_snapshot(self) -> FrozenMapping:
        return json_safe_mapping(
            {
                "requested_backend": self.requested_backend,
                "actual_backend": self.actual_backend,
                "requested_precision": self.requested_precision,
                "actual_precision": self.actual_precision,
                "result_dtype": self.result_dtype,
                "device_kind": self.device_kind,
                "compilation_used": self.compilation_used,
            }
        )


def _component_names(value: object) -> tuple[str, ...]:
    """Return a validated tuple of component names from a field or snapshot."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvalidResultError("components must be a sequence of names")
    names = tuple(cast(Sequence[object], value))
    if any(type(name) is not str for name in names):
        raise InvalidResultError("components must be a sequence of names")
    return cast(tuple[str, ...], names)


def _component_counts(value: object) -> tuple[int, ...]:
    """Return a validated tuple of nonnegative component element counts."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvalidResultError(
            "component_element_counts must be a sequence of integers"
        )
    counts: list[int] = []
    for item in cast(Sequence[object], value):
        if isinstance(item, (bool, np.bool_)) or type(item) is not int:
            raise InvalidResultError(
                "component_element_counts must be a sequence of integers"
            )
        if item < 0:
            raise InvalidResultError("component_element_counts must be nonnegative")
        counts.append(item)
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class SolverResultProvenance:
    """Solver identity and scientific execution convention."""

    solver: Literal["rime"]
    sky_representation: Literal["point_sources", "healpix_map", "hybrid"]
    convention: Literal["radiosim.rime-zenith-drift.v1"]
    execution_path: Literal["scalar", "polarized"]
    #: The solved components, in the fixed ``("point", "healpix")`` order of
    #: ``Tier6HybridRuntimePlan.md`` Section 8.3.  Deterministic, so it enters
    #: ``scientific_sha256`` and a hybrid result can never collide with a
    #: single-component result over the same instrument and sky numbers.
    components: tuple[str, ...]
    #: Each component's true element count, in the same order.
    component_element_counts: tuple[int, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("SolverResultProvenance")

    def __post_init__(self) -> None:
        from radiosim.core.hybrid import component_names_for_representation

        if self.solver != "rime":
            raise InvalidResultError("solver must be 'rime'")
        if self.sky_representation not in {"point_sources", "healpix_map", "hybrid"}:
            raise InvalidResultError("sky_representation is unsupported")
        if self.convention != "radiosim.rime-zenith-drift.v1":
            raise InvalidResultError("convention is unsupported")
        if self.execution_path not in {"scalar", "polarized"}:
            raise InvalidResultError("execution_path is unsupported")
        components = _component_names(self.components)
        if components != component_names_for_representation(self.sky_representation):
            raise InvalidResultError("components do not match sky_representation")
        counts = _component_counts(self.component_element_counts)
        if len(counts) != len(components):
            raise InvalidResultError(
                "component_element_counts must cover every component"
            )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "component_element_counts", counts)

    def to_snapshot(self) -> FrozenMapping:
        return json_safe_mapping(
            {
                "solver": self.solver,
                "sky_representation": self.sky_representation,
                "convention": self.convention,
                "execution_path": self.execution_path,
                "components": self.components,
                "component_element_counts": self.component_element_counts,
            }
        )


@dataclass(frozen=True, slots=True)
class MModeSolverResultProvenance:
    """The m-mode arm of Section 10's strict tagged solver record.

    ``docs/development/sci004_mmode_design.md`` Section 10 widens the solver
    record to a tagged union in which the current ``rime`` snapshot stays
    exactly as it is -- ``RIMESimulator`` serialization must be byte-identical
    after the union is introduced -- and an m-mode snapshot carries the exact
    common fields ``solver``, ``sky_representation``, ``convention``,
    ``execution_path``, ``components`` and ``component_element_counts``,
    followed by the exact m-mode block.

    Neither ``tangent_polarization_frame`` nor ``stokes_v_basis_bridge`` is
    nullable.  In M1 the first is the exact literal
    ``not_applicable_scalar_m1``, because the phase carries no polarized
    payload at all; after M2 it becomes the six-key Section 5.1 object.  The
    second is always ``radiosim.stokes-ne-theta-phi.v1``.

    The scientific snapshot excludes worker count and memory budget but
    includes every scientific convention and truncation dimension; backend,
    device, worker, chunk, library-version, IERS-path and timing details remain
    provenance.
    """

    snapshot: Any

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("MModeSolverResultProvenance")

    @property
    def solver(self) -> str:
        """Return the registry key this arm of the union records."""
        return "mmode"

    @property
    def sky_representation(self) -> str:
        """Return the solved representation, which stays solver provenance."""
        return str(self.snapshot.sky_representation)

    @property
    def convention(self) -> str:
        """Return the m-mode execution convention literal."""
        return str(self.snapshot.convention)

    @property
    def execution_path(self) -> str:
        """Return the solved execution path, a Section 10 common field."""
        return str(self.snapshot.execution_path)

    @property
    def components(self) -> tuple[str, ...]:
        """Return the solved components, a Section 10 common field.

        ``docs/development/sci004_mmode_design.md`` Section 10 gives the m-mode
        arm "the exact common fields ``solver``, ``sky_representation``,
        ``convention``, ``execution_path``, ``components`` and
        ``component_element_counts``", so the two arms of the tagged union
        present the same attribute surface for them.  Every consumer that reads
        a solved component list -- the standard projection's HISTORY lines, most
        of all -- then works on either arm without asking which one it holds.
        """
        return tuple(str(name) for name in self.snapshot.components)

    @property
    def component_element_counts(self) -> tuple[int, ...]:
        """Return each solved component's element count, likewise."""
        return tuple(int(count) for count in self.snapshot.component_element_counts)

    @property
    def direct_gate(self) -> Any:
        """Return Section 7.3's every-run complete frozen-direct comparison."""
        return self.snapshot.direct_gate

    @property
    def frozen_gauss128_cube_sha256(self) -> str:
        """Return the certificate's retained final frozen direct-cube identity."""
        return str(self.snapshot.frozen_gauss128_cube_sha256)

    @property
    def frozen_enclosure_error_cube_sha256(self) -> str:
        """Return the certificate's retained frozen enclosure-error identity."""
        return str(self.snapshot.frozen_enclosure_error_cube_sha256)

    def as_mapping(self) -> dict[str, Any]:
        """Return Section 10's exact tagged m-mode snapshot, in key order."""
        return dict(self.snapshot.as_mapping())

    def to_snapshot(self) -> FrozenMapping:
        """Return the JSON-safe scientific snapshot the fingerprint hashes."""
        return json_safe_mapping(self.as_mapping())


def _json_shaped(value: Any) -> Any:
    """Return one frozen snapshot value in the JSON shape it was written in."""
    if isinstance(value, Mapping):
        return {str(key): _json_shaped(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_shaped(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class LoadedMModeSolverSnapshot:
    """A deserialized Section 10 m-mode snapshot, replayed from stored bytes.

    ``docs/development/sci004_mmode_design.md`` Section 10: "Reader round trips
    must reconstruct and authenticate the m-mode solver snapshot; a reader that
    silently labels it ``rime`` fails acceptance."  Reconstruction needs an
    object with the snapshot's own surface, and the *solver's*
    ``MModeSolverSnapshot`` is not it: that class also carries the Section 7.3
    gate record and the frozen certificate cubes, which Section 10 deliberately
    keeps out of the serialized twenty-key set.  This is the reader's arm --
    exactly the stored keys, in the stored order, and an explicit refusal for
    the run-time-only fields rather than a plausible-looking zero.
    """

    stored: Mapping[str, Any]

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("LoadedMModeSolverSnapshot")

    def __post_init__(self) -> None:
        mapping = dict(self.stored)
        if tuple(mapping) != MMODE_SOLVER_SNAPSHOT_KEYS:
            raise InvalidResultError(
                "a stored m-mode solver snapshot must carry Section 10's exact "
                "key set, in order"
            )
        object.__setattr__(self, "stored", MappingProxyType(mapping))

    @property
    def solver(self) -> str:
        """Return the registry key the stored snapshot records."""
        return str(self.stored["solver"])

    @property
    def sky_representation(self) -> str:
        return str(self.stored["sky_representation"])

    @property
    def convention(self) -> str:
        return str(self.stored["convention"])

    @property
    def execution_path(self) -> str:
        return str(self.stored["execution_path"])

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.stored["components"])

    @property
    def component_element_counts(self) -> tuple[int, ...]:
        return tuple(int(count) for count in self.stored["component_element_counts"])

    @property
    def direct_gate(self) -> Any:
        """Refuse: Section 10 keeps the gate record out of the stored snapshot."""
        raise InvalidResultError(
            "a deserialized m-mode snapshot carries no Section 7.3 gate record; "
            "read it from the run that produced the file"
        )

    def as_mapping(self) -> dict[str, Any]:
        """Return the stored Section 10 snapshot, in its stored key order.

        The values come back in their JSON shape.  A frozen snapshot stores an
        array as a tuple, but the written document held a JSON array and the
        in-memory arm's ``as_mapping`` produces a list, so a round trip that
        handed back tuples would compare unequal to the record it replays for
        no reason a reader could act on.
        """
        return {key: _json_shaped(value) for key, value in self.stored.items()}

    def to_snapshot(self) -> dict[str, Any]:
        return self.as_mapping()


#: ``docs/development/sci004_mmode_design.md`` Section 11's four new
#: characterized families, in the order the memo prints them.  "The set names
#: exactly the capability accepted M2 licenses through the public solve path."
#: The former HEALPix, hybrid and east-X families were removed by the accepted
#: accepted-capability-characterization-envelope correction: two published
#: identically zero cubes, one silently dropped its diffuse half, and the last
#: reproduced ``mmode_point_full_stokes`` byte for byte.
MMODE_CHARACTERIZATION_FAMILIES: Final[tuple[str, ...]] = (
    "mmode_single_scalar_mode",
    "mmode_point_stokes_i",
    "mmode_point_full_stokes",
    "mmode_circular_receptor",
)

#: Section 11's two namespaced characterization domains.  "The family record's
#: grid and input-identity digests use the namespaced domains
#: ``radiosim.sci004.characterization-time.v1`` and
#: ``radiosim.sci004.characterization-input.v1``, computed from the retained
#: ``SimulationResult`` exactly as the strict validator re-derives them; Section
#: 14.0's solver-internal domains do not apply to a result-derived record."
#: The distinction is not cosmetic: Section 14.0's ``radiosim.mmode-utc-grid.v1``
#: preimage is the exact-turn manifest with its exposure edges, which a
#: published result does not carry, and reusing that domain over a different
#: preimage would be a collision rather than a shortcut.
MMODE_CHARACTERIZATION_TIME_DOMAIN: Final = "radiosim.sci004.characterization-time.v1"
MMODE_CHARACTERIZATION_INPUT_DOMAIN: Final = "radiosim.sci004.characterization-input.v1"

#: Section 11's family record key set, in the order the record is built.
MMODE_CHARACTERIZATION_RECORD_KEYS: Final[tuple[str, ...]] = (
    "family_id",
    "raw_cube_sha256",
    "scientific_sha256",
    "solver_snapshot",
    "era_utc_grid_sha256",
    "harmonic_index_table_sha256",
    "input_identity_sha256",
)

#: Section 11's initial harvest: "The initial harvest binds exactly the
#: platform/Python cells this phase's acceptance actually runs on; every other
#: cell and every newly observed NumPy/OpenBLAS dispatch class enters afterwards
#: by the standing admission discipline, exactly as the accepted AVX-512
#: admissions did."
#:
#: These four ``scientific_sha256`` values were measured on 2026-08-24 from the
#: live family fixtures on ``osx-arm64-py311``, at the
#: accepted Section 7.3 dimensions -- ``49`` sidereal samples, ``lmax = mmax =
#: 16``, ``quadrature_nside = 8``, one 50 MHz channel, the qualified two-antenna
#: compact geometry -- and each family passed the every-run two-tier gate on
#: real numbers rather than through its exact-zero corner:
#:
#: ===========================  ==================  ==============  ========
#: family                       tier-1a max / limit  deficit q>h>f   factor
#: ===========================  ==================  ==============  ========
#: ``mmode_single_scalar_mode``  1.210e-13/2.261e-08  0.6931>0.1847>0.1170  5.925
#: ``mmode_point_stokes_i``      3.473e-13/6.391e-08  1.5480>0.4771>0.2987  5.182
#: ``mmode_point_full_stokes``   3.464e-13/7.799e-08  2.5113>1.0108>0.4113  6.105
#: ``mmode_circular_receptor``   3.686e-13/7.029e-08  1.7028>0.9033>0.3286  5.182
#: ===========================  ==================  ==============  ========
#:
#: No other environment cell is claimed.  A second cell is admitted only by the
#: standing adjudication, never by appending a digest CI printed.
_MMODE_CHARACTERIZATION_OBSERVATIONS: Final[
    Mapping[str, Mapping[str, tuple[str, ...]]]
] = MappingProxyType(
    {
        "mmode_single_scalar_mode": MappingProxyType(
            {
                "osx-arm64-py311": (
                    "a8852fd181e8f1ddd08e24e01066bcede2e1e1541fba9a067d47f0febc51c344",
                )
            }
        ),
        "mmode_point_stokes_i": MappingProxyType(
            {
                "osx-arm64-py311": (
                    "1750b098f5bb8894bde8aed2b0b52addeff0f1cc44c7d980930624b552fe5640",
                )
            }
        ),
        "mmode_point_full_stokes": MappingProxyType(
            {
                "osx-arm64-py311": (
                    "2ac26b66787b5844de488bb5f8cd161e14b15282c5b66759262bc4e39b94d262",
                )
            }
        ),
        "mmode_circular_receptor": MappingProxyType(
            {
                "osx-arm64-py311": (
                    "e4c4abb439f5c08b451e30bce96bb28eb9c1fd5fc4f23b15e27d1f007e14979b",
                )
            }
        ),
    }
)


def mmode_characterization_observation_set(
    family_id: str,
) -> dict[str, tuple[str, ...]]:
    """Return one Section 11 family's CI-001 observation set.

    The value is a mapping from environment cell to the admitted
    ``scientific_sha256`` values for that cell -- never a bare digest, because a
    single-machine digest is not a pin.  A cell absent from the mapping has not
    been harvested; adding one is the standing admission discipline's work, not
    this function's.
    """
    if family_id not in MMODE_CHARACTERIZATION_FAMILIES:
        raise InvalidResultError(f"unknown characterization family {family_id!r}")
    return {
        cell: tuple(digests)
        for cell, digests in _MMODE_CHARACTERIZATION_OBSERVATIONS[family_id].items()
    }


def _characterization_time_manifest(grid: Any) -> dict[str, Any]:
    """Return the result-derived UTC-grid manifest the time domain covers."""
    from radiosim.core.mmode.types import f64be

    jd1 = np.asarray(grid.utc_jd1, dtype=np.float64)
    jd2 = np.asarray(grid.utc_jd2, dtype=np.float64)
    widths = np.asarray(grid.integration_time_seconds, dtype=np.float64)
    return {
        "schema_version": MMODE_CHARACTERIZATION_TIME_DOMAIN,
        "axis_order": ["sample"],
        "shape": [int(jd1.shape[0])],
        "interval_semantics": str(grid.interval_semantics),
        "start_time_iso": str(grid.start_time_iso),
        "center_jd1_f64be": [f64be(value) for value in jd1.tolist()],
        "center_jd2_f64be": [f64be(value) for value in jd2.tolist()],
        "integration_time_seconds_f64be": [f64be(value) for value in widths.tolist()],
    }


def _characterization_input_manifest(
    result: SimulationResult,
    family_id: str,
) -> dict[str, Any]:
    """Return the result-derived input manifest the input domain covers.

    Two families differ exactly in their inputs, so the manifest joins the
    resolved configuration with the instrument, receptor and beam identities the
    result already carries.  ``mmode_circular_receptor`` and
    ``mmode_point_full_stokes`` share a sky and differ only in the receptor
    basis, which is why the receptor fingerprint is part of the preimage rather
    than an afterthought.
    """
    from radiosim.core.mmode.types import f64be

    return {
        "schema_version": MMODE_CHARACTERIZATION_INPUT_DOMAIN,
        "family_id": family_id,
        "resolved_config": _json_shaped(dict(result.resolved_config)),
        "instrument_sha256": result.instrument.provenance.instrument_sha256,
        "receptor_sha256": result.receptors.provenance.receptor_sha256,
        "beam_loaded_fingerprint": result.beam_state.loaded_fingerprint,
        "correlations": list(result.correlations),
        "polarization_basis": str(result.polarization_basis),
        "frequencies_hz_f64be": [
            f64be(value)
            for value in np.asarray(result.frequencies_hz, dtype=np.float64).tolist()
        ],
    }


def mmode_characterization_record(
    result: SimulationResult,
    *,
    family_id: str,
) -> dict[str, Any]:
    """Return Section 11's characterization record for one m-mode family.

    "Each family records the raw cube, ``scientific_sha256``, solver snapshot,
    ERA/UTC grid, harmonic index table, and input identity."  Every part is
    derived from the retained result, deterministically, so two calls on one
    result are equal and the strict validator re-derives each digest from the
    same preimage.

    The raw cube uses Section 14.0's own ``A`` rule under the generic
    visibility-cube domain; the harmonic index table is the packed block table
    the solver itself builds for the run's ``(lmax, mmax)``, under its own
    Section 14.0 domain; the two remaining digests use Section 11's namespaced
    characterization domains.
    """
    from radiosim.core.mmode.harmonics import scalar_packed_block_table
    from radiosim.core.mmode.types import array_digest, object_digest

    if family_id not in MMODE_CHARACTERIZATION_FAMILIES:
        raise InvalidResultError(f"unknown characterization family {family_id!r}")
    if type(result.solver) is not MModeSolverResultProvenance:
        raise InvalidResultError(
            "a characterization family record requires an m-mode result"
        )
    snapshot = result.solver.as_mapping()
    cube = np.asarray(result.visibilities)
    dtype = "complex64-be" if cube.dtype.itemsize == 8 else "complex128-be"
    table = scalar_packed_block_table(
        lmax=int(snapshot["lmax"]), mmax=int(snapshot["mmax"])
    )
    return {
        "family_id": family_id,
        "raw_cube_sha256": array_digest(
            "radiosim.mmode-visibility-cube.v1",
            "visibility_cube",
            ["time", "baseline", "frequency", "correlation"],
            "Jy",
            cube,
            dtype=dtype,
        ),
        "scientific_sha256": result.scientific_sha256,
        "solver_snapshot": snapshot,
        "era_utc_grid_sha256": object_digest(
            MMODE_CHARACTERIZATION_TIME_DOMAIN,
            _characterization_time_manifest(result.time_grid),
        ),
        "harmonic_index_table_sha256": table.block_table_sha256,
        "input_identity_sha256": object_digest(
            MMODE_CHARACTERIZATION_INPUT_DOMAIN,
            _characterization_input_manifest(result, family_id),
        ),
    }


#: The Section 10 tagged solver union.  ``rime`` results keep the unchanged
#: ``SolverResultProvenance`` arm, byte for byte.
SolverProvenanceUnion = SolverResultProvenance | MModeSolverResultProvenance

#: Section 10's exact m-mode snapshot key set, in order.  Unknown or missing
#: keys reject on both the written and the read side.
MMODE_SOLVER_SNAPSHOT_KEYS: tuple[str, ...] = (
    "solver",
    "sky_representation",
    "convention",
    "execution_path",
    "components",
    "component_element_counts",
    "time_grid_convention",
    "frame_model",
    "harmonic_convention",
    "sidereal_samples",
    "lmax",
    "mmax",
    "quadrature_nside",
    "quadrature_policy",
    "truncation_policy",
    "tangent_polarization_frame",
    "stokes_v_basis_bridge",
    "iers_table_sha256",
    "frame_certificate_sha256",
    "transform_execution_policy",
)


@dataclass(frozen=True, slots=True)
class ResultPerformance:
    """Finite nonnegative timings for one result construction."""

    setup_seconds: float
    solver_seconds: float
    #: Wall time of the ``point`` component; ``0.0`` when it did not run.
    #: Component timings are nondeterministic and therefore stay out of both
    #: fingerprints (``Tier6HybridRuntimePlan.md`` Section 9.4).
    solver_point_seconds: float
    #: Wall time of the ``healpix`` component; ``0.0`` when it did not run.
    solver_healpix_seconds: float
    result_construction_seconds: float
    host_transfer_seconds: float
    total_seconds: float

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("ResultPerformance")

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (bool, np.bool_)):
                raise InvalidResultError(f"{field.name} must be finite and nonnegative")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InvalidResultError(
                    f"{field.name} must be finite and nonnegative"
                ) from exc
            if not math.isfinite(number) or number < 0.0:
                raise InvalidResultError(f"{field.name} must be finite and nonnegative")
            normalized[field.name] = number
            object.__setattr__(self, field.name, number)
        minimum_total = (
            normalized["setup_seconds"]
            + normalized["solver_seconds"]
            + normalized["result_construction_seconds"]
            + normalized["host_transfer_seconds"]
        )
        allowance = 32.0 * np.finfo(np.float64).eps * max(1.0, minimum_total)
        if normalized["total_seconds"] + allowance < minimum_total:
            raise InvalidResultError(
                "total_seconds is not coherent with component times"
            )
        component_total = (
            normalized["solver_point_seconds"] + normalized["solver_healpix_seconds"]
        )
        component_allowance = (
            32.0 * np.finfo(np.float64).eps * max(1.0, component_total)
        )
        if component_total > normalized["solver_seconds"] + component_allowance:
            raise InvalidResultError(
                "solver component times are not coherent with solver_seconds"
            )

    def to_snapshot(self) -> FrozenMapping:
        return json_safe_mapping(
            {field.name: getattr(self, field.name) for field in fields(self)}
        )


def _immutable_array(
    value: object,
    *,
    dtype: DTypeLike | None = None,
) -> NDArray[np.generic]:
    try:
        array = np.array(
            value,
            dtype=dtype,
            order="C",
            copy=True,
            subok=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidResultError("result array could not be normalized") from exc
    if array.dtype.hasobject:
        raise InvalidResultError("object arrays are not supported")
    return np.ndarray(array.shape, dtype=array.dtype, buffer=array.tobytes(order="C"))


def _coordinates(
    frequencies_hz: object,
    channel_widths_hz: object,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies = cast(
        NDArray[np.float64],
        _immutable_array(frequencies_hz, dtype=np.float64),
    )
    widths = cast(
        NDArray[np.float64],
        _immutable_array(channel_widths_hz, dtype=np.float64),
    )
    if frequencies.ndim != 1 or not frequencies.size:
        raise ResultCoordinateError("frequencies_hz must be nonempty and 1-dimensional")
    if widths.shape != frequencies.shape:
        raise ResultCoordinateError("channel_widths_hz must match frequencies_hz shape")
    if (
        not np.all(np.isfinite(frequencies))
        or not np.all(frequencies > 0.0)
        or not np.all(np.diff(frequencies) > 0.0)
    ):
        raise ResultCoordinateError(
            "frequencies_hz must be finite, positive, and strictly increasing"
        )
    if not np.all(np.isfinite(widths)) or not np.all(widths > 0.0):
        raise ResultCoordinateError("channel_widths_hz must be finite and positive")
    return frequencies, widths


def _history(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise InvalidResultError("history must be a sequence of strings")
    return tuple(
        _nonblank(item, field_name=f"history[{index}]")
        for index, item in enumerate(value)
    )


def _runtime_snapshot(value: object) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise TypeError("resolved_config must be a mapping")
    mapping = cast(Mapping[str, object], value)
    return json_safe_mapping(
        {key: item for key, item in mapping.items() if key != "workflow"}
    )


def _optional_snapshot(
    value: object,
) -> FrozenMapping | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("configuration_provenance must be a mapping or None")
    mapping = cast(Mapping[str, object], value)
    snapshot = {key: item for key, item in mapping.items() if key != "workflow"}
    input_snapshot = snapshot.get("input_snapshot")
    if isinstance(input_snapshot, Mapping):
        snapshot["input_snapshot"] = {
            key: item
            for key, item in cast(Mapping[str, object], input_snapshot).items()
            if key != "workflow"
        }
    for field_name in ("override_origins", "path_resolutions"):
        provenance_values = snapshot.get(field_name)
        if isinstance(provenance_values, Mapping):
            snapshot[field_name] = {
                key: item
                for key, item in cast(
                    Mapping[str, object],
                    provenance_values,
                ).items()
                if not key.startswith("workflow.")
            }
    return json_safe_mapping(snapshot)


def _json_tree(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_tree(item) for key, item in mapping.items()}
    if isinstance(value, tuple):
        sequence = cast(Sequence[object], value)
        return [_json_tree(item) for item in sequence]
    if isinstance(value, list):
        sequence = cast(Sequence[object], value)
        return [_json_tree(item) for item in sequence]
    if isinstance(value, np.generic):
        return cast(object, value.item())
    return value


def _tagged_update(digest: Any, tag: str, payload: bytes) -> None:
    tag_bytes = tag.encode("utf-8")
    digest.update(len(tag_bytes).to_bytes(8, "little"))
    digest.update(tag_bytes)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _hash_json(digest: Any, tag: str, value: object) -> None:
    try:
        encoded = json.dumps(
            _json_tree(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise InvalidResultError(f"{tag} is not compact finite JSON") from exc
    _tagged_update(digest, tag, encoded)


def _hash_array(digest: Any, tag: str, value: np.ndarray) -> None:
    dtype = value.dtype.newbyteorder("<")
    canonical = np.array(value, dtype=dtype, order="C", copy=True, subok=False)
    _hash_json(
        digest,
        f"{tag}.metadata",
        {"dtype": dtype.str, "shape": list(value.shape)},
    )
    _tagged_update(digest, f"{tag}.data", canonical.tobytes(order="C"))


def _package_version() -> str:
    try:
        return version("radiosim")
    except PackageNotFoundError:
        return "unknown"


def _receptor_result_snapshot(snapshot: object) -> dict[str, object]:
    """Return the exact result-bearing projection of a receptor snapshot.

    The projection is the same set of values the HDF5 ``receptors/`` group
    stores (Section 21), so an in-memory result and a deserialized one produce
    byte-identical fingerprint input.  Configuration-only fields of
    :meth:`~radiosim.core.receptor.ResolvedReceptorSet.to_snapshot` -- the
    requested basis, the resolution rule, the override applications, and the
    per-antenna ``source`` and derived ``feed_array`` -- are excluded: they
    explain how the receptor set was chosen, not what it is.
    """
    if not isinstance(snapshot, Mapping):
        raise InvalidResultError("receptor snapshot must be a mapping")
    typed = cast(Mapping[str, object], snapshot)
    schema_version = typed.get("schema_version", _RECEPTOR_SCHEMA_VERSION)
    if schema_version != _RECEPTOR_SCHEMA_VERSION:
        raise InvalidResultError(
            f"receptor snapshot schema_version must be {_RECEPTOR_SCHEMA_VERSION!r}"
        )
    output_basis = typed.get("output_basis")
    if output_basis not in CORRELATION_LABELS:
        raise InvalidResultError(
            f"receptor snapshot output_basis must be one of {POLARIZATION_BASES!r}"
        )
    receptor_sha256 = typed.get("receptor_sha256")
    if type(receptor_sha256) is not str or _SHA256.fullmatch(receptor_sha256) is None:
        raise InvalidResultError(
            "receptor snapshot receptor_sha256 must be a lower-case SHA-256"
        )
    rows_value = typed.get("receptors")
    if isinstance(rows_value, (str, bytes)) or not isinstance(rows_value, Sequence):
        raise InvalidResultError("receptor snapshot receptors must be a sequence")
    rows: list[dict[str, object]] = []
    seen_numbers: set[int] = set()
    for index, row in enumerate(cast(Sequence[object], rows_value)):
        if not isinstance(row, Mapping):
            raise InvalidResultError(f"receptor snapshot receptors[{index}] is invalid")
        typed_row = cast(Mapping[str, object], row)
        missing = [key for key in _RECEPTOR_ROW_KEYS if key not in typed_row]
        if missing:
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] is missing {missing[0]}"
            )
        number = typed_row["antenna_number"]
        name = typed_row["antenna_name"]
        basis = typed_row["basis"]
        rotation = typed_row["feed_rotation_rad"]
        angles = typed_row["feed_angle_rad"]
        if type(number) is not int or number in seen_numbers:
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] antenna_number is invalid"
            )
        seen_numbers.add(number)
        if type(name) is not str or not name:
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] antenna_name is invalid"
            )
        if basis not in _RECEPTOR_BASES:
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] basis must be "
                f"one of {_RECEPTOR_BASES!r}"
            )
        if type(rotation) is not float or not math.isfinite(rotation):
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] feed_rotation_rad is invalid"
            )
        if isinstance(angles, (str, bytes)) or not isinstance(angles, Sequence):
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] feed_angle_rad is invalid"
            )
        angle_values = [
            float(value)
            for value in cast(Sequence[object], angles)
            if type(value) is float and math.isfinite(value)
        ]
        if len(angle_values) != 2 or len(cast(Sequence[object], angles)) != 2:
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] feed_angle_rad is invalid"
            )
        rows.append(
            {
                "antenna_number": number,
                "antenna_name": name,
                "basis": basis,
                "feed_rotation_rad": rotation,
                "feed_angle_rad": angle_values,
            }
        )
    if not rows:
        raise InvalidResultError("receptor snapshot must contain at least one antenna")
    return {
        "schema_version": _RECEPTOR_SCHEMA_VERSION,
        "output_basis": output_basis,
        "receptor_sha256": receptor_sha256,
        "receptors": rows,
    }


def _result_receptor_snapshot(
    result: SimulationResult | LoadedSimulationResult,
) -> dict[str, object]:
    receptors = result.receptors
    if isinstance(receptors, Mapping):
        return _receptor_result_snapshot(receptors)
    return _receptor_result_snapshot(receptors.to_snapshot())


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


_INSTRUMENT_LOCATION_SCIENTIFIC_KEYS = (
    "longitude_deg",
    "latitude_deg",
    "height_m",
    "itrs_xyz_m",
    "source",
    "location_source",
)
_INSTRUMENT_ANTENNA_SCIENTIFIC_KEYS = (
    "number",
    "name",
    "position_enu_m",
    "diameter_m",
    "mount_type",
    "beam_id",
)
_INSTRUMENT_ANTENNA_SOURCE_KEYS = (
    "identity_source",
    "position_source",
    "diameter_source",
    "mount_source",
    "beam_id_source",
)


def _scientific_instrument_projection(
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    """Project an instrument snapshot onto its transport-free scientific facts.

    The scientific fingerprint must depend only on the resolved science, never
    on where a source file happened to live on disk, so equal science hashes
    equally in every checkout.  This projection keeps exactly the facts that
    :func:`radiosim.core.instrument._canonical_instrument_fingerprint_payload`
    declares scientific -- the resolved values plus the field-source labels
    that explain them, bound by ``instrument_sha256`` -- and drops the
    transport facts that snapshot alongside them: the source path and locator,
    the raw source hash, dependency versions, the registry policy, pre-override
    source diameters, per-antenna source-record locators, and location
    transport diagnostics.  The full snapshot, transport facts included,
    remains stored on the result and inside ``provenance_sha256``.
    """
    source = _mapping_or_empty(snapshot.get("source"))
    location = _mapping_or_empty(snapshot.get("location"))
    antennas_value = snapshot.get("antennas")
    antennas: Sequence[object]
    if isinstance(antennas_value, (str, bytes)) or not isinstance(
        antennas_value, Sequence
    ):
        antennas = ()
    else:
        antennas = cast(Sequence[object], antennas_value)
    projected_antennas: list[dict[str, object]] = []
    for antenna_value in antennas:
        antenna = _mapping_or_empty(antenna_value)
        provenance = _mapping_or_empty(antenna.get("provenance"))
        projected: dict[str, object] = {
            key: antenna.get(key) for key in _INSTRUMENT_ANTENNA_SCIENTIFIC_KEYS
        }
        projected["provenance"] = {
            key: provenance.get(key) for key in _INSTRUMENT_ANTENNA_SOURCE_KEYS
        }
        projected_antennas.append(projected)
    return {
        "schema_version": snapshot.get("schema_version"),
        "instrument_sha256": snapshot.get("instrument_sha256"),
        "name": snapshot.get("name"),
        "source": {
            "telescope_name_source": source.get("telescope_name_source"),
        },
        "location": {
            key: location.get(key) for key in _INSTRUMENT_LOCATION_SCIENTIFIC_KEYS
        },
        "antennas": projected_antennas,
    }


# Beam snapshot keys excluded from the scientific hash.  The FITS source
# paths and the config locator that authored them are filesystem transport,
# not science.  The four beam fingerprints no longer hash any path (a FITS
# ``definition_fingerprint`` binds only the load settings; content binds at
# load time through the handler ``scientific_fingerprint``), but they remain
# excluded as redundant digests: every fact they bind survives the projection
# directly as sibling keys (analytic model parameters, FITS handler content
# hashes and validated metadata, and the antenna-to-definition mapping), so
# hashing them would only couple ``scientific_sha256`` to fingerprint-payload
# evolution.
_BEAM_TRANSPORT_KEYS = frozenset(
    {
        "path",
        "resolved_path",
        "path_provenance_key",
        "definition_fingerprint",
        "assignment_fingerprint",
        "state_fingerprint",
        "loaded_fingerprint",
    }
)


def _scientific_beam_projection(value: object) -> object:
    """Recursively drop filesystem-transport keys from a beam snapshot."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {
            key: _scientific_beam_projection(item)
            for key, item in mapping.items()
            if key not in _BEAM_TRANSPORT_KEYS
        }
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        sequence = cast(Sequence[object], value)
        return [_scientific_beam_projection(item) for item in sequence]
    return value


#: The exact keys a Jones snapshot carries.  Checked rather than assumed, for
#: the same reason the receptor snapshot is: a loaded file is untrusted input,
#: and a snapshot that silently gained or lost a key would change the
#: fingerprint without changing the science.
_JONES_SNAPSHOT_KEYS = (
    "schema_version",
    "enabled_terms",
    "chain_order",
    "term_snapshots",
    "mount_types",
    "jones_sha256",
)


def _jones_result_snapshot(jones_terms: object) -> Mapping[str, object]:
    """Return the JSON-safe Jones snapshot for a live resolved inventory.

    Returns an empty mapping when no optional term was configured, so equivalent
    current runs with an empty optional-term inventory add no Jones snapshot to
    the scientific fingerprint (I1).  Always-present factors are fingerprinted
    independently.
    """
    from radiosim.core.jones_terms import ResolvedJonesTerms

    if type(jones_terms) is not ResolvedJonesTerms:
        raise TypeError("jones_terms must be an exact ResolvedJonesTerms")
    return jones_terms.to_snapshot()


def _loaded_jones_snapshot(
    snapshot: Mapping[str, object] | None,
) -> Mapping[str, object]:
    """Validate a Jones snapshot read back from a file.

    ``None`` and an empty mapping both mean "no optional Jones or baseline term
    was enabled", which is how a file written without a ``jones/`` group is read
    (Section 25.2).  Anything else must carry exactly the six recorded keys and
    a well-formed digest.
    """
    if snapshot is None:
        return MappingProxyType({})
    if not isinstance(snapshot, Mapping):
        raise InvalidResultError("jones snapshot must be a mapping")
    mapping = cast(Mapping[str, object], snapshot)
    if not mapping:
        return MappingProxyType({})
    if set(mapping) != set(_JONES_SNAPSHOT_KEYS):
        raise InvalidResultError("jones snapshot has unexpected fields")
    digest = mapping["jones_sha256"]
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        raise InvalidResultError("jones_sha256 must be a lower-case SHA-256")
    enabled = mapping["enabled_terms"]
    if isinstance(enabled, (str, bytes)) or not isinstance(enabled, Sequence):
        raise InvalidResultError("jones enabled_terms must be a sequence")
    if not enabled:
        raise InvalidResultError(
            "a jones snapshot with no enabled term must be absent, not empty"
        )
    return json_safe_mapping(mapping)


def _jones_summary_block(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Return the bounded summary view of one Jones snapshot.

    Bounded on purpose: the per-term parameters can be arbitrarily large (a
    tabulated bandpass carries every node), and the summary is a metadata
    document.  The full record lives in the HDF5 ``jones/`` group.
    """
    if not snapshot:
        return {"enabled_terms": [], "chain_order": [], "jones_sha256": None}
    return {
        "enabled_terms": list(cast(Sequence[str], snapshot["enabled_terms"])),
        "chain_order": list(cast(Sequence[str], snapshot["chain_order"])),
        "jones_sha256": snapshot["jones_sha256"],
    }


def _scientific_hash(
    *,
    visibilities: np.ndarray,
    flags: np.ndarray,
    weights: np.ndarray,
    time_grid: ObservationTimeGrid,
    frequencies: np.ndarray,
    widths: np.ndarray,
    correlations: tuple[str, ...],
    polarization_basis: PolarizationBasis,
    receptor_snapshot: Mapping[str, object],
    phase_snapshot: Mapping[str, object],
    instrument_snapshot: Mapping[str, object],
    selection_snapshot: Mapping[str, object],
    beam_snapshot: Mapping[str, object],
    solver_snapshot: Mapping[str, object],
    jones_snapshot: Mapping[str, object] = MappingProxyType({}),
) -> str:
    digest = hashlib.sha256()
    _hash_json(digest, "schema", "radiosim.result.v1")
    for tag, array in (
        ("visibilities", visibilities),
        ("flags", flags),
        ("weights", weights),
        ("time.utc_jd1", time_grid.utc_jd1),
        ("time.utc_jd2", time_grid.utc_jd2),
        ("time.integration_time_seconds", time_grid.integration_time_seconds),
        ("frequency_hz", frequencies),
        ("channel_width_hz", widths),
    ):
        _hash_array(digest, tag, array)
    _hash_json(digest, "correlations", correlations)
    _hash_json(digest, "polarization_basis", polarization_basis)
    _hash_json(digest, "receptor", receptor_snapshot)
    _hash_json(
        digest,
        "instrument",
        _scientific_instrument_projection(instrument_snapshot),
    )
    _hash_json(digest, "selection", selection_snapshot)
    _hash_json(digest, "beam", _scientific_beam_projection(beam_snapshot))
    _hash_json(digest, "phase_center", phase_snapshot)
    _hash_json(digest, "solver", solver_snapshot)
    # The Jones record is hashed only when a term was actually configured.  An
    # empty snapshot contributes *nothing* -- not an empty object, not a null --
    # so the current empty optional-term inventory adds no Jones component
    # (``Tier7JonesSciencePlan.md`` Section 25.1, invariant I1).  Other
    # scientific inputs, including the always-present receptor convention,
    # remain part of the digest; this is not a compatibility promise for
    # pre-SCI-006 fingerprints.  Hashing an empty placeholder would add a
    # scientifically meaningless distinction between otherwise equivalent
    # current runs.
    if jones_snapshot:
        _hash_json(digest, "jones", jones_snapshot)
    return digest.hexdigest()


def _provenance_hash(
    *,
    scientific_sha256: str,
    backend_snapshot: Mapping[str, object],
    resolved_config: Mapping[str, object],
    configuration_provenance: Mapping[str, object] | None,
    history: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    _hash_json(digest, "scientific_sha256", scientific_sha256)
    _hash_json(digest, "backend", backend_snapshot)
    _hash_json(digest, "resolved_config", resolved_config)
    _hash_json(digest, "configuration_provenance", configuration_provenance)
    _hash_json(digest, "package_version", _package_version())
    _hash_json(digest, "history", history)
    return digest.hexdigest()


class _ResultMethods:
    schema_version: str
    visibilities: np.ndarray
    flags: np.ndarray
    weights: np.ndarray
    time_grid: ObservationTimeGrid
    frequencies_hz: np.ndarray
    channel_widths_hz: np.ndarray
    correlations: tuple[str, str, str, str]
    polarization_basis: PolarizationBasis
    receptors: ResolvedReceptorSet | FrozenMapping
    phase_center: PhaseCenter
    scientific_sha256: str
    provenance_sha256: str

    @override
    def __hash__(self) -> int:
        raise TypeError("canonical results are unhashable")

    def scientifically_equal(
        self,
        other: SimulationResult | LoadedSimulationResult,
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> bool:
        if type(other) not in {SimulationResult, LoadedSimulationResult}:
            return False
        if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
            raise ValueError("rtol and atol must be finite and nonnegative")
        if (
            self.schema_version != other.schema_version
            or self.visibilities.dtype != other.visibilities.dtype
            or self.visibilities.shape != other.visibilities.shape
            or self.flags.shape != other.flags.shape
            or self.weights.dtype != other.weights.dtype
            or self.correlations != other.correlations
            or self.polarization_basis != other.polarization_basis
            or _json_tree(self.phase_center.to_snapshot())
            != _json_tree(other.phase_center.to_snapshot())
            or _json_tree(
                _identity_snapshots(
                    cast(SimulationResult | LoadedSimulationResult, self)
                )
            )
            != _json_tree(_identity_snapshots(other))
        ):
            return False
        exact_pairs = (
            (self.flags, other.flags),
            (self.time_grid.utc_jd1, other.time_grid.utc_jd1),
            (self.time_grid.utc_jd2, other.time_grid.utc_jd2),
            (
                self.time_grid.integration_time_seconds,
                other.time_grid.integration_time_seconds,
            ),
            (self.frequencies_hz, other.frequencies_hz),
            (self.channel_widths_hz, other.channel_widths_hz),
        )
        if any(not np.array_equal(left, right) for left, right in exact_pairs):
            return False
        return bool(
            np.allclose(
                self.visibilities,
                other.visibilities,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
            and np.allclose(
                self.weights,
                other.weights,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
        )

    def stokes_i(self) -> np.ndarray:
        """Return a newly owned parallel-hand sum for the published basis.

        The two indices are derived from :attr:`correlations` through
        :func:`~radiosim.core.polarization_basis.parallel_hand_indices`, so the
        sum is ``XX + YY`` in ``linear_xy`` and ``RR + LL`` in ``circular_rl``
        without either literal appearing here.
        """
        first, second = parallel_hand_indices(self.correlations)
        if self.visibilities.shape[-1] != len(self.correlations):
            raise InvalidResultError(
                "the correlation axis does not match the correlation labels"
            )
        return np.array(
            self.visibilities[..., first] + self.visibilities[..., second],
            copy=True,
            order="C",
            subok=False,
        )

    def to_summary_snapshot(self) -> dict[str, object]:
        """Return bounded JSON-safe metadata without embedding science arrays."""
        receptor_snapshot = _result_receptor_snapshot(
            cast("SimulationResult | LoadedSimulationResult", self)
        )
        receptor_rows = cast(
            list[Mapping[str, object]],
            receptor_snapshot["receptors"],
        )
        native_counts = dict.fromkeys(_RECEPTOR_BASES, 0)
        for row in receptor_rows:
            native_counts[cast(str, row["basis"])] += 1
        return {
            "schema_version": self.schema_version,
            "shape": list(self.visibilities.shape),
            "dtype": self.visibilities.dtype.name,
            "correlations": list(self.correlations),
            "polarization_basis": self.polarization_basis,
            "receptor": {
                "output_basis": receptor_snapshot["output_basis"],
                "receptor_sha256": receptor_snapshot["receptor_sha256"],
                "native_basis_counts": native_counts,
                "antenna_count": len(receptor_rows),
            },
            "time": {
                "start_time_iso": self.time_grid.start_time_iso,
                "duration_seconds": self.time_grid.duration_seconds,
                "cadence_seconds": self.time_grid.cadence_seconds,
                "sample_count": len(self.time_grid),
            },
            "frequency": {
                "channel_count": int(self.frequencies_hz.size),
                "minimum_hz": float(self.frequencies_hz[0]),
                "maximum_hz": float(self.frequencies_hz[-1]),
            },
            "array_summaries": {
                "visibilities": {
                    "shape": list(self.visibilities.shape),
                    "dtype": self.visibilities.dtype.name,
                },
                "flags": {
                    "shape": list(self.flags.shape),
                    "dtype": self.flags.dtype.name,
                },
                "weights": {
                    "shape": list(self.weights.shape),
                    "dtype": self.weights.dtype.name,
                },
            },
            "jones": _jones_summary_block(self.jones),
            "scientific_sha256": self.scientific_sha256,
            "provenance_sha256": self.provenance_sha256,
        }


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SimulationResult(_ResultMethods):
    """A canonical result retaining exact live runtime identity objects."""

    schema_version: str
    visibilities: np.ndarray
    flags: np.ndarray
    weights: np.ndarray
    time_grid: ObservationTimeGrid
    frequencies_hz: np.ndarray
    channel_widths_hz: np.ndarray
    correlations: tuple[str, str, str, str]
    polarization_basis: PolarizationBasis
    instrument: ResolvedInstrument
    selection: ResolvedBaselineSelection
    beam_state: LoadedBeamState
    receptors: ResolvedReceptorSet
    jones: FrozenMapping
    phase_center: PhaseCenter
    backend: BackendResultProvenance
    solver: SolverProvenanceUnion
    resolved_config: FrozenMapping
    configuration_provenance: FrozenMapping | None
    performance: ResultPerformance
    history: tuple[str, ...]
    scientific_sha256: str
    provenance_sha256: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("SimulationResult")

    def __init__(self) -> None:
        raise TypeError("SimulationResult must be built by build_simulation_result")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class LoadedSimulationResult(_ResultMethods):
    """A canonical deserialized result containing frozen identity snapshots."""

    schema_version: str
    visibilities: np.ndarray
    flags: np.ndarray
    weights: np.ndarray
    time_grid: ObservationTimeGrid
    frequencies_hz: np.ndarray
    channel_widths_hz: np.ndarray
    correlations: tuple[str, str, str, str]
    polarization_basis: PolarizationBasis
    phase_center: PhaseCenter
    instrument_snapshot: FrozenMapping
    selection_snapshot: FrozenMapping
    beam_snapshot: FrozenMapping
    receptors: FrozenMapping
    jones: FrozenMapping
    backend_snapshot: FrozenMapping
    solver_snapshot: FrozenMapping
    resolved_config_snapshot: FrozenMapping
    configuration_provenance_snapshot: FrozenMapping | None
    performance: ResultPerformance
    history: tuple[str, ...]
    scientific_sha256: str
    provenance_sha256: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("LoadedSimulationResult")

    def __init__(self) -> None:
        raise TypeError(
            "LoadedSimulationResult must be built by build_loaded_simulation_result"
        )

    @property
    def solver(self) -> SolverProvenanceUnion:
        """Return the stored solver record as the arm of the union it names.

        ``docs/development/sci004_mmode_design.md`` Section 10: "Reader round
        trips must reconstruct and authenticate the m-mode solver snapshot; a
        reader that silently labels it ``rime`` fails acceptance."  The stored
        mapping stays exactly where it was -- ``solver_snapshot`` is unchanged --
        and this reconstructs the typed arm from it, so a caller holding a
        deserialized result reads the same attribute surface it would have read
        on the in-memory one.  The arm is chosen by the stored ``solver`` key
        alone, and an unknown key is refused rather than defaulted.
        """
        stored = dict(self.solver_snapshot)
        key = stored.get("solver")
        if key == "mmode":
            return MModeSolverResultProvenance(
                snapshot=LoadedMModeSolverSnapshot(stored=stored)
            )
        if key == "rime":
            try:
                return SolverResultProvenance(**cast(dict[str, Any], stored))
            except (TypeError, ValueError) as exc:
                raise InvalidResultError(
                    "the stored rime solver snapshot is invalid"
                ) from exc
        raise InvalidResultError(f"unknown stored solver arm {key!r}")


def _identity_snapshots(
    result: SimulationResult | LoadedSimulationResult,
) -> tuple[object, ...]:
    """Return the scientific identity snapshots used by ``scientifically_equal``.

    The instrument and beam snapshots pass through the same transport-free
    scientific projections as ``scientific_sha256``, so equal science compares
    equal regardless of which checkout resolved the source files.
    """
    receptor_snapshot = _result_receptor_snapshot(result)
    if isinstance(result, SimulationResult):
        instrument_snapshot: Mapping[str, object] = result.instrument.to_snapshot()
        selection_snapshot: Mapping[str, object] = result.selection.to_snapshot()
        beam_snapshot: Mapping[str, object] = result.beam_state.to_snapshot()
        backend_snapshot: Mapping[str, object] = result.backend.to_snapshot()
        solver_snapshot: Mapping[str, object] = result.solver.to_snapshot()
    else:
        instrument_snapshot = result.instrument_snapshot
        selection_snapshot = result.selection_snapshot
        beam_snapshot = result.beam_snapshot
        backend_snapshot = result.backend_snapshot
        solver_snapshot = result.solver_snapshot
    return (
        _scientific_instrument_projection(instrument_snapshot),
        selection_snapshot,
        _scientific_beam_projection(beam_snapshot),
        receptor_snapshot,
        backend_snapshot,
        solver_snapshot,
        dict(result.jones),
    )


def _require_exact(value: object, expected: type[Any], field_name: str) -> None:
    if type(value) is not expected:
        raise TypeError(f"{field_name} must be an exact {expected.__name__}")


def _require_backend(value: object) -> ArrayBackend:
    from radiosim.backends.base import ArrayBackend

    if not isinstance(value, ArrayBackend):
        raise TypeError("backend must be an ArrayBackend")
    return value


def _required_snapshot(value: object, *, field_name: str) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return json_safe_mapping(cast(Mapping[str, object], value))


def _snapshot_mapping(
    snapshot: Mapping[str, object],
    key: str,
    *,
    field_name: str,
) -> Mapping[str, object]:
    value = snapshot.get(key)
    if not isinstance(value, Mapping):
        raise InvalidResultError(f"{field_name}.{key} must be a mapping")
    return cast(Mapping[str, object], value)


def _snapshot_sequence(
    snapshot: Mapping[str, object],
    key: str,
    *,
    field_name: str,
) -> Sequence[object]:
    value = snapshot.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvalidResultError(f"{field_name}.{key} must be a sequence")
    return cast(Sequence[object], value)


def _validate_loaded_identity_snapshots(
    *,
    instrument: FrozenMapping,
    selection: FrozenMapping,
    beam: FrozenMapping,
    receptor: Mapping[str, object],
    backend: FrozenMapping,
    solver: FrozenMapping,
    visibility_dtype: np.dtype[Any],
    baseline_count: int,
) -> None:
    if instrument.get("schema_version") != "radiosim.instrument.v1":
        raise InvalidResultError("instrument_snapshot has an invalid schema")
    instrument_sha256 = instrument.get("instrument_sha256")
    if (
        type(instrument_sha256) is not str
        or _SHA256.fullmatch(instrument_sha256) is None
    ):
        raise InvalidResultError("instrument_snapshot has an invalid fingerprint")
    antennas = _snapshot_sequence(
        instrument,
        "antennas",
        field_name="instrument_snapshot",
    )
    if not antennas:
        raise InvalidResultError("instrument_snapshot.antennas must be nonempty")
    antenna_numbers: set[int] = set()
    antenna_identity: list[tuple[object, object]] = []
    for index, antenna in enumerate(antennas):
        if not isinstance(antenna, Mapping):
            raise InvalidResultError(
                f"instrument_snapshot.antennas[{index}] must be a mapping"
            )
        typed_antenna = cast(Mapping[str, object], antenna)
        number = typed_antenna.get("number")
        if type(number) is not int:
            raise InvalidResultError(
                "instrument_snapshot antenna numbers must be unique integers"
            )
        if number in antenna_numbers:
            raise InvalidResultError(
                "instrument_snapshot antenna numbers must be unique integers"
            )
        antenna_numbers.add(number)
        antenna_identity.append((number, typed_antenna.get("name")))

    receptor_rows = cast(Sequence[Mapping[str, object]], receptor["receptors"])
    if [
        (row["antenna_number"], row["antenna_name"]) for row in receptor_rows
    ] != antenna_identity:
        raise InvalidResultError(
            "the receptor snapshot does not cover instrument_snapshot in "
            "canonical antenna order"
        )

    if selection.get("schema_version") != "radiosim.baseline-selection.v1":
        raise InvalidResultError("selection_snapshot has an invalid schema")
    selected_ids = _snapshot_sequence(
        selection,
        "selected_ids",
        field_name="selection_snapshot",
    )
    if len(selected_ids) != baseline_count:
        raise ResultShapeError(
            "selection_snapshot baseline count does not match visibilities"
        )
    seen_pairs: set[tuple[int, int]] = set()
    for index, pair_value in enumerate(selected_ids):
        if isinstance(pair_value, (str, bytes)) or not isinstance(pair_value, Sequence):
            raise InvalidResultError(
                f"selection_snapshot.selected_ids[{index}] must contain two integers"
            )
        pair_items = cast(Sequence[object], pair_value)
        if len(pair_items) != 2:
            raise InvalidResultError(
                f"selection_snapshot.selected_ids[{index}] must contain two integers"
            )
        ant1, ant2 = pair_items[0], pair_items[1]
        if type(ant1) is not int or type(ant2) is not int:
            raise InvalidResultError(
                f"selection_snapshot.selected_ids[{index}] must contain two integers"
            )
        pair = (ant1, ant2)
        if ant1 not in antenna_numbers or ant2 not in antenna_numbers:
            raise InvalidResultError(
                "selection_snapshot contains an antenna outside instrument_snapshot"
            )
        if pair in seen_pairs:
            raise InvalidResultError(
                "selection_snapshot contains duplicate selected baselines"
            )
        seen_pairs.add(pair)

    resolved_beam = _snapshot_mapping(
        beam,
        "resolved",
        field_name="beam_snapshot",
    )
    if resolved_beam.get("instrument_fingerprint") != instrument_sha256:
        raise InvalidResultError("beam_snapshot does not belong to instrument_snapshot")

    backend_fields = {
        "requested_backend",
        "actual_backend",
        "requested_precision",
        "actual_precision",
        "result_dtype",
        # Tier 6H (plan Section 14.3): execution facts, not scientific ones.
        "device_kind",
        "compilation_used",
    }
    if set(backend) != backend_fields:
        raise InvalidResultError("backend_snapshot has unexpected fields")
    try:
        backend_identity = BackendResultProvenance(**dict(backend))
    except (TypeError, ValueError, InvalidResultError) as exc:
        raise InvalidResultError("backend_snapshot is invalid") from exc
    if np.dtype(backend_identity.result_dtype) != visibility_dtype:
        raise InvalidResultError(
            "backend_snapshot result dtype does not match visibilities"
        )

    if solver.get("solver") == "mmode":
        # Section 10's m-mode arm of the tagged union.  Unknown or missing keys
        # reject, and written and read solver records use the same rule, so a
        # reader can never silently relabel an m-mode record as ``rime``.
        if tuple(solver) != MMODE_SOLVER_SNAPSHOT_KEYS:
            raise InvalidResultError("solver_snapshot has unexpected fields")
        for key in (
            "tangent_polarization_frame",
            "stokes_v_basis_bridge",
            "frame_certificate_sha256",
            "iers_table_sha256",
        ):
            if solver.get(key) in (None, ""):
                raise InvalidResultError("solver_snapshot is invalid")
        return

    solver_fields = {
        "solver",
        "sky_representation",
        "convention",
        "execution_path",
        "components",
        "component_element_counts",
    }
    if set(solver) != solver_fields:
        raise InvalidResultError("solver_snapshot has unexpected fields")
    try:
        _ = SolverResultProvenance(**dict(solver))
    except (TypeError, ValueError, InvalidResultError) as exc:
        raise InvalidResultError("solver_snapshot is invalid") from exc


def _assign(target: object, **values: object) -> None:
    for key, value in values.items():
        object.__setattr__(target, key, value)


def _validate_common_shapes(
    *,
    visibilities: np.ndarray,
    flags: np.ndarray,
    weights: np.ndarray,
    time_grid: ObservationTimeGrid,
    frequencies: np.ndarray,
    baseline_count: int | None,
) -> None:
    if visibilities.ndim != 4 or any(size <= 0 for size in visibilities.shape):
        raise ResultShapeError("visibilities must have nonempty shape (T,B,F,4)")
    if visibilities.shape[-1] != 4:
        raise ResultShapeError("the correlation axis must contain exactly four values")
    if flags.shape != visibilities.shape or weights.shape != visibilities.shape:
        raise ResultShapeError("flags and weights must match visibility shape")
    if visibilities.shape[0] != len(time_grid):
        raise ResultShapeError("visibility time axis does not match time_grid")
    if visibilities.shape[2] != frequencies.size:
        raise ResultShapeError("visibility frequency axis does not match coordinates")
    if baseline_count is not None and visibilities.shape[1] != baseline_count:
        raise ResultShapeError("visibility baseline axis does not match selection")
    if not np.all(np.isfinite(visibilities)) or not np.all(np.isfinite(weights)):
        raise InvalidResultError("result arrays must contain only finite values")


def build_simulation_result(
    *,
    receptor_visibilities: object,
    backend: ArrayBackend,
    time_grid: ObservationTimeGrid,
    frequencies_hz: Sequence[float],
    channel_widths_hz: Sequence[float],
    instrument: ResolvedInstrument,
    selection: ResolvedBaselineSelection,
    beam_state: LoadedBeamState,
    receptors: ResolvedReceptorSet,
    jones_terms: ResolvedJonesTerms = EMPTY_JONES_TERMS,
    phase_center: PhaseCenter,
    backend_provenance: BackendResultProvenance,
    solver_provenance: SolverProvenanceUnion,
    resolved_config: Mapping[str, object],
    configuration_provenance: Mapping[str, object] | None,
    performance: ResultPerformance,
    history: Sequence[str] = (),
) -> SimulationResult:
    """Validate, transfer once, flatten, harden, and fingerprint a result."""
    from radiosim.core.beam.models import LoadedBeamState
    from radiosim.core.instrument import ResolvedBaselineSelection, ResolvedInstrument
    from radiosim.core.phase_center import PhaseCenter
    from radiosim.core.receptor import ResolvedReceptorSet
    from radiosim.core.time_grid import ObservationTimeGrid

    construction_started = time.perf_counter()
    checked_backend = _require_backend(backend)
    for value, expected, field_name in (
        (time_grid, ObservationTimeGrid, "time_grid"),
        (instrument, ResolvedInstrument, "instrument"),
        (selection, ResolvedBaselineSelection, "selection"),
        (beam_state, LoadedBeamState, "beam_state"),
        (receptors, ResolvedReceptorSet, "receptors"),
        (phase_center, PhaseCenter, "phase_center"),
        (backend_provenance, BackendResultProvenance, "backend_provenance"),
        (performance, ResultPerformance, "performance"),
    ):
        _require_exact(value, expected, field_name)
    # Section 10's tagged union: either arm is accepted, and neither is a
    # subclass of the other, so the check stays an exact-type check.
    if type(solver_provenance) not in (
        SolverResultProvenance,
        MModeSolverResultProvenance,
    ):
        raise TypeError(
            "solver_provenance must be an exact SolverResultProvenance or "
            "MModeSolverResultProvenance"
        )
    if (
        selection.provenance.instrument_sha256
        != instrument.provenance.instrument_sha256
    ):
        raise InvalidResultError("selection does not belong to instrument")
    antenna_ids = {antenna.id for antenna in instrument.antennas}
    if any(
        baseline.ant1 not in antenna_ids or baseline.ant2 not in antenna_ids
        for baseline in selection.baselines
    ):
        raise InvalidResultError("selection contains a baseline outside instrument")
    if set(receptors.receptor_by_antenna) != antenna_ids:
        raise InvalidResultError("receptors do not belong to instrument")
    polarization_basis = receptors.output_basis
    if polarization_basis not in CORRELATION_LABELS:
        raise InvalidResultError(
            f"polarization_basis must be one of {POLARIZATION_BASES!r}"
        )
    receptor_snapshot = _receptor_result_snapshot(receptors.to_snapshot())
    jones_snapshot = _jones_result_snapshot(jones_terms)

    frequencies, widths = _coordinates(frequencies_hz, channel_widths_hz)
    transfer_started = time.perf_counter()
    try:
        host = checked_backend.to_numpy(receptor_visibilities)
    except Exception as exc:
        raise InvalidResultError("backend host transfer failed") from exc
    host_transfer_seconds = time.perf_counter() - transfer_started
    if type(host) is not np.ndarray:
        host = np.asarray(host)
    expected_shape = (
        len(time_grid),
        len(selection.baselines),
        frequencies.size,
        2,
        2,
    )
    if host.shape != expected_shape:
        raise ResultShapeError(
            f"receptor_visibilities must have shape {expected_shape}, got {host.shape}"
        )
    dtype = np.dtype(backend_provenance.result_dtype)
    if dtype.kind != "c":
        raise InvalidResultError("result dtype must be complex")
    try:
        cast_host = np.array(host, dtype=dtype, order="C", copy=True, subok=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidResultError(
            "receptor visibilities cannot use result dtype"
        ) from exc
    if not np.all(np.isfinite(cast_host)):
        raise InvalidResultError("receptor visibilities must be finite")
    flattened = cast_host.reshape(expected_shape[:3] + (4,))
    visibilities = _immutable_array(flattened, dtype=dtype)
    flags = _immutable_array(np.zeros(visibilities.shape, dtype=np.bool_))
    weight_dtype = np.float32 if dtype.itemsize == 8 else np.float64
    weights = _immutable_array(
        np.ones(visibilities.shape, dtype=weight_dtype),
        dtype=weight_dtype,
    )
    _validate_common_shapes(
        visibilities=visibilities,
        flags=flags,
        weights=weights,
        time_grid=time_grid,
        frequencies=frequencies,
        baseline_count=len(selection.baselines),
    )
    frozen_config = _runtime_snapshot(resolved_config)
    frozen_provenance = _optional_snapshot(configuration_provenance)
    frozen_history = _history(history)
    instrument_snapshot = instrument.to_snapshot()
    selection_snapshot = selection.to_snapshot()
    beam_snapshot = beam_state.to_snapshot()
    backend_snapshot = backend_provenance.to_snapshot()
    solver_snapshot = solver_provenance.to_snapshot()
    scientific = _scientific_hash(
        visibilities=visibilities,
        flags=flags,
        weights=weights,
        time_grid=time_grid,
        frequencies=frequencies,
        widths=widths,
        correlations=CORRELATION_LABELS[polarization_basis],
        polarization_basis=polarization_basis,
        receptor_snapshot=receptor_snapshot,
        phase_snapshot=phase_center.to_snapshot(),
        instrument_snapshot=instrument_snapshot,
        selection_snapshot=selection_snapshot,
        beam_snapshot=beam_snapshot,
        solver_snapshot=solver_snapshot,
        jones_snapshot=jones_snapshot,
    )
    provenance_hash = _provenance_hash(
        scientific_sha256=scientific,
        backend_snapshot=backend_snapshot,
        resolved_config=frozen_config,
        configuration_provenance=frozen_provenance,
        history=frozen_history,
    )
    construction_elapsed = time.perf_counter() - construction_started
    result_construction_seconds = max(
        0.0,
        construction_elapsed - host_transfer_seconds,
    )
    measured_performance = ResultPerformance(
        setup_seconds=performance.setup_seconds,
        solver_seconds=performance.solver_seconds,
        solver_point_seconds=performance.solver_point_seconds,
        solver_healpix_seconds=performance.solver_healpix_seconds,
        result_construction_seconds=result_construction_seconds,
        host_transfer_seconds=host_transfer_seconds,
        total_seconds=performance.total_seconds + construction_elapsed,
    )
    result = object.__new__(SimulationResult)
    _assign(
        result,
        schema_version="radiosim.result.v1",
        visibilities=visibilities,
        flags=flags,
        weights=weights,
        time_grid=time_grid,
        frequencies_hz=frequencies,
        channel_widths_hz=widths,
        correlations=CORRELATION_LABELS[polarization_basis],
        polarization_basis=polarization_basis,
        instrument=instrument,
        selection=selection,
        beam_state=beam_state,
        receptors=receptors,
        jones=json_safe_mapping(jones_snapshot),
        phase_center=phase_center,
        backend=backend_provenance,
        solver=solver_provenance,
        resolved_config=frozen_config,
        configuration_provenance=frozen_provenance,
        performance=measured_performance,
        history=frozen_history,
        scientific_sha256=scientific,
        provenance_sha256=provenance_hash,
    )
    return result


def _performance_from_snapshot(snapshot: object) -> ResultPerformance:
    if not isinstance(snapshot, Mapping):
        raise TypeError("performance_snapshot must be a mapping")
    typed_snapshot = cast(Mapping[str, object], snapshot)
    expected = {field.name for field in fields(ResultPerformance)}
    if set(typed_snapshot) != expected:
        raise InvalidResultError("performance_snapshot has unexpected fields")
    return ResultPerformance(**cast(dict[str, Any], dict(typed_snapshot)))


def build_loaded_simulation_result(
    *,
    visibilities: object,
    flags: object,
    weights: object,
    time_grid: ObservationTimeGrid,
    frequencies_hz: object,
    channel_widths_hz: object,
    correlations: Sequence[str],
    phase_center: PhaseCenter,
    instrument_snapshot: Mapping[str, object],
    selection_snapshot: Mapping[str, object],
    beam_snapshot: Mapping[str, object],
    receptors_snapshot: Mapping[str, object],
    backend_snapshot: Mapping[str, object],
    solver_snapshot: Mapping[str, object],
    resolved_config_snapshot: Mapping[str, object],
    configuration_provenance_snapshot: Mapping[str, object] | None,
    performance_snapshot: Mapping[str, object],
    history: Sequence[str],
    jones_snapshot: Mapping[str, object] | None = None,
    expected_scientific_sha256: str,
    expected_provenance_sha256: str,
) -> LoadedSimulationResult:
    """Build and independently verify an immutable deserialized result."""
    from radiosim.core.phase_center import PhaseCenter
    from radiosim.core.time_grid import ObservationTimeGrid

    _require_exact(time_grid, ObservationTimeGrid, "time_grid")
    _require_exact(phase_center, PhaseCenter, "phase_center")
    if isinstance(correlations, (str, bytes)) or not isinstance(correlations, Sequence):
        raise InvalidResultError("correlations must be a sequence of labels")
    correlation_labels = tuple(
        label for label in cast(Sequence[object], correlations) if type(label) is str
    )
    try:
        polarization_basis = basis_for_correlations(correlation_labels)
    except (TypeError, ValueError) as exc:
        raise InvalidResultError(
            f"correlations must be exactly {_accepted_correlations_text()}"
        ) from exc
    loaded_jones = _loaded_jones_snapshot(jones_snapshot)
    receptor_snapshot = _receptor_result_snapshot(receptors_snapshot)
    if receptor_snapshot["output_basis"] != polarization_basis:
        raise InvalidResultError(
            "the receptor output basis does not match the correlation labels"
        )
    frequency_array, width_array = _coordinates(frequencies_hz, channel_widths_hz)
    visibility_array = _immutable_array(visibilities)
    if visibility_array.dtype.kind != "c" or visibility_array.dtype.itemsize not in {
        8,
        16,
        32,
    }:
        raise InvalidResultError("visibilities must use a supported complex dtype")
    try:
        flag_input = np.asarray(flags)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidResultError("flags could not be normalized") from exc
    if flag_input.dtype != np.dtype("bool"):
        raise InvalidResultError("flags must use bool dtype")
    flag_array = _immutable_array(flag_input, dtype=np.bool_)
    expected_weight_dtype = (
        np.dtype("float32")
        if visibility_array.dtype.itemsize == 8
        else np.dtype("float64")
    )
    weight_input = np.asarray(weights)
    if weight_input.dtype != expected_weight_dtype:
        raise InvalidResultError("weights dtype does not match visibility dtype")
    weight_array = _immutable_array(weight_input, dtype=expected_weight_dtype)
    _validate_common_shapes(
        visibilities=visibility_array,
        flags=flag_array,
        weights=weight_array,
        time_grid=time_grid,
        frequencies=frequency_array,
        baseline_count=None,
    )
    snapshots: list[FrozenMapping] = []
    for field_name, snapshot in (
        ("instrument_snapshot", instrument_snapshot),
        ("selection_snapshot", selection_snapshot),
        ("beam_snapshot", beam_snapshot),
        ("backend_snapshot", backend_snapshot),
        ("solver_snapshot", solver_snapshot),
    ):
        snapshots.append(_required_snapshot(snapshot, field_name=field_name))
    frozen_instrument, frozen_selection, frozen_beam, frozen_backend, frozen_solver = (
        snapshots
    )
    _validate_loaded_identity_snapshots(
        instrument=frozen_instrument,
        selection=frozen_selection,
        beam=frozen_beam,
        receptor=receptor_snapshot,
        backend=frozen_backend,
        solver=frozen_solver,
        visibility_dtype=visibility_array.dtype,
        baseline_count=visibility_array.shape[1],
    )
    frozen_config = _runtime_snapshot(resolved_config_snapshot)
    frozen_configuration_provenance = _optional_snapshot(
        configuration_provenance_snapshot
    )
    performance = _performance_from_snapshot(performance_snapshot)
    frozen_history = _history(history)
    for field_name, expected in (
        ("expected_scientific_sha256", expected_scientific_sha256),
        ("expected_provenance_sha256", expected_provenance_sha256),
    ):
        if type(expected) is not str or _SHA256.fullmatch(expected) is None:
            raise InvalidResultError(f"{field_name} must be a lower-case SHA-256")
    scientific = _scientific_hash(
        visibilities=visibility_array,
        flags=flag_array,
        weights=weight_array,
        time_grid=time_grid,
        frequencies=frequency_array,
        widths=width_array,
        correlations=correlation_labels,
        polarization_basis=polarization_basis,
        receptor_snapshot=receptor_snapshot,
        phase_snapshot=phase_center.to_snapshot(),
        instrument_snapshot=frozen_instrument,
        selection_snapshot=frozen_selection,
        beam_snapshot=frozen_beam,
        solver_snapshot=frozen_solver,
        jones_snapshot=loaded_jones,
    )
    provenance_hash = _provenance_hash(
        scientific_sha256=scientific,
        backend_snapshot=frozen_backend,
        resolved_config=frozen_config,
        configuration_provenance=frozen_configuration_provenance,
        history=frozen_history,
    )
    if scientific != expected_scientific_sha256:
        raise InvalidResultError("scientific fingerprint mismatch")
    if provenance_hash != expected_provenance_sha256:
        raise InvalidResultError("provenance fingerprint mismatch")
    result = object.__new__(LoadedSimulationResult)
    _assign(
        result,
        schema_version="radiosim.result.v1",
        visibilities=visibility_array,
        flags=flag_array,
        weights=weight_array,
        time_grid=time_grid,
        frequencies_hz=frequency_array,
        channel_widths_hz=width_array,
        correlations=correlation_labels,
        polarization_basis=polarization_basis,
        phase_center=phase_center,
        instrument_snapshot=frozen_instrument,
        selection_snapshot=frozen_selection,
        beam_snapshot=frozen_beam,
        receptors=json_safe_mapping(receptor_snapshot),
        jones=json_safe_mapping(loaded_jones),
        backend_snapshot=frozen_backend,
        solver_snapshot=frozen_solver,
        resolved_config_snapshot=frozen_config,
        configuration_provenance_snapshot=frozen_configuration_provenance,
        performance=performance,
        history=frozen_history,
        scientific_sha256=scientific,
        provenance_sha256=provenance_hash,
    )
    return result


__all__ = [
    "BackendResultProvenance",
    "InvalidPhaseCenterError",
    "InvalidResultError",
    "InvalidTimeGridError",
    "LoadedSimulationResult",
    "ResultCoordinateError",
    "ResultError",
    "ResultPerformance",
    "ResultShapeError",
    "ResultUnavailableError",
    "SimulationResult",
    "MModeSolverResultProvenance",
    "SolverProvenanceUnion",
    "SolverResultProvenance",
    "TimeGridLimitError",
    "build_loaded_simulation_result",
    "build_simulation_result",
]
