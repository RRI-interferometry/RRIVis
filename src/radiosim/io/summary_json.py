"""Truthful, bounded metadata-only JSON summaries for canonical results."""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from radiosim.core.result import SimulationResult
from radiosim.core.runtime_config import FrozenMapping
from radiosim.io.atomic_paths import (
    create_sibling_temporary,
    fsync_directory,
    fsync_file,
    open_parent_directory,
    publish_no_clobber,
    publish_replace,
    unlink_temporary,
)
from radiosim.io.result_errors import (
    AtomicWriteError,
    OutputPathError,
    OverwriteRefusedError,
    PartialCleanupError,
    ResultIOError,
    SummaryContractError,
)
from radiosim.io.result_format import ResultFormat, normalize_result_path

SUMMARY_SCHEMA_NAME = "radiosim.result-summary"
SUMMARY_SCHEMA_VERSION = "1.2.0"
_MAX_SUMMARY_BYTES = 16 * 1024 * 1024
_MAX_SUMMARY_NESTING = 32
_MAX_SUMMARY_NODES = 100_000
_EXCLUDED_PAYLOADS = [
    "visibility_samples",
    "flags_array",
    "weights_array",
    "full_time_coordinate",
    "full_frequency_coordinate",
    "per_baseline_geometry",
    "per_antenna_geometry",
    "per_antenna_receptor_definitions",
]
# Feed rotations are configured in degrees and stored in radians, so the round
# trip through radians must not leak representation noise into the summary.  A
# 1e-12-degree quantum is four orders of magnitude finer than any physically
# meaningful feed orientation.
_FEED_ROTATION_DECIMALS = 12


def _receptor_summary(result: SimulationResult) -> dict[str, object]:
    """Return the bounded receptor block for one canonical result.

    Per-antenna receptor rows are deliberately absent: the summary is bounded
    metadata by Tier 4 contract, and the complete per-antenna set lives in the
    HDF5 ``receptors/`` group.  The distinct-rotation list is bounded by the
    configured ``receptors`` section, which the embedded ``resolved_config``
    already reproduces in full.
    """
    receptors = result.receptors
    rotations = sorted(
        {
            round(math.degrees(receptor.feed_rotation_rad), _FEED_ROTATION_DECIMALS)
            for receptor in receptors.receptor_by_antenna.values()
        }
    )
    return {
        "output_basis": receptors.output_basis,
        "receptor_sha256": receptors.provenance.receptor_sha256,
        "native_basis_counts": receptors.native_basis_counts,
        "distinct_feed_rotations_deg": rotations,
    }


def _jones_summary(result: SimulationResult) -> dict[str, object]:
    """Return the bounded Jones block for one canonical result.

    Bounded for the same reason the receptor block is: a tabulated bandpass
    carries every node it was measured at, and the summary is metadata.  The
    enabled terms, the composed chain order, the digest, and each term's
    resolved parameters are here; a run that enabled no optional Jones or
    baseline term reports empty lists and a ``null`` digest rather than omitting
    the block, because a reader should be able to tell "no optional terms" from
    "an older summary".  Solver-owned ``H``, ``C``, and ``E`` remain active and
    are represented by their owning result records.

    ``Tier7JonesSciencePlan.md`` Section 25.2.
    """
    snapshot = dict(result.jones)
    if not snapshot:
        return {
            "enabled_terms": [],
            "chain_order": [],
            "jones_sha256": None,
            "terms": {},
        }
    return {
        "enabled_terms": list(snapshot["enabled_terms"]),
        "chain_order": list(snapshot["chain_order"]),
        "jones_sha256": snapshot["jones_sha256"],
        "terms": dict(snapshot["term_snapshots"]),
    }


def _execution_summary(result: SimulationResult) -> dict[str, object]:
    """Return the bounded execution block for one canonical result.

    Plan Section 19: the *requested* worker policy comes from the resolved
    configuration, while the *executed* loader policy comes from the encoded
    history line the simulator wrote.  Both are needed, because ``executor:
    auto`` resolves to a concrete pool class only at dispatch time and may
    degrade with a recorded reason.
    """
    from radiosim.core.sky.operations.parallel import LoaderExecutionRecord

    execution = result.resolved_config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise SummaryContractError("resolved execution configuration is not a mapping")
    record = LoaderExecutionRecord.from_history(result.history)
    return {
        "offline": execution.get("offline"),
        "sky_loading": dict(execution.get("sky_loading", {})),
        "solver": dict(execution.get("solver", {})),
        "loader": None if record is None else record.to_snapshot(),
    }


def _summary_scalar(value: object) -> str | int | float | bool | None:
    value_type = type(value)
    if value is None or value_type is bool or value_type is int:
        return cast(str | int | float | bool | None, value)
    if value_type is str:
        text = cast(str, value)
        if "\x00" in text:
            raise SummaryContractError(
                "canonical result contains invalid summary metadata: "
                "summary strings must be NUL-free"
            )
        try:
            _ = text.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise SummaryContractError(
                "canonical result contains invalid summary metadata: "
                "summary strings must contain valid UTF-8 text"
            ) from exc
        return text
    if value_type is float:
        number = cast(float, value)
        if not math.isfinite(number):
            raise SummaryContractError(
                "canonical result contains invalid summary metadata: "
                "summary floats must be finite"
            )
        return number
    if isinstance(value, np.bool_):
        return cast(bool, value.item())
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.floating):
        number = value.item()
        if not math.isfinite(number):
            raise SummaryContractError(
                "canonical result contains invalid summary metadata: "
                "summary NumPy floats must be finite"
            )
        return number
    if isinstance(value, np.str_):
        text = value.item()
        if "\x00" in text:
            raise SummaryContractError(
                "canonical result contains invalid summary metadata: "
                "summary strings must be NUL-free"
            )
        try:
            _ = text.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise SummaryContractError(
                "canonical result contains invalid summary metadata: "
                "summary strings must contain valid UTF-8 text"
            ) from exc
        return text
    raise SummaryContractError(
        "canonical result contains invalid summary metadata: "
        f"summary contains unsupported exact value type: {value_type.__name__}"
    )


def _set_tree_value(
    destination: dict[str, object] | list[object],
    slot: str | int,
    value: object,
) -> None:
    if isinstance(destination, list):
        if type(slot) is not int:
            raise RuntimeError("internal summary sequence slot must be an integer")
        destination[slot] = value
        return
    if type(slot) is not str:
        raise RuntimeError("internal summary mapping slot must be a string")
    destination[slot] = value


def _json_tree(value: object) -> object:
    holder: list[object] = [None]
    worklist: list[
        tuple[
            object,
            int,
            frozenset[int],
            dict[str, object] | list[object],
            str | int,
        ]
    ]
    worklist = [(value, 1, frozenset(), holder, 0)]
    scheduled_nodes = 1
    while worklist:
        source, nesting, ancestors, destination, slot = worklist.pop()
        source_type = type(source)
        if source_type is dict or source_type is FrozenMapping:
            if nesting > _MAX_SUMMARY_NESTING:
                raise SummaryContractError(
                    "canonical result contains invalid summary metadata: "
                    "summary nesting exceeds the explicit "
                    f"{_MAX_SUMMARY_NESTING}-level limit"
                )
            source_id = id(source)
            if source_id in ancestors:
                raise SummaryContractError(
                    "canonical result contains invalid summary metadata: "
                    "summary container cycle is not supported"
                )
            child_ancestors = ancestors | {source_id}
            mapping = cast(dict[object, object] | FrozenMapping, source)
            child_count = len(mapping)
            if scheduled_nodes + child_count > _MAX_SUMMARY_NODES:
                raise SummaryContractError(
                    "canonical result contains invalid summary metadata: "
                    "summary node count exceeds the explicit "
                    f"{_MAX_SUMMARY_NODES}-node limit"
                )
            scheduled_nodes += child_count
            normalized_mapping: dict[str, object] = {}
            _set_tree_value(destination, slot, normalized_mapping)
            items: list[tuple[str, object]] = []
            for key, item in mapping.items():
                if type(key) is not str or "\x00" in key:
                    raise SummaryContractError(
                        "canonical result contains invalid summary metadata: "
                        "summary mapping keys must be NUL-free exact strings"
                    )
                try:
                    _ = key.encode("utf-8", errors="strict")
                except UnicodeError as exc:
                    raise SummaryContractError(
                        "canonical result contains invalid summary metadata: "
                        "summary mapping keys must contain valid UTF-8 text"
                    ) from exc
                items.append((key, item))
            for key, item in reversed(items):
                worklist.append(
                    (
                        item,
                        nesting + 1,
                        child_ancestors,
                        normalized_mapping,
                        key,
                    )
                )
            continue
        if source_type is list or source_type is tuple:
            if nesting > _MAX_SUMMARY_NESTING:
                raise SummaryContractError(
                    "canonical result contains invalid summary metadata: "
                    "summary nesting exceeds the explicit "
                    f"{_MAX_SUMMARY_NESTING}-level limit"
                )
            source_id = id(source)
            if source_id in ancestors:
                raise SummaryContractError(
                    "canonical result contains invalid summary metadata: "
                    "summary container cycle is not supported"
                )
            child_ancestors = ancestors | {source_id}
            sequence = cast(list[object] | tuple[object, ...], source)
            child_count = len(sequence)
            if scheduled_nodes + child_count > _MAX_SUMMARY_NODES:
                raise SummaryContractError(
                    "canonical result contains invalid summary metadata: "
                    "summary node count exceeds the explicit "
                    f"{_MAX_SUMMARY_NODES}-node limit"
                )
            scheduled_nodes += child_count
            normalized_sequence: list[object] = [None] * child_count
            _set_tree_value(destination, slot, normalized_sequence)
            for index in range(child_count - 1, -1, -1):
                worklist.append(
                    (
                        sequence[index],
                        nesting + 1,
                        child_ancestors,
                        normalized_sequence,
                        index,
                    )
                )
            continue
        _set_tree_value(destination, slot, _summary_scalar(source))
    return holder[0]


def _summary_payload(result: SimulationResult) -> dict[str, object]:
    if type(result) is not SimulationResult:
        raise TypeError("result must be an exact SimulationResult")
    try:
        centers = result.time_grid.as_astropy().utc
        center_iso = np.atleast_1d(np.asarray(centers.isot, dtype=str))
        selection_snapshot = result.selection.to_snapshot()
        payload: dict[str, object] = {
            # Tier 6 grew this document: an ``execution`` block (6C) and, in
            # the ``solver`` and ``performance`` blocks below, the solved
            # components, their element counts, and the two per-component
            # timings (6F, serialized here by 6G).  The bump is deliberately
            # *minor* where the HDF5 schema takes a major one in the same
            # slice.  The HDF5 reader rejects any version but its own and any
            # solver field set but its own, so old and new files are mutually
            # unreadable; this document has no reader, nothing was removed or
            # retyped, and every ``1.0.0`` key survives at the same path with
            # the same meaning.  See ``Tier6HybridRuntimePlan.md`` Section 19.
            "schema": {
                "name": SUMMARY_SCHEMA_NAME,
                "version": SUMMARY_SCHEMA_VERSION,
            },
            "result": {
                "schema": result.schema_version,
                "shape": list(result.visibilities.shape),
                "dtype": result.visibilities.dtype.name,
                "units": {"visibility": "Jy", "weight": "dimensionless"},
                "scientific_sha256": result.scientific_sha256,
                "provenance_sha256": result.provenance_sha256,
                "flag_count": int(np.count_nonzero(result.flags)),
                "weight_minimum": float(np.min(result.weights)),
                "weight_maximum": float(np.max(result.weights)),
                "axis_counts": {
                    "time": int(result.visibilities.shape[0]),
                    "baseline": int(result.visibilities.shape[1]),
                    "frequency": int(result.visibilities.shape[2]),
                    "correlation": int(result.visibilities.shape[3]),
                },
            },
            "observation": {
                "first_center_iso_utc": str(center_iso[0]),
                "last_center_iso_utc": str(center_iso[-1]),
                "count": len(result.time_grid),
                "cadence_seconds": result.time_grid.cadence_seconds,
                "duration_seconds": result.time_grid.duration_seconds,
                "interval_semantics": result.time_grid.interval_semantics,
                "exposure_rule": (
                    "minimum of cadence_seconds and remaining observation duration"
                ),
            },
            "frequency": {
                "count": int(result.frequencies_hz.size),
                "minimum_center_hz": float(np.min(result.frequencies_hz)),
                "maximum_center_hz": float(np.max(result.frequencies_hz)),
                "minimum_width_hz": float(np.min(result.channel_widths_hz)),
                "maximum_width_hz": float(np.max(result.channel_widths_hz)),
            },
            "correlation": {
                "labels": list(result.correlations),
                "basis": result.polarization_basis,
            },
            "receptors": _receptor_summary(result),
            "jones": _jones_summary(result),
            "instrument": {
                "name": result.instrument.name,
                "instrument_sha256": result.instrument.provenance.instrument_sha256,
                "antenna_count": len(result.instrument.antennas),
                "selected_baseline_count": len(result.selection.baselines),
                "selection": selection_snapshot,
            },
            "phase_center": result.phase_center.to_snapshot(),
            "beam": result.beam_state.to_snapshot(),
            "backend": result.backend.to_snapshot(),
            "solver": result.solver.to_snapshot(),
            "execution": _execution_summary(result),
            "resolved_config": result.resolved_config,
            "configuration_provenance": result.configuration_provenance,
            "performance": result.performance.to_snapshot(),
            "history": list(result.history),
            "excluded_payloads": list(_EXCLUDED_PAYLOADS),
        }
        tree = _json_tree(payload)
        if type(tree) is not dict:
            raise TypeError("summary root did not normalize to an object")
        normalized = cast(dict[str, object], tree)
        return dict(sorted(normalized.items()))
    except SummaryContractError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise SummaryContractError(
            "canonical result could not be represented by the summary schema"
        ) from exc


def _encode_summary(result: SimulationResult) -> bytes:
    try:
        payload = _json_tree(_summary_payload(result))
        if type(payload) is not dict:
            raise TypeError("summary root must be an object")
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8", errors="strict")
    except SummaryContractError:
        raise
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise SummaryContractError(
            "canonical result contains invalid summary metadata"
        ) from exc
    if len(encoded) > _MAX_SUMMARY_BYTES:
        raise SummaryContractError(
            "encoded result summary exceeds the 16 MiB contract limit"
        )
    return encoded


def _validate_summary_target(path: Path, *, overwrite: object) -> Path:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean")
    current = Path(path.anchor)
    for component in path.parent.parts[1:]:
        current = current / component
        try:
            status = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise OutputPathError(f"summary output ancestor is unsafe: {current}")
    try:
        status = path.lstat()
    except FileNotFoundError:
        return path
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise OutputPathError(
            f"summary output target is not a regular non-symlink file: {path}"
        )
    if not overwrite:
        raise OverwriteRefusedError(f"output target already exists: {path}")
    return path


def write_result_summary_json(
    result: SimulationResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one bounded metadata-only result summary atomically."""
    encoded = _encode_summary(result)
    normalized = normalize_result_path(path, ResultFormat.SUMMARY_JSON)
    final = _validate_summary_target(normalized, overwrite=overwrite)
    parent_fd = -1
    temporary_fd = -1
    temporary: Path | None = None
    published = False
    try:
        parent_fd = open_parent_directory(final.parent, create=True)
        temporary_fd, temporary = create_sibling_temporary(final, parent_fd)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(temporary_fd, view)
                if written <= 0:
                    raise AtomicWriteError(
                        "summary temporary file write made no progress"
                    )
                view = view[written:]
            fsync_file(temporary_fd)
            os.close(temporary_fd)
            temporary_fd = -1
            observed = temporary.read_bytes()
            if observed != encoded:
                raise SummaryContractError(
                    "temporary result summary failed exact read-back verification"
                )
            if overwrite:
                publish_replace(temporary, final, parent_fd)
                published = True
            else:
                publish_no_clobber(temporary, final, parent_fd)
                published = True
                try:
                    unlink_temporary(temporary, parent_fd)
                except OSError as exc:
                    raise PartialCleanupError(temporary) from exc
            fsync_directory(parent_fd)
            return final
        except Exception as exc:
            if not published:
                try:
                    unlink_temporary(temporary, parent_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    raise PartialCleanupError(temporary) from exc
            if isinstance(exc, ResultIOError):
                raise
            raise AtomicWriteError(
                f"atomic summary transaction failed before publication: {final}"
            ) from exc
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


__all__ = [
    "SUMMARY_SCHEMA_NAME",
    "SUMMARY_SCHEMA_VERSION",
    "write_result_summary_json",
]
