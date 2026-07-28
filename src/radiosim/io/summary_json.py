"""Truthful, bounded metadata-only JSON summaries for canonical results."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from radiosim.core.result import SimulationResult
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

_MAX_SUMMARY_BYTES = 16 * 1024 * 1024
_EXCLUDED_PAYLOADS = [
    "visibility_samples",
    "flags_array",
    "weights_array",
    "full_time_coordinate",
    "full_frequency_coordinate",
    "per_baseline_geometry",
    "per_antenna_geometry",
]


def _json_tree(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        normalized: dict[str, object] = {}
        for key, item in mapping.items():
            if type(key) is not str or "\x00" in key:
                raise ValueError("summary mapping keys must be NUL-free strings")
            normalized[key] = _json_tree(item)
        return normalized
    if isinstance(value, (tuple, list)):
        sequence = cast(Sequence[object], value)
        return [_json_tree(item) for item in sequence]
    if isinstance(value, np.generic):
        return _json_tree(cast(object, value.item()))
    if isinstance(value, str):
        if type(value) is not str or "\x00" in value:
            raise ValueError("summary strings must be exact and NUL-free")
        _ = value.encode("utf-8", errors="strict")
    return value


def _summary_payload(result: SimulationResult) -> dict[str, object]:
    if type(result) is not SimulationResult:
        raise TypeError("result must be an exact SimulationResult")
    try:
        centers = result.time_grid.as_astropy().utc
        center_iso = np.atleast_1d(np.asarray(centers.isot, dtype=str))
        selection_snapshot = result.selection.to_snapshot()
        payload: dict[str, object] = {
            "schema": {
                "name": "radiosim.result-summary",
                "version": "1.0.0",
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
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
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
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
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


__all__ = ["write_result_summary_json"]
