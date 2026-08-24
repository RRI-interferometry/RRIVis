"""Canonical UVFITS projection, reader, and publication contracts."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from pyuvdata import UVData

import radiosim.io.uvfits as uvfits_module
from radiosim.core.result import LoadedSimulationResult, SimulationResult
from radiosim.io.result_errors import (
    FormatRepresentationError,
    OptionalResultDependencyError,
    OutputPathError,
    OverwriteRefusedError,
    UnsafeResultInputError,
)
from radiosim.io.standard_visibility import (
    PROJECTION_HISTORY_PREFIX,
    StandardReadLimits,
    StandardVisibilityData,
    project_simulation_result,
    projection_record_from_history,
)
from radiosim.io.uvfits import read_uvfits, write_uvfits
from tests.unit.test_io.test_standard_visibility import build_standard_result


def test_uvfits_public_signatures() -> None:
    assert str(inspect.signature(write_uvfits)) == (
        "(result: 'SimulationResult', path: 'str | Path', *, "
        "overwrite: 'bool' = False) -> 'Path'"
    )
    assert str(inspect.signature(read_uvfits)) == (
        "(path: 'str | Path', *, limits: 'StandardReadLimits' = "
        "StandardReadLimits()) -> 'StandardVisibilityData'"
    )


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_uvfits_round_trip_preserves_supported_dtype_and_raw_contract(
    tmp_path: Path,
    dtype: str,
) -> None:
    result = build_standard_result(tmp_path, dtype=dtype)
    target = tmp_path / f"canonical-{dtype}.uvfits"

    assert write_uvfits(result, target) == target.absolute()
    loaded = read_uvfits(target)
    expected = project_simulation_result(result, format="uvfits").data

    assert type(loaded) is StandardVisibilityData
    assert not isinstance(loaded, (SimulationResult, LoadedSimulationResult))
    assert loaded.format == "uvfits"
    assert loaded.visibilities.dtype == np.dtype(dtype)
    assert loaded.weights.dtype == np.dtype(np.float32)
    assert loaded.correlations == ("XX", "XY", "YX", "YY")
    assert loaded.source_scientific_sha256 == result.scientific_sha256
    assert loaded.source_provenance_sha256 == result.provenance_sha256
    np.testing.assert_allclose(
        loaded.visibilities,
        expected.visibilities,
        rtol=2e-6 if dtype == "complex64" else 5e-13,
        atol=2e-6 if dtype == "complex64" else 5e-13,
    )

    raw = UVData()
    raw.read_uvfits(str(target))
    assert raw.data_array.dtype == np.dtype(dtype)
    assert np.asarray(raw.polarization_array).tolist() == [-5, -6, -7, -8]
    with fits.open(target, mode="readonly", memmap=True) as handle:
        assert handle[0].header["GROUPS"]
        assert any(hdu.name == "AIPS AN" for hdu in handle)


# ---------------------------------------------------------------------------
# Tier 5F: UVFITS carries the resolved basis (Sections 14.2, 22)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("receptors", "labels", "file_codes", "feed_letters"),
    [
        (None, ("XX", "XY", "YX", "YY"), [-5, -6, -7, -8], ["x", "y"]),
        (
            {"default": {"basis": "circular"}},
            ("RR", "RL", "LR", "LL"),
            [-1, -2, -3, -4],
            ["r", "l"],
        ),
    ],
    ids=["linear", "circular"],
)
def test_uvfits_polarization_metadata_round_trips_both_bases(
    tmp_path: Path,
    receptors: dict[str, object] | None,
    labels: tuple[str, ...],
    file_codes: list[int],
    feed_letters: list[str],
) -> None:
    """UVFITS stores the descending Section 14.2 code order in both bases."""
    result = build_standard_result(tmp_path, receptors=receptors)
    assert result.correlations == labels
    target = tmp_path / "basis.uvfits"

    assert write_uvfits(result, target) == target.absolute()
    loaded = read_uvfits(target)
    assert loaded.correlations == labels

    raw = UVData()
    raw.read_uvfits(str(target))
    antenna_count = int(raw.telescope.Nants)
    assert np.asarray(raw.polarization_array).tolist() == file_codes
    assert raw.telescope.feed_array.tolist() == [feed_letters] * antenna_count
    assert list(raw.telescope.mount_type) == ["fixed"] * antenna_count

    with fits.open(target, mode="readonly", memmap=True) as handle:
        primary = handle[0].header
        assert abs(int(primary["NAXIS3"])) == 4
        # The UVFITS polarization axis is a monotonic CRVAL/CDELT sequence.
        assert int(primary["CRVAL3"]) == file_codes[0]
        assert int(primary["CDELT3"]) == -1

    expected = project_simulation_result(result, format="uvfits").data
    np.testing.assert_allclose(
        loaded.visibilities,
        expected.visibilities,
        rtol=5e-13,
        atol=5e-13,
    )


def test_uvfits_history_records_the_resolved_basis(tmp_path: Path) -> None:
    result = build_standard_result(
        tmp_path,
        receptors={"default": {"basis": "circular"}},
    )
    target = tmp_path / "circular-history.uvfits"
    write_uvfits(result, target)

    loaded = read_uvfits(target)
    # UVFITS wraps long HISTORY cards, so the record spans several lines.
    record, _lines = projection_record_from_history("\n".join(loaded.history))
    assert record["polarization_basis"] == "circular_rl"
    assert record["receptor_sha256"] == result.receptors.provenance.receptor_sha256


def test_uvfits_rejects_a_record_basis_that_contradicts_the_code_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forged record cannot relabel a linear file as circular."""
    result = build_standard_result(tmp_path)
    target = tmp_path / "forged.uvfits"
    write_uvfits(result, target)

    original = uvfits_module.projection_record_from_history

    def forge(history: object) -> tuple[dict[str, object], tuple[str, ...]]:
        record, lines = original(history)
        record["polarization_basis"] = "circular_rl"
        return record, lines

    # Both the bounded preflight and the loaded read must see the same forged
    # record, so the rejection comes from the basis check and not from the
    # preflight/loaded comparison.
    monkeypatch.setattr(
        "radiosim.io.standard_visibility.projection_record_from_history",
        forge,
    )
    monkeypatch.setattr(uvfits_module, "projection_record_from_history", forge)
    with pytest.raises(UnsafeResultInputError, match="polarization axis carries"):
        _ = read_uvfits(target)


@pytest.mark.parametrize(
    ("frequencies", "widths", "expected"),
    [
        ((100e6, 101e6, 103e6), (1e6, 1e6, 1e6), "evenly spaced"),
        ((100e6, 101e6), (1e6, 2e6), "equal channel widths"),
        ((100e6, 101e6), (2e6, 2e6), "spacing must equal channel width"),
    ],
)
def test_uvfits_representability_rejects_each_spectral_constraint_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frequencies: tuple[float, ...],
    widths: tuple[float, ...],
    expected: str,
) -> None:
    result = build_standard_result(
        tmp_path,
        frequencies_hz=frequencies,
        channel_widths_hz=widths,
    )
    imported = False

    def fail_import():
        nonlocal imported
        imported = True
        raise AssertionError("dependency import must not run")

    monkeypatch.setattr(uvfits_module, "_import_pyuvdata", fail_import)
    target = tmp_path / "invalid.uvfits"
    with pytest.raises(FormatRepresentationError, match=expected):
        write_uvfits(result, target)
    assert not imported
    assert not target.exists()


def test_uvfits_aggregates_multiple_representability_failures(
    tmp_path: Path,
) -> None:
    result = build_standard_result(
        tmp_path,
        frequencies_hz=(100e6, 101e6, 103e6),
        channel_widths_hz=(2e6, 3e6, 4e6),
    )
    with pytest.raises(FormatRepresentationError) as caught:
        write_uvfits(result, tmp_path / "invalid.uvfits")
    message = str(caught.value)
    assert "evenly spaced" in message
    assert "equal channel widths" in message
    assert "spacing must equal channel width" in message
    assert "HDF5 or Measurement Set" in message


def test_uvfits_writer_passes_force_phase_false_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real_write = uvfits_module._write_uvfits

    def recording_write(uvdata, path, **kwargs):
        calls.append(((uvdata, path), kwargs))
        return real_write(uvdata, path, **kwargs)

    monkeypatch.setattr(uvfits_module, "_write_uvfits", recording_write)
    write_uvfits(result, tmp_path / "arguments.uvfits")
    assert len(calls) == 1
    assert calls[0][1] == {"force_phase": False}
    assert isinstance(calls[0][0][0].polarization_array, np.ndarray)
    assert np.issubdtype(calls[0][0][0].polarization_array.dtype, np.integer)


def test_uvfits_collision_replace_and_prepublication_failure_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build_standard_result(tmp_path / "first", dtype="complex64")
    second = build_standard_result(tmp_path / "second", dtype="complex64")
    target = tmp_path / "replace.uvfits"
    write_uvfits(first, target)
    original = target.read_bytes()
    with pytest.raises(OverwriteRefusedError):
        write_uvfits(second, target)
    assert target.read_bytes() == original

    monkeypatch.setattr(
        uvfits_module,
        "_verify_temporary_uvfits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("verify")),
    )
    with pytest.raises(Exception, match="verify|atomic"):
        write_uvfits(second, target, overwrite=True)
    assert target.read_bytes() == original


def test_uvfits_header_limits_precede_pyuvdata_data_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "limited.uvfits"
    write_uvfits(result, target)
    data_reads = 0
    real_read = uvfits_module._read_uvfits

    def recording_read(path, *, read_data):
        nonlocal data_reads
        if read_data:
            data_reads += 1
        return real_read(path, read_data=read_data)

    monkeypatch.setattr(uvfits_module, "_read_uvfits", recording_read)
    with pytest.raises(UnsafeResultInputError, match="max_times"):
        read_uvfits(target, limits=StandardReadLimits(max_times=1))
    assert data_reads == 0


def _insert_primary_history_bytes(
    source: Path,
    target: Path,
    *,
    byte_count: int,
) -> None:
    original = source.read_bytes()
    end_offset = next(
        offset
        for offset in range(0, len(original), 80)
        if original[offset : offset + 8] == b"END     "
    )
    original_header_size = ((end_offset + 80 + 2879) // 2880) * 2880
    history_cards = b"".join(
        b"HISTORY " + (b"X" * min(72, byte_count - offset)).ljust(72)
        for offset in range(0, byte_count, 72)
    )
    new_header = (
        original[:end_offset] + history_cards + original[end_offset : end_offset + 80]
    )
    new_header += b" " * (-len(new_header) % 2880)
    target.write_bytes(new_header + original[original_header_size:])


def test_uvfits_oversized_primary_history_rejects_before_json_or_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    valid = write_uvfits(result, tmp_path / "bounded.uvfits")
    hostile = tmp_path / "oversized-history.uvfits"
    _insert_primary_history_bytes(valid, hostile, byte_count=1024 * 1024)
    json_reads = 0
    science_reads = 0
    real_projection_record = uvfits_module.projection_record_from_history
    real_read = uvfits_module._read_uvfits

    def recording_projection_record(history):
        nonlocal json_reads
        json_reads += 1
        return real_projection_record(history)

    def recording_read(path, *, read_data):
        nonlocal science_reads
        if read_data:
            science_reads += 1
        return real_read(path, read_data=read_data)

    monkeypatch.setattr(
        uvfits_module,
        "projection_record_from_history",
        recording_projection_record,
    )
    monkeypatch.setattr(uvfits_module, "_read_uvfits", recording_read)
    with pytest.raises(UnsafeResultInputError, match="HISTORY|header"):
        read_uvfits(hostile)
    assert json_reads == 0
    assert science_reads == 0


def test_uvfits_hostile_history_subprocess_allocation_is_bounded(
    tmp_path: Path,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    valid = write_uvfits(result, tmp_path / "subprocess-bounded.uvfits")
    hostile = tmp_path / "subprocess-hostile.uvfits"
    _insert_primary_history_bytes(valid, hostile, byte_count=1024 * 1024)
    script = """
import json
import resource
import sys
import tracemalloc
import radiosim.io.uvfits as module

science_reads = 0
json_reads = 0
real_read = module._read_uvfits
real_projection = module.projection_record_from_history

def recording_read(path, *, read_data):
    global science_reads
    science_reads += int(read_data)
    return real_read(path, read_data=read_data)

def recording_projection(history):
    global json_reads
    json_reads += 1
    return real_projection(history)

module._read_uvfits = recording_read
module.projection_record_from_history = recording_projection
module._import_pyuvdata()
tracemalloc.start()
try:
    module.read_uvfits(sys.argv[1])
except Exception as exc:
    rejection = type(exc).__name__
    message = str(exc)
else:
    rejection = None
    message = ""
_current, peak = tracemalloc.get_traced_memory()
print(json.dumps({
    "rejection": rejection,
    "message": message,
    "python_peak": peak,
    "native_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "science_reads": science_reads,
    "json_reads": json_reads,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(hostile)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout.splitlines()[-1])
    assert observed["rejection"] == "UnsafeResultInputError"
    assert "HISTORY" in observed["message"]
    assert observed["python_peak"] < 4 * 1024 * 1024
    assert observed["native_rss"] > 0
    assert observed["science_reads"] == 0
    assert observed["json_reads"] == 0


def test_uvfits_projection_history_rejects_before_science_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    valid = write_uvfits(result, tmp_path / "valid-history.uvfits")
    raw = UVData()
    raw.read_uvfits(str(valid))
    record, _lines = projection_record_from_history(raw.history)
    record["schema"] = "attacker.projection.v1"
    raw.history = PROJECTION_HISTORY_PREFIX + json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
    )
    hostile = tmp_path / "hostile-history.uvfits"
    raw.write_uvfits(str(hostile), force_phase=False)

    science_reads = 0
    real_read = uvfits_module._read_uvfits

    def recording_read(path, *, read_data):
        nonlocal science_reads
        if read_data:
            science_reads += 1
        return real_read(path, read_data=read_data)

    monkeypatch.setattr(uvfits_module, "_read_uvfits", recording_read)
    with pytest.raises((UnsafeResultInputError, FormatRepresentationError)):
        read_uvfits(hostile)
    assert science_reads == 0


def test_uvfits_rejects_truncated_and_symlink_inputs(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.uvfits"
    truncated.write_bytes(b"SIMPLE  =                    T")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(UnsafeResultInputError):
            read_uvfits(truncated)
    assert caught == []

    destination = tmp_path / "destination.uvfits"
    destination.write_bytes(b"safe")
    linked = tmp_path / "linked.uvfits"
    linked.symlink_to(destination)
    with pytest.raises(UnsafeResultInputError):
        read_uvfits(linked)


def test_uvfits_rejects_partial_trailing_block_before_science_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    valid = write_uvfits(result, tmp_path / "valid-tail.uvfits")
    hostile = tmp_path / "partial-tail.uvfits"
    hostile.write_bytes(valid.read_bytes() + b"BAD")

    science_reads = 0
    real_read = uvfits_module._read_uvfits

    def recording_read(path: Path, *, read_data: bool) -> object:
        nonlocal science_reads
        science_reads += int(read_data)
        return real_read(path, read_data=read_data)

    monkeypatch.setattr(uvfits_module, "_read_uvfits", recording_read)
    with pytest.raises(UnsafeResultInputError, match="trailing|block"):
        read_uvfits(hostile)
    assert science_reads == 0


def test_uvfits_rejects_extra_full_trailing_block_before_science_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    valid = write_uvfits(result, tmp_path / "valid-full-tail.uvfits")
    hostile = tmp_path / "full-tail.uvfits"
    hostile.write_bytes(valid.read_bytes() + b"\x00" * 2880)

    science_reads = 0
    real_read = uvfits_module._read_uvfits

    def recording_read(path: Path, *, read_data: bool) -> object:
        nonlocal science_reads
        science_reads += int(read_data)
        return real_read(path, read_data=read_data)

    monkeypatch.setattr(uvfits_module, "_read_uvfits", recording_read)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(UnsafeResultInputError, match="trailing|HDU|header"):
            read_uvfits(hostile)
    assert [(item.category.__name__, str(item.message)) for item in caught] == [
        (
            "AstropyUserWarning",
            "Unexpected extra padding at the end of the file.  "
            "This padding may not be preserved when saving changes.",
        )
    ]
    assert science_reads == 0


def test_uvfits_optional_dependency_failure_precedes_parent_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "missing" / "result.uvfits"
    monkeypatch.setattr(
        uvfits_module,
        "_import_pyuvdata",
        lambda: (_ for _ in ()).throw(
            OptionalResultDependencyError(
                "format=uvfits missing_package=pyuvdata "
                "pyuvdata_version=unavailable install_extra=radiosim"
            )
        ),
    )
    with pytest.raises(OptionalResultDependencyError, match="format=uvfits"):
        write_uvfits(result, target)
    assert not target.parent.exists()


def test_uvfits_one_channel_round_trip(tmp_path: Path) -> None:
    result = build_standard_result(
        tmp_path,
        dtype="complex128",
        frequencies_hz=(100e6,),
        channel_widths_hz=(2e6,),
    )

    target = write_uvfits(result, tmp_path / "one-channel.uvfits")
    loaded = read_uvfits(target)

    assert loaded.frequencies_hz.tolist() == [100e6]
    assert loaded.channel_widths_hz.tolist() == [2e6]
    assert loaded.visibilities.dtype == np.dtype("complex128")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("unsupported_dtype", "float64"),
        ("antenna_count", "1 through 255"),
        ("antenna_number", "0..254"),
        ("nonfinite_weight", "finite"),
    ],
)
def test_uvfits_each_nonspectral_constraint_fails_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex128")
    if mutation == "unsupported_dtype":
        values = np.asarray(result.visibilities.real, dtype=np.float64)
        object.__setattr__(
            result,
            "visibilities",
            np.ndarray(values.shape, dtype=values.dtype, buffer=values.tobytes()),
        )
    elif mutation == "antenna_count":
        object.__setattr__(
            result.instrument,
            "antennas",
            tuple(result.instrument.antennas[0] for _ in range(256)),
        )
    elif mutation == "antenna_number":
        object.__setattr__(result.instrument.antennas[0].id, "number", 255)
    else:
        values = np.asarray(result.weights).copy()
        values.flat[0] = np.nan
        object.__setattr__(
            result,
            "weights",
            np.ndarray(values.shape, dtype=values.dtype, buffer=values.tobytes()),
        )
    imported = False

    def fail_import() -> object:
        nonlocal imported
        imported = True
        raise AssertionError("dependency import must not run")

    monkeypatch.setattr(uvfits_module, "_import_pyuvdata", fail_import)
    with pytest.raises(FormatRepresentationError, match=expected):
        write_uvfits(result, tmp_path / "unsupported.uvfits")
    assert not imported


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo"])
def test_uvfits_writer_rejects_nonregular_target(
    tmp_path: Path,
    kind: str,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = tmp_path / "wrong.uvfits"
    if kind == "directory":
        target.mkdir()
    elif kind == "symlink":
        destination = tmp_path / "destination.uvfits"
        destination.write_bytes(b"safe")
        target.symlink_to(destination)
    else:
        os.mkfifo(target)

    with pytest.raises(OutputPathError):
        write_uvfits(result, target, overwrite=True)


def test_uvfits_oversized_header_precedes_pyuvdata_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    target = write_uvfits(result, tmp_path / "oversized.uvfits")
    read_called = False

    def fail_read(*_args: object, **_kwargs: object) -> object:
        nonlocal read_called
        read_called = True
        raise AssertionError("pyuvdata read must not run")

    monkeypatch.setattr(uvfits_module, "_read_uvfits", fail_read)
    with pytest.raises(UnsafeResultInputError, match="max_visibility_elements"):
        read_uvfits(
            target,
            limits=StandardReadLimits(max_visibility_elements=1),
        )
    assert not read_called


def test_standard_writers_never_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("writer must not prompt")
        ),
    )
    target = write_uvfits(result, tmp_path / "no-prompt.uvfits")
    assert target.is_file()
    assert any(
        item.startswith("RADIOSIM_PROJECTION_JSON=")
        for item in read_uvfits(target).history
    )


def test_uvfits_history_names_every_solved_component(tmp_path: Path) -> None:
    """Tier 6G, plan Section 19 / row H10: UVFITS HISTORY records them too."""
    result = build_standard_result(
        tmp_path,
        sky_representation="hybrid",
        components=("point", "healpix"),
        component_element_counts=(3, 3072),
    )
    target = tmp_path / "hybrid-components.uvfits"
    write_uvfits(result, target)

    loaded = read_uvfits(target)
    joined = "\n".join(loaded.history)

    assert "sky_representation=hybrid" in joined
    assert "solver_components=point,healpix" in joined
    assert "solver_component_element_counts=3,3072" in joined
    record, _lines = projection_record_from_history(joined)
    assert record["solver"]["components"] == ["point", "healpix"]
    assert record["solver"]["component_element_counts"] == [3, 3072]


# ==============================================================================
# SCI-005 Stage 3: the UVFITS half of the frozen output matrix
# ==============================================================================
#
# ``docs/development/sci005_beam_physics_plan.md`` Section 5.4 freezes one
# predicate for ``uvfits`` and ``measurement_set`` together: "Both serialize the
# four already-corrupted correlation products in the result's declared output
# basis. The predicate is that every per-correlation round-trip difference lies
# within that writer's **existing accepted** output tolerance, which is not
# widened; that the antenna feed metadata equals the resolved
# ``ResolvedReceptorSet`` values rather than any file row label; and that the
# existing HISTORY provenance carries ``source_scientific_sha256``. Neither
# format is claimed to preserve a reusable efield beam model, and no invented
# standard table is added."
#
# The tolerance below is the one this module's own accepted ``complex128``
# round-trip case already uses; Section 7.5 makes ``io/uvfits.py`` unwritable at
# Stage 3, so nothing here asks the writer to change.

_STAGE3_UVFITS_TOLERANCE = 5e-13

_STAGE3_UVFITS_EXPECTATION = {
    "linear": ("linear_xy", ("XX", "XY", "YX", "YY"), ["x", "y"], (np.pi / 2.0, 0.0)),
    "circular": ("circular_rl", ("RR", "RL", "LR", "LL"), ["r", "l"], (0.0, 0.0)),
}


@pytest.mark.parametrize("authored_basis", sorted(_STAGE3_UVFITS_EXPECTATION))
def test_uvfits_serializes_a_full_efield_run_in_the_declared_basis(
    tmp_path: Path,
    authored_basis: str,
) -> None:
    """Section 5.4's ``uvfits`` predicate, on both output bases.

    ``efield_uvfits_linear_xy`` and ``efield_uvfits_circular_rl``.  The beam
    file's own feed row label is ``('x', 'y')`` in both rows, so the circular
    row is the one that proves the feed metadata is taken from the resolved
    receptor set's single declared output basis and never from the file.
    """
    from tests.fixtures.beamfits import run_full_efield_workload

    basis, labels, feeds, feed_angles = _STAGE3_UVFITS_EXPECTATION[authored_basis]
    workload = run_full_efield_workload(tmp_path, output_basis=authored_basis)
    result = workload.result
    assert result.correlations == labels
    assert result.polarization_basis == basis

    target = tmp_path / f"efield-{authored_basis}.uvfits"
    assert write_uvfits(result, target) == target.absolute()
    loaded = read_uvfits(target)
    expected = project_simulation_result(result, format="uvfits").data

    assert loaded.correlations == labels
    assert loaded.visibilities.shape[-1] == 4
    for index, label in enumerate(labels):
        np.testing.assert_allclose(
            loaded.visibilities[..., index],
            expected.visibilities[..., index],
            rtol=_STAGE3_UVFITS_TOLERANCE,
            atol=_STAGE3_UVFITS_TOLERANCE,
            err_msg=f"correlation {label}",
        )

    raw = UVData()
    raw.read_uvfits(str(target))
    observed_feeds = np.asarray(raw.telescope.feed_array).tolist()
    assert observed_feeds == [feeds for _ in observed_feeds]
    np.testing.assert_allclose(
        np.asarray(raw.telescope.feed_angle, dtype=np.float64),
        np.tile(np.asarray(feed_angles, dtype=np.float64), (len(observed_feeds), 1)),
        rtol=0.0,
        atol=1e-6,
    )
    # The transport the beam came from labels its own rows ``('x', 'y')``; the
    # circular row proves the writer did not read that label.
    if basis == "circular_rl":
        assert feeds != ["x", "y"]

    assert loaded.source_scientific_sha256 == result.scientific_sha256
    record, _lines = projection_record_from_history("\n".join(loaded.history))
    assert record["source_scientific_sha256"] == result.scientific_sha256
    assert record["polarization_basis"] == basis
    assert record["receptor_sha256"] == result.receptors.provenance.receptor_sha256
    # No invented standard table is added for the efield beam model: the file
    # carries exactly the HDUs an accepted scalar run already produces.
    with fits.open(target, mode="readonly", memmap=True) as handle:
        assert {hdu.name for hdu in handle if hdu.name} == {
            "PRIMARY",
            "AIPS AN",
            "AIPS SU",
        }


def test_a_scalar_peak_uvfits_run_keeps_its_zero_cross_hands(tmp_path: Path) -> None:
    """The disabled half of the same evidence.

    Section 5.1.1 "changes no byte of the accepted ``peak`` path": the scalar
    ``E = e I2`` still writes exactly zero in both cross-hand products, which is
    precisely the property the full-efield subset above breaks.
    """
    from tests.fixtures.beamfits import run_scalar_beamfits_workload

    workload = run_scalar_beamfits_workload(tmp_path)
    target = tmp_path / "scalar.uvfits"
    assert write_uvfits(workload.result, target) == target.absolute()
    loaded = read_uvfits(target)

    assert loaded.correlations == ("XX", "XY", "YX", "YY")
    for index in (1, 2):
        np.testing.assert_array_equal(
            loaded.visibilities[..., index],
            np.zeros_like(loaded.visibilities[..., index]),
        )
    assert loaded.source_scientific_sha256 == workload.result.scientific_sha256


# ==============================================================================
# SCI-004 phase M3: Section 10's UVFITS contract for an m-mode result
# ==============================================================================
#
# ``docs/development/sci004_mmode_design.md`` Section 10: "UVFITS/MS keep the
# canonical zenith phase centre, east-X/circular feed metadata, four correlation
# products, UTC coordinates, and history lines naming the m-mode/frame/harmonic
# conventions. Reader round trips must reconstruct and authenticate the m-mode
# solver snapshot; a reader that silently labels it ``rime`` fails acceptance."

_PHASE3_UVFITS_GREEN_CONTROL = (
    "tests/unit/test_io/test_uvfits.py::"
    "test_uvfits_round_trip_preserves_supported_dtype_and_raw_contract[complex128]"
)

_PHASE3_UVFITS_PATTERN = r"has no attribute 'components'"


def _phase3_uvfits_case(
    case_id: str,
    requirement_id: str,
    function: str,
) -> dict[str, object]:
    from tests.unit.test_io.test_standard_visibility import MMODE_FIXTURE_BYTES

    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": f"tests/unit/test_io/test_uvfits.py::{function}",
        "expected_failure_kind": "missing-symbol",
        "expected_failure_pattern": _PHASE3_UVFITS_PATTERN,
        "fixture_defect_excluded_by": _PHASE3_UVFITS_GREEN_CONTROL,
        "fixture_bytes": MMODE_FIXTURE_BYTES,
    }


SCI004_PHASE3_RED_CASES: tuple[dict[str, object], ...] = (
    _phase3_uvfits_case(
        "m3.uvfits.mmode-round-trip",
        "sci004.section-10.uvfits-round-trips-an-mmode-result",
        "test_an_mmode_result_round_trips_through_uvfits",
    ),
    _phase3_uvfits_case(
        "m3.uvfits.history-names-the-mmode-conventions",
        "sci004.section-10.uvfits-history-names-the-mmode-conventions",
        "test_the_published_uvfits_history_names_the_mmode_conventions",
    ),
    _phase3_uvfits_case(
        "m3.uvfits.synthesized-utc-grid",
        "sci004.section-10.uvfits-carries-the-synthesized-utc-grid",
        "test_the_published_uvfits_carries_the_synthesized_utc_grid",
    ),
)

SCI004_PHASE3_RED_GREEN_CONTROLS: tuple[str, ...] = (_PHASE3_UVFITS_GREEN_CONTROL,)


def test_an_mmode_result_round_trips_through_uvfits(tmp_path: Path) -> None:
    """Section 10: four correlation products, zenith phase centre, m-mode arm.

    "UVFITS/MS keep the canonical zenith phase centre" is a statement about
    what the file *retains*, not about the frame it is phased to.  UVFITS has
    no altaz phase model, so the projection is a fixed ICRS reference -- the
    accepted ``ProjectedPhaseCenter`` fixes ``frame`` as the exact literal
    ``icrs`` and refuses anything else -- and the canonical zenith drift is kept
    beside it, byte for byte, in ``original_phase_snapshot``.  Both halves are
    asserted here, because only the pair is the requirement.
    """
    from radiosim.core.result import MMODE_SOLVER_SNAPSHOT_KEYS
    from tests.unit.test_io.test_standard_visibility import build_mmode_result

    result = build_mmode_result(tmp_path)
    target = tmp_path / "mmode.uvfits"

    assert write_uvfits(result, target) == target.absolute()
    loaded = read_uvfits(target)

    assert loaded.format == "uvfits"
    assert len(loaded.correlations) == 4
    assert loaded.visibilities.shape[-1] == 4
    assert loaded.source_scientific_sha256 == result.scientific_sha256
    assert loaded.phase_center.frame == "icrs"
    assert dict(loaded.phase_center.original_phase_snapshot) == dict(
        result.phase_center.to_snapshot()
    )
    assert loaded.phase_center.original_phase_snapshot["kind"] == "zenith_drift"

    record, _lines = projection_record_from_history("\n".join(loaded.history))
    solver = record["solver"]
    assert tuple(solver) == MMODE_SOLVER_SNAPSHOT_KEYS
    assert solver["solver"] == "mmode"
    assert solver["solver"] != "rime"


def test_the_published_uvfits_history_names_the_mmode_conventions(
    tmp_path: Path,
) -> None:
    """Section 10: "history lines naming the m-mode/frame/harmonic conventions"."""
    from tests.unit.test_io.test_standard_visibility import (
        MMODE_HISTORY_CONVENTIONS,
        build_mmode_result,
    )

    result = build_mmode_result(tmp_path)
    target = tmp_path / "mmode-history.uvfits"
    write_uvfits(result, target)

    with fits.open(target, memmap=False) as handle:
        assert handle[0].header["HISTORY"] is not None
    loaded = read_uvfits(target)
    # "History *lines*", so the embedded projection record does not count: that
    # JSON already carried the snapshot before this phase.
    plain = [
        line
        for line in loaded.history
        if not line.startswith(PROJECTION_HISTORY_PREFIX)
    ]

    for literal in MMODE_HISTORY_CONVENTIONS:
        assert any(literal in line for line in plain), literal


def test_the_published_uvfits_carries_the_synthesized_utc_grid(
    tmp_path: Path,
) -> None:
    """Section 10: every path writes the same synthesized centres and widths."""
    from tests.unit.test_io.test_standard_visibility import build_mmode_result

    result = build_mmode_result(tmp_path)
    grid = result.time_grid
    widths = np.asarray(grid.integration_time_seconds, dtype=np.float64)
    target = tmp_path / "mmode-time.uvfits"
    write_uvfits(result, target)

    loaded = read_uvfits(target)
    exposures = np.asarray(loaded.exposure_seconds, dtype=np.float64)
    centres = np.asarray(loaded.utc_jd1, dtype=np.float64) + np.asarray(
        loaded.utc_jd2, dtype=np.float64
    )
    expected = np.asarray(grid.utc_jd1, dtype=np.float64) + np.asarray(
        grid.utc_jd2, dtype=np.float64
    )

    assert len(set(widths.tolist())) > 1
    assert np.allclose(np.unique(exposures), np.unique(widths), rtol=0.0, atol=1e-6)
    assert np.allclose(np.unique(centres), np.unique(expected), rtol=0.0, atol=1e-9)
