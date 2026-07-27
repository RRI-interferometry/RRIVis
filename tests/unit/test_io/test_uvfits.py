"""Canonical UVFITS projection, reader, and publication contracts."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyDeprecationWarning
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
    StandardReadLimits,
    StandardVisibilityData,
    project_simulation_result,
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


def test_uvfits_rejects_truncated_and_symlink_inputs(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.uvfits"
    truncated.write_bytes(b"SIMPLE  =                    T")
    with pytest.warns(
        (AstropyDeprecationWarning, VerifyWarning),
    ) as caught:
        with pytest.raises(UnsafeResultInputError):
            read_uvfits(truncated)
    expected_categories = (
        [VerifyWarning]
        if sys.version_info >= (3, 12)
        else [AstropyDeprecationWarning, VerifyWarning]
    )
    assert [type(item.message) for item in caught] == expected_categories
    if sys.version_info < (3, 12):
        assert "indent function is deprecated" in str(caught[0].message)
    assert "Header size is not multiple of 2880" in str(caught[-1].message)

    destination = tmp_path / "destination.uvfits"
    destination.write_bytes(b"safe")
    linked = tmp_path / "linked.uvfits"
    linked.symlink_to(destination)
    with pytest.raises(UnsafeResultInputError):
        read_uvfits(linked)


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
