"""Tier 4F result-format and truthful summary JSON contracts."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

from radiosim.api.simulator import Simulator
from radiosim.core.result import ResultUnavailableError
from radiosim.io.result_errors import (
    AtomicWriteError,
    OutputPathError,
    OverwriteRefusedError,
    SummaryContractError,
)
from tests.unit.test_core.test_result import _build


def test_result_format_module_exposes_exact_typed_values_and_extensions():
    module = importlib.import_module("radiosim.io.result_format")
    result_format = module.ResultFormat

    assert [item.value for item in result_format] == [
        "hdf5",
        "summary_json",
        "ms",
        "uvfits",
    ]
    assert result_format.HDF5.extension == ".h5"
    assert result_format.SUMMARY_JSON.extension == ".summary.json"
    assert result_format.MS.extension == ".ms"
    assert result_format.UVFITS.extension == ".uvfits"


@pytest.mark.parametrize(
    ("format_name", "extension"),
    [
        ("HDF5", ".h5"),
        ("SUMMARY_JSON", ".summary.json"),
        ("MS", ".ms"),
        ("UVFITS", ".uvfits"),
    ],
)
def test_every_result_format_has_exact_extension_normalization(
    tmp_path,
    format_name,
    extension,
):
    module = importlib.import_module("radiosim.io.result_format")
    result_format = getattr(module.ResultFormat, format_name)

    assert module.normalize_result_path(tmp_path / "result", result_format) == (
        tmp_path / f"result{extension}"
    )
    assert module.normalize_result_path(
        tmp_path / f"result{extension}",
        result_format,
    ) == (tmp_path / f"result{extension}")
    with pytest.raises(OutputPathError, match="conflicts"):
        module.normalize_result_path(tmp_path / "result.wrong", result_format)


def test_result_format_exports_preserve_identity_without_heavy_imports():
    import radiosim
    import radiosim.api
    import radiosim.io

    result_format = importlib.import_module("radiosim.io.result_format").ResultFormat

    assert radiosim.ResultFormat is result_format
    assert radiosim.api.ResultFormat is result_format
    assert radiosim.io.ResultFormat is result_format


def test_summary_writer_has_exact_public_signature():
    writer = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json

    assert list(inspect.signature(writer).parameters) == [
        "result",
        "path",
        "overwrite",
    ]


def test_summary_json_is_exact_bounded_metadata_contract(tmp_path):
    result, _ = _build(tmp_path)
    writer = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json

    target = writer(result, tmp_path / "result", overwrite=False)
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert target.name == "result.summary.json"
    assert list(payload) == sorted(payload)
    assert set(payload) == {
        "schema",
        "result",
        "observation",
        "frequency",
        "correlation",
        "instrument",
        "phase_center",
        "beam",
        "backend",
        "solver",
        "resolved_config",
        "configuration_provenance",
        "performance",
        "history",
        "excluded_payloads",
    }
    assert payload["schema"] == {
        "name": "radiosim.result-summary",
        "version": "1.0.0",
    }
    assert payload["excluded_payloads"] == [
        "visibility_samples",
        "flags_array",
        "weights_array",
        "full_time_coordinate",
        "full_frequency_coordinate",
        "per_baseline_geometry",
        "per_antenna_geometry",
    ]
    encoded = target.read_bytes()
    assert encoded.endswith(b"\n")
    assert len(encoded) <= 16 * 1024 * 1024
    assert "visibilities" not in payload
    assert "workflow" not in payload["resolved_config"]
    if payload["configuration_provenance"] is not None:
        assert "workflow" not in payload["configuration_provenance"]
    assert result.scientific_sha256 == payload["result"]["scientific_sha256"]
    assert result.provenance_sha256 == payload["result"]["provenance_sha256"]
    payload["resolved_config"].clear()
    payload["history"].append("mutated parsed payload")
    assert result.resolved_config
    assert "mutated parsed payload" not in result.history


def test_simulator_save_rejects_absent_result_before_path_or_writer_work(
    tmp_path, monkeypatch
):
    from radiosim.io.result_format import ResultFormat

    simulator = object.__new__(Simulator)
    simulator._result = None

    def forbidden(*args, **kwargs):
        pytest.fail("absent result crossed the filesystem/writer boundary")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    with pytest.raises(ResultUnavailableError):
        simulator.save(tmp_path / "result", format=ResultFormat.HDF5)


def test_simulator_save_signature_is_final_path_and_typed_format():
    from radiosim.io.result_format import ResultFormat

    parameters = inspect.signature(Simulator.save).parameters

    assert list(parameters) == ["self", "path", "format", "overwrite"]
    assert parameters["path"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert parameters["format"].default is ResultFormat.HDF5


def test_summary_rejects_wrong_result_type_before_path_mutation(tmp_path):
    writer = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json
    output_parent = tmp_path / "must-not-exist"

    with pytest.raises(SummaryContractError, match="invalid summary metadata"):
        writer(object(), output_parent / "summary")

    assert not output_parent.exists()


def test_summary_wrong_extension_and_collision_leave_existing_bytes_unchanged(
    tmp_path,
):
    result, _ = _build(tmp_path)
    writer = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json
    wrong = tmp_path / "result.json"

    with pytest.raises(OutputPathError, match="conflicts"):
        writer(result, wrong)
    assert not wrong.exists()

    target = writer(result, tmp_path / "result")
    original = target.read_bytes()
    with pytest.raises(OverwriteRefusedError):
        writer(result, target)
    assert target.read_bytes() == original
    assert writer(result, target, overwrite=True) == target


def test_summary_write_failure_removes_temporary_and_final_paths(tmp_path, monkeypatch):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    target = tmp_path / "failed.summary.json"

    def fail_write(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(module.os, "write", fail_write)
    with pytest.raises(AtomicWriteError):
        module.write_result_summary_json(result, target)

    assert not target.exists()
    assert not list(tmp_path.glob(".failed.summary.json.*.tmp"))


def test_summary_encoded_limit_accepts_exact_boundary_and_rejects_one_byte_more(
    tmp_path, monkeypatch
):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    overhead = len(
        (
            json.dumps(
                {"padding": ""},
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    exact_padding = "x" * (16 * 1024 * 1024 - overhead)

    monkeypatch.setattr(
        module,
        "_summary_payload",
        lambda observed: {"padding": exact_padding},
    )
    assert len(module._encode_summary(result)) == 16 * 1024 * 1024
    exact_target = module.write_result_summary_json(result, tmp_path / "exact")
    assert exact_target.stat().st_size == 16 * 1024 * 1024

    monkeypatch.setattr(
        module,
        "_summary_payload",
        lambda observed: {"padding": exact_padding + "x"},
    )
    rejected_parent = tmp_path / "rejected"
    with pytest.raises(SummaryContractError, match="16 MiB"):
        module.write_result_summary_json(result, rejected_parent / "summary")
    assert not rejected_parent.exists()


def test_summary_nonfinite_serialization_failure_precedes_path_mutation(
    tmp_path, monkeypatch
):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    output_parent = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        module,
        "_summary_payload",
        lambda observed: {"invalid": float("nan")},
    )

    with pytest.raises(SummaryContractError, match="invalid summary metadata"):
        module.write_result_summary_json(result, output_parent / "summary")

    assert not output_parent.exists()


def test_summary_recursion_failure_is_typed_and_precedes_path_mutation(
    tmp_path,
    monkeypatch,
):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    output_parent = tmp_path / "must-not-exist"
    nested: object = None
    for _ in range(2_000):
        nested = [nested]
    monkeypatch.setattr(
        module,
        "_summary_payload",
        lambda observed: {"nested": nested},
    )

    with pytest.raises(SummaryContractError, match="invalid summary metadata"):
        module.write_result_summary_json(result, output_parent / "summary")

    assert not output_parent.exists()


def test_summary_nul_text_is_rejected_before_path_mutation(tmp_path, monkeypatch):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    output_parent = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        module,
        "_summary_payload",
        lambda observed: {"invalid": "contains\x00nul"},
    )

    with pytest.raises(SummaryContractError, match="invalid summary metadata"):
        module.write_result_summary_json(result, output_parent / "summary")

    assert not output_parent.exists()
