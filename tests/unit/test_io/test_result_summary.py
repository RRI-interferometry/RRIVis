"""Tier 4F result-format and truthful summary JSON contracts."""

from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
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


def _exception_chain(error: BaseException) -> tuple[type[BaseException], ...]:
    chain: list[type[BaseException]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(type(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _nested_summary_payload(nesting: int, kind: str) -> dict[str, object]:
    value: object = None
    for level in range(nesting - 1):
        container = kind
        if kind == "alternating":
            container = "list" if level % 2 == 0 else "dict"
        if container == "list":
            value = [value]
        elif container == "tuple":
            value = (value,)
        else:
            value = {"nested": value}
    return {"nested": value}


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
        "receptors",
        "instrument",
        "phase_center",
        "beam",
        "backend",
        "solver",
        "execution",
        "resolved_config",
        "configuration_provenance",
        "performance",
        "history",
        "excluded_payloads",
    }
    # Tier 6G, plan Section 19: additive growth over ``1.0.0`` (the Tier 6C
    # ``execution`` block and the Tier 6F component fields), so the summary
    # takes a minor bump where the HDF5 schema takes a major one.
    assert payload["schema"] == {
        "name": "radiosim.result-summary",
        "version": "1.1.0",
    }
    assert payload["excluded_payloads"] == [
        "visibility_samples",
        "flags_array",
        "weights_array",
        "full_time_coordinate",
        "full_frequency_coordinate",
        "per_baseline_geometry",
        "per_antenna_geometry",
        "per_antenna_receptor_definitions",
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


# ---------------------------------------------------------------------------
# Tier 5F: the bounded receptor block (Section 23)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("receptors", "basis", "labels", "native_counts", "rotations"),
    [
        (
            None,
            "linear_xy",
            ["XX", "XY", "YX", "YY"],
            {"linear": 2, "circular": 0},
            [0.0],
        ),
        (
            {"default": {"basis": "circular"}},
            "circular_rl",
            ["RR", "RL", "LR", "LL"],
            {"linear": 0, "circular": 2},
            [0.0],
        ),
        (
            {
                "default": {"basis": "linear", "feed_rotation_deg": 30.0},
                "overrides": [
                    {
                        "antenna": {"kind": "number", "number": 1},
                        "feed_rotation_deg": -15.0,
                    }
                ],
            },
            "linear_xy",
            ["XX", "XY", "YX", "YY"],
            {"linear": 2, "circular": 0},
            [-15.0, 30.0],
        ),
    ],
    ids=["default_linear", "circular", "rotated_heterogeneous"],
)
def test_summary_receptor_block_is_truthful_and_bounded(
    tmp_path,
    receptors,
    basis,
    labels,
    native_counts,
    rotations,
):
    result, _ = _build(tmp_path, receptors=receptors)
    writer = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json

    target = writer(result, tmp_path / "result", overwrite=False)
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["correlation"] == {"labels": labels, "basis": basis}
    assert set(payload["receptors"]) == {
        "output_basis",
        "receptor_sha256",
        "native_basis_counts",
        "distinct_feed_rotations_deg",
    }
    assert payload["receptors"]["output_basis"] == basis
    assert (
        payload["receptors"]["receptor_sha256"]
        == result.receptors.provenance.receptor_sha256
    )
    assert payload["receptors"]["native_basis_counts"] == native_counts
    assert payload["receptors"]["distinct_feed_rotations_deg"] == rotations
    # Per-antenna receptor rows stay out of the bounded summary.
    assert "receptors" not in payload["receptors"]
    assert "feed_angle_rad" not in target.read_text(encoding="utf-8")
    assert "per_antenna_receptor_definitions" in payload["excluded_payloads"]


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

    with pytest.raises(SummaryContractError, match="nesting") as caught:
        module.write_result_summary_json(result, output_parent / "summary")

    assert RecursionError not in _exception_chain(caught.value)
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


@pytest.mark.parametrize("kind", ["list", "dict", "alternating", "tuple"])
def test_summary_explicit_nesting_boundary_is_exact_and_pre_filesystem(
    tmp_path,
    monkeypatch,
    kind,
):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    limit = module._MAX_SUMMARY_NESTING
    accepted = _nested_summary_payload(limit, kind)
    monkeypatch.setattr(module, "_summary_payload", lambda observed: accepted)
    accepted_target = module.write_result_summary_json(
        result,
        tmp_path / f"accepted-{kind}",
    )
    assert accepted_target.is_file()

    rejected = _nested_summary_payload(limit + 1, kind)
    monkeypatch.setattr(module, "_summary_payload", lambda observed: rejected)
    rejected_parent = tmp_path / f"rejected-{kind}"
    with pytest.raises(SummaryContractError, match="nesting") as caught:
        module.write_result_summary_json(result, rejected_parent / "summary")

    assert RecursionError not in _exception_chain(caught.value)
    assert not rejected_parent.exists()


def test_summary_explicit_node_boundary_precedes_json_serialization(
    tmp_path,
    monkeypatch,
):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    limit = module._MAX_SUMMARY_NODES
    exact = {"items": [None] * (limit - 2)}
    monkeypatch.setattr(module, "_summary_payload", lambda observed: exact)
    assert module._encode_summary(result)

    over = {"items": [None] * (limit - 1)}
    monkeypatch.setattr(module, "_summary_payload", lambda observed: over)
    monkeypatch.setattr(
        module.json,
        "dumps",
        lambda *args, **kwargs: pytest.fail(
            "over-limit summary reached JSON serialization"
        ),
    )
    rejected_parent = tmp_path / "node-limit-rejected"
    with pytest.raises(SummaryContractError, match="node"):
        module.write_result_summary_json(result, rejected_parent / "summary")
    assert not rejected_parent.exists()


def test_very_deep_summary_is_explicitly_rejected_in_fresh_process(tmp_path):
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        from radiosim.io import summary_json
        from radiosim.io.result_errors import SummaryContractError
        from tests.unit.test_core.test_result import _build

        root = Path({str(tmp_path)!r})
        result, _ = _build(root)
        nested = None
        for _ in range(2_000):
            nested = [nested]
        summary_json._summary_payload = lambda observed: {{"nested": nested}}
        sys.setrecursionlimit(100)
        parent = root / "subprocess-must-not-exist"
        try:
            summary_json.write_result_summary_json(result, parent / "summary")
        except SummaryContractError as error:
            current = error
            seen = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                if isinstance(current, RecursionError):
                    raise AssertionError("raw RecursionError in exception chain")
                current = current.__cause__ or current.__context__
            assert "nesting" in str(error)
            assert not parent.exists()
        else:
            raise AssertionError("deep summary was accepted")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("cycle_kind", ["list", "dict", "mutual"])
def test_summary_cycles_are_rejected_without_recursion_or_path_mutation(
    tmp_path,
    monkeypatch,
    cycle_kind,
):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    if cycle_kind == "list":
        value: object = []
        value.append(value)
    elif cycle_kind == "dict":
        value = {}
        value["self"] = value
    else:
        list_value: list[object] = []
        dict_value = {"list": list_value}
        list_value.append(dict_value)
        value = list_value
    monkeypatch.setattr(
        module,
        "_summary_payload",
        lambda observed: {"cycle": value},
    )
    output_parent = tmp_path / f"cycle-{cycle_kind}"

    with pytest.raises(SummaryContractError, match="cycle|alias") as caught:
        module.write_result_summary_json(result, output_parent / "summary")

    assert RecursionError not in _exception_chain(caught.value)
    assert not output_parent.exists()


def test_summary_detaches_repeated_container_aliases_within_node_limit(
    tmp_path,
    monkeypatch,
):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    shared = ["detached value"]
    monkeypatch.setattr(
        module,
        "_summary_payload",
        lambda observed: {"first": shared, "second": shared},
    )
    target = module.write_result_summary_json(result, tmp_path / "alias-bounded")
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload == {
        "first": ["detached value"],
        "second": ["detached value"],
    }
    assert payload["first"] is not payload["second"]


def test_summary_rejects_wide_mapping_before_serialization(
    tmp_path,
    monkeypatch,
):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    wide = {f"key-{index}": None for index in range(module._MAX_SUMMARY_NODES)}
    monkeypatch.setattr(module, "_summary_payload", lambda observed: wide)
    monkeypatch.setattr(
        module.json,
        "dumps",
        lambda *args, **kwargs: pytest.fail(
            "wide over-limit mapping reached JSON serialization"
        ),
    )
    output_parent = tmp_path / "wide-rejected"

    with pytest.raises(SummaryContractError, match="node"):
        module.write_result_summary_json(result, output_parent / "summary")

    assert not output_parent.exists()


@pytest.mark.parametrize(
    "invalid",
    [
        {"nul-key\x00": "value"},
        {"invalid-unicode": "\ud800"},
        {"nan": float("nan")},
        {"positive-infinity": float("inf")},
        {"negative-infinity": float("-inf")},
        {1: "invalid key"},
        {"object-array": np.asarray([object()], dtype=object)},
        {"datetime-scalar": np.datetime64("2026-01-01")},
    ],
)
def test_summary_rejects_broad_invalid_values_before_path_mutation(
    tmp_path,
    monkeypatch,
    invalid,
):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    monkeypatch.setattr(module, "_summary_payload", lambda observed: invalid)
    output_parent = tmp_path / "invalid-rejected"

    with pytest.raises(SummaryContractError):
        module.write_result_summary_json(result, output_parent / "summary")

    assert not output_parent.exists()


def test_summary_accepts_supported_finite_numpy_scalars(tmp_path, monkeypatch):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")
    monkeypatch.setattr(
        module,
        "_summary_payload",
        lambda observed: {
            "boolean": np.bool_(True),
            "float": np.float64(1.25),
            "integer": np.int64(7),
            "string": np.str_("valid"),
        },
    )

    target = module.write_result_summary_json(result, tmp_path / "numpy-scalars")

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "boolean": True,
        "float": 1.25,
        "integer": 7,
        "string": "valid",
    }


def test_summary_rejects_hostile_container_subclasses_without_calling_hooks(
    tmp_path,
    monkeypatch,
):
    result, _ = _build(tmp_path)
    module = importlib.import_module("radiosim.io.summary_json")

    class HostileList(list):
        def __iter__(self):
            pytest.fail("unsupported list subclass iterator executed")

    class HostileDict(dict):
        def items(self):
            pytest.fail("unsupported dict subclass items executed")

    for index, invalid in enumerate((HostileList([1]), HostileDict(value=1))):
        monkeypatch.setattr(
            module,
            "_summary_payload",
            lambda observed, invalid=invalid: {"invalid": invalid},
        )
        output_parent = tmp_path / f"hostile-{index}"
        with pytest.raises(SummaryContractError, match="unsupported"):
            module.write_result_summary_json(result, output_parent / "summary")
        assert not output_parent.exists()


# ---------------------------------------------------------------------------
# Tier 6G: the summary reports every solved component (plan Section 19; H10)
# ---------------------------------------------------------------------------


def test_summary_solver_block_reports_the_solved_components(tmp_path):
    """H10 half one: the components and counts reach the summary document."""
    result, _ = _build(tmp_path)
    writer = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json

    target = writer(result, tmp_path / "components", overwrite=False)
    payload = json.loads(target.read_text(encoding="utf-8"))

    solver = payload["solver"]
    assert set(solver) == {
        "solver",
        "sky_representation",
        "convention",
        "execution_path",
        "components",
        "component_element_counts",
    }
    assert solver["sky_representation"] == result.solver.sky_representation
    assert solver["components"] == list(result.solver.components)
    assert solver["component_element_counts"] == list(
        result.solver.component_element_counts
    )
    assert (
        solver["sky_representation"]
        == (payload["resolved_config"]["visibility"]["sky_representation"])
    )


def test_summary_performance_block_reports_both_component_timings(tmp_path):
    """The two per-component timings travel with the run that produced them."""
    result, _ = _build(tmp_path)
    writer = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json

    target = writer(result, tmp_path / "timings", overwrite=False)
    payload = json.loads(target.read_text(encoding="utf-8"))

    performance = payload["performance"]
    assert performance["solver_point_seconds"] == (
        result.performance.solver_point_seconds
    )
    assert performance["solver_healpix_seconds"] == (
        result.performance.solver_healpix_seconds
    )
    assert (
        performance["solver_point_seconds"] + performance["solver_healpix_seconds"]
        <= performance["solver_seconds"]
    )
    # Nondeterministic timings stay out of both fingerprints (Section 9.4).
    assert "solver_point_seconds" not in json.dumps(payload["result"])
