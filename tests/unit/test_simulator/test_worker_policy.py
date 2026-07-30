"""Tier 6B: the resolved worker policy is observable and not yet consumed.

Tier 6B owns configuration and resolution only.  Its explicit interim position
(``Tier6HybridRuntimePlan.md`` Section 32.2, "without changing any behavior
yet") is that both worker knobs are typed, resolved, clamped, and recorded while
no loader driver or solver reads them.  The tests below assert exactly that, so
the slice that makes each knob effective (6C for the loader, 6E for the solver)
must flip the interim assertion it invalidates.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from radiosim.api import Simulator
from tests.fixtures.configs import valid_config_mapping


def _run(tmp_path: Path, **overrides: object):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return Simulator.from_mapping(
        valid_config_mapping(tmp_path, **overrides),
        base_dir=tmp_path,
    ).run(progress=False)


def test_tier6b_resolved_worker_policy_appears_in_the_result_snapshot(tmp_path):
    result = _run(
        tmp_path,
        execution={
            "sky_loading": {"max_workers": 4, "executor": "thread"},
            "solver": {"workers": 2},
        },
    )
    execution = result.resolved_config["execution"]

    assert execution["sky_loading"]["max_workers"] == 4
    assert execution["sky_loading"]["executor"] == "thread"
    assert execution["solver"]["workers"] == 2
    assert execution["solver"]["executor"] == "thread"


def test_tier6b_auto_loader_policy_is_resolved_before_the_result_is_recorded(tmp_path):
    result = _run(tmp_path)
    sky_loading = result.resolved_config["execution"]["sky_loading"]

    assert sky_loading["max_workers"] == min(1, os.cpu_count() or 1, 8)
    assert sky_loading["max_workers"] is not None
    assert sky_loading["executor"] == "auto"


def test_tier6b_resolved_worker_policy_appears_in_the_summary_json(tmp_path):
    result = _run(
        tmp_path,
        execution={
            "sky_loading": {"max_workers": 6, "executor": "process"},
            "solver": {"workers": 1},
        },
    )
    write_result_summary_json = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json
    target = tmp_path / "result.summary.json"

    write_result_summary_json(result, target)
    document = json.loads(target.read_text(encoding="utf-8"))
    execution = document["resolved_config"]["execution"]

    assert execution["sky_loading"] == {"max_workers": 6, "executor": "process"}
    assert execution["solver"] == {"workers": 1, "executor": "thread"}


def test_tier6b_worker_policy_never_changes_the_scientific_fingerprint(tmp_path):
    serial = _run(tmp_path)
    parallel = _run(
        tmp_path,
        execution={
            "sky_loading": {"max_workers": 4, "executor": "thread"},
            "solver": {"workers": 2},
        },
    )

    assert serial.scientific_sha256 == parallel.scientific_sha256
    assert serial.provenance_sha256 != parallel.provenance_sha256


def test_tier6b_loader_driver_does_not_yet_read_the_resolved_policy(
    tmp_path, monkeypatch
):
    """Interim boundary: OWNED BY Tier 6C, which makes the loader policy real."""
    parallel = importlib.import_module("radiosim.core.sky.operations.parallel")
    observed: list[object] = []
    real_loader = parallel.load_models_parallel

    def recording_loader(requests, *args, **kwargs):
        observed.append(kwargs.get("max_workers"))
        return real_loader(requests, *args, **kwargs)

    monkeypatch.setattr(parallel, "load_models_parallel", recording_loader)
    simulator = Simulator.from_mapping(
        valid_config_mapping(
            tmp_path,
            execution={"sky_loading": {"max_workers": 2}},
        ),
        base_dir=tmp_path,
    )
    simulator.setup()

    assert observed == [8]
    assert simulator.config.execution.sky_loading.max_workers == 2


def test_tier6b_solver_does_not_yet_read_the_resolved_worker_count(tmp_path):
    """Interim boundary: OWNED BY Tier 6E, which makes the solver policy real."""
    import inspect

    from radiosim.core import visibility, visibility_healpix
    from radiosim.simulator import rime

    for module in (visibility, visibility_healpix, rime):
        source = inspect.getsource(module)
        assert "ThreadPoolExecutor" not in source
        assert "solver.workers" not in source
