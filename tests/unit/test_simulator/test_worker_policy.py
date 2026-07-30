"""Worker policy: typed and resolved in Tier 6B, effective for the loader in 6C.

Tier 6B owned configuration and resolution only -- both worker knobs were typed,
resolved, clamped, and recorded while no loader driver or solver read them.  Tier
6C makes the *loader* half effective (``Tier6HybridRuntimePlan.md`` Sections 11.2,
16.1, 32.3) and flips the interim pin that recorded the hard-coded pool size.
The solver half stays interim until 6E, which must flip its own pin below.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from radiosim.api import Simulator
from radiosim.core.sky.operations.parallel import LoaderExecutionRecord
from radiosim.utils.network import set_offline_policy
from tests.fixtures.configs import valid_config_mapping

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _never_leak_an_offline_policy():
    """A forced-offline run must not silence a later test's network gate."""
    set_offline_policy(False)
    yield
    set_offline_policy(False)


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


def test_tier6c_loader_driver_receives_the_resolved_worker_count(tmp_path, monkeypatch):
    """Flipped by Tier 6C from the 6B interim pin.

    The 6B pin asserted ``observed == [8]``: the loader driver was handed a
    literal while the typed policy sat unread.  Tier 6C removed the literal
    (plan Section 11.2), so the driver now receives exactly the resolved value.
    """
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

    assert observed == [2]
    assert simulator.config.execution.sky_loading.max_workers == 2
    assert simulator._resolved.execution.sky_loading.max_workers == 2


def test_tier6c_no_worker_literal_survives_in_the_simulator():
    """Section 27 W7 -- the api layer names no pool size of its own."""
    import re

    source = (_REPO_ROOT / "src" / "radiosim" / "api" / "simulator.py").read_text(
        encoding="utf-8"
    )

    assert "max_workers=8" not in source
    assert re.search(r"max_workers\s*=\s*\d", source) is None
    assert "sky_loading.max_workers" in source


def test_tier6c_the_loader_execution_record_is_observable_on_the_simulator(tmp_path):
    """Section 11.2 -- the driver reports what it actually did."""
    simulator = Simulator.from_mapping(
        valid_config_mapping(
            tmp_path,
            execution={"sky_loading": {"max_workers": 3, "executor": "thread"}},
        ),
        base_dir=tmp_path,
    )
    assert simulator.loader_execution is None

    simulator.setup()
    record = simulator.loader_execution

    assert record is not None
    assert record.requested_executor == "thread"
    assert record.actual_executor == "thread"
    assert record.max_workers == 3
    assert record.degraded_reason is None


def test_tier6c_auto_loader_policy_records_the_registry_choice(tmp_path):
    """Section 27 W2 -- the *actual* executor is recorded, not just the request."""
    result = _run(tmp_path)
    record = LoaderExecutionRecord.from_history(result.history)

    assert record is not None
    assert record.requested_executor == "auto"
    # The default fixture requests one ``test_sources`` loader, a synthetic
    # (GIL-bound) category, so the registry recommends a process pool.
    assert record.actual_executor == "process"
    assert record.max_workers == min(1, os.cpu_count() or 1, 8)
    assert record.degraded_reason is None


def test_tier6c_the_summary_json_execution_block_reports_the_policy(tmp_path):
    """Section 19 / Section 27 W2 -- resolved policy *and* executed record."""
    result = _run(
        tmp_path,
        execution={
            "sky_loading": {"max_workers": 2, "executor": "thread"},
            "solver": {"workers": 1},
        },
    )
    write_result_summary_json = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json
    target = tmp_path / "execution.summary.json"

    write_result_summary_json(result, target)
    document = json.loads(target.read_text(encoding="utf-8"))
    execution = document["execution"]

    assert execution["sky_loading"] == {"max_workers": 2, "executor": "thread"}
    assert execution["solver"] == {"workers": 1, "executor": "thread"}
    assert execution["loader"] == {
        "requested_executor": "thread",
        "actual_executor": "thread",
        "max_workers": 2,
        "degraded_reason": None,
    }
    # The shared fixture forces offline, and the executed policy says so.
    assert execution["offline"] is True


@pytest.mark.parametrize("max_workers", [1, 2, 4, 8])
@pytest.mark.parametrize("executor", ["thread", "process"])
def test_tier6c_loader_worker_invariance(tmp_path, max_workers, executor):
    """Section 27 W1 / invariant S7 -- the loader policy is scientifically inert.

    Reference and varied runs share one ``tmp_path`` because the antenna-layout
    path is part of the instrument identity, and therefore of the scientific
    digest; only the worker policy may differ between the two runs.
    """
    reference = _run(tmp_path)
    varied = _run(
        tmp_path,
        execution={"sky_loading": {"max_workers": max_workers, "executor": executor}},
    )

    assert varied.scientific_sha256 == reference.scientific_sha256
    assert np.array_equal(varied.visibilities, reference.visibilities)


def test_tier6c_offline_policy_is_installed_before_any_loader_runs(
    tmp_path, monkeypatch
):
    """Section 16.1 / Section 20.1 step 6 precedes step 7."""
    network = importlib.import_module("radiosim.utils.network")
    parallel = importlib.import_module("radiosim.core.sky.operations.parallel")
    events: list[str] = []

    real_set = network.set_offline_policy
    real_loader = parallel.load_models_parallel

    def recording_set(offline):
        events.append(f"offline:{offline}")
        return real_set(offline)

    def recording_loader(*args, **kwargs):
        events.append("load")
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(network, "set_offline_policy", recording_set)
    monkeypatch.setattr(parallel, "load_models_parallel", recording_loader)

    Simulator.from_mapping(
        valid_config_mapping(tmp_path, execution={"offline": True}),
        base_dir=tmp_path,
    ).setup()

    assert events == ["offline:True", "load"]
    network.set_offline_policy(False)


def test_tier6b_solver_does_not_yet_read_the_resolved_worker_count(tmp_path):
    """Interim boundary: OWNED BY Tier 6E, which makes the solver policy real."""
    import inspect

    from radiosim.core import visibility, visibility_healpix
    from radiosim.simulator import rime

    for module in (visibility, visibility_healpix, rime):
        source = inspect.getsource(module)
        assert "ThreadPoolExecutor" not in source
        assert "solver.workers" not in source
