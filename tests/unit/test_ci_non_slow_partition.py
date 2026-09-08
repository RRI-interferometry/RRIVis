"""Synthetic subprocess controls: no scientific package or actual diagnostic run."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools/ci_non_slow_partition.py"
NAMES = (
    "test_production_v2_retains_the_complete_owned_input_preimage",
    "test_production_v2_rejects_a_rehashed_foreign_phase_against_unchanged_result",
)


def _case(
    tmp_path: Path, body: str, conftest: str = ""
) -> tuple[list[str], dict[str, str]]:
    git_env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    for command in (
        ["git", "init", "-q", str(tmp_path)],
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=synthetic@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "synthetic",
        ],
    ):
        _ = subprocess.run(
            command, env=git_env, check=True, capture_output=True, timeout=10
        )
    (tmp_path / "tools").mkdir()
    _ = shutil.copyfile(TOOL, tmp_path / "tools/ci_non_slow_partition.py")
    (tmp_path / "src/radiosim").mkdir(parents=True)
    _ = (tmp_path / "src/radiosim/__init__.py").write_text("VALUE = 1\n")
    (tmp_path / "tests/unit").mkdir(parents=True)
    _ = (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nmarkers=["synthetic: infrastructure control"]\n'
    )
    _ = (tmp_path / "conftest.py").write_text(conftest)
    owner = "tests/unit/test_sci004_phase3_evidence.py"
    _ = (tmp_path / owner).write_text(body)
    env = os.environ.copy()
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONPATH=str(tmp_path) + os.pathsep + str(tmp_path / "src"),
        PYTHONNOUSERSITE="1",
    )
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        _ = env.pop(name, None)
    args = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "tools.ci_non_slow_partition",
        "--ci-diagnostic-output=records",
        "-q",
        *(owner + "::" + name for name in NAMES),
    ]
    return args, env


def _run(
    tmp_path: Path, body: str, conftest: str = "", extra: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    args, env = _case(tmp_path, body, conftest)
    return subprocess.run(
        [*args, *extra],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _events(tmp_path: Path) -> list[dict[str, object]]:
    snapshot = (tmp_path / "records/events.jsonl").read_bytes()
    if snapshot and not snapshot.endswith(b"\n"):
        raise ValueError("unterminated terminal record")
    return [json.loads(line) for line in snapshot.split(b"\n")[:-1]]


def _poll_events(tmp_path: Path) -> list[dict[str, object]]:
    snapshot = (tmp_path / "records/events.jsonl").read_bytes()
    prefix = snapshot[: snapshot.rfind(b"\n") + 1]
    return [json.loads(line) for line in prefix.split(b"\n")[:-1]]


def test_diagnostic_preserves_natural_import_and_records(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        f"def {NAMES[0]}():\n import radiosim\n assert radiosim.VALUE == 1\ndef {NAMES[1]}():\n pass\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = _events(tmp_path)
    origins = [row for row in events if row["event"] == "origins"]
    assert origins[0]["not_loaded"] is True
    assert any("radiosim" in cast(dict[str, object], row["modules"]) for row in origins)
    assert [row["nodeid"] for row in events if row["event"] == "end"] == [
        "tests/unit/test_sci004_phase3_evidence.py::" + name for name in NAMES
    ]


@pytest.mark.parametrize("extra", [("-x",), ("--collect-only",), ("-k", NAMES[0])])
def test_diagnostic_refuses_changed_execution(
    tmp_path: Path, extra: tuple[str, ...]
) -> None:
    result = _run(
        tmp_path,
        "\n".join(f"def {name}():\n assert False, 'BODY_EXECUTED'" for name in NAMES),
        extra=extra,
    )
    assert result.returncode != 0
    assert "BODY_EXECUTED" not in result.stdout + result.stderr
    assert any(row["event"] == "infrastructure_failure" for row in _events(tmp_path))


def test_diagnostic_refuses_foreign_natural_owner(tmp_path: Path) -> None:
    body = f"def {NAMES[0]}():\n import sys, types\n m = types.ModuleType('radiosim')\n m.__file__ = '/foreign/radiosim.py'\n sys.modules['radiosim'] = m\ndef {NAMES[1]}():\n assert False, 'SECOND_BODY'\n"
    result = _run(tmp_path, body)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "foreign origin: radiosim" in result.stderr
    assert "SECOND_BODY" not in result.stdout + result.stderr


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_failure_is_flushed_before_blocked_later_test(
    tmp_path: Path, phase: str
) -> None:
    fixture = ""
    argument = ""
    if phase != "call":
        argument = "guard"
        code = (
            "assert False, 'FIRST_FAILURE'"
            if phase == "setup"
            else "yield\n assert False, 'FIRST_FAILURE'"
        )
        fixture = f"import pytest\n@pytest.fixture\ndef guard():\n {code}\n"
    first = "assert False, 'FIRST_FAILURE'" if phase == "call" else "pass"
    body = f"def {NAMES[0]}({argument}):\n {first}\ndef {NAMES[1]}():\n import time\n time.sleep(60)\n"
    args, env = _case(tmp_path, body, fixture)
    with (
        (tmp_path / "stdout.log").open("w") as stdout,
        (tmp_path / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            args, cwd=tmp_path, env=env, stdout=stdout, stderr=stderr, text=True
        )
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if (tmp_path / "records/events.jsonl").exists():
                    events = _poll_events(tmp_path)
                    if any(
                        row["event"] == "start"
                        and str(row["nodeid"]).endswith(NAMES[1])
                        for row in events
                    ):
                        break
                if process.poll() is not None:
                    pytest.fail((tmp_path / "stderr.log").read_text())
                time.sleep(0.02)
            else:
                pytest.fail("second synthetic test did not start")
            assert process.poll() is None
            failures = [
                row
                for row in events
                if row["event"] == "report" and row["outcome"] == "failed"
            ]
            assert failures and failures[0]["phase"] == phase
            assert "FIRST_FAILURE" in str(failures[0]["longrepr"])
            assert "FIRST_FAILURE" in (tmp_path / "stderr.log").read_text()
        finally:
            process.terminate()
            _ = process.wait(timeout=5)


@pytest.mark.parametrize(
    "mutation",
    [
        "session.config.option.maxfail = 1",
        "session.config.option.plugins.append('late-option')",
    ],
)
def test_diagnostic_refuses_collection_option_mutation(
    tmp_path: Path, mutation: str
) -> None:
    result = _run(
        tmp_path,
        "\n".join(f"def {name}():\n assert False, 'BODY_EXECUTED'" for name in NAMES),
        f"def pytest_collection_modifyitems(session):\n {mutation}\n",
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "collection configuration drift: options" in result.stderr
    assert "BODY_EXECUTED" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "mutation, reason",
    [
        ("radiosim.__file__ = None", "missing/string origin required"),
        ("radiosim.__file__ = 123", "missing/string origin required"),
        ("Path(radiosim.__file__).unlink()", "source unreadable"),
        ("Path(radiosim.__file__).write_text('CHANGED = 1')", "source drift"),
        ("sys.modules.pop('radiosim')", "loaded origin disappeared"),
    ],
)
def test_natural_owner_refusals(tmp_path: Path, mutation: str, reason: str) -> None:
    body = f"def {NAMES[0]}():\n import radiosim\ndef {NAMES[1]}():\n import radiosim, sys\n from pathlib import Path\n {mutation}\n"
    result = _run(tmp_path, body)
    assert result.returncode == 3, result.stdout + result.stderr
    assert reason in result.stderr
    assert any(row["event"] == "infrastructure_failure" for row in _events(tmp_path))


def test_reporting_error_retains_original_stream(tmp_path: Path) -> None:
    body = f"def {NAMES[0]}():\n from pathlib import Path\n p = Path('records/events.jsonl')\n p.rename('records/events.saved')\n p.mkdir()\ndef {NAMES[1]}():\n assert False, 'SECOND_BODY'\n"
    result = _run(tmp_path, body)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "diagnostic reporting failed" in result.stderr
    assert (tmp_path / "records/events.saved").is_file()
    assert "SECOND_BODY" not in result.stdout + result.stderr


def test_diagnostic_refuses_late_collection_wrapper(tmp_path: Path) -> None:
    conftest = "import pytest\n@pytest.hookimpl(hookwrapper=True, tryfirst=True)\ndef pytest_collection_finish(session):\n yield\n session.items.reverse()\n"
    result = _run(
        tmp_path,
        "\n".join(f"def {name}():\n assert False, 'BODY_EXECUTED'" for name in NAMES),
        conftest,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "late diagnostic collection drift" in result.stderr
    assert not any(row["event"] == "start" for row in _events(tmp_path))
    assert "BODY_EXECUTED" not in result.stdout + result.stderr


def test_diagnostic_refuses_initialized_file_mutation(tmp_path: Path) -> None:
    conftest = "from pathlib import Path\ndef pytest_collection_modifyitems():\n p = Path('tools/ci_non_slow_partition.py')\n p.write_text(p.read_text() + '\\n# changed after initialization\\n')\n"
    result = _run(
        tmp_path,
        "\n".join(f"def {name}():\n assert False, 'BODY_EXECUTED'" for name in NAMES),
        conftest,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "initialized file identity drift" in result.stderr
    assert not any(row["event"] == "start" for row in _events(tmp_path))
    assert "BODY_EXECUTED" not in result.stdout + result.stderr


def test_polling_defers_only_the_unterminated_tail(tmp_path: Path) -> None:
    (tmp_path / "records").mkdir()
    path = tmp_path / "records/events.jsonl"
    first = b'{"event":"report","outcome":"failed"}\n'
    _ = path.write_bytes(first + b'{"event":"sta')
    assert _poll_events(tmp_path) == [{"event": "report", "outcome": "failed"}]
    _ = path.write_bytes(first + b'{"event":"start"}\n')
    expected = [{"event": "report", "outcome": "failed"}, {"event": "start"}]
    assert _poll_events(tmp_path) == expected
    assert _events(tmp_path) == expected


@pytest.mark.parametrize("tail", [b'{"event":', b'{"event":"start"}'])
def test_terminal_reader_refuses_unterminated_records(
    tmp_path: Path, tail: bytes
) -> None:
    (tmp_path / "records").mkdir()
    _ = (tmp_path / "records/events.jsonl").write_bytes(tail)
    assert _poll_events(tmp_path) == []
    with pytest.raises(ValueError, match="unterminated terminal record"):
        _ = _events(tmp_path)


def test_both_readers_reject_malformed_complete_record(tmp_path: Path) -> None:
    (tmp_path / "records").mkdir()
    _ = (tmp_path / "records/events.jsonl").write_bytes(b'{"event":"report"}\n{bad}\n')
    with pytest.raises(json.JSONDecodeError):
        _ = _poll_events(tmp_path)
    with pytest.raises(json.JSONDecodeError):
        _ = _events(tmp_path)


def test_toml_configuration_settles_after_core_and_late_hooks(tmp_path: Path) -> None:
    conftest = 'import pytest\n@pytest.hookimpl(trylast=True)\ndef pytest_configure(config):\n config.addinivalue_line("markers", "late: synthetic")\n'
    result = _run(
        tmp_path, "\n".join(f"def {name}():\n pass" for name in NAMES), conftest
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = _events(tmp_path)
    initial = cast(dict[str, Any], events[0]["snapshot"])
    settled_event = next(
        row for row in events if row["event"] == "settled_configuration"
    )
    settled = cast(dict[str, Any], settled_event["snapshot"])
    assert initial["ini"]["markers"] == ["synthetic: infrastructure control"]
    assert "late: synthetic" in settled["ini"]["markers"]
    assert any(mark.startswith("parametrize(") for mark in settled["ini"]["markers"])
    assert (
        cast(dict[str, Any], settled_event["ini_delta"])["markers"]["initial"]
        == initial["ini"]["markers"]
    )
    assert len([row for row in events if row["event"] == "end"]) == 2


@pytest.mark.parametrize("phase", ["collection", "runtime"])
def test_actual_ini_drift_retains_attempt_before_refusal(
    tmp_path: Path, phase: str
) -> None:
    conftest = ""
    body = "\n".join(f"def {name}():\n pass" for name in NAMES)
    if phase == "collection":
        conftest = 'def pytest_collection_modifyitems(config):\n config.inicfg["markers"].append("changed: collection")\n'
    else:
        body = f'def {NAMES[0]}(request):\n request.config.inicfg["markers"].append("changed: runtime")\ndef {NAMES[1]}():\n assert False, "SECOND_BODY"\n'
    result = _run(tmp_path, body, conftest)
    assert result.returncode == 3, result.stdout + result.stderr
    events = _events(tmp_path)
    failure = next(
        i for i, row in enumerate(events) if row["event"] == "infrastructure_failure"
    )
    attempted = next(
        row
        for row in reversed(events[:failure])
        if row["event"] == "attempted_configuration"
    )
    assert "ini" in cast(list[str], attempted["changed_fields"])
    assert (
        "changed: " + phase
        in cast(dict[str, Any], attempted["snapshot"])["ini"]["markers"]
    )
    settled = next(row for row in events if row["event"] == "settled_configuration")
    assert (
        "changed: " + phase
        not in cast(dict[str, Any], settled["snapshot"])["ini"]["markers"]
    )
    assert "SECOND_BODY" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "mutation, reason",
    [
        ("config.option.maxfail = 1", "initial configuration drift: options"),
        ("config.args.reverse()", "initial configuration drift: args"),
        (
            'p=Path("tools/ci_non_slow_partition.py"); p.write_text(p.read_text()+"\\n# changed")',
            "initialized file identity drift",
        ),
    ],
)
def test_initial_identity_survives_settlement(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    conftest = (
        "import pytest\nfrom pathlib import Path\n@pytest.hookimpl(trylast=True)\ndef pytest_configure(config):\n "
        + mutation
        + "\n"
    )
    result = _run(
        tmp_path,
        "\n".join(f"def {name}():\n assert False, 'BODY_EXECUTED'" for name in NAMES),
        conftest,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert reason in result.stderr
    events = _events(tmp_path)
    assert any(row["event"] == "settled_configuration" for row in events)
    assert not any(row["event"] == "start" for row in events)


def test_missing_settlement_is_refused(tmp_path: Path) -> None:
    conftest = 'def pytest_collection_modifyitems(config):\n config.pluginmanager.get_plugin("ci-diagnostic").settled = False\n'
    result = _run(
        tmp_path,
        "\n".join(f"def {name}():\n assert False, 'BODY_EXECUTED'" for name in NAMES),
        conftest,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "missing settled configuration" in result.stderr
    events = _events(tmp_path)
    assert not any(row["event"] == "start" for row in events)
    assert any(row["event"] == "attempted_configuration" for row in events)


def _partition_tool() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("partition_kernel", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_collection_uses_observed_file_and_parameter_order(tmp_path: Path) -> None:
    kernel = _partition_tool()
    _ = (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nmarkers=["slow: excluded"]\n'
    )
    observer = """import hashlib, inspect, json, os, pathlib, pytest, sys
@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
 yield
 manager=session.config.pluginmanager
 assert manager.get_name(manager) == str(id(manager))
 rows=[]
 for name,plugin in manager.list_name_plugin():
  owner=plugin if inspect.ismodule(plugin) else type(plugin)
  try:
   source=inspect.getsourcefile(owner)
  except TypeError:
   source=None
  identity={"module":getattr(owner,"__module__",getattr(owner,"__name__",None)),
   "class":None if inspect.ismodule(plugin) else type(plugin).__qualname__,
   "source":source,"sha256":hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest() if source else None}
  if plugin is manager:
   assert source is not None
   identity["registration"]="self-registered-plugin-manager"
  else:
   identity["registration"]=name
  rows.append({"raw_name":name,"manager":plugin is manager,"identity":identity})
 record={"nodes":[item.nodeid for item in session.items],"exit":int(exitstatus),
  "args":session.config.args,"options":vars(session.config.option),
  "raw_plugins":rows,"plugins":sorted((r["identity"] for r in rows),key=lambda r:json.dumps(r,sort_keys=True)),
  "distributions":sorted((d.project_name,d.version) for _,d in manager.list_plugin_distinfo()),
  "executable":sys.executable,"pytest":pytest.__version__}
 with pathlib.Path(os.environ["PARTITION_TEST_RECORD"]).open("x") as stream:
  json.dump(record,stream,default=str)
"""
    _ = (tmp_path / "conftest.py").write_text(observer)
    body = """import pytest
@pytest.mark.parametrize("value", [0,1], ids=["p0","p1"])
def test_parameter(value):
 raise AssertionError("BODY_EXECUTED")
@pytest.fixture(params=[0,1], ids=["f0","f1"])
def value(request):
 return request.param
def test_fixture(value):
 raise AssertionError("BODY_EXECUTED")
@pytest.mark.skip(reason="retained member")
def test_skipped():
 raise AssertionError("BODY_EXECUTED")
@pytest.mark.slow
def test_slow():
 raise AssertionError("BODY_EXECUTED")
"""
    for name in ("test_a.py", "test_b.py", "test_c.py"):
        _ = (tmp_path / name).write_text(body)
    records: list[dict[str, Any]] = []

    def collect(selection: tuple[str, ...]) -> tuple[str, ...]:
        args = [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-m",
            "not slow",
            *selection,
        ]
        record_path = tmp_path / f"observation-{len(records)}.json"
        env = os.environ.copy()
        env["PARTITION_TEST_RECORD"] = str(record_path)
        run = subprocess.run(
            args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
        )
        _ = record_path.with_suffix(".stdout").write_text(run.stdout)
        _ = record_path.with_suffix(".stderr").write_text(run.stderr)
        assert run.returncode == 0, run.stdout + run.stderr
        record = cast(dict[str, Any], json.loads(record_path.read_text()))
        records.append(record)
        assert record["exit"] == 0
        assert "BODY_EXECUTED" not in run.stdout + run.stderr
        return tuple(record["nodes"])

    baseline = collect((".",))
    selected, empty = kernel._ordered_files(
        baseline, ("test_b.py", "test_a.py"), ("test_a.py", "test_b.py", "test_c.py")
    )
    assert selected == ("test_a.py", "test_b.py") and empty == ()
    actual = collect(selected)
    assert actual == tuple(n for n in baseline if n.partition("::")[0] in selected)
    assert any("[p0]" in n for n in actual) and any("[f1]" in n for n in actual)
    assert any("test_skipped" in n for n in actual)
    assert not any("test_slow" in n for n in actual)
    assert records[0]["plugins"] == records[1]["plugins"]
    assert records[0]["distributions"] == records[1]["distributions"]


def _partition_case() -> tuple[
    Any, tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]]
]:
    kernel = _partition_tool()
    roster = tuple(p for paths in kernel.PARTITION_FILES.values() for p in paths) + (
        "tests/unit/test_ci_non_slow_partition.py",
    )
    baseline = (
        roster[0] + "::test_example[p::0]",
        roster[0] + "::test_example[p::1]",
    ) + tuple(p + "::test_example[p::0]" for p in roster[1:])
    groups = {
        g: tuple(n for n in baseline if n.partition("::")[0] in paths)
        for g, paths in kernel.PARTITION_FILES.items()
    }
    groups["general"] = (baseline[-1],)
    return kernel, roster, baseline, groups


def test_partition_exact_cover_and_empty_tree_order() -> None:
    kernel, roster, baseline, groups = _partition_case()
    before = groups.copy()
    result = kernel.prove_partition(baseline, groups, roster)
    assert groups == before
    assert result["general"][0][0] == "tests"
    assert len(result) == 6
    history = kernel.PARTITION_FILES["history"]
    selected, empty = kernel._ordered_files((), history, tuple(reversed(roster)))
    assert selected == empty == tuple(reversed(history))
    all_empty = dict.fromkeys(groups, ())
    assert len(kernel.prove_partition((), all_empty, roster)) == 6


@pytest.mark.parametrize(
    "mutation",
    [
        "reorder",
        "missing",
        "extra",
        "duplicate",
        "foreign",
        "group",
        "missing-owner",
        "duplicate-roster",
        "wrong-membership",
    ],
)
def test_partition_refuses_changed_observations(mutation: str) -> None:
    kernel, roster, baseline, groups = _partition_case()
    if mutation == "reorder":
        groups["history"] = (
            groups["history"][1],
            groups["history"][0],
            *groups["history"][2:],
        )
    elif mutation == "missing":
        groups["history"] = groups["history"][:-1]
    elif mutation == "extra":
        groups["history"] += (roster[0] + "::test_extra",)
    elif mutation == "duplicate":
        baseline += (baseline[0],)
    elif mutation == "foreign":
        groups["general"] = ("tests/unit/foreign.py::test_x",)
    elif mutation == "group":
        groups["unknown"] = ()
    elif mutation == "missing-owner":
        roster = roster[1:]
    elif mutation == "duplicate-roster":
        roster += (roster[0],)
    else:
        groups["general"] += groups["history"]
    before = groups.copy()
    with pytest.raises(ValueError):
        _ = kernel.prove_partition(baseline, groups, roster)
    assert groups == before


def test_partition_requires_an_owner_even_with_zero_selected_nodes() -> None:
    kernel, roster, _, groups = _partition_case()
    empty_groups = dict.fromkeys(groups, ())
    assert len(kernel.prove_partition((), empty_groups, roster)) == 6
    missing_owner = roster[1:]
    with pytest.raises(ValueError, match="missing or duplicate required owner"):
        _ = kernel.prove_partition((), empty_groups, missing_owner)


def _request_document() -> dict[str, Any]:
    return {
        "schema": "radiosim-ci-collection-request-v1",
        "nonce": "a" * 32,
        "index": 0,
        "role": "baseline",
        "root": "/synthetic/checkout",
        "commit": "b" * 40,
        "source_manifest_sha256": "c" * 64,
        "argv": ["-n", "auto", "--collect-only", "-m", "not slow", "tests"],
    }


def test_request_raw_identity_preserves_whitespace_and_key_order() -> None:
    kernel = _partition_tool()
    document = _request_document()
    raw = (
        json.dumps(
            document,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    alternate = json.dumps(dict(reversed(tuple(document.items()))), indent=2).encode()
    expected = hashlib.sha256(raw).hexdigest()
    decoded, digest = kernel.parse_collection_request(raw, expected_sha256=expected)
    other_decoded, other_digest = kernel.parse_collection_request(alternate)
    assert decoded == other_decoded == document
    assert digest == expected and other_digest == hashlib.sha256(alternate).hexdigest()
    assert digest != other_digest
    with pytest.raises(ValueError, match="raw request digest mismatch"):
        _ = kernel.parse_collection_request(alternate, expected_sha256=expected)


@pytest.mark.parametrize(
    "key,value",
    [
        ("request_sha256", "d" * 64),
        ("index", True),
        ("index", 1.0),
        ("index", -1),
        ("index", 7),
        ("role", "general"),
        ("schema", "other"),
        ("nonce", "A" * 32),
        ("commit", "b" * 39),
        ("source_manifest_sha256", "not-a-hash"),
        ("root", "relative"),
        ("root", 1),
        ("argv", "tests"),
        ("argv", ["tests", 1]),
    ],
)
def test_request_refuses_schema_and_role_mutations(key: str, value: Any) -> None:
    document = _request_document()
    document[key] = value
    with pytest.raises(ValueError):
        _ = _partition_tool().parse_collection_request(json.dumps(document).encode())


def test_request_refuses_missing_key_and_duplicate_key() -> None:
    document = _request_document()
    del document["commit"]
    with pytest.raises(ValueError, match="exactly the eight schema keys"):
        _ = _partition_tool().parse_collection_request(json.dumps(document).encode())
    raw = json.dumps(_request_document()).encode()
    duplicate = raw[:-1] + b', "index":0}'
    with pytest.raises(ValueError, match="duplicate request key"):
        _ = _partition_tool().parse_collection_request(duplicate)


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity", b"1e999"])
def test_request_refuses_nonfinite_numbers(token: bytes) -> None:
    raw = (
        json.dumps(_request_document())
        .encode()
        .replace(b'"index": 0', b'"index": ' + token)
    )
    with pytest.raises(ValueError, match="not an integer"):
        _ = _partition_tool().parse_collection_request(raw)


@pytest.mark.parametrize("raw", [b"\xff", b"{", b"null", b"[]"])
def test_request_refuses_nonobject_and_malformed_utf8(raw: bytes) -> None:
    with pytest.raises(ValueError):
        _ = _partition_tool().parse_collection_request(raw)


@pytest.mark.parametrize(
    "role,index",
    [
        ("baseline", 0),
        ("general", 1),
        ("history", 2),
        ("evidence", 3),
        ("red-replay", 4),
        ("public-integration", 5),
        ("public-characterization", 6),
    ],
)
def test_request_accepts_exact_role_index_pairs(role: str, index: int) -> None:
    document = _request_document()
    document.update(role=role, index=index)
    parsed, _ = _partition_tool().parse_collection_request(
        json.dumps(document).encode()
    )
    assert parsed == document
