"""Synthetic subprocess controls: no scientific package or actual diagnostic run."""

from __future__ import annotations

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
