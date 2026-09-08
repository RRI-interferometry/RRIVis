"""Serial diagnostic reporting only; main-suite partitioning is not implemented."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

DIAGNOSTIC_IDS = (
    "tests/unit/test_sci004_phase3_evidence.py::"
    "test_production_v2_retains_the_complete_owned_input_preimage",
    "tests/unit/test_sci004_phase3_evidence.py::"
    "test_production_v2_rejects_a_rehashed_foreign_phase_against_unchanged_result",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--ci-diagnostic-output", help="Owned serial diagnostic records")


def pytest_configure(config: pytest.Config) -> None:
    output = config.getoption("ci_diagnostic_output")
    if output is not None:
        _ = config.pluginmanager.register(
            _Diagnostic(config, Path(output)), "ci-diagnostic"
        )


class _Diagnostic:
    def __init__(self, config: pytest.Config, output: Path) -> None:
        self.config = config
        self.root = Path(__file__).resolve().parents[1]
        self.sequence = 0
        self.session: pytest.Session | None = None
        self.started: list[str] = []
        self.origins: dict[str, str] = {}
        output.mkdir(parents=True, exist_ok=True)
        self.record = output / "events.jsonl"
        with self.record.open("x", encoding="utf-8"):
            pass
        self.sources = {
            str(p.resolve()): _sha(p)
            for p in (self.root / "src/radiosim").rglob("*.py")
        }
        self.configuration = self._configuration()
        self._emit("configuration", snapshot=self.configuration)
        self._require(config.rootpath.resolve() == self.root, "foreign pytest root")
        self._require(tuple(config.args) == DIAGNOSTIC_IDS, "diagnostic argv differs")
        for option in (
            "numprocesses",
            "maxfail",
            "collectonly",
            "setuponly",
            "lf",
            "stepwise",
        ):
            self._require(
                not config.getoption(option, default=None), f"unsupported {option}"
            )
        self._require(not hasattr(config, "workerinput"), "diagnostic worker forbidden")
        self._owners()

    def _configuration(self) -> dict[str, Any]:
        try:
            commit = subprocess.check_output(
                [
                    "git",
                    "--no-replace-objects",
                    "-C",
                    str(self.root),
                    "rev-parse",
                    "--verify",
                    "HEAD^{commit}",
                ],
                env={
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith("GIT_")
                },
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError) as error:
            self._emit(
                "infrastructure_failure", reason=f"Git identity unavailable: {error}"
            )
            pytest.exit("Git identity unavailable", returncode=3)
        files = {str(Path(__file__).resolve()): _sha(Path(__file__).resolve())}
        for _, plugin in self.config.pluginmanager.list_name_plugin():
            source = getattr(plugin, "__file__", None)
            if source is not None and Path(source).is_file():
                files[str(Path(source).resolve())] = _sha(Path(source))
        if self.config.inipath is not None:
            files[str(self.config.inipath)] = _sha(self.config.inipath)
        return {
            "commit": commit,
            "root": str(self.root),
            "executable": sys.executable,
            "python": sys.version,
            "pytest": pytest.__version__,
            "args": list(self.config.args),
            "options": copy.deepcopy(vars(self.config.option)),
            "ini": copy.deepcopy(dict(self.config.inicfg)),
            "plugins": sorted(
                name for name, _ in self.config.pluginmanager.list_name_plugin()
            ),
            "distributions": sorted(
                (dist.project_name, dist.version)
                for _, dist in self.config.pluginmanager.list_plugin_distinfo()
            ),
            "files": files,
        }

    def _emit(self, event: str, **fields: Any) -> None:
        self.sequence += 1
        line = json.dumps(
            {
                "sequence": self.sequence,
                "time": time.monotonic(),
                "event": event,
                **fields,
            },
            default=str,
            sort_keys=True,
        )
        try:
            assert sys.__stderr__ is not None
            print(line, file=sys.__stderr__, flush=True)
            with self.record.open("a", encoding="utf-8") as stream:
                _ = stream.write(line + "\n")
                stream.flush()
        except OSError as error:
            pytest.exit(f"diagnostic reporting failed: {error}", returncode=3)

    def _require(self, condition: bool, reason: str) -> None:
        if not condition:
            self._emit("infrastructure_failure", reason=reason)
            pytest.exit(reason, returncode=3)

    def _owners(self) -> None:
        observed: dict[str, str] = {}
        for name, module in tuple(sys.modules.items()):
            if name == "radiosim" or name.startswith("radiosim."):
                source = getattr(module, "__file__", None)
                self._require(
                    isinstance(source, str), f"missing/string origin required: {name}"
                )
                path = Path(cast(str, source)).resolve()
                self._require(str(path) in self.sources, f"foreign origin: {name}")
                try:
                    actual = _sha(path)
                except OSError as error:
                    self._emit(
                        "infrastructure_failure",
                        reason=f"source unreadable: {name}: {error}",
                    )
                    pytest.exit("source unreadable", returncode=3)
                self._require(
                    actual == self.sources[str(path)], f"source drift: {name}"
                )
                self._require(
                    name not in self.origins or self.origins[name] == str(path),
                    f"origin drift: {name}",
                )
                observed[name] = str(path)
        self._require(
            self.origins.keys() <= observed.keys(), "loaded origin disappeared"
        )
        self.origins.update(observed)
        self._emit(
            "origins",
            modules=observed,
            hashes={p: self.sources[p] for p in observed.values()},
            not_loaded=not observed,
        )

    @pytest.hookimpl(trylast=True)
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.session = session
        actual = tuple(item.nodeid for item in session.items)
        self._emit("collection", nodeids=actual)
        self._require(actual == DIAGNOSTIC_IDS, "diagnostic collection differs")
        self._owners()
        current = self._configuration()
        for field in ("commit", "root", "args", "options", "ini"):
            self._require(
                current[field] == self.configuration[field],
                f"collection configuration drift: {field}",
            )
        initial_files = cast(dict[str, str], self.configuration["files"])
        current_files = cast(dict[str, str], current["files"])
        self._require(
            all(
                current_files.get(path) == digest
                for path, digest in initial_files.items()
            ),
            "initialized file identity drift",
        )
        self.configuration = current
        self._emit("execution_configuration", snapshot=self.configuration)

    def pytest_runtest_logstart(self, nodeid: str) -> None:
        session = self.session
        self._require(session is not None, "missing diagnostic collection")
        assert session is not None
        self._require(
            tuple(item.nodeid for item in session.items) == DIAGNOSTIC_IDS,
            "late diagnostic collection drift",
        )
        self._require(
            len(self.started) < len(DIAGNOSTIC_IDS)
            and nodeid == DIAGNOSTIC_IDS[len(self.started)],
            "incoming diagnostic node differs",
        )
        self.started.append(nodeid)
        self._require(
            self._configuration() == self.configuration, "configuration drift"
        )
        self._owners()
        self._emit("start", nodeid=nodeid, worker="serial")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.failed:
            print(report.longreprtext, file=sys.__stderr__, flush=True)
        self._emit(
            "report",
            nodeid=report.nodeid,
            phase=report.when,
            outcome=report.outcome,
            longrepr=report.longreprtext if report.failed else "",
            worker="serial",
        )

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        self._require(
            self._configuration() == self.configuration, "configuration drift"
        )
        self._owners()
        self._emit("end", nodeid=nodeid, worker="serial")

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self._owners()
        self._emit("session_end", exitstatus=exitstatus)
