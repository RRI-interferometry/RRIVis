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
        self.settled = False
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

    def _check_initialized_files(self, current: dict[str, Any]) -> None:
        initial_files = cast(dict[str, str], self.configuration["files"])
        current_files = cast(dict[str, str], current["files"])
        self._require(
            all(
                current_files.get(path) == digest
                for path, digest in initial_files.items()
            ),
            "initialized file identity drift",
        )

    @pytest.hookimpl(trylast=True)
    def pytest_sessionstart(self, session: pytest.Session) -> None:
        current = self._configuration()
        initial_ini = cast(dict[str, Any], self.configuration["ini"])
        settled_ini = cast(dict[str, Any], current["ini"])
        delta = {
            key: {
                "initial_present": key in initial_ini,
                "initial": initial_ini.get(key),
                "settled_present": key in settled_ini,
                "settled": settled_ini.get(key),
            }
            for key in sorted(initial_ini.keys() | settled_ini.keys())
            if key not in initial_ini
            or key not in settled_ini
            or initial_ini[key] != settled_ini[key]
        }
        self._emit("settled_configuration", snapshot=current, ini_delta=delta)
        for field in ("commit", "root", "args", "options"):
            self._require(
                current[field] == self.configuration[field],
                f"initial configuration drift: {field}",
            )
        self._check_initialized_files(current)
        self._owners()
        self.configuration = current
        self.settled = True

    def _attempted_configuration(self, phase: str) -> dict[str, Any]:
        current = self._configuration()
        self._emit(
            "attempted_configuration",
            phase=phase,
            snapshot=current,
            changed_fields=sorted(
                key for key in current if current[key] != self.configuration[key]
            ),
        )
        return current

    @pytest.hookimpl(trylast=True)
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        current = self._attempted_configuration("collection")
        self._require(self.settled, "missing settled configuration")
        self.session = session
        actual = tuple(item.nodeid for item in session.items)
        self._emit("collection", nodeids=actual)
        self._require(actual == DIAGNOSTIC_IDS, "diagnostic collection differs")
        self._owners()
        for field in ("commit", "root", "args", "options", "ini"):
            self._require(
                current[field] == self.configuration[field],
                f"collection configuration drift: {field}",
            )
        self._check_initialized_files(current)
        self.configuration = current
        self._emit("execution_configuration", snapshot=self.configuration)

    def pytest_runtest_logstart(self, nodeid: str) -> None:
        current = self._attempted_configuration("start")
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
        self._require(current == self.configuration, "configuration drift")
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
        current = self._attempted_configuration("end")
        self._require(current == self.configuration, "configuration drift")
        self._owners()
        self._emit("end", nodeid=nodeid, worker="serial")

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self._owners()
        self._emit("session_end", exitstatus=exitstatus)


# Membership only: actual baseline first occurrence determines each file argv.
PARTITION_FILES = {
    "history": (
        "tests/unit/test_sci004_phase3_history.py",
        "tests/unit/test_sci004_phase3_dependency.py",
    ),
    "evidence": ("tests/unit/test_sci004_phase3_evidence.py",),
    "red-replay": ("tests/unit/test_sci004_phase3_red_failures.py",),
    "public-integration": ("tests/integration/test_sci004_mmode.py",),
    "public-characterization": ("tests/characterization/test_sci004_mmode.py",),
}


def _node_owners(nodes: tuple[str, ...], roster: tuple[str, ...]) -> tuple[str, ...]:
    """Validate supplied observations without authenticating their transport."""
    if len(set(roster)) != len(roster) or any(
        not isinstance(cast(object, path), str)
        or not path.endswith(".py")
        or "\\" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
        for path in roster
    ):
        raise ValueError("invalid or duplicate source roster")
    if any(not isinstance(cast(object, node), str) for node in nodes):
        raise ValueError("node IDs must be strings")
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate node ID")
    owners = tuple(node.partition("::")[0] for node in nodes)
    if any(owner not in roster for owner in owners):
        raise ValueError("unknown node owner")
    return owners


def _ordered_files(
    baseline: tuple[str, ...], members: tuple[str, ...], roster: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive argv from raw baseline order, then zero-node files in tree order."""
    owners = _node_owners(baseline, roster)
    if len(set(members)) != len(members) or any(p not in roster for p in members):
        raise ValueError("invalid membership catalog")
    represented = tuple(dict.fromkeys(owner for owner in owners if owner in members))
    empty = tuple(
        path for path in roster if path in members and path not in represented
    )
    return represented + empty, empty


def _partition_selections(
    baseline: tuple[str, ...], roster: tuple[str, ...]
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Build closed selections; supplied roster/source provenance is not proved here."""
    _ = _node_owners(baseline, roster)
    dedicated = tuple(path for paths in PARTITION_FILES.values() for path in paths)
    if len(set(dedicated)) != len(dedicated) or any(p not in roster for p in dedicated):
        raise ValueError("missing or duplicate required owner")
    selections = {
        group: _ordered_files(baseline, members, roster)
        for group, members in PARTITION_FILES.items()
    }
    general = tuple(path for path in roster if path not in dedicated)
    _, empty = _ordered_files(baseline, general, roster)
    selections["general"] = (
        ("tests", *("--ignore=" + path for path in dedicated)),
        empty,
    )
    return selections


def prove_partition(
    baseline: tuple[str, ...],
    observed: dict[str, tuple[str, ...]],
    roster: tuple[str, ...],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Prove exact raw ordered coverage; do not normalize or repair observations."""
    selections = _partition_selections(baseline, roster)
    if set(observed) != set(selections):
        raise ValueError("partition requires exactly the six groups")
    baseline_owners = _node_owners(baseline, roster)
    dedicated = tuple(path for paths in PARTITION_FILES.values() for path in paths)
    all_nodes: list[str] = []
    for group, nodes in observed.items():
        _ = _node_owners(nodes, roster)
        members = (
            tuple(path for path in roster if path not in dedicated)
            if group == "general"
            else PARTITION_FILES[group]
        )
        expected = tuple(
            node
            for node, owner in zip(baseline, baseline_owners, strict=True)
            if owner in members
        )
        if nodes != expected:
            raise ValueError(f"ordered partition mismatch: {group}")
        all_nodes.extend(nodes)
    if len(set(all_nodes)) != len(all_nodes) or set(all_nodes) != set(baseline):
        raise ValueError("partition is not a disjoint exact cover")
    return selections


def _request_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate request key: {key}")
        result[key] = value
    return result


def _request_noninteger(value: str) -> Any:
    raise ValueError(f"request numeric token is not an integer: {value}")


def parse_collection_request(
    raw: bytes, *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    """Validate request shape and raw identity; source/argv authority is separate."""
    if not isinstance(cast(object, raw), bytes):
        raise ValueError("request must be raw bytes")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("raw request digest mismatch")
    try:
        request = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_request_pairs,
            parse_constant=_request_noninteger,
            parse_float=_request_noninteger,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("request is not UTF-8 JSON") from error
    keys = {
        "schema",
        "nonce",
        "index",
        "role",
        "root",
        "commit",
        "source_manifest_sha256",
        "argv",
    }
    if not isinstance(request, dict) or set(cast(dict[str, Any], request)) != keys:
        raise ValueError("request must have exactly the eight schema keys")
    request = cast(dict[str, Any], request)
    if request["schema"] != "radiosim-ci-collection-request-v1":
        raise ValueError("unknown request schema")
    for field, length in (
        ("nonce", 32),
        ("commit", 40),
        ("source_manifest_sha256", 64),
    ):
        value = request[field]
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"invalid request {field}")
    roles = (
        "baseline",
        "general",
        "history",
        "evidence",
        "red-replay",
        "public-integration",
        "public-characterization",
    )
    index = request["index"]
    if type(index) is not int or not 0 <= index < len(roles):
        raise ValueError("invalid request index")
    if request["role"] != roles[index]:
        raise ValueError("request role/index mismatch")
    root = request["root"]
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("request root must be an absolute path string")
    argv = request["argv"]
    if not isinstance(argv, list) or any(
        not isinstance(arg, str) for arg in cast(list[Any], argv)
    ):
        raise ValueError("request argv must be an array of strings")
    return request, digest


def bind_collection_request(
    raw: bytes,
    *,
    expected_sha256: str,
    root: str,
    commit: str,
    source_manifest_sha256: str,
    baseline: tuple[str, ...],
    roster: tuple[str, ...],
) -> tuple[dict[str, Any], str, tuple[str, ...] | None]:
    """Join an independently authenticated authority; do not authenticate Git here."""
    request, digest = parse_collection_request(raw, expected_sha256=expected_sha256)
    for key, value in (
        ("root", root),
        ("commit", commit),
        ("source_manifest_sha256", source_manifest_sha256),
    ):
        if request[key] != value:
            raise ValueError(f"collection authority mismatch: {key}")
    role = request["role"]
    if role == "baseline":
        if baseline:
            raise ValueError("baseline request cannot consume a prior baseline")
        _ = _partition_selections((), roster)  # Required files, including empty owners.
        selection = ("tests",)
        expected = None
    else:
        if not baseline:
            raise ValueError("selected collection requires a nonempty proven baseline")
        selection = _partition_selections(baseline, roster)[role][0]
        owners = _node_owners(baseline, roster)
        dedicated = tuple(p for paths in PARTITION_FILES.values() for p in paths)
        members = (
            tuple(p for p in roster if p not in dedicated)
            if role == "general"
            else PARTITION_FILES[role]
        )
        expected = tuple(
            n for n, owner in zip(baseline, owners, strict=True) if owner in members
        )
    if request["argv"] != [
        "-n",
        "auto",
        "--collect-only",
        "-m",
        "not slow",
        *selection,
    ]:
        raise ValueError("collection argv differs from derived role selection")
    return request, digest, expected


def collection_exit_code(value: object) -> int:
    """Normalize only actual pytest ExitCode or exact int; do not admit a result."""
    if type(value) not in (int, pytest.ExitCode):
        raise ValueError("collection status must be exact int or pytest.ExitCode")
    return int(cast(int, value))
