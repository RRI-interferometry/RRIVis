"""Enforce RadioSim's strict Pyright error ceiling without hiding diagnostics."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "pyright-baseline.json"


def _run_pyright() -> tuple[int, dict[str, Any], str]:
    command = [sys.executable, "-m", "pyright", "--outputjson"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload: dict[str, Any] = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        print("Pyright did not produce valid JSON output.", file=sys.stderr)
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise SystemExit(2) from exc
    return completed.returncode, payload, completed.stderr


def _load_baseline() -> dict[str, Any]:
    with BASELINE_PATH.open(encoding="utf-8") as stream:
        baseline: dict[str, Any] = json.load(stream)
    return baseline


def _relative_path(value: object) -> str:
    path = Path(str(value))
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _format_diagnostic(diagnostic: dict[str, Any]) -> str:
    location = diagnostic.get("range", {}).get("start", {})
    line = int(location.get("line", 0)) + 1
    column = int(location.get("character", 0)) + 1
    rule = diagnostic.get("rule") or "unclassified"
    message = " ".join(str(diagnostic.get("message", "")).split())
    return (
        f"{_relative_path(diagnostic.get('file', 'unknown'))}:{line}:{column}: "
        f"{rule}: {message}"
    )


def _write_lower_baseline(
    baseline: dict[str, Any], error_count: int, pyright_version: str
) -> None:
    previous = int(baseline["maximum_errors"])
    if error_count > previous:
        raise ValueError(
            f"Refusing to raise the Pyright ceiling from {previous} to {error_count}."
        )
    baseline["maximum_errors"] = error_count
    baseline["recorded_pyright"] = pyright_version
    baseline["recorded_python"] = platform.python_version()
    _ = BASELINE_PATH.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when strict Pyright errors exceed the checked-in ceiling."
    )
    _ = parser.add_argument(
        "--update",
        action="store_true",
        help="Lower the checked-in ceiling to the current error count.",
    )
    args = parser.parse_args()

    baseline = _load_baseline()
    maximum_errors = int(baseline["maximum_errors"])
    return_code, payload, stderr = _run_pyright()
    summary = payload.get("summary", {})
    error_count = int(summary.get("errorCount", -1))
    pyright_version = str(payload.get("version", ""))

    if return_code not in (0, 1) or error_count < 0 or not pyright_version:
        print(
            f"Pyright execution failed unexpectedly with exit status {return_code}.",
            file=sys.stderr,
        )
        if stderr:
            print(stderr, file=sys.stderr)
        return 2

    if args.update:
        try:
            _write_lower_baseline(baseline, error_count, pyright_version)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Updated Pyright ceiling: {maximum_errors} -> {error_count}")
        return 0

    recorded_pyright = str(baseline["recorded_pyright"])
    if pyright_version != recorded_pyright:
        print(
            "Pyright version differs from the checked-in baseline: "
            f"running {pyright_version}, recorded {recorded_pyright}.",
            file=sys.stderr,
        )
        print(
            "Restore the recorded checker or review the new diagnostics and run "
            "'pixi run typecheck-update' without raising the error ceiling.",
            file=sys.stderr,
        )
        return 1

    if error_count <= maximum_errors:
        print(
            "Strict Pyright error ceiling satisfied: "
            f"{error_count} errors <= {maximum_errors}."
        )
        if error_count < maximum_errors:
            print(
                "Type debt decreased. Run "
                "'pixi run typecheck-update' to lower the checked-in ceiling."
            )
        return 0

    increase = error_count - maximum_errors
    print(
        "Strict Pyright error ceiling exceeded: "
        f"{error_count} errors > {maximum_errors} ({increase} new error(s)).",
        file=sys.stderr,
    )
    print(
        "The first diagnostics are shown with repository-relative paths:",
        file=sys.stderr,
    )
    diagnostics = payload.get("generalDiagnostics", [])
    for diagnostic in diagnostics[:20]:
        print(f"- {_format_diagnostic(diagnostic)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
