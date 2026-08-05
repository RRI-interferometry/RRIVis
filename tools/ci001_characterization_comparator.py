"""CI-001 comparator: classify harvested characterization runs by digest class.

Post-Tier-8 WP-2 evidence tooling (``PostTier8RemediationPlan.md`` §5.2 item 1).
Given a directory of harvested CI evidence, this script:

1. **classifies runs by digest class** — two runs share a class exactly when
   every pin they both measured carries the same digest;
2. **diffs machine-fingerprint fields between classes** — which recorded
   facts (CPU model, dispatched features, thread environment, BLAS build, and
   the WP-2 extension fields) separate one class from another;
3. **computes cube deltas wherever two classes both have a captured cube**
   for the same pin — ``max|dV|``, max relative delta, differing-element
   count, and an explicit verdict against the ``rtol=1e-12`` backend
   tolerance named by ``Tier8ReleasePlan.md`` §14 — and says so honestly
   when no such pair exists (red runs do not persist cubes; only the pass
   path captures references).

Expected harvest layout (working data, gitignored under ``output/``)::

    <harvest>/artifacts/run-<id>/machine-fingerprint-<env>-<worker>.txt
    <harvest>/artifacts/run-<id>/reference_cubes/<pin-slug>/<env>/<digest>.npy
    <harvest>/logs/run-<id>-failed.log        # `gh run view <id> --log-failed`

Artifacts come from ``gh run download <id> --name characterization-<cell>``;
logs from ``gh run view <id> --log-failed``. Both are optional per run: a
green run contributes fingerprints and reference cubes, a red run contributes
the failure log's measured digests and its fail-path fingerprint block.

This tool changes nothing: no assertion, no digest, no pin table. It only
reads harvested evidence and reports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The Tier8ReleasePlan.md §14 adjudication criterion, verbatim from the
#: Section 13.5 backend tolerance the project already uses.
RTOL_CRITERION = 1e-12

_FINGERPRINT_FIELDS = (
    "environment key",
    "cpu model",
    "numpy dispatched features",
    "thread environment",
    "blas build",
    "libc",
    "runner image",
    "numpy runtime",
    "cpu topology",
    "cache topology",
)

_MEASURED_RE = re.compile(r"measured:\s+([0-9a-f]{64})")
_RECORDED_RE = re.compile(r"recorded:\s+([0-9a-f]{64})")
_PIN_LABEL_RE = re.compile(
    r"(?:Failed:\s+)?([^:]+?): digest not among those recorded for environment"
)
#: A `gh run view --log-failed` line is `<job>\t<step>\t<ISO timestamp> <text>`,
#: and pytest failure bodies additionally carry an `E ` marker.
_LOG_PREFIX_RE = re.compile(
    r"^.*?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s?(?:E\s+)?"
)


def _slug(label: str) -> str:
    """The same filesystem slug ``test_tier6_current_behavior.py`` uses."""
    return re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower()


@dataclass
class RunEvidence:
    """Everything one CI run contributed to the harvest."""

    run_id: str
    kind: str  # "artifact" (pass-path record) or "log" (fail-path record)
    digests: dict[str, str] = field(default_factory=dict)  # pin slug -> digest
    fingerprint: dict[str, str] = field(default_factory=dict)
    cube_paths: dict[str, Path] = field(default_factory=dict)  # slug -> .npy


def _parse_fingerprint_text(text: str) -> dict[str, str]:
    """Pull known ``key: value`` fingerprint fields out of arbitrary text.

    Works on both the artifact fingerprint files and the fail-path fingerprint
    block embedded in a ``gh run view --log-failed`` dump (where each line is
    prefixed with job name, step, timestamp, and pytest's ``E``).
    """
    fields: dict[str, str] = {}
    for line in text.splitlines():
        for name in _FINGERPRINT_FIELDS:
            marker = f"{name}: "
            index = line.find(marker)
            if index == -1:
                continue
            value = line[index + len(marker) :].strip()
            if value and fields.get(name) is None:
                fields[name] = value
    return fields


def _load_artifact_run(run_dir: Path) -> RunEvidence:
    """Read one downloaded ``characterization-<cell>`` artifact directory."""
    evidence = RunEvidence(run_id=run_dir.name.removeprefix("run-"), kind="artifact")
    for record in sorted(run_dir.glob("machine-fingerprint-*.txt")):
        parsed = _parse_fingerprint_text(record.read_text(encoding="utf-8"))
        for key, value in parsed.items():
            evidence.fingerprint.setdefault(key, value)
    for cube in sorted(run_dir.glob("reference_cubes/*/*/*.npy")):
        pin_slug = cube.parent.parent.name
        evidence.digests[pin_slug] = cube.stem
        evidence.cube_paths[pin_slug] = cube
    return evidence


def _load_log_run(log_path: Path) -> RunEvidence:
    """Read one failed-job log: measured digests per pin + fingerprint block."""
    run_id = re.sub(r"^run-|-failed$", "", log_path.stem)
    evidence = RunEvidence(run_id=run_id, kind="log")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    evidence.fingerprint = _parse_fingerprint_text(text)

    current_pin: str | None = None
    for raw_line in text.splitlines():
        line = _LOG_PREFIX_RE.sub("", raw_line)
        pin_match = _PIN_LABEL_RE.search(line)
        if pin_match:
            current_pin = _slug(pin_match.group(1).strip())
            continue
        measured = _MEASURED_RE.search(line)
        if measured and current_pin is not None:
            evidence.digests.setdefault(current_pin, measured.group(1))
    return evidence


def _same_class(left: RunEvidence, right: RunEvidence) -> bool | None:
    """Same class iff every shared pin agrees; ``None`` when nothing overlaps."""
    shared = set(left.digests) & set(right.digests)
    if not shared:
        return None
    return all(left.digests[pin] == right.digests[pin] for pin in shared)


def _classify(runs: list[RunEvidence]) -> dict[str, list[RunEvidence]]:
    """Group runs into digest classes by pairwise shared-pin agreement."""
    classes: list[list[RunEvidence]] = []
    unclassified: list[RunEvidence] = []
    for run in runs:
        if not run.digests:
            unclassified.append(run)
            continue
        placed = False
        for members in classes:
            verdicts = [_same_class(run, member) for member in members]
            if any(verdict is True for verdict in verdicts) and not any(
                verdict is False for verdict in verdicts
            ):
                members.append(run)
                placed = True
                break
        if not placed:
            classes.append([run])
    labelled = {
        f"class-{index + 1}": members
        for index, members in enumerate(
            sorted(classes, key=lambda members: min(run.run_id for run in members))
        )
    }
    if unclassified:
        labelled["unclassified (no digest evidence)"] = unclassified
    return labelled


def _fingerprint_diff(
    classes: dict[str, list[RunEvidence]],
) -> dict[str, dict[str, list[str]]]:
    """Per fingerprint field, the distinct values observed in each class."""
    diff: dict[str, dict[str, list[str]]] = {}
    for name in _FINGERPRINT_FIELDS:
        per_class: dict[str, list[str]] = {}
        for label, members in classes.items():
            values = sorted(
                {run.fingerprint[name] for run in members if name in run.fingerprint}
            )
            if values:
                per_class[label] = values
        if per_class:
            diff[name] = per_class
    return diff


def _cube_delta(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    """The §14 numeric probe: max|dV|, max relative, count, first index."""
    if left.shape != right.shape:
        return {"comparable": False, "reason": f"shape {left.shape} vs {right.shape}"}
    difference = np.abs(left - right)
    max_absolute = float(np.max(difference)) if difference.size else 0.0
    scale = np.maximum(np.abs(left), np.abs(right))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(scale > 0.0, difference / scale, 0.0)
    max_relative = float(np.max(relative)) if relative.size else 0.0
    differing = np.flatnonzero(left.ravel() != right.ravel())
    first = (
        tuple(int(axis) for axis in np.unravel_index(int(differing[0]), left.shape))
        if differing.size
        else None
    )
    return {
        "comparable": True,
        "max_abs_dV": max_absolute,
        "max_relative": max_relative,
        "differing_elements": int(differing.size),
        "total_elements": int(left.size),
        "first_differing_index": first,
        "within_rtol_1e-12": bool(max_relative <= RTOL_CRITERION),
    }


def _cross_class_deltas(
    classes: dict[str, list[RunEvidence]],
) -> list[dict[str, object]]:
    """Deltas for every pin where two different digests both have a cube."""
    cubes: dict[str, dict[str, Path]] = defaultdict(dict)  # slug -> digest -> path
    owners: dict[tuple[str, str], str] = {}
    for label, members in classes.items():
        for run in members:
            for slug, path in run.cube_paths.items():
                digest = run.digests[slug]
                cubes[slug].setdefault(digest, path)
                owners.setdefault((slug, digest), label)
    reports: list[dict[str, object]] = []
    for slug, by_digest in sorted(cubes.items()):
        digests = sorted(by_digest)
        if len(digests) < 2:
            continue
        for i, left_digest in enumerate(digests):
            for right_digest in digests[i + 1 :]:
                left = np.load(by_digest[left_digest])
                right = np.load(by_digest[right_digest])
                reports.append(
                    {
                        "pin": slug,
                        "left": {
                            "digest": left_digest,
                            "class": owners[(slug, left_digest)],
                        },
                        "right": {
                            "digest": right_digest,
                            "class": owners[(slug, right_digest)],
                        },
                        "delta": _cube_delta(left, right),
                    }
                )
    return reports


def _collect(harvest: Path) -> list[RunEvidence]:
    runs: list[RunEvidence] = []
    for run_dir in sorted((harvest / "artifacts").glob("run-*")):
        if run_dir.is_dir():
            runs.append(_load_artifact_run(run_dir))
    for log_path in sorted((harvest / "logs").glob("run-*-failed.log")):
        runs.append(_load_log_run(log_path))
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "harvest",
        nargs="?",
        default="output/ci001-harvest",
        type=Path,
        help="harvest directory (default: output/ci001-harvest)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the full report as JSON"
    )
    args = parser.parse_args(argv)

    if not args.harvest.exists():
        print(f"harvest directory {args.harvest} does not exist", file=sys.stderr)
        return 2

    runs = _collect(args.harvest)
    classes = _classify(runs)
    diff = _fingerprint_diff(classes)
    deltas = _cross_class_deltas(classes)

    if args.json:
        payload = {
            "criterion_rtol": RTOL_CRITERION,
            "classes": {
                label: [
                    {"run": run.run_id, "kind": run.kind, "digests": run.digests}
                    for run in members
                ]
                for label, members in classes.items()
            },
            "fingerprint_fields": diff,
            "cube_deltas": deltas,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"harvest: {args.harvest} ({len(runs)} runs with evidence)")
    print()
    print("== digest classes (shared pins must agree byte-for-byte) ==")
    for label, members in classes.items():
        print(f"  {label}:")
        for run in members:
            print(f"    run {run.run_id} [{run.kind}] {len(run.digests)} pins")
        pins: dict[str, set[str]] = defaultdict(set)
        for run in members:
            for slug, digest in run.digests.items():
                pins[slug].add(digest)
        for slug, values in sorted(pins.items()):
            for digest in sorted(values):
                print(f"      {slug}: {digest[:16]}...")
    print()
    print("== machine-fingerprint fields by class ==")
    for name, per_class in diff.items():
        values = {tuple(v) for v in per_class.values()}
        marker = "DIFFERS" if len(values) > 1 else "same"
        print(f"  {name} [{marker}]")
        for label, observed in per_class.items():
            for value in observed:
                print(f"    {label}: {value}")
    print()
    print(f"== cross-class cube deltas (criterion: rtol={RTOL_CRITERION}) ==")
    if not deltas:
        print(
            "  none computable: no pin has captured cubes under two different\n"
            "  digests. Reference cubes are captured on the pass path only, so\n"
            "  this stays empty until a red run coincides with restored\n"
            "  references (the §14 adjudication moment) or a second class's\n"
            "  cube is harvested from a passing run of that class."
        )
    for report in deltas:
        left = report["left"]
        right = report["right"]
        assert isinstance(left, dict) and isinstance(right, dict)
        print(
            f"  {report['pin']}: {str(left['digest'])[:12]}... ({left['class']})"
            f" vs {str(right['digest'])[:12]}... ({right['class']})"
        )
        print(f"    {report['delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
