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
   count, and an explicit verdict against the full Section 13.5 float64
   predicate named by ``Tier8ReleasePlan.md`` §14 — and says so honestly
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

WP-3 extension: ``--experiment <dir>`` additionally reads the artifacts of a
``ci001-forced-experiment.yml`` run (downloaded as ``<dir>/draw-<n>/``, one
per matrix job) and reports the per-job, per-variant table the experiment
exists to produce: dispatched tier, OpenBLAS runtime core, pass/fail, digest
class, and the maximum relative delta against the green reference cubes.  A
variant that passed matched the recorded observation set byte-for-byte (its
delta is exactly zero); a variant that failed carries measured digests and
delta lines in its pytest log, and its digests are classified against the
harvest's digest classes.  A digest matching no known class is reported as
``NOVEL``: it is a new observation to record and adjudicate, never a value to
append so that a failure goes away.

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
#: Section 13.5 float64 tolerance the project already uses.
RTOL_CRITERION = 1e-12
ATOL_SCALE_CRITERION = 1e-12

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
    cube_paths: dict[tuple[str, str], Path] = field(default_factory=dict)


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
    for manifest in sorted(run_dir.glob("observed-digests-*.tsv")):
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                pin_slug, digest = line.split("\t", maxsplit=1)
            except ValueError:
                continue
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                evidence.digests.setdefault(pin_slug, digest)
    for cube in sorted(run_dir.glob("reference_cubes/*/*/*.npy")):
        pin_slug = cube.parent.parent.name
        evidence.digests.setdefault(pin_slug, cube.stem)
        evidence.cube_paths[(pin_slug, cube.stem)] = cube
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
    """The §14 probe, including the full Section 13.5 float64 predicate."""
    if left.shape != right.shape:
        return {"comparable": False, "reason": f"shape {left.shape} vs {right.shape}"}
    difference = np.abs(left - right)
    max_absolute = float(np.max(difference)) if difference.size else 0.0
    scale = np.maximum(np.abs(left), np.abs(right))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(scale > 0.0, difference / scale, 0.0)
    max_relative = float(np.max(relative)) if relative.size else 0.0
    reference_scale = max(1.0, float(np.max(np.abs(right))) if right.size else 0.0)
    atol = ATOL_SCALE_CRITERION * reference_scale
    allowed = atol + RTOL_CRITERION * np.abs(right)
    within = bool(np.all(difference <= allowed))
    with np.errstate(divide="ignore", invalid="ignore"):
        tolerance_ratio = np.where(allowed > 0.0, difference / allowed, 0.0)
    max_tolerance_ratio = (
        float(np.max(tolerance_ratio)) if tolerance_ratio.size else 0.0
    )
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
        "section_13_5_atol": atol,
        "max_tolerance_ratio": max_tolerance_ratio,
        "differing_elements": int(differing.size),
        "total_elements": int(left.size),
        "first_differing_index": first,
        "within_section_13_5": within,
    }


def _cross_class_deltas(
    classes: dict[str, list[RunEvidence]],
) -> list[dict[str, object]]:
    """Deltas for every pin where two different digests both have a cube."""
    cubes: dict[str, dict[str, Path]] = defaultdict(dict)  # slug -> digest -> path
    owners: dict[tuple[str, str], str] = {}
    for label, members in classes.items():
        for run in members:
            for (slug, digest), path in run.cube_paths.items():
                cubes[slug].setdefault(digest, path)
                key = (slug, digest)
                if run.digests.get(slug) == digest:
                    owners[key] = label
                else:
                    owners.setdefault(key, "retained reference")
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


_PROBE_FEATURES_RE = re.compile(r"numpy dispatched features: (.+)")
_PROBE_CORE_RE = re.compile(r"openblas runtime core: (\S+)")
_MAX_ABSOLUTE_RE = re.compile(r"max\|dV\| = ([0-9.eE+-]+)")
_MAX_RELATIVE_RE = re.compile(r"max relative d = ([0-9.eE+-]+)")
_NEAREST_RE = re.compile(r"nearest recorded observation: ([0-9a-f]{64})")
_SECTION_13_5_VERDICT_RE = re.compile(r"Section 13\.5 verdict: (WITHIN|OUTSIDE)")


@dataclass
class VariantEvidence:
    """Everything one experiment variant (one pytest run) contributed."""

    draw: str
    variant: str
    cpu_model: str = "?"
    exit_code: int | None = None
    dispatched_features: str = "?"
    openblas_core: str = "?"
    env_overrides: dict[str, str] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)  # pin slug -> digest
    max_absolute_delta: float | None = None
    max_relative_delta: float | None = None
    within_section_13_5: bool | None = None
    nearest_observation: str | None = None


def _dispatch_tier(features: str) -> str:
    """Summarize a dispatched-feature list as the tier CI-001 cares about."""
    names = {name.strip() for name in features.split(",") if name.strip()}
    if any(name.startswith("AVX512") for name in names):
        return "AVX512"
    if "AVX2" in names:
        return "AVX2"
    return "unknown"


def _load_experiment_variant(draw_dir: Path, variant: str) -> VariantEvidence:
    """Read one variant's probe, exit code, and pytest log from a draw."""
    evidence = VariantEvidence(draw=draw_dir.name, variant=variant)
    variant_dir = draw_dir / "experiment" / variant

    probe = variant_dir / "runtime-probe.txt"
    if probe.exists():
        text = probe.read_text(encoding="utf-8", errors="replace")
        match = _PROBE_FEATURES_RE.search(text)
        if match:
            evidence.dispatched_features = match.group(1).strip()
        match = _PROBE_CORE_RE.search(text)
        if match:
            evidence.openblas_core = match.group(1)
        for name in ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE"):
            override = re.search(rf"{name}=(.+)", text)
            if override and override.group(1).strip() not in ("<unset>", ""):
                evidence.env_overrides[name] = override.group(1).strip()

    exit_file = variant_dir / "pytest-exit-code.txt"
    if exit_file.exists():
        try:
            evidence.exit_code = int(exit_file.read_text().strip())
        except ValueError:
            evidence.exit_code = None

    log = variant_dir / "pytest.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        current_pin: str | None = None
        absolute_deltas: list[float] = []
        deltas: list[float] = []
        verdicts: list[bool] = []
        for line in text.splitlines():
            pin_match = _PIN_LABEL_RE.search(line)
            if pin_match:
                current_pin = _slug(pin_match.group(1).strip())
                continue
            measured = _MEASURED_RE.search(line)
            if measured and current_pin is not None:
                evidence.digests.setdefault(current_pin, measured.group(1))
            for value in _MAX_ABSOLUTE_RE.findall(line):
                try:
                    absolute_deltas.append(float(value))
                except ValueError:
                    continue
            for value in _MAX_RELATIVE_RE.findall(line):
                try:
                    deltas.append(float(value))
                except ValueError:
                    continue
            nearest = _NEAREST_RE.search(line)
            if nearest:
                evidence.nearest_observation = nearest.group(1)
            verdicts.extend(
                verdict == "WITHIN"
                for verdict in _SECTION_13_5_VERDICT_RE.findall(line)
            )
        if absolute_deltas:
            evidence.max_absolute_delta = max(absolute_deltas)
        if deltas:
            evidence.max_relative_delta = max(deltas)
        if verdicts:
            evidence.within_section_13_5 = all(verdicts)
        elif absolute_deltas and max(absolute_deltas) <= ATOL_SCALE_CRITERION:
            # Historical logs predate the explicit verdict.  Since Section
            # 13.5's absolute allowance is always at least 1e-12, this bound
            # proves the full predicate without guessing the reference scale.
            evidence.within_section_13_5 = True

    for manifest in sorted((variant_dir / "record").glob("observed-digests-*.tsv")):
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                pin_slug, digest = line.split("\t", maxsplit=1)
            except ValueError:
                continue
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                evidence.digests.setdefault(pin_slug, digest)

    for record in sorted(
        (draw_dir / "experiment" / "baseline").glob("machine-fingerprint-*.txt")
    ):
        parsed = _parse_fingerprint_text(
            record.read_text(encoding="utf-8", errors="replace")
        )
        if "cpu model" in parsed:
            evidence.cpu_model = parsed["cpu model"]
            break
    return evidence


def _load_experiment(experiment: Path) -> list[VariantEvidence]:
    variants: list[VariantEvidence] = []
    for draw_dir in sorted(experiment.glob("draw-*")):
        if not draw_dir.is_dir():
            continue
        for variant in ("V1", "V2", "V3", "V4"):
            if (draw_dir / "experiment" / variant).is_dir():
                variants.append(_load_experiment_variant(draw_dir, variant))
    return variants


def _classify_experiment_variant(
    evidence: VariantEvidence, classes: dict[str, list[RunEvidence]]
) -> str:
    """Name the digest class one variant landed in.

    A variant is classified from its retained observed-digest manifest, whether
    pytest passed or failed.  This remains meaningful after WP-3 accepts a
    second class.  Older passing artifacts without that manifest can only be
    called accepted, not assigned to one class.  A failing digest that matches
    no harvested class is a NOVEL observation, reported for adjudication and
    never appended to make the failure go away.
    """
    if not evidence.digests and evidence.exit_code == 0:
        return "accepted class (digest manifest unavailable)"
    if not evidence.digests:
        return "failed without digest evidence"
    signatures: dict[str, dict[str, str]] = {}
    for label, members in classes.items():
        signature: dict[str, str] = {}
        for member in members:
            signature.update(member.digests)
        signatures[label] = signature

    support: dict[str, int] = {}
    conflict: dict[str, int] = {}
    for label, signature in signatures.items():
        shared = set(evidence.digests) & set(signature)
        support[label] = sum(evidence.digests[pin] == signature[pin] for pin in shared)
        conflict[label] = len(shared) - support[label]

    exact = [
        label
        for label, signature in signatures.items()
        if support[label] == len(signature) and conflict[label] == 0
    ]
    if exact:
        return "/".join(exact)

    partial = [
        label for label in signatures if support[label] > 0 and conflict[label] == 0
    ]
    if len(partial) == 1:
        label = partial[0]
        return (
            f"partial {label} signature "
            f"({support[label]}/{len(signatures[label])} class pins observed)"
        )

    supported = [label for label in signatures if support[label] > 0]
    if len(supported) > 1:
        details = ", ".join(f"{label}:{support[label]}" for label in supported)
        return f"MIXED signature ({details} matching pins)"
    if evidence.exit_code == 0:
        return "ACCEPTED BUT UNCLASSIFIED (harvest is incomplete)"
    return "NOVEL (matches no known class)"


def _print_experiment_report(
    variants: list[VariantEvidence], classes: dict[str, list[RunEvidence]]
) -> None:
    print()
    print(
        "== forced-experiment variants "
        "(criterion: Section 13.5 float64; delta is vs green references) =="
    )
    header = (
        f"  {'draw':<8} {'variant':<8} {'tier':<7} {'blas core':<10} "
        f"{'result':<7} {'max|dV|':<12} {'§13.5':<8} class"
    )
    print(header)
    for evidence in variants:
        result = (
            "pass"
            if evidence.exit_code == 0
            else "FAIL"
            if evidence.exit_code is not None
            else "?"
        )
        if evidence.exit_code == 0:
            delta = "0"
        elif evidence.max_absolute_delta is not None:
            delta = f"{evidence.max_absolute_delta:.3e}"
        else:
            delta = "n/a"
        criterion = (
            "within"
            if evidence.within_section_13_5 is True
            else "OUTSIDE"
            if evidence.within_section_13_5 is False
            else "n/a"
        )
        label = _classify_experiment_variant(evidence, classes)
        print(
            f"  {evidence.draw:<8} {evidence.variant:<8} "
            f"{_dispatch_tier(evidence.dispatched_features):<7} "
            f"{evidence.openblas_core:<10} {result:<7} {delta:<12} "
            f"{criterion:<8} {label}"
        )
    models = sorted({evidence.cpu_model for evidence in variants})
    print()
    print("  cpu models drawn:")
    for model in models:
        draws = sorted(
            {evidence.draw for evidence in variants if evidence.cpu_model == model}
        )
        print(f"    {model}: {', '.join(draws)}")
    novel = [
        evidence
        for evidence in variants
        if evidence.exit_code not in (0, None)
        and _classify_experiment_variant(evidence, classes).startswith("NOVEL")
    ]
    if novel:
        print()
        print(
            "  NOVEL digests observed (record and adjudicate; a set never grows"
            " to make a failure go away):"
        )
        for evidence in novel:
            for slug, digest in sorted(evidence.digests.items()):
                print(f"    {evidence.draw}/{evidence.variant} {slug}: {digest}")


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
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="directory of downloaded ci001-forced-experiment artifacts "
        "(one draw-<n>/ subdirectory per matrix job)",
    )
    args = parser.parse_args(argv)

    if not args.harvest.exists():
        print(f"harvest directory {args.harvest} does not exist", file=sys.stderr)
        return 2
    if args.experiment is not None and not args.experiment.exists():
        print(f"experiment directory {args.experiment} does not exist", file=sys.stderr)
        return 2

    runs = _collect(args.harvest)
    classes = _classify(runs)
    diff = _fingerprint_diff(classes)
    deltas = _cross_class_deltas(classes)
    variants = _load_experiment(args.experiment) if args.experiment is not None else []

    if args.json:
        payload = {
            "criterion_rtol": RTOL_CRITERION,
            "criterion_atol_scale": ATOL_SCALE_CRITERION,
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
        if variants:
            payload["experiment"] = [
                {
                    "draw": evidence.draw,
                    "variant": evidence.variant,
                    "cpu_model": evidence.cpu_model,
                    "dispatch_tier": _dispatch_tier(evidence.dispatched_features),
                    "dispatched_features": evidence.dispatched_features,
                    "openblas_core": evidence.openblas_core,
                    "env_overrides": evidence.env_overrides,
                    "exit_code": evidence.exit_code,
                    "digests": evidence.digests,
                    "max_absolute_delta": evidence.max_absolute_delta,
                    "max_relative_delta": evidence.max_relative_delta,
                    "within_section_13_5": evidence.within_section_13_5,
                    "nearest_observation": evidence.nearest_observation,
                    "digest_class": _classify_experiment_variant(evidence, classes),
                }
                for evidence in variants
            ]
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
    print(
        "== cross-class cube deltas "
        f"(Section 13.5: rtol={RTOL_CRITERION}, "
        f"atol={ATOL_SCALE_CRITERION}*max(1,max|reference|)) =="
    )
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
    if variants:
        _print_experiment_report(variants, classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
