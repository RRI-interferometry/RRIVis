"""Capture and compare the SCI-006 characterization evidence set.

The capture is intentionally built from the same runners and digest recipes as
``tests/characterization/test_tier6_current_behavior.py``.  It writes generated
``.npy`` cubes and a JSON manifest beneath a caller-selected directory.  The
comparison classifies the two unpolarized workloads and the two hermetic shipped
configurations as byte-preserving, and the four polarized linear-output
workloads as the deliberate ``V_new = P V_old P^H`` change.  A separate
feed-asymmetric gain workload proves that native feed 0 remains physical X/east
and feed 1 remains Y/north; it is deliberately not classified by the simple
permutation relation because ``G`` and ``P`` do not commute.

This tool never edits the characterization pin tables.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Importing the characterization module normally records diagnostic files.
# This tool owns its output tree, so disable that independent side effect.
os.environ.setdefault("RADIOSIM_CHARACTERIZATION_RECORD_DIR", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.characterization import (
    test_tier6_current_behavior as characterization,  # noqa: E402
)

_UNCHANGED_WORKLOADS = (
    "healpix_scalar",
    "point_unpolarized_1time_2freq",
)
_PERMUTED_WORKLOADS = (
    "healpix_polarized",
    "heterogeneous_receptor_bases",
    "point_gaussian_morphology",
    "point_polarized_2times",
)
_FEED_ASYMMETRIC_WORKLOAD = "feed_asymmetric_gain"
_EAST_X_GAIN = 2.0
_NORTH_Y_GAIN = 0.5
_UNCHANGED_CONFIGS = (
    "config.yaml",
    "receptor_circular_example.yaml",
)
_WORKLOAD_SLUGS = {
    name: "section-13-4-workload-" + name.replace("_", "-")
    for name in (*_UNCHANGED_WORKLOADS, *_PERMUTED_WORKLOADS)
}
_CONFIG_SLUGS = {
    "config.yaml": "configs-config-yaml-raw-cube-sha256",
    "receptor_circular_example.yaml": (
        "configs-receptor-circular-example-yaml-raw-cube-sha256"
    ),
}


def _head_sha() -> str:
    override = os.environ.get("RADIOSIM_WP5_SOURCE_SHA")
    if override:
        if len(override) != 40 or any(
            char not in "0123456789abcdef" for char in override
        ):
            raise ValueError("RADIOSIM_WP5_SOURCE_SHA must be a lowercase 40-hex SHA")
        return override
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=characterization.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _array_record(path: Path, array: np.ndarray, *, digest: str) -> dict[str, Any]:
    np.save(path, array)
    return {
        "digest": digest,
        "dtype": str(array.dtype),
        "path": path.name,
        "shape": list(array.shape),
    }


def _run_feed_asymmetric_gain(output_dir: Path) -> np.ndarray:
    """Run the production point solver with exact, non-commuting feed gains."""
    from tests.unit.test_jones.test_gain import _cube

    per_antenna = [
        {
            "antenna": antenna,
            "feed": feed,
            "amplitude_error": gain - 1.0,
        }
        for antenna in (0, 1)
        for feed, gain in ((0, _EAST_X_GAIN), (1, _NORTH_Y_GAIN))
    ]
    return np.ascontiguousarray(
        _cube(
            output_dir,
            {"G": {"amplitude_error": 0.0, "per_antenna": per_antenna}},
        )
    )


def _feed_asymmetric_expected(before: np.ndarray) -> np.ndarray:
    """Map the old north-first result to fixed physical east-X/north-Y gains."""
    expected = np.empty_like(before)
    gain_ratio_squared = (_EAST_X_GAIN / _NORTH_Y_GAIN) ** 2
    expected[..., 0, 0] = gain_ratio_squared * before[..., 1, 1]
    expected[..., 0, 1] = before[..., 1, 0]
    expected[..., 1, 0] = before[..., 0, 1]
    expected[..., 1, 1] = before[..., 0, 0] / gain_ratio_squared
    return expected


def capture(output_dir: Path, *, phase: str) -> dict[str, Any]:
    """Capture all nine locally reproducible SCI-006 evidence cubes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "environment": characterization._ENVIRONMENT_KEY,
        "git_head": _head_sha(),
        "numpy": np.__version__,
        "phase": phase,
        "python": platform.python_version(),
        "workloads": {},
        "configs": {},
    }

    for name, runner in sorted(characterization._WORKLOAD_RUNNERS.items()):
        with tempfile.TemporaryDirectory(prefix=f"wp5-{name}-") as raw:
            cube = np.ascontiguousarray(np.asarray(runner(Path(raw))))
        manifest["workloads"][name] = _array_record(
            output_dir / f"{name}.npy",
            cube,
            digest=characterization._cube_digest(cube),
        )

    with tempfile.TemporaryDirectory(prefix="wp5-feed-asymmetric-") as raw:
        cube = _run_feed_asymmetric_gain(Path(raw))
    manifest["workloads"][_FEED_ASYMMETRIC_WORKLOAD] = _array_record(
        output_dir / f"{_FEED_ASYMMETRIC_WORKLOAD}.npy",
        cube,
        digest=characterization._cube_digest(cube),
    )

    for name in _UNCHANGED_CONFIGS:
        with tempfile.TemporaryDirectory(prefix="wp5-config-") as raw:
            result = characterization._run_shipped_config(name, Path(raw))
        cube = np.ascontiguousarray(np.asarray(result.visibilities))
        slug = name.removesuffix(".yaml").replace("_", "-")
        record = _array_record(
            output_dir / f"{slug}.npy",
            cube,
            digest=characterization._raw_cube_digest(cube),
        )
        record["scientific_sha256"] = result.scientific_sha256
        manifest["configs"][name] = record

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_array(root: Path, record: dict[str, Any]) -> np.ndarray:
    return np.load(root / record["path"], allow_pickle=False)


def _comparison_record(
    before: np.ndarray,
    after: np.ndarray,
    expected: np.ndarray,
) -> dict[str, Any]:
    shape_matches = after.shape == expected.shape
    difference = np.abs(after - expected) if shape_matches else np.array([np.inf])
    return {
        "after_matches_expected_bytes": bool(
            shape_matches and np.array_equal(after, expected)
        ),
        "before_equals_after_bytes": bool(np.array_equal(before, after)),
        "max_abs_after_minus_expected": float(np.max(difference)),
        "shape": list(after.shape),
    }


def compare(before_dir: Path, after_dir: Path) -> dict[str, Any]:
    """Compare a candidate capture with its same-environment parent capture."""
    before_manifest = json.loads(
        (before_dir / "manifest.json").read_text(encoding="utf-8")
    )
    after_manifest = json.loads(
        (after_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if before_manifest["environment"] != after_manifest["environment"]:
        raise ValueError("before and after captures must use the same environment")

    report: dict[str, Any] = {
        "before_git_head": before_manifest["git_head"],
        "after_git_head": after_manifest["git_head"],
        "environment": before_manifest["environment"],
        "workloads": {},
        "configs": {},
    }
    all_pass = True

    for name in _UNCHANGED_WORKLOADS:
        before = _load_array(before_dir, before_manifest["workloads"][name])
        after = _load_array(after_dir, after_manifest["workloads"][name])
        record = _comparison_record(before, after, before)
        record["expected_relation"] = "V_new = V_old (byte-identical)"
        record["passed"] = record["after_matches_expected_bytes"]
        all_pass = all_pass and record["passed"]
        report["workloads"][name] = record

    permutation = np.array([1, 0])
    for name in _PERMUTED_WORKLOADS:
        before = _load_array(before_dir, before_manifest["workloads"][name])
        after = _load_array(after_dir, after_manifest["workloads"][name])
        expected = np.take(np.take(before, permutation, axis=-2), permutation, axis=-1)
        record = _comparison_record(before, after, expected)
        record["expected_relation"] = "V_new = P V_old P^H (byte-identical)"
        record["passed"] = record["after_matches_expected_bytes"]
        all_pass = all_pass and record["passed"]
        report["workloads"][name] = record

    before = _load_array(
        before_dir,
        before_manifest["workloads"][_FEED_ASYMMETRIC_WORKLOAD],
    )
    after = _load_array(
        after_dir,
        after_manifest["workloads"][_FEED_ASYMMETRIC_WORKLOAD],
    )
    record = _comparison_record(before, after, _feed_asymmetric_expected(before))
    record["expected_relation"] = (
        "feed 0 remains X/east with gain 2; feed 1 remains Y/north with gain 0.5; "
        "G and P do not commute"
    )
    record["passed"] = record["after_matches_expected_bytes"]
    all_pass = all_pass and record["passed"]
    report["workloads"][_FEED_ASYMMETRIC_WORKLOAD] = record

    for name in _UNCHANGED_CONFIGS:
        before = _load_array(before_dir, before_manifest["configs"][name])
        after = _load_array(after_dir, after_manifest["configs"][name])
        record = _comparison_record(before, after, before)
        record["expected_relation"] = "V_new = V_old (byte-identical)"
        record["scientific_sha256_unchanged"] = (
            before_manifest["configs"][name]["scientific_sha256"]
            == after_manifest["configs"][name]["scientific_sha256"]
        )
        record["passed"] = (
            record["after_matches_expected_bytes"]
            and record["scientific_sha256_unchanged"]
        )
        all_pass = all_pass and record["passed"]
        report["configs"][name] = record

    report["passed"] = all_pass
    return report


def _observed_digests(record_dir: Path) -> dict[str, set[str]]:
    observed: dict[str, set[str]] = {}
    for manifest in record_dir.glob("observed-digests-*.tsv"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                slug, digest = line.split("\t", maxsplit=1)
            except ValueError:
                continue
            observed.setdefault(slug, set()).add(digest)
    return observed


def compare_ci_artifact(record_dir: Path) -> dict[str, Any]:
    """Adjudicate one CI cell before any SCI-006 pin regeneration."""
    environments = {
        path.parent.name for path in record_dir.glob("observed_cubes/*/*/*.npy")
    }
    if len(environments) != 1:
        raise ValueError(
            "CI artifact must contain candidate cubes for exactly one environment"
        )
    environment = environments.pop()
    digests = _observed_digests(record_dir)
    report: dict[str, Any] = {
        "environment": environment,
        "workloads": {},
        "configs": {},
    }
    all_pass = True
    permutation = np.array([1, 0])

    for name in _PERMUTED_WORKLOADS:
        slug = _WORKLOAD_SLUGS[name]
        references = sorted(
            (record_dir / "reference_cubes" / slug / environment).glob("*.npy")
        )
        candidates = sorted(
            (record_dir / "observed_cubes" / slug / environment).glob("*.npy")
        )
        matches: list[dict[str, str]] = []
        for candidate_path in candidates:
            candidate = np.load(candidate_path, allow_pickle=False)
            for reference_path in references:
                reference = np.load(reference_path, allow_pickle=False)
                expected = np.take(
                    np.take(reference, permutation, axis=-2),
                    permutation,
                    axis=-1,
                )
                if np.array_equal(candidate, expected):
                    matches.append(
                        {
                            "candidate_digest": candidate_path.stem,
                            "reference_digest": reference_path.stem,
                        }
                    )
        passed = bool(matches) and len(candidates) == 1
        all_pass = all_pass and passed
        report["workloads"][name] = {
            "candidate_count": len(candidates),
            "expected_relation": "V_new = P V_old P^H (byte-identical)",
            "matches": matches,
            "passed": passed,
            "reference_count": len(references),
        }

    unchanged = {
        **{name: _WORKLOAD_SLUGS[name] for name in _UNCHANGED_WORKLOADS},
        **_CONFIG_SLUGS,
    }
    for name, slug in unchanged.items():
        measured = digests.get(slug, set())
        references = {
            path.stem
            for path in (record_dir / "reference_cubes" / slug / environment).glob(
                "*.npy"
            )
        }
        passed = len(measured) == 1 and measured <= references
        all_pass = all_pass and passed
        target = "configs" if name in _UNCHANGED_CONFIGS else "workloads"
        report[target][name] = {
            "expected_relation": "V_new = V_old (byte-identical)",
            "measured_digests": sorted(measured),
            "passed": passed,
            "reference_digests": sorted(references),
        }

    report["passed"] = all_pass
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("output_dir", type=Path)
    capture_parser.add_argument("--phase", choices=("before", "after"), required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("before_dir", type=Path)
    compare_parser.add_argument("after_dir", type=Path)
    compare_parser.add_argument("--output", type=Path)

    artifact_parser = subparsers.add_parser("compare-ci-artifact")
    artifact_parser.add_argument("record_dir", type=Path)
    artifact_parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "capture":
        payload = capture(args.output_dir, phase=args.phase)
    elif args.command == "compare":
        payload = compare(args.before_dir, args.after_dir)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        payload = compare_ci_artifact(args.record_dir)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
