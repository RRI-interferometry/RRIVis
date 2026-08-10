"""Regression tests for the post-Tier-8 CI-001 WP-2/WP-3 mechanics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import ci001_characterization_comparator as comparator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_section_13_5_verdict_uses_the_absolute_term_near_zero() -> None:
    """A relative delta of one at 1e-21 scale is not a scientific failure."""
    reference = np.zeros(2, dtype=np.complex128)
    candidate = reference.copy()
    candidate[1] = 1.6e-21

    report = comparator._cube_delta(candidate, reference)

    assert report["max_relative"] == 1.0
    assert report["within_section_13_5"] is True
    assert float(report["max_tolerance_ratio"]) < 1e-8


def test_section_13_5_verdict_rejects_a_real_regression() -> None:
    reference = np.ones(2, dtype=np.complex128)
    candidate = reference.copy()
    candidate[0] += 1e-8

    report = comparator._cube_delta(candidate, reference)

    assert report["within_section_13_5"] is False
    assert float(report["max_tolerance_ratio"]) > 1.0


def test_passing_experiment_is_classified_from_its_digest_manifest() -> None:
    green = comparator.RunEvidence(
        run_id="green", kind="artifact", digests={"pin": "a" * 64}
    )
    divergent = comparator.RunEvidence(
        run_id="divergent", kind="artifact", digests={"pin": "b" * 64}
    )
    evidence = comparator.VariantEvidence(
        draw="draw-1", variant="V1", exit_code=0, digests={"pin": "b" * 64}
    )

    assert (
        comparator._classify_experiment_variant(
            evidence, {"green": [green], "divergent": [divergent]}
        )
        == "divergent"
    )


def test_partial_failure_log_does_not_overstate_the_whole_variant_class() -> None:
    divergent = comparator.RunEvidence(
        run_id="divergent",
        kind="log",
        digests={"pin-a": "a" * 64, "pin-b": "b" * 64},
    )
    evidence = comparator.VariantEvidence(
        draw="draw-1",
        variant="V2",
        exit_code=1,
        digests={"pin-a": "a" * 64},
    )

    assert (
        comparator._classify_experiment_variant(evidence, {"divergent": [divergent]})
        == "partial divergent signature (1/2 class pins observed)"
    )


def test_manifest_can_report_a_control_variant_with_a_mixed_signature() -> None:
    green = comparator.RunEvidence(
        run_id="green",
        kind="artifact",
        digests={"pin-a": "g" * 64, "pin-b": "h" * 64},
    )
    divergent = comparator.RunEvidence(
        run_id="divergent",
        kind="log",
        digests={"pin-a": "a" * 64, "pin-b": "b" * 64},
    )
    evidence = comparator.VariantEvidence(
        draw="draw-1",
        variant="V2",
        exit_code=0,
        digests={"pin-a": "g" * 64, "pin-b": "b" * 64},
    )

    assert (
        comparator._classify_experiment_variant(
            evidence, {"green": [green], "divergent": [divergent]}
        )
        == "MIXED signature (green:1, divergent:1 matching pins)"
    )


def test_artifact_manifest_wins_over_cumulative_reference_cubes(tmp_path) -> None:
    """A restored artifact can hold both class cubes but observed only one."""
    run = tmp_path / "run-42"
    run.mkdir()
    (run / "observed-digests-linux-64-py311-gw0.tsv").write_text(
        f"pin\t{'b' * 64}\n", encoding="utf-8"
    )
    for digest in ("a" * 64, "b" * 64):
        cube = run / "reference_cubes" / "pin" / "linux-64-py311" / f"{digest}.npy"
        cube.parent.mkdir(parents=True, exist_ok=True)
        np.save(cube, np.ones(1))

    evidence = comparator._load_artifact_run(run)

    assert evidence.digests == {"pin": "b" * 64}
    assert set(evidence.cube_paths) == {("pin", "a" * 64), ("pin", "b" * 64)}


def test_observed_digest_manifest_survives_a_passing_pin(tmp_path, monkeypatch) -> None:
    from tests.characterization import test_tier6_current_behavior as tier6

    monkeypatch.setattr(tier6, "_record_dir", lambda: tmp_path)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")

    tier6._record_observed_digest("a measured pin", "c" * 64)

    manifest = tmp_path / f"observed-digests-{tier6._ENVIRONMENT_KEY}-gw3.tsv"
    assert manifest.read_text(encoding="utf-8") == f"a-measured-pin\t{'c' * 64}\n"


def test_failing_pin_retains_its_candidate_cube(tmp_path, monkeypatch) -> None:
    """SCI-006 CI evidence keeps the after-cube without accepting its digest."""
    from tests.characterization import test_tier6_current_behavior as tier6

    monkeypatch.setattr(tier6, "_record_dir", lambda: tmp_path)
    measured = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)
    digest = "d" * 64

    with pytest.raises(pytest.fail.Exception):
        tier6._assert_pinned_digests(
            (
                {tier6._ENVIRONMENT_KEY: ("a" * 64,)},
                "SCI-006 candidate",
                digest,
                measured,
            )
        )

    path = (
        tmp_path
        / "observed_cubes"
        / "sci-006-candidate"
        / tier6._ENVIRONMENT_KEY
        / f"{digest}.npy"
    )
    np.testing.assert_array_equal(np.load(path), measured)
    assert digest not in {"a" * 64}


def test_failure_path_prints_the_full_section_13_5_verdict() -> None:
    from tests.characterization.test_tier6_current_behavior import _cube_delta

    reference = np.zeros(2, dtype=np.complex128)
    candidate = reference.copy()
    candidate[0] = 1e-21

    report = _cube_delta(candidate, reference)

    assert "max relative d = 1.0" in report
    assert "Section 13.5 verdict: WITHIN" in report
    assert "rtol=1e-12" in report


def test_wp2_and_wp3_workflows_are_retained_and_non_gating() -> None:
    nightly = (
        REPOSITORY_ROOT / ".github/workflows/characterization-nightly.yml"
    ).read_text(encoding="utf-8")
    experiment = (
        REPOSITORY_ROOT / ".github/workflows/ci001-forced-experiment.yml"
    ).read_text(encoding="utf-8")
    ci = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'cron: "17 2 * * *"' in nightly
    assert "pull_request:" not in nightly and "push:" not in nightly
    assert "characterization-linux-64-py311" in nightly
    assert "original recurrent linux-64-py311 second digest class" in nightly
    assert "second linux-64-py312 heterogeneous-receptor" in nightly
    assert "on-demand py312 forced-discrimination path" in nightly
    assert "one cell with a second byte-stable digest class" not in nightly
    assert "retention-days: 30" in nightly
    assert "workflow_dispatch:" in experiment
    assert "pixi_environment:" in experiment
    assert "characterization-linux-64-py312" in experiment
    assert 'pixi run --environment "${PIXI_ENVIRONMENT:?}"' in experiment
    assert "continue-on-error: true" in experiment
    assert "ci001-experiment-${{ github.run_id }}-draw${{ matrix.draw }}" in experiment
    assert "observed-digest TSV manifests are retained" in experiment
    assert "characterization-${{ matrix.key }}" in ci
    assert "Diff the fingerprint against the previous green run" in ci
    assert "mv output/characterization/observed-digests-*.tsv" in ci
    assert "mv output/characterization/observed-digests-*.tsv" in nightly
