"""Regression tests for the SCI-006 before/after evidence comparator."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from tools import wp5_sci006_fingerprint as wp5

_HISTORICAL_HETEROGENEOUS_PY311 = (
    "c7b51d022de6c917ee8a3359d2f5f20600a8259e52977555b5148dc32a4718c1"
)
_POST_SCI006_HETEROGENEOUS_PY311 = (
    "9f07661c3348515e5fd1acc478606badd2f4c8a143f67008f8922aabedff04c5"
)
_FINGERPRINT_DIFF_SHA256 = (
    "2a053a4fedeb426ebfeb261ac6e33586121fd931a2299fbbad3d166a08f92ef3"
)


def test_controlled_before_after_diff_is_retained_machine_readably() -> None:
    """Pin the exact Section 10.5 comparison report as durable evidence."""
    path = (
        wp5.characterization.REPO_ROOT
        / "docs"
        / "development"
        / "sci006_fingerprint_diff.json"
    )
    raw = path.read_bytes()
    report = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == _FINGERPRINT_DIFF_SHA256
    assert report["before_git_head"] == "22908c1ef1ab04749e0fddbf7c128ff76c1e30bb"
    assert report["after_git_head"] == "1efcbc6b1d9b5eeedacaaa01f41dfe02b5999a27"
    assert report["environment"] == "osx-arm64-py311"
    assert report["passed"] is True

    unchanged = {
        "healpix_scalar",
        "point_unpolarized_1time_2freq",
    }
    permuted = {
        "healpix_polarized",
        "heterogeneous_receptor_bases",
        "point_gaussian_morphology",
        "point_polarized_2times",
    }
    assert set(report["workloads"]) == unchanged | permuted | {"feed_asymmetric_gain"}
    assert {name: row["shape"] for name, row in report["workloads"].items()} == {
        "feed_asymmetric_gain": [2, 3, 3, 2, 2],
        "healpix_polarized": [2, 3, 2, 2, 2],
        "healpix_scalar": [2, 3, 2, 2, 2],
        "heterogeneous_receptor_bases": [2, 3, 2, 2, 2],
        "point_gaussian_morphology": [2, 3, 2, 2, 2],
        "point_polarized_2times": [2, 3, 2, 2, 2],
        "point_unpolarized_1time_2freq": [1, 3, 2, 2, 2],
    }
    for name in unchanged:
        row = report["workloads"][name]
        assert row["expected_relation"] == "V_new = V_old (byte-identical)"
        assert row["before_equals_after_bytes"] is True
        assert row["after_matches_expected_bytes"] is True
        assert row["max_abs_after_minus_expected"] == 0.0
        assert row["passed"] is True
    for name in permuted:
        row = report["workloads"][name]
        assert row["expected_relation"] == "V_new = P V_old P^H (byte-identical)"
        assert row["before_equals_after_bytes"] is False
        assert row["after_matches_expected_bytes"] is True
        assert row["max_abs_after_minus_expected"] == 0.0
        assert row["passed"] is True

    feed = report["workloads"]["feed_asymmetric_gain"]
    assert "G and P do not commute" in feed["expected_relation"]
    assert feed["before_equals_after_bytes"] is False
    assert feed["after_matches_expected_bytes"] is True
    assert feed["max_abs_after_minus_expected"] == 0.0
    assert feed["passed"] is True

    assert set(report["configs"]) == {
        "config.yaml",
        "receptor_circular_example.yaml",
    }
    assert {name: row["shape"] for name, row in report["configs"].items()} == {
        "config.yaml": [60, 15, 101, 4],
        "receptor_circular_example.yaml": [6, 15, 3, 4],
    }
    for row in report["configs"].values():
        assert row["expected_relation"] == "V_new = V_old (byte-identical)"
        assert row["before_equals_after_bytes"] is True
        assert row["after_matches_expected_bytes"] is True
        assert row["scientific_sha256_unchanged"] is True
        assert row["max_abs_after_minus_expected"] == 0.0
        assert row["passed"] is True


def test_historical_dispatch_digest_is_retained_but_not_an_active_pin() -> None:
    """Keep the old CI-001 proof without accepting pre-correction physics."""
    adjudication = (
        wp5.characterization.REPO_ROOT
        / "docs"
        / "development"
        / "ci001_adjudication.md"
    ).read_text(encoding="utf-8")
    active = wp5.characterization._WORKLOAD_DIGESTS["heterogeneous_receptor_bases"][
        "linux-64-py311"
    ]

    assert _HISTORICAL_HETEROGENEOUS_PY311 in adjudication
    assert _POST_SCI006_HETEROGENEOUS_PY311 in adjudication
    for provenance_token in (
        "31273416758",
        "424b4b90dcb162f4d54a3cb4f4abf2516269ca44",
        "93143054023",
        "9026308532",
        "ci001-experiment-31273416758-draw3",
        "AVX512_SKX",
        "Cooperlake",
    ):
        assert provenance_token in adjudication
    assert _HISTORICAL_HETEROGENEOUS_PY311 not in repr(
        wp5.characterization._WORKLOAD_DIGESTS
    )
    assert _HISTORICAL_HETEROGENEOUS_PY311 not in active
    assert _POST_SCI006_HETEROGENEOUS_PY311 in active


def test_feed_asymmetric_mapping_keeps_gains_on_physical_east_x_and_north_y() -> None:
    before = np.array(
        [[[[[3.0 + 1.0j, 5.0 + 2.0j], [7.0 - 2.0j, 11.0 - 1.0j]]]]],
        dtype=np.complex128,
    )
    after = wp5._feed_asymmetric_expected(before)

    expected = np.array(
        [[[[[176.0 - 16.0j, 7.0 - 2.0j], [5.0 + 2.0j, 0.1875 + 0.0625j]]]]],
        dtype=np.complex128,
    )
    np.testing.assert_array_equal(after, expected)


def test_ci_artifact_requires_exact_permutation_and_unchanged_relations(
    tmp_path,
) -> None:
    environment = "linux-64-py311"
    manifest_lines: list[str] = []

    for index, name in enumerate(wp5._PERMUTED_WORKLOADS):
        slug = wp5._WORKLOAD_SLUGS[name]
        reference_digest = f"{index + 1:064x}"
        candidate_digest = f"{index + 101:064x}"
        reference = np.arange(16, dtype=np.float64).reshape(1, 2, 2, 2, 2) + 1j * (
            index + 1
        )
        candidate = reference[..., [1, 0], :][..., [1, 0]]
        reference_path = (
            tmp_path
            / "reference_cubes"
            / slug
            / environment
            / f"{reference_digest}.npy"
        )
        candidate_path = (
            tmp_path / "observed_cubes" / slug / environment / f"{candidate_digest}.npy"
        )
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(reference_path, reference)
        np.save(candidate_path, candidate)
        manifest_lines.append(f"{slug}\t{candidate_digest}\n")

    unchanged = {
        **{name: wp5._WORKLOAD_SLUGS[name] for name in wp5._UNCHANGED_WORKLOADS},
        **wp5._CONFIG_SLUGS,
    }
    for index, slug in enumerate(unchanged.values(), start=201):
        digest = f"{index:064x}"
        path = tmp_path / "reference_cubes" / slug / environment / f"{digest}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.ones((1, 1, 1, 2, 2), dtype=np.complex128))
        manifest_lines.append(f"{slug}\t{digest}\n")

    (tmp_path / f"observed-digests-{environment}-gw0.tsv").write_text(
        "".join(manifest_lines), encoding="utf-8"
    )

    report = wp5.compare_ci_artifact(tmp_path)

    assert report["environment"] == environment
    assert report["passed"] is True
    assert all(item["passed"] for item in report["workloads"].values())
    assert all(item["passed"] for item in report["configs"].values())
