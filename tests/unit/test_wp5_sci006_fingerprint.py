"""Regression tests for the SCI-006 before/after evidence comparator."""

from __future__ import annotations

import numpy as np

from tools import wp5_sci006_fingerprint as wp5


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
