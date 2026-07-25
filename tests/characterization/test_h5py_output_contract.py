"""Characterize pinned h5py behavior relevant to future Tier 4 output."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest


def test_dependency_h5py_identity_matches_locked_environment() -> None:
    """Characterizes the exact h5py version in each locked environment."""
    expected = {
        (3, 11): "3.14.0",
        (3, 12): "3.16.0",
    }
    assert h5py.__version__ == expected[sys.version_info[:2]]


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_dependency_h5py_preserves_complex_dtype_and_values(
    tmp_path: Path,
    dtype: type[np.complexfloating],
) -> None:
    """Characterizes raw dtype preservation, not a RadioSim schema."""
    values = np.array(
        [[1.25 + 2.5j, -3.75 + 0.125j], [8.5 - 4.25j, -0.5 - 9.0j]],
        dtype=dtype,
    )
    path = tmp_path / f"{np.dtype(dtype).name}.h5"

    with h5py.File(path, "w") as handle:
        handle.create_dataset("visibilities", data=values)
    with h5py.File(path, "r") as handle:
        stored = handle["visibilities"]
        assert stored.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(stored[:], values)


def test_dependency_dimension_labels_survive_reopening(tmp_path: Path) -> None:
    """Characterizes h5py labels for planned semantic numeric axes."""
    path = tmp_path / "labels.h5"
    with h5py.File(path, "w") as handle:
        vis = handle.create_dataset(
            "visibilities",
            data=np.zeros((2, 3, 2, 4), dtype=np.complex64),
        )
        for dimension, label in zip(
            vis.dims,
            ("time", "baseline", "frequency", "correlation"),
            strict=True,
        ):
            dimension.label = label
        baseline = handle.create_dataset(
            "baseline_vectors",
            data=np.zeros((3, 3), dtype=np.float64),
        )
        baseline.dims[0].label = "baseline"
        baseline.dims[1].label = "enu_component"

    with h5py.File(path, "r") as handle:
        assert tuple(dim.label for dim in handle["visibilities"].dims) == (
            "time",
            "baseline",
            "frequency",
            "correlation",
        )
        assert tuple(dim.label for dim in handle["baseline_vectors"].dims) == (
            "baseline",
            "enu_component",
        )


def test_dependency_h5py_string_paths_are_explicit_and_nonexecutable(
    tmp_path: Path,
) -> None:
    """Characterizes supported strings; object serialization is not a contract."""
    path = tmp_path / "strings.h5"
    utf8 = h5py.string_dtype(encoding="utf-8")
    encoded_labels = np.array([b"XX", b"XY", b"YX", b"YY"], dtype="S2")

    with h5py.File(path, "w") as handle:
        handle.create_dataset("ascii", data="RadioSim", dtype=utf8)
        handle.create_dataset("unicode", data="provenance-β", dtype=utf8)
        handle.create_dataset("labels", data=encoded_labels, dtype="S2")
        with pytest.raises(OSError, match="conversion path"):
            handle.create_dataset(
                "object_to_fixed",
                data=np.array(["XX", "YY"], dtype=object),
                dtype="S2",
            )
        # h5py creates a link before its failed data conversion; remove that
        # incomplete object inside this temporary characterization file.
        del handle["object_to_fixed"]

    with h5py.File(path, "r") as handle:
        assert handle["ascii"].asstr()[()] == "RadioSim"
        assert handle["unicode"].asstr()[()] == "provenance-β"
        np.testing.assert_array_equal(handle["labels"][:], encoded_labels)
        assert "object_to_fixed" not in handle


def test_dependency_h5py_stores_large_json_attribute_and_dataset(
    tmp_path: Path,
) -> None:
    """Characterizes capacity; future provenance remains bounded datasets."""
    payload = json.dumps(
        {"payload": "x" * (1024 * 1024)},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert 1024 * 1024 <= len(payload.encode("utf-8")) < 1024 * 1024 + 100
    path = tmp_path / "large-json.h5"
    utf8 = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w") as handle:
        handle.attrs["large_json"] = payload
        handle.create_dataset("large_json", data=payload, dtype=utf8)

    with h5py.File(path, "r") as handle:
        assert handle.attrs["large_json"] == payload
        assert handle["large_json"].asstr()[()] == payload


def test_dependency_raw_h5py_accepts_semantically_invalid_radiosim_storage(
    tmp_path: Path,
) -> None:
    """Characterizes h5py permissiveness, not future reader acceptance."""
    path = tmp_path / "permissive.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_name"] = "radiosim.visibility"
        handle.attrs["schema_version"] = "999.0.0"
        handle.create_dataset(
            "visibilities",
            data=np.ones((2, 3), dtype=np.complex64),
        )

    with h5py.File(path, "r") as handle:
        assert handle.attrs["schema_version"] == "999.0.0"
        assert handle["visibilities"].shape == (2, 3)


def test_dependency_open_old_inode_survives_closed_file_replacement(
    tmp_path: Path,
) -> None:
    """Characterizes why verification handles must close before publication."""
    temporary_root: Path
    with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
        temporary_root = Path(directory)
        final = temporary_root / "result.h5"
        replacement = temporary_root / "replacement.h5"

        with h5py.File(final, "w") as handle:
            handle.create_dataset("generation", data=np.int64(1))
        old_handle = h5py.File(final, "r")
        try:
            with h5py.File(replacement, "w") as handle:
                handle.create_dataset("generation", data=np.int64(2))
            os.replace(replacement, final)

            assert old_handle["generation"][()] == 1
            with h5py.File(final, "r") as new_handle:
                assert new_handle["generation"][()] == 2
        finally:
            old_handle.close()

        assert not replacement.exists()
        assert final.exists()

    assert not temporary_root.exists()
