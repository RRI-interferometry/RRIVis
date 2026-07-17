"""Tier 1C immutable resolved-runtime and provenance tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiosim.core.runtime_config import FrozenMapping, ResolvedConfiguration
from radiosim.io.config_resolution import (
    ConfigurationSource,
    resolve_config,
)
from tests.fixtures.configs import (
    resolved_config,
    valid_config_mapping,
)


def _contains_numpy_array(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_numpy_array(getattr(value, item.name)) for item in fields(value)
        )
    if isinstance(value, Mapping):
        return any(_contains_numpy_array(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_numpy_array(item) for item in value)
    return False


def test_resolved_fixture_returns_real_bundle(tmp_path):
    bundle = resolved_config(tmp_path)

    assert isinstance(bundle, ResolvedConfiguration)
    assert bundle.runtime.antenna_layout.antenna_positions_file.is_absolute()
    assert bundle.workflow.output_dir.is_absolute()


def test_resolved_dataclasses_and_nested_mappings_are_immutable(tmp_path):
    bundle = resolved_config(tmp_path)

    with pytest.raises(FrozenInstanceError):
        bundle.runtime.location.lat_deg = 0.0
    with pytest.raises(FrozenInstanceError):
        bundle.runtime.execution.offline = False
    with pytest.raises(TypeError, match="immutable"):
        bundle.runtime.visibility["calculation_type"] = "other"
    with pytest.raises(TypeError, match="immutable"):
        bundle.provenance.override_origins["execution.backend"] = "override"


def test_runtime_has_no_workflow_field_or_numpy_array(tmp_path):
    bundle = resolved_config(tmp_path)

    assert not hasattr(bundle.runtime, "workflow")
    assert _contains_numpy_array(bundle.runtime) is False
    assert _contains_numpy_array(bundle.provenance) is False


def test_frequency_as_numpy_returns_independent_float64_arrays(tmp_path):
    frequency = resolved_config(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101.25e6, 109e6],
        },
    ).runtime.frequency

    first = frequency.as_numpy()
    second = frequency.as_numpy()
    first[1] = 999e6

    assert first.dtype == np.float64
    assert second.dtype == np.float64
    assert second.tolist() == [100e6, 101.25e6, 109e6]
    assert frequency.channel_frequencies_hz == (100e6, 101.25e6, 109e6)
    assert not np.shares_memory(first, second)


@pytest.mark.parametrize(
    "unit,start,interval,bandwidth",
    [
        ("Hz", 100e6, 1e6, 2e6),
        ("kHz", 100_000.0, 1_000.0, 2_000.0),
        ("MHz", 100.0, 1.0, 2.0),
        ("GHz", 0.1, 0.001, 0.002),
    ],
)
def test_grid_frequency_resolution_preserves_requested_interval(
    tmp_path, unit, start, interval, bandwidth
):
    bundle = resolved_config(
        tmp_path,
        frequency={
            "mode": "grid",
            "starting_frequency": start,
            "frequency_interval": interval,
            "frequency_bandwidth": bandwidth,
            "frequency_unit": unit,
        },
    )

    assert bundle.runtime.frequency.channel_frequencies_hz == (
        100e6,
        101e6,
        102e6,
    )
    assert bundle.runtime.frequency.source_mode == "grid"


def test_one_explicit_channel_is_not_reconstructed(tmp_path):
    bundle = resolved_config(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [123_456_789.125],
        },
    )

    assert bundle.runtime.frequency.channel_frequencies_hz == (123_456_789.125,)
    assert bundle.runtime.frequency.source_mode == "explicit"


def test_resolution_copies_caller_mapping_list_array_and_nested_map(tmp_path):
    channels = np.array([100e6, 101.5e6, 108e6])
    sources = [{"kind": "test_sources", "num_sources": 2}]
    data = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": channels,
        },
        sky_sources=sources,
    )
    source = ConfigurationSource.for_mapping(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )

    bundle = resolve_config(data, source=source)
    channels[1] = 999e6
    sources[0]["num_sources"] = 999
    data["sky_model"]["sources"][0]["num_sources"] = 777

    assert bundle.runtime.frequency.channel_frequencies_hz == (
        100e6,
        101.5e6,
        108e6,
    )
    assert bundle.runtime.sky_model.sources[0].options["num_sources"] == 2
    assert (
        bundle.provenance.input_snapshot["sky_model"]["sources"][0]["num_sources"] == 2
    )


def test_frozen_mapping_constructor_deeply_copies_existing_frozen_mapping():
    nested = {"items": [1, 2]}
    first = FrozenMapping(nested)
    second = FrozenMapping(first)
    nested["items"].append(3)

    assert first["items"] == (1, 2)
    assert second["items"] == (1, 2)
    assert first is not second


def test_frozen_mapping_has_no_dict_base_mutation_escape_hatch():
    frozen = FrozenMapping({"nested": {"value": 1}})

    assert not isinstance(frozen, dict)
    with pytest.raises(TypeError):
        dict.__setitem__(frozen, "new", 2)


def test_provenance_is_versioned_json_safe_and_workflow_distinguishable(tmp_path):
    bundle = resolved_config(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101.5e6, 108e6],
        },
    )

    serialized = bundle.provenance.to_json_safe()
    encoded = json.dumps(serialized, sort_keys=True)

    assert serialized["schema_version"] == 1
    assert serialized["input_snapshot"]["obs_frequency"]["channel_frequencies_hz"] == [
        100e6,
        101.5e6,
        108e6,
    ]
    assert set(bundle.provenance.workflow_origins) == {"workflow.output_dir"}
    assert "workflow.output_dir" not in bundle.provenance.runtime_origins
    assert "numpy" in encoded
    assert "object at 0x" not in encoded
    assert str(Path(tmp_path).resolve()) in encoded
    assert not hasattr(bundle.provenance, "to_json_dict")
