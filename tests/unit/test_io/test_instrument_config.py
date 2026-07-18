"""Tests for the inactive Tier 2B instrument input contract."""

from __future__ import annotations

import builtins
import json
import socket
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

import radiosim
import radiosim.io as public_io
from radiosim.io.config import (
    BaselineSelectionConfig as LegacyBaselineSelectionConfig,
)
from radiosim.io.config import RadioSimConfig, StrictFrozenModel
from radiosim.io.config_resolution import ConfigurationSource, resolve_config
from radiosim.io.instrument_config import (
    AntennaDiameterOverrideConfig,
    AntennaNameReference,
    AntennaNumberReference,
    AntennaReference,
    AzimuthRangeConfig,
    BaselineSelectionConfig,
    InstrumentConfig,
    InstrumentLocationConfig,
    InstrumentSourceConfig,
    KnownTelescopeSourceConfig,
    LayoutFileSourceConfig,
    LengthFilterConfig,
    LengthRangeConfig,
    LengthRangesConfig,
    LengthTargetsConfig,
)
from tests.fixtures.configs import valid_config_mapping

SOURCE_ADAPTER = TypeAdapter(InstrumentSourceConfig)
REFERENCE_ADAPTER = TypeAdapter(AntennaReference)
LENGTH_FILTER_ADAPTER = TypeAdapter(LengthFilterConfig)


def _location() -> dict[str, float]:
    return {
        "longitude_deg": 116.67,
        "latitude_deg": -26.70,
        "height_m": 377.8,
    }


def _local_source(
    *,
    format: str = "radiosim",
    path: str = "layouts/array.txt",
    telescope_name: str = "Example Array",
) -> dict[str, Any]:
    return {
        "kind": "layout_file",
        "path": path,
        "format": format,
        "telescope_name": telescope_name,
    }


def _instrument(**updates: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source": _local_source(),
        "location": _location(),
    }
    data.update(updates)
    return data


def _all_concrete_models() -> tuple[type[StrictFrozenModel], ...]:
    return (
        LayoutFileSourceConfig,
        KnownTelescopeSourceConfig,
        InstrumentLocationConfig,
        AntennaNumberReference,
        AntennaNameReference,
        AntennaDiameterOverrideConfig,
        InstrumentConfig,
        LengthTargetsConfig,
        LengthRangeConfig,
        LengthRangesConfig,
        AzimuthRangeConfig,
        BaselineSelectionConfig,
    )


def _valid_model_inputs() -> dict[type[StrictFrozenModel], dict[str, Any]]:
    return {
        LayoutFileSourceConfig: _local_source(),
        KnownTelescopeSourceConfig: {
            "kind": "known_telescope",
            "name": "HERA",
        },
        InstrumentLocationConfig: _location(),
        AntennaNumberReference: {"kind": "number", "number": 0},
        AntennaNameReference: {"kind": "name", "name": "ANT-0"},
        AntennaDiameterOverrideConfig: {
            "antenna": {"kind": "number", "number": 0},
            "diameter_m": 14.0,
        },
        InstrumentConfig: _instrument(),
        LengthTargetsConfig: {
            "mode": "targets",
            "targets_m": [0.0, 14.0],
            "tolerance_m": 0.5,
        },
        LengthRangeConfig: {"min_m": 0.0, "max_m": 14.0},
        LengthRangesConfig: {
            "mode": "ranges",
            "ranges_m": [{"min_m": 0.0, "max_m": 14.0}],
        },
        AzimuthRangeConfig: {"start_deg": 170.0, "end_deg": 10.0},
        BaselineSelectionConfig: {},
    }


@pytest.mark.parametrize(
    "format",
    ["radiosim", "casa_loc", "measurement_set", "uvfits", "mwa_metafits"],
)
def test_layout_source_accepts_exact_retained_formats(format):
    source_data = _local_source(format=format)
    if format in {"measurement_set", "uvfits"}:
        source_data.pop("telescope_name")

    source = SOURCE_ADAPTER.validate_python(source_data)

    assert isinstance(source, LayoutFileSourceConfig)
    assert source.kind == "layout_file"
    assert source.format == format
    assert source.path == Path("layouts/array.txt")


@pytest.mark.parametrize("format", ["radiosim", "casa_loc", "mwa_metafits"])
def test_local_layout_formats_require_explicit_identity(format):
    source = _local_source(format=format)
    source.pop("telescope_name")

    with pytest.raises(ValidationError, match="telescope_name"):
        SOURCE_ADAPTER.validate_python(source)


@pytest.mark.parametrize("format", ["measurement_set", "uvfits"])
def test_dataset_layout_identity_is_optional_but_allowed(format):
    without_name = _local_source(format=format)
    without_name.pop("telescope_name")

    assert SOURCE_ADAPTER.validate_python(without_name).telescope_name is None
    assert (
        SOURCE_ADAPTER.validate_python(
            _local_source(format=format, telescope_name="Dataset Array")
        ).telescope_name
        == "Dataset Array"
    )


@pytest.mark.parametrize("format", ["casa", "mwa", "pyuvdata", "unknown"])
def test_layout_source_rejects_legacy_and_unknown_format_literals(format):
    with pytest.raises(ValidationError):
        SOURCE_ADAPTER.validate_python(_local_source(format=format))


@pytest.mark.parametrize(
    "path",
    ["", "   ", "$DATA/layout.txt", "${DATA}/layout.txt", Path("$DATA/file")],
)
def test_layout_source_rejects_empty_and_environment_paths(path):
    with pytest.raises(ValidationError, match="path|environment"):
        LayoutFileSourceConfig.model_validate(
            _local_source(path=path)  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("path", [b"layout.txt", 1, 1.0, object()])
def test_layout_source_rejects_non_path_input_types(path):
    with pytest.raises(ValidationError):
        LayoutFileSourceConfig.model_validate(_local_source(path=path))


def test_layout_source_preserves_tilde_without_resolving(monkeypatch):
    monkeypatch.setenv("HOME", "/must/not/be/used")

    source = LayoutFileSourceConfig.model_validate(
        _local_source(path="~/layouts/array.txt")
    )

    assert source.path == Path("~/layouts/array.txt")
    assert not source.path.is_absolute()


def test_layout_source_validation_performs_no_filesystem_io(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.txt"

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Tier 2B source validation performed filesystem I/O")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "resolve", forbidden)
        guard.setattr(Path, "exists", forbidden)
        guard.setattr(Path, "is_file", forbidden)
        guard.setattr(Path, "is_dir", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(builtins, "open", forbidden)
        source = LayoutFileSourceConfig.model_validate(_local_source(path=str(missing)))

    assert source.path == missing


@pytest.mark.parametrize("field", ["name", "registry_policy"])
def test_layout_source_rejects_known_telescope_only_fields(field):
    data = _local_source()
    data[field] = "HERA" if field == "name" else "offline"

    with pytest.raises(ValidationError, match=field):
        SOURCE_ADAPTER.validate_python(data)


def test_known_telescope_source_normalizes_name_and_defaults_offline():
    source = SOURCE_ADAPTER.validate_python(
        {"kind": "known_telescope", "name": "  He\u0301RA  "}
    )

    assert isinstance(source, KnownTelescopeSourceConfig)
    assert source.name == "H\u00e9RA"
    assert source.registry_policy == "offline"


@pytest.mark.parametrize("policy", ["offline", "allow_network"])
def test_known_telescope_accepts_only_exact_registry_policies(policy):
    source = KnownTelescopeSourceConfig(
        name="Unknown-but-structurally-valid", registry_policy=policy
    )

    assert source.registry_policy == policy
    assert source.name == "Unknown-but-structurally-valid"


@pytest.mark.parametrize("name", ["", "   ", 123, None])
def test_known_telescope_requires_a_real_nonblank_name(name):
    with pytest.raises(ValidationError):
        KnownTelescopeSourceConfig.model_validate(
            {"kind": "known_telescope", "name": name}
        )


@pytest.mark.parametrize("policy", ["online", "network", True, None])
def test_known_telescope_rejects_unknown_registry_policy(policy):
    with pytest.raises(ValidationError):
        KnownTelescopeSourceConfig.model_validate(
            {
                "kind": "known_telescope",
                "name": "HERA",
                "registry_policy": policy,
            }
        )


@pytest.mark.parametrize("field", ["path", "format", "telescope_name"])
def test_known_telescope_rejects_layout_only_fields(field):
    data: dict[str, Any] = {"kind": "known_telescope", "name": "HERA"}
    data[field] = "layout.txt" if field == "path" else "radiosim"

    with pytest.raises(ValidationError, match=field):
        SOURCE_ADAPTER.validate_python(data)


def test_known_telescope_validation_does_not_import_registry_or_use_network(
    monkeypatch,
):
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("pyuvdata", "astropy.coordinates")):
            pytest.fail(f"Tier 2B validation imported registry dependency {name}")
        return real_import(name, *args, **kwargs)

    def forbidden_network(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Tier 2B validation attempted network access")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)

    source = KnownTelescopeSourceConfig(name="NotEnumerated")

    assert source.name == "NotEnumerated"


@pytest.mark.parametrize(
    "data",
    [
        {"path": "layout.txt", "format": "radiosim", "telescope_name": "A"},
        {"kind": "mystery", "name": "A"},
        {
            "kind": "layout_file",
            "path": "layout.txt",
            "format": "radiosim",
            "telescope_name": "A",
            "name": "B",
        },
        {
            "kind": "known_telescope",
            "name": "A",
            "path": "layout.txt",
        },
    ],
)
def test_instrument_source_discriminator_is_required_and_exclusive(data):
    with pytest.raises(ValidationError):
        SOURCE_ADAPTER.validate_python(data)


def test_source_validation_does_not_mutate_caller_mapping():
    caller = _local_source(telescope_name="  A\u0301rray  ")
    before = deepcopy(caller)

    source = SOURCE_ADAPTER.validate_python(caller)

    assert caller == before
    assert source.telescope_name == "\u00c1rray"


@pytest.mark.parametrize("format", ["radiosim", "measurement_set"])
def test_layout_source_rejects_blank_supplied_identity(format):
    with pytest.raises(ValidationError, match="telescope_name"):
        LayoutFileSourceConfig.model_validate(
            _local_source(format=format, telescope_name="   ")
        )


def test_location_requires_exact_three_fields_and_allows_negative_height():
    location = InstrumentLocationConfig(
        longitude_deg=1,
        latitude_deg=2.5,
        height_m=-430,
    )

    assert location.longitude_deg == 1.0
    assert location.latitude_deg == 2.5
    assert location.height_m == -430.0


@pytest.mark.parametrize("missing", ["longitude_deg", "latitude_deg", "height_m"])
def test_location_rejects_missing_fields(missing):
    data = _location()
    data.pop(missing)

    with pytest.raises(ValidationError, match=missing):
        InstrumentLocationConfig.model_validate(data)


@pytest.mark.parametrize("legacy", ["lon", "lat", "height"])
def test_location_rejects_legacy_field_names(legacy):
    data = _location()
    replacement = {
        "lon": "longitude_deg",
        "lat": "latitude_deg",
        "height": "height_m",
    }[legacy]
    data[legacy] = data.pop(replacement)

    with pytest.raises(ValidationError, match=legacy):
        InstrumentLocationConfig.model_validate(data)


@pytest.mark.parametrize("field", ["longitude_deg", "latitude_deg", "height_m"])
@pytest.mark.parametrize(
    "value",
    [True, False, "1.0", float("nan"), float("inf"), float("-inf")],
)
def test_location_rejects_non_strict_or_nonfinite_values(field, value):
    data = _location()
    data[field] = value

    with pytest.raises(ValidationError):
        InstrumentLocationConfig.model_validate(data)


@pytest.mark.parametrize("format", ["radiosim", "casa_loc", "mwa_metafits"])
def test_instrument_requires_location_for_local_layout_formats(format):
    with pytest.raises(ValidationError, match="location"):
        InstrumentConfig(source=_local_source(format=format), location=None)


@pytest.mark.parametrize("format", ["measurement_set", "uvfits"])
def test_instrument_allows_missing_or_explicit_location_for_dataset_formats(format):
    source = _local_source(format=format)
    source.pop("telescope_name")

    assert InstrumentConfig(source=source).location is None
    assert InstrumentConfig(source=source, location=_location()).location is not None


def test_instrument_allows_missing_or_explicit_location_for_known_telescope():
    source = {"kind": "known_telescope", "name": "HERA"}

    assert InstrumentConfig(source=source).location is None
    assert InstrumentConfig(source=source, location=_location()).location is not None


def test_instrument_diameter_defaults_distinguish_no_hidden_fallback():
    omitted = InstrumentConfig.model_validate(_instrument())
    explicit_null = InstrumentConfig.model_validate(
        _instrument(default_diameter_m=None)
    )

    assert omitted.default_diameter_m is None
    assert explicit_null.default_diameter_m is None
    assert omitted.diameter_overrides == ()
    assert omitted == explicit_null
    assert 14.0 not in omitted.model_dump(mode="json").values()


@pytest.mark.parametrize("value", [1, 14.0, 0.001])
def test_instrument_accepts_positive_finite_default_diameter(value):
    model = InstrumentConfig.model_validate(_instrument(default_diameter_m=value))

    assert model.default_diameter_m == float(value)


@pytest.mark.parametrize(
    "value",
    [0, -1, True, False, "14", float("nan"), float("inf"), float("-inf")],
)
def test_instrument_rejects_invalid_default_diameter(value):
    with pytest.raises(ValidationError):
        InstrumentConfig.model_validate(_instrument(default_diameter_m=value))


@pytest.mark.parametrize("number", [0, 2_147_483_647])
def test_antenna_number_reference_accepts_exact_boundaries(number):
    reference = REFERENCE_ADAPTER.validate_python({"kind": "number", "number": number})

    assert isinstance(reference, AntennaNumberReference)
    assert reference.number == number


@pytest.mark.parametrize(
    "number",
    [True, False, "1", 1.0, 1.5, -1, 2_147_483_648],
)
def test_antenna_number_reference_rejects_non_integer_or_out_of_range(number):
    with pytest.raises(ValidationError):
        REFERENCE_ADAPTER.validate_python({"kind": "number", "number": number})


def test_antenna_name_reference_uses_shared_nfc_case_preserving_normalization():
    reference = REFERENCE_ADAPTER.validate_python(
        {"kind": "name", "name": "  A\u0301nt-ONE  "}
    )

    assert isinstance(reference, AntennaNameReference)
    assert reference.name == "\u00c1nt-ONE"


@pytest.mark.parametrize("name", ["", "  ", 1, 1.0, True, None])
def test_antenna_name_reference_requires_real_nonblank_string(name):
    with pytest.raises(ValidationError):
        REFERENCE_ADAPTER.validate_python({"kind": "name", "name": name})


def test_numeric_looking_antenna_name_remains_a_name():
    reference = REFERENCE_ADAPTER.validate_python({"kind": "name", "name": "007"})

    assert isinstance(reference, AntennaNameReference)
    assert reference.name == "007"


@pytest.mark.parametrize(
    "data",
    [
        {"number": 1},
        {"kind": "other", "number": 1},
        {"kind": "number", "number": 1, "name": "one"},
        {"kind": "name", "name": "one", "number": 1},
    ],
)
def test_antenna_reference_discriminator_is_required_and_exclusive(data):
    with pytest.raises(ValidationError):
        REFERENCE_ADAPTER.validate_python(data)


@pytest.mark.parametrize("value", [1, 14.0, 0.001])
def test_diameter_override_accepts_positive_finite_value(value):
    override = AntennaDiameterOverrideConfig(
        antenna={"kind": "number", "number": 3},
        diameter_m=value,
    )

    assert override.diameter_m == float(value)


@pytest.mark.parametrize(
    "value",
    [0, -1, True, False, "14", float("nan"), float("inf"), float("-inf")],
)
def test_diameter_override_rejects_invalid_value(value):
    with pytest.raises(ValidationError):
        AntennaDiameterOverrideConfig.model_validate(
            {
                "antenna": {"kind": "number", "number": 3},
                "diameter_m": value,
            }
        )


def test_override_collection_is_copy_owned_frozen_and_inventory_agnostic():
    antenna = {"kind": "name", "name": "  ANT-A  "}
    override = {"antenna": antenna, "diameter_m": 12.0}
    overrides = [override]
    caller = _instrument(diameter_overrides=overrides)
    before = deepcopy(caller)

    model = InstrumentConfig.model_validate(caller)

    assert caller == before
    overrides.append({"antenna": {"kind": "number", "number": 99}, "diameter_m": 20.0})
    antenna["name"] = "CHANGED"
    override["diameter_m"] = 99.0
    assert isinstance(model.diameter_overrides, tuple)
    assert len(model.diameter_overrides) == 1
    assert model.diameter_overrides[0].antenna.name == "ANT-A"
    assert model.diameter_overrides[0].diameter_m == 12.0


def test_mixed_name_and_number_overrides_are_structurally_allowed():
    model = InstrumentConfig.model_validate(
        _instrument(
            diameter_overrides=[
                {"antenna": {"kind": "number", "number": 1}, "diameter_m": 12},
                {"antenna": {"kind": "name", "name": "ANT-1"}, "diameter_m": 13},
            ]
        )
    )

    assert len(model.diameter_overrides) == 2


@pytest.mark.parametrize(
    ("targets", "tolerance"),
    [([0.0], 0.0), ([0, 10.5, 20], 0.25)],
)
def test_length_targets_accept_zero_positive_values_and_required_tolerance(
    targets, tolerance
):
    config = LengthTargetsConfig(targets_m=targets, tolerance_m=tolerance)

    assert config.targets_m == tuple(float(value) for value in targets)
    assert config.tolerance_m == float(tolerance)


@pytest.mark.parametrize(
    "data",
    [
        {"targets_m": [], "tolerance_m": 0.0},
        {"targets_m": [1.0, 1.0], "tolerance_m": 0.0},
        {"targets_m": [-1.0], "tolerance_m": 0.0},
        {"targets_m": [True], "tolerance_m": 0.0},
        {"targets_m": ["1"], "tolerance_m": 0.0},
        {"targets_m": [float("nan")], "tolerance_m": 0.0},
        {"targets_m": [float("inf")], "tolerance_m": 0.0},
        {"targets_m": [1.0]},
        {"targets_m": [1.0], "tolerance_m": -1.0},
        {"targets_m": [1.0], "tolerance_m": True},
        {"targets_m": [1.0], "tolerance_m": "0.1"},
        {"targets_m": [1.0], "tolerance_m": float("inf")},
    ],
)
def test_length_targets_reject_invalid_values_duplicates_and_missing_tolerance(data):
    with pytest.raises(ValidationError):
        LengthTargetsConfig.model_validate(data)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(0, 0), (0.0, 10.0), (10, 10), (1.5, 25.5)],
)
def test_length_range_accepts_exact_and_normal_nonnegative_ranges(minimum, maximum):
    range_config = LengthRangeConfig(min_m=minimum, max_m=maximum)

    assert range_config.min_m == float(minimum)
    assert range_config.max_m == float(maximum)


@pytest.mark.parametrize(
    "data",
    [
        {"min_m": 2.0, "max_m": 1.0},
        {"min_m": -1.0, "max_m": 1.0},
        {"min_m": 0.0, "max_m": -1.0},
        {"min_m": True, "max_m": 1.0},
        {"min_m": "0", "max_m": 1.0},
        {"min_m": 0.0, "max_m": float("nan")},
        {"min_m": 0.0, "max_m": float("inf")},
    ],
)
def test_length_range_rejects_reversed_negative_nonstrict_and_nonfinite(data):
    with pytest.raises(ValidationError):
        LengthRangeConfig.model_validate(data)


def test_length_ranges_keep_order_and_allow_overlaps():
    ranges = [
        {"min_m": 20.0, "max_m": 30.0},
        {"min_m": 0.0, "max_m": 10.0},
        {"min_m": 5.0, "max_m": 25.0},
    ]

    config = LengthRangesConfig(ranges_m=ranges)

    assert [(item.min_m, item.max_m) for item in config.ranges_m] == [
        (20.0, 30.0),
        (0.0, 10.0),
        (5.0, 25.0),
    ]


@pytest.mark.parametrize(
    "ranges",
    [
        [],
        [
            {"min_m": 0.0, "max_m": 1.0},
            {"min_m": 0.0, "max_m": 1.0},
        ],
    ],
)
def test_length_ranges_reject_empty_and_exact_duplicates(ranges):
    with pytest.raises(ValidationError):
        LengthRangesConfig(ranges_m=ranges)


@pytest.mark.parametrize(
    "data",
    [
        {"targets_m": [1.0], "tolerance_m": 0.0},
        {"mode": "unknown", "targets_m": [1.0], "tolerance_m": 0.0},
        {
            "mode": "targets",
            "targets_m": [1.0],
            "tolerance_m": 0.0,
            "ranges_m": [{"min_m": 0.0, "max_m": 1.0}],
        },
        {
            "mode": "ranges",
            "ranges_m": [{"min_m": 0.0, "max_m": 1.0}],
            "targets_m": [1.0],
        },
    ],
)
def test_length_filter_discriminator_is_required_and_exclusive(data):
    with pytest.raises(ValidationError):
        LENGTH_FILTER_ADAPTER.validate_python(data)


@pytest.mark.parametrize(
    ("start", "end"),
    [(0.0, 10.0), (10.0, 170.0), (170.0, 10.0), (0, 179.999999999)],
)
def test_azimuth_range_accepts_normal_wrapped_and_boundary_values(start, end):
    range_config = AzimuthRangeConfig(start_deg=start, end_deg=end)

    assert range_config.start_deg == float(start)
    assert range_config.end_deg == float(end)


@pytest.mark.parametrize(
    "data",
    [
        {"start_deg": -0.001, "end_deg": 10.0},
        {"start_deg": 0.0, "end_deg": 180.0},
        {"start_deg": 180.0, "end_deg": 0.0},
        {"start_deg": 10.0, "end_deg": 10.0},
        {"start_deg": True, "end_deg": 10.0},
        {"start_deg": "0", "end_deg": 10.0},
        {"start_deg": float("nan"), "end_deg": 10.0},
        {"start_deg": 0.0, "end_deg": float("inf")},
    ],
)
def test_azimuth_range_rejects_invalid_endpoints(data):
    with pytest.raises(ValidationError):
        AzimuthRangeConfig.model_validate(data)


def test_baseline_selection_defaults_and_all_correlations():
    assert BaselineSelectionConfig() == BaselineSelectionConfig(
        correlations="all",
        length_filter=None,
        azimuth_ranges_deg=(),
    )

    for correlations in ("all", "cross", "auto"):
        assert BaselineSelectionConfig(correlations=correlations).correlations == (
            correlations
        )


@pytest.mark.parametrize("correlations", ["neither", "both", "", None, True])
def test_baseline_selection_rejects_invalid_correlation(correlations):
    with pytest.raises(ValidationError):
        BaselineSelectionConfig.model_validate({"correlations": correlations})


def test_baseline_selection_accepts_none_targets_ranges_and_overlapping_azimuths():
    no_filter = BaselineSelectionConfig(length_filter=None)
    targets = BaselineSelectionConfig(
        length_filter={
            "mode": "targets",
            "targets_m": [0, 30, 10],
            "tolerance_m": 0,
        }
    )
    ranges = BaselineSelectionConfig(
        length_filter={
            "mode": "ranges",
            "ranges_m": [
                {"min_m": 20, "max_m": 40},
                {"min_m": 0, "max_m": 30},
            ],
        },
        azimuth_ranges_deg=[
            {"start_deg": 10, "end_deg": 80},
            {"start_deg": 40, "end_deg": 100},
        ],
    )

    assert no_filter.length_filter is None
    assert targets.length_filter.targets_m == (0.0, 30.0, 10.0)
    assert isinstance(ranges.length_filter, LengthRangesConfig)
    assert len(ranges.azimuth_ranges_deg) == 2


def test_baseline_selection_rejects_exact_duplicate_azimuth_pairs():
    with pytest.raises(ValidationError):
        BaselineSelectionConfig(
            azimuth_ranges_deg=[
                {"start_deg": 170.0, "end_deg": 10.0},
                {"start_deg": 170.0, "end_deg": 10.0},
            ]
        )


def test_baseline_collections_are_copy_owned_tuples_and_retain_input_order():
    target_values = [30.0, 0.0, 10.0]
    first_range = {"start_deg": 170.0, "end_deg": 10.0}
    azimuth_values = [first_range, {"start_deg": 20.0, "end_deg": 40.0}]
    caller = {
        "length_filter": {
            "mode": "targets",
            "targets_m": target_values,
            "tolerance_m": 0.5,
        },
        "azimuth_ranges_deg": azimuth_values,
    }
    before = deepcopy(caller)

    model = BaselineSelectionConfig.model_validate(caller)

    assert caller == before
    target_values.reverse()
    first_range["start_deg"] = 1.0
    azimuth_values.clear()
    assert model.length_filter.targets_m == (30.0, 0.0, 10.0)
    assert [(item.start_deg, item.end_deg) for item in model.azimuth_ranges_deg] == [
        (170.0, 10.0),
        (20.0, 40.0),
    ]


@pytest.mark.parametrize(
    "legacy_field",
    [
        "antenna_positions_file",
        "antenna_file_format",
        "all_antenna_diameter",
        "use_different_diameters",
        "diameters",
        "use_pyuvdata_telescope",
        "use_pyuvdata_location",
        "use_pyuvdata_antennas",
        "use_pyuvdata_diameters",
    ],
)
def test_instrument_rejects_removed_legacy_fields(legacy_field):
    data = _instrument()
    data[legacy_field] = True

    with pytest.raises(ValidationError) as exc_info:
        InstrumentConfig.model_validate(data)

    assert any(error["type"] == "extra_forbidden" for error in exc_info.value.errors())


@pytest.mark.parametrize(
    "legacy_field",
    [
        "use_autocorrelations",
        "use_crosscorrelations",
        "only_selective_baseline_length",
        "selective_baseline_lengths",
        "selective_baseline_tolerance_meters",
        "trim_by_angle_ranges",
        "selective_angle_ranges_deg",
    ],
)
def test_baseline_selection_rejects_removed_legacy_fields(legacy_field):
    with pytest.raises(ValidationError) as exc_info:
        BaselineSelectionConfig.model_validate({legacy_field: True})

    assert any(error["type"] == "extra_forbidden" for error in exc_info.value.errors())


def test_every_concrete_model_is_strict_frozen_and_rejects_unknown_fields():
    valid_inputs = _valid_model_inputs()

    for model_type in _all_concrete_models():
        assert issubclass(model_type, StrictFrozenModel)
        assert model_type.model_config["extra"] == "forbid"
        assert model_type.model_config["frozen"] is True
        data = deepcopy(valid_inputs[model_type])
        data["tier2b_unknown"] = True
        with pytest.raises(ValidationError) as exc_info:
            model_type.model_validate(data)
        assert any(
            error["type"] == "extra_forbidden" for error in exc_info.value.errors()
        ), model_type.__name__


def test_nested_models_and_model_copies_remain_frozen():
    model = InstrumentConfig.model_validate(
        _instrument(
            diameter_overrides=[
                {"antenna": {"kind": "number", "number": 1}, "diameter_m": 14}
            ]
        )
    )
    copied = model.model_copy(deep=True)

    with pytest.raises(ValidationError, match="frozen"):
        model.default_diameter_m = 12.0
    with pytest.raises(ValidationError, match="frozen"):
        model.source.path = Path("other.txt")
    with pytest.raises(ValidationError, match="frozen"):
        model.diameter_overrides[0].antenna.number = 2
    with pytest.raises(ValidationError, match="frozen"):
        copied.location.height_m = 0.0


def test_no_mutable_list_or_mapping_escapes_through_model_fields():
    model = InstrumentConfig.model_validate(
        _instrument(
            diameter_overrides=[
                {"antenna": {"kind": "name", "name": "A"}, "diameter_m": 14}
            ]
        )
    )
    selection = BaselineSelectionConfig(
        length_filter={
            "mode": "ranges",
            "ranges_m": [{"min_m": 0, "max_m": 10}],
        },
        azimuth_ranges_deg=[{"start_deg": 170, "end_deg": 10}],
    )

    for value in (*model.__dict__.values(), *selection.__dict__.values()):
        assert not isinstance(value, (list, dict))


def test_layout_json_serialization_uses_path_string_and_discriminator():
    source = LayoutFileSourceConfig.model_validate(_local_source())

    assert source.model_dump(mode="json") == {
        "kind": "layout_file",
        "path": "layouts/array.txt",
        "format": "radiosim",
        "telescope_name": "Example Array",
    }
    assert '"kind":"layout_file"' in source.model_dump_json()


def test_known_source_json_serialization_includes_offline_default():
    source = KnownTelescopeSourceConfig(name="HERA")

    assert source.model_dump(mode="json") == {
        "kind": "known_telescope",
        "name": "HERA",
        "registry_policy": "offline",
    }


def test_instrument_json_serialization_round_trips_exact_tagged_shape():
    model = InstrumentConfig.model_validate(
        _instrument(
            default_diameter_m=None,
            diameter_overrides=[
                {"antenna": {"kind": "number", "number": 1}, "diameter_m": 12},
                {"antenna": {"kind": "name", "name": "ANT-A"}, "diameter_m": 14},
            ],
        )
    )
    before = model.model_dump(mode="json")

    dumped = model.model_dump(mode="json")
    reparsed = InstrumentConfig.model_validate(dumped)

    assert dumped == {
        "source": {
            "kind": "layout_file",
            "path": "layouts/array.txt",
            "format": "radiosim",
            "telescope_name": "Example Array",
        },
        "location": _location(),
        "default_diameter_m": None,
        "diameter_overrides": [
            {"antenna": {"kind": "number", "number": 1}, "diameter_m": 12.0},
            {"antenna": {"kind": "name", "name": "ANT-A"}, "diameter_m": 14.0},
        ],
    }
    assert reparsed == model
    assert json.loads(model.model_dump_json()) == dumped
    assert model.model_dump(mode="json") == before


def test_empty_override_and_default_selection_serialization_are_canonical():
    instrument = InstrumentConfig.model_validate(_instrument())
    selection = BaselineSelectionConfig()

    assert instrument.model_dump(mode="json")["diameter_overrides"] == []
    assert selection.model_dump(mode="json") == {
        "correlations": "all",
        "length_filter": None,
        "azimuth_ranges_deg": [],
    }


@pytest.mark.parametrize(
    "selection",
    [
        {
            "correlations": "cross",
            "length_filter": {
                "mode": "targets",
                "targets_m": [30, 0, 10],
                "tolerance_m": 0.5,
            },
            "azimuth_ranges_deg": [{"start_deg": 170, "end_deg": 10}],
        },
        {
            "correlations": "auto",
            "length_filter": {
                "mode": "ranges",
                "ranges_m": [
                    {"min_m": 20, "max_m": 30},
                    {"min_m": 0, "max_m": 10},
                ],
            },
            "azimuth_ranges_deg": [],
        },
    ],
)
def test_selection_json_serialization_round_trips_without_sorting(selection):
    model = BaselineSelectionConfig.model_validate(selection)

    dumped = model.model_dump(mode="json")

    assert BaselineSelectionConfig.model_validate(dumped) == model
    assert json.loads(model.model_dump_json()) == dumped
    if dumped["length_filter"]["mode"] == "targets":
        assert dumped["length_filter"]["targets_m"] == [30.0, 0.0, 10.0]
    else:
        assert dumped["length_filter"]["ranges_m"] == [
            {"min_m": 20.0, "max_m": 30.0},
            {"min_m": 0.0, "max_m": 10.0},
        ]


def test_serialized_values_are_new_mutable_json_containers_not_model_aliases():
    model = InstrumentConfig.model_validate(
        _instrument(
            diameter_overrides=[
                {"antenna": {"kind": "name", "name": "A"}, "diameter_m": 14}
            ]
        )
    )

    dumped = model.model_dump(mode="json")
    dumped["source"]["telescope_name"] = "Changed"
    dumped["diameter_overrides"][0]["antenna"]["name"] = "Changed"
    dumped["diameter_overrides"].append({"changed": True})

    assert model.source.telescope_name == "Example Array"
    assert model.diameter_overrides[0].antenna.name == "A"
    assert len(model.diameter_overrides) == 1


def test_active_top_level_schema_and_legacy_selection_remain_unchanged(tmp_path):
    expected_fields = (
        "telescope",
        "antenna_layout",
        "feeds",
        "beams",
        "baseline_selection",
        "location",
        "sky_model",
        "obs_time",
        "obs_frequency",
        "visibility",
        "execution",
        "workflow",
    )

    assert tuple(RadioSimConfig.model_fields) == expected_fields
    assert "instrument" not in RadioSimConfig.model_fields
    assert LegacyBaselineSelectionConfig is not BaselineSelectionConfig
    assert "use_autocorrelations" in LegacyBaselineSelectionConfig.model_fields
    assert "correlations" not in LegacyBaselineSelectionConfig.model_fields

    data = valid_config_mapping(tmp_path)
    data["instrument"] = _instrument()
    with pytest.raises(ValidationError) as exc_info:
        RadioSimConfig.model_validate(data)
    assert any(
        error["loc"] == ("instrument",) and error["type"] == "extra_forbidden"
        for error in exc_info.value.errors()
    )


def test_importing_internal_module_does_not_mutate_active_schema_or_reexport_types():
    probe = """
import radiosim
import radiosim.io as public_io
from radiosim.io.config import RadioSimConfig

fields_before = tuple(RadioSimConfig.model_fields)
import radiosim.io.instrument_config as instrument_config_module

assert tuple(RadioSimConfig.model_fields) == fields_before
assert instrument_config_module.InstrumentConfig.__module__ == (
    "radiosim.io.instrument_config"
)
for module in (radiosim, public_io):
    assert not hasattr(module, "InstrumentConfig")
    assert "InstrumentConfig" not in module.__all__
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert tuple(RadioSimConfig.model_fields) == (
        "telescope",
        "antenna_layout",
        "feeds",
        "beams",
        "baseline_selection",
        "location",
        "sky_model",
        "obs_time",
        "obs_frequency",
        "visibility",
        "execution",
        "workflow",
    )
    for module in (radiosim, public_io):
        assert not hasattr(module, "InstrumentConfig")
        assert "InstrumentConfig" not in module.__all__


def test_existing_configuration_resolution_path_keeps_legacy_shape(tmp_path):
    data = valid_config_mapping(tmp_path)

    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    assert bundle.runtime.baseline_selection["use_autocorrelations"] is True
    assert bundle.runtime.baseline_selection["use_crosscorrelations"] is True
    assert "correlations" not in bundle.runtime.baseline_selection
    assert not hasattr(bundle.runtime, "instrument")
