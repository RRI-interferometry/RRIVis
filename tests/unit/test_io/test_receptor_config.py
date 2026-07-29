"""Contract tests for the strict Tier 5 ``receptors:`` configuration section.

Every accepted document in ``Tier5ReceptorFeedPlan.md`` Section 16.1 must
validate, and every rejection message in Section 27 is asserted verbatim.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from radiosim.io.config import RadioSimConfig, collect_schema_issues
from radiosim.io.receptor_config import (
    ReceptorDefinitionConfig,
    ReceptorOverrideConfig,
    ReceptorsConfig,
)
from tests.fixtures.configs import valid_config_mapping

# The exact YAML blocks published in Section 16.1, in document order.
_OMITTED = ""

_EXPLICIT_HOMOGENEOUS_LINEAR = """
receptors:
  default:
    basis: linear
    feed_rotation_deg: 0.0
  output_basis: auto
"""

_HOMOGENEOUS_CIRCULAR = """
receptors:
  default:
    basis: circular
  output_basis: auto
"""

_HOMOGENEOUS_LINEAR_ROTATED = """
receptors:
  default:
    basis: linear
    feed_rotation_deg: 45.0
"""

_HETEROGENEOUS_ROTATIONS = """
receptors:
  default:
    basis: linear
  overrides:
    - antenna: {kind: number, number: 3}
      feed_rotation_deg: 30.0
    - antenna: {kind: name, name: HERA-11}
      feed_rotation_deg: -15.0
"""

_HETEROGENEOUS_BASES = """
receptors:
  default:
    basis: linear
  overrides:
    - antenna: {kind: number, number: 7}
      basis: circular
  output_basis: circular
"""

_CIRCULAR_NATIVE_LINEAR_OUTPUT = """
receptors:
  default:
    basis: circular
  output_basis: linear
"""

_ACCEPTED_YAML = (
    _OMITTED,
    _EXPLICIT_HOMOGENEOUS_LINEAR,
    _HOMOGENEOUS_CIRCULAR,
    _HOMOGENEOUS_LINEAR_ROTATED,
    _HETEROGENEOUS_ROTATIONS,
    _HETEROGENEOUS_BASES,
    _CIRCULAR_NATIVE_LINEAR_OUTPUT,
)


def _document(tmp_path, block: str) -> dict[str, object]:
    data = valid_config_mapping(tmp_path)
    parsed = yaml.safe_load(block) if block.strip() else None
    if parsed is not None:
        data.update(parsed)
    return data


def _issue(issues, path: str):
    matches = [item for item in issues if item.path == path]
    assert matches, f"no issue reported for {path!r}: {[i.path for i in issues]}"
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("block", _ACCEPTED_YAML)
def test_every_documented_receptor_mode_validates(tmp_path, block):
    data = _document(tmp_path, block)

    assert collect_schema_issues(data) == ()

    config = RadioSimConfig.model_validate(data)

    assert type(config.receptors) is ReceptorsConfig


def test_omitted_section_is_exactly_the_explicit_default(tmp_path):
    omitted = RadioSimConfig.model_validate(_document(tmp_path, _OMITTED))
    explicit = RadioSimConfig.model_validate(
        _document(tmp_path, _EXPLICIT_HOMOGENEOUS_LINEAR)
    )

    assert omitted.receptors == explicit.receptors
    assert omitted.receptors == ReceptorsConfig()
    assert omitted.receptors.default == ReceptorDefinitionConfig()
    assert omitted.receptors.default.basis == "linear"
    assert omitted.receptors.default.feed_rotation_deg == 0.0
    assert omitted.receptors.overrides == ()
    assert omitted.receptors.output_basis == "auto"


def test_receptors_is_the_tenth_top_level_section():
    assert "receptors" in RadioSimConfig.model_fields
    assert len(RadioSimConfig.model_fields) == 10


def test_receptor_models_are_strict_and_frozen():
    config = ReceptorsConfig()

    assert config.model_config["extra"] == "forbid"
    assert config.model_config["frozen"] is True
    assert ReceptorDefinitionConfig.model_config["extra"] == "forbid"
    assert ReceptorOverrideConfig.model_config["extra"] == "forbid"

    with pytest.raises(ValidationError):
        config.output_basis = "linear"  # type: ignore[misc]


def test_overrides_are_an_immutable_tuple():
    config = ReceptorsConfig.model_validate(
        {
            "overrides": [
                {"antenna": {"kind": "number", "number": 3}, "basis": "circular"}
            ]
        }
    )

    assert type(config.overrides) is tuple
    assert type(config.overrides[0]) is ReceptorOverrideConfig


def test_feeds_alias_points_at_the_receptor_section(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["feeds"] = {}

    issue = _issue(collect_schema_issues(data), "feeds")

    assert issue.code == "removed_field"
    assert issue.message == (
        "top-level 'feeds' was replaced by the Tier 5 receptor model"
    )
    assert issue.hint == (
        "Use the 'receptors' section with 'default.basis', "
        "'default.feed_rotation_deg', and 'output_basis'."
    )


@pytest.mark.parametrize(
    ("field", "value", "message", "hint"),
    [
        (
            "feed_type",
            "linear",
            "removed before v1.0; use 'basis'",
            "Set receptors.default.basis to 'linear' or 'circular'.",
        ),
        (
            "n_feeds",
            2,
            "removed before v1.0; every antenna has exactly two feeds",
            (
                "Single-feed and multi-feed antennas are rejected until Tier 7 "
                "implements them."
            ),
        ),
        (
            "feed_angle_deg",
            0.0,
            "removed before v1.0; use 'feed_rotation_deg'",
            (
                "feed_rotation_deg is an offset from the nominal orientation for "
                "the selected basis."
            ),
        ),
    ],
)
def test_removed_receptor_default_fields_have_exact_messages(
    tmp_path, field, value, message, hint
):
    data = valid_config_mapping(tmp_path)
    data["receptors"] = {"default": {field: value}}

    issue = _issue(collect_schema_issues(data), f"receptors.default.{field}")

    assert issue.code == "removed_field"
    assert issue.message == message
    assert issue.hint == hint


def test_unsupported_basis_has_the_exact_message(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["receptors"] = {"default": {"basis": "stokes"}}

    issue = _issue(collect_schema_issues(data), "receptors.default.basis")

    assert issue.message == "input should be 'linear' or 'circular'"
    assert issue.hint == (
        "Tier 5 supports exactly two receptor bases; elliptical and mixed-feed "
        "receptors are Tier 7."
    )


def test_unsupported_output_basis_has_the_exact_message(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["receptors"] = {"output_basis": "stokes"}

    issue = _issue(collect_schema_issues(data), "receptors.output_basis")

    assert issue.message == "input should be 'auto', 'linear' or 'circular'"
    assert issue.hint == (
        "Use 'auto' for a homogeneous array; name a basis explicitly for a mixed array."
    )


def test_override_basis_uses_the_same_exact_message(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["receptors"] = {
        "overrides": [
            {"antenna": {"kind": "number", "number": 0}, "basis": "elliptical"}
        ]
    }

    issue = _issue(collect_schema_issues(data), "receptors.overrides[0].basis")

    assert issue.message == "input should be 'linear' or 'circular'"
    assert issue.hint == (
        "Tier 5 supports exactly two receptor bases; elliptical and mixed-feed "
        "receptors are Tier 7."
    )


def test_unknown_receptor_field_is_rejected(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["receptors"] = {"basis": "linear"}

    issue = _issue(collect_schema_issues(data), "receptors.basis")

    assert issue.code == "extra_forbidden"
    assert issue.message == "unknown or removed field"


def test_empty_override_is_rejected():
    with pytest.raises(ValueError) as error:
        ReceptorsConfig.model_validate(
            {"overrides": [{"antenna": {"kind": "number", "number": 3}}]}
        )

    assert (
        "receptors.overrides[0] must set at least one of 'basis' or "
        "'feed_rotation_deg'" in str(error.value)
    )


@pytest.mark.parametrize(
    "reference",
    [
        {"kind": "number", "number": 3},
        {"kind": "name", "name": "HERA-11"},
    ],
)
def test_duplicate_override_reference_is_rejected(reference):
    with pytest.raises(ValueError) as error:
        ReceptorsConfig.model_validate(
            {
                "overrides": [
                    {"antenna": reference, "basis": "circular"},
                    {"antenna": reference, "feed_rotation_deg": 10.0},
                ]
            }
        )

    assert (
        "receptors.overrides[1] repeats the antenna reference already used by "
        "receptors.overrides[0]" in str(error.value)
    )


def test_override_names_are_normalized_before_duplicate_comparison():
    with pytest.raises(ValueError) as error:
        ReceptorsConfig.model_validate(
            {
                "overrides": [
                    {"antenna": {"kind": "name", "name": "ANT0"}, "basis": "circular"},
                    {
                        "antenna": {"kind": "name", "name": "  ANT0  "},
                        "feed_rotation_deg": 5.0,
                    },
                ]
            }
        )

    assert "receptors.overrides[1] repeats the antenna reference" in str(error.value)


def test_mixed_reference_kinds_are_not_a_schema_duplicate():
    config = ReceptorsConfig.model_validate(
        {
            "overrides": [
                {"antenna": {"kind": "number", "number": 0}, "basis": "circular"},
                {"antenna": {"kind": "name", "name": "ANT0"}, "basis": "circular"},
            ]
        }
    )

    assert len(config.overrides) == 2


@pytest.mark.parametrize(
    "value", ["45.0", True, float("nan"), float("inf"), float("-inf")]
)
def test_feed_rotation_requires_a_strict_finite_float(value):
    with pytest.raises(ValueError):
        ReceptorDefinitionConfig.model_validate({"feed_rotation_deg": value})


def test_feed_rotation_accepts_an_exact_integer_like_the_instrument_schema():
    definition = ReceptorDefinitionConfig.model_validate({"feed_rotation_deg": 45})

    assert definition.feed_rotation_deg == 45.0
    assert type(definition.feed_rotation_deg) is float


def test_receptor_config_models_are_exported_from_the_io_package():
    import radiosim.io as io_package

    assert io_package.ReceptorsConfig is ReceptorsConfig
    assert io_package.ReceptorDefinitionConfig is ReceptorDefinitionConfig
    assert io_package.ReceptorOverrideConfig is ReceptorOverrideConfig
    for name in (
        "ReceptorsConfig",
        "ReceptorDefinitionConfig",
        "ReceptorOverrideConfig",
    ):
        assert name in io_package.__all__
