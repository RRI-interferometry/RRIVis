"""Tier 7D: the strict ``jones:`` configuration schema.

``Tier7JonesSciencePlan.md`` Section 21.  This file owns what a *schema* can
decide on its own -- shape, types, unknown keys, unknown ``kind`` -- and nothing
that needs a resolved instrument or a resolved band.  Those rejections (R2,
R4-R7, R11) live in ``tests/unit/test_core/test_jones_resolution.py``, because
Section 26.1 makes the stage a failure is raised at part of the contract, and
splitting the tests the same way is what keeps that honest.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from radiosim.io.config import RadioSimConfig, collect_schema_issues
from radiosim.io.jones_config import (
    BandpassPolynomialModel,
    BandpassTabulatedModel,
    BandpassTermConfig,
    CableReflectionTermConfig,
    CrosshandTermConfig,
    DelayTermConfig,
    ExplicitLeakageModel,
    FrequencyPolynomialLeakageModel,
    GainTermConfig,
    IXRLeakageModel,
    JonesConfig,
    LeakageTermConfig,
    LinearDriftTimeModel,
    ParallacticTermConfig,
    SinusoidalTimeModel,
    StaticTimeModel,
    as_complex,
)
from tests.fixtures.configs import valid_config_mapping

_GAIN_WITH_EVERYTHING = """
jones:
  G:
    amplitude_error: 0.02
    phase_error_rad: 0.0
    per_antenna:
      - antenna: 1
        feed: 0
        amplitude_error: 0.05
        phase_error_rad: 0.13
    elevation_curve: [1.0, -1.0e-4]
    time_model:
      kind: linear_drift
      rate_per_hour: 0.01
"""

_BANDPASS_POLYNOMIAL = """
jones:
  B:
    model:
      kind: polynomial
      coefficients: [1.0, 0.0, -0.05]
      reference_frequency_hz: null
      scale_frequency_hz: null
    per_antenna: []
"""

_BANDPASS_TABULATED = """
jones:
  B:
    model:
      kind: tabulated
      node_frequencies_hz: [9.0e+7, 1.0e+8, 1.1e+8, 1.2e+8]
      gains: [[0.9, 0.0], [1.0, 0.05], [0.98, 0.0], [0.9, -0.05]]
"""

_BOTH_TERMS = """
jones:
  G:
    amplitude_error: 0.02
  B:
    model:
      kind: polynomial
      coefficients: [1.0, 0.0, -0.05]
"""

_ACCEPTED_YAML = (
    _GAIN_WITH_EVERYTHING,
    _BANDPASS_POLYNOMIAL,
    _BANDPASS_TABULATED,
    _BOTH_TERMS,
)


def _document(tmp_path, block: str) -> dict[str, Any]:
    data = valid_config_mapping(tmp_path)
    parsed = yaml.safe_load(block) if block.strip() else None
    if parsed is not None:
        data.update(parsed)
    return data


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", _ACCEPTED_YAML)
def test_every_documented_jones_block_validates(tmp_path, block: str) -> None:
    """Section 21.2's YAML, as written, must parse."""
    data = _document(tmp_path, block)

    assert collect_schema_issues(data) == ()

    config = RadioSimConfig.model_validate(data)

    assert type(config.jones) is JonesConfig


def test_an_omitted_section_is_none_and_not_an_empty_model(tmp_path) -> None:
    """The distinction R2 rests on.

    A ``default_factory`` would make an absent section and ``jones: {}``
    indistinguishable, and then either both would be rejected -- breaking every
    configuration ever written -- or neither would, which is the silent no-op
    this tier removes.
    """
    config = RadioSimConfig.model_validate(valid_config_mapping(tmp_path))

    assert config.jones is None


def test_an_empty_section_parses_and_is_left_for_resolution(tmp_path) -> None:
    """``jones: {}`` is well formed; it is R2, at resolution, that rejects it."""
    data = _document(tmp_path, "jones: {}")

    config = RadioSimConfig.model_validate(data)

    assert type(config.jones) is JonesConfig
    assert config.jones.configured_terms == ()


# ---------------------------------------------------------------------------
# Strictness
# ---------------------------------------------------------------------------


def test_jones_models_are_strict_and_frozen() -> None:
    """Tier 1/5/6 discipline: extra keys forbidden, instances frozen."""
    for model in (
        JonesConfig,
        GainTermConfig,
        BandpassTermConfig,
        BandpassPolynomialModel,
        BandpassTabulatedModel,
        StaticTimeModel,
        LinearDriftTimeModel,
        SinusoidalTimeModel,
    ):
        assert model.model_config["extra"] == "forbid"
        assert model.model_config["frozen"] is True

    config = GainTermConfig(amplitude_error=0.02)
    with pytest.raises(ValidationError):
        config.amplitude_error = 0.5  # type: ignore[misc]


def test_an_unimplemented_term_letter_is_rejected(tmp_path) -> None:
    """R3: a term no slice has implemented yet is not silently accepted.

    ``Z`` is a real Jones term with a real Section 20.8 definition, and it is
    still rejected here, because RadioSim cannot yet honour it.  A schema field
    for a term whose evaluation raises would be a configuration surface that
    accepts a value and discards it -- defect D2, one level up.

    FLIPPED BY: Tier 7E.  ``D`` was the probe until this slice implemented it.

    FLIPPED BY: Tier 7F, which implemented ``P``; the probe moved to ``Z``
    rather than being deleted, because the property being pinned is "the schema
    surface never runs ahead of the physics", and that property needs a witness
    for as long as any term is still planned.
    """
    data = _document(
        tmp_path, "jones:\n  Z:\n    tec:\n      vertical_tec_tecu: 10.0\n"
    )

    with pytest.raises(ValidationError) as caught:
        RadioSimConfig.model_validate(data)

    assert "jones" in str(caught.value)


def test_an_unknown_key_inside_a_term_is_rejected(tmp_path) -> None:
    """The same strictness one level down."""
    data = _document(tmp_path, "jones:\n  G:\n    amplitude_erorr: 0.02\n")

    with pytest.raises(ValidationError):
        RadioSimConfig.model_validate(data)


def test_the_known_field_table_covers_the_new_section(tmp_path) -> None:
    """The unknown-field reporter must know the section it is reporting on."""
    from radiosim.io.config import _KNOWN_FIELDS_BY_PARENT

    assert set(_KNOWN_FIELDS_BY_PARENT["jones"]) == {
        "G",
        "B",
        "Rc",
        "Kd",
        "X",
        "D",
        "P",
    }
    assert set(_KNOWN_FIELDS_BY_PARENT["jones.P"]) == {"enabled"}
    for letter, model in (
        ("G", GainTermConfig),
        ("B", BandpassTermConfig),
        ("Rc", CableReflectionTermConfig),
        ("Kd", DelayTermConfig),
        ("X", CrosshandTermConfig),
        ("D", LeakageTermConfig),
    ):
        assert set(_KNOWN_FIELDS_BY_PARENT[f"jones.{letter}"]) == set(
            model.model_fields
        )


# ---------------------------------------------------------------------------
# Discriminated unions on ``kind``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({"kind": "constant"}, StaticTimeModel),
        ({"kind": "linear_drift", "rate_per_hour": 0.01}, LinearDriftTimeModel),
        (
            {"kind": "sinusoidal", "depth": 0.1, "period_hours": 2.0},
            SinusoidalTimeModel,
        ),
    ],
)
def test_each_time_model_kind_selects_its_own_model(
    block: dict[str, Any],
    expected: type,
) -> None:
    """The union is discriminated by Pydantic, not by a hand-written branch."""
    config = GainTermConfig.model_validate(
        {"amplitude_error": 0.02, "time_model": block}
    )

    assert type(config.time_model) is expected


def test_the_default_time_model_is_the_constant_one() -> None:
    """A ``G`` with no time model does not vary, and says so by its type."""
    config = GainTermConfig(amplitude_error=0.02)

    assert type(config.time_model) is StaticTimeModel


@pytest.mark.parametrize("kind", ["exponential", "random_walk", ""])
def test_an_unknown_time_model_kind_is_rejected(kind: str) -> None:
    """An unmodelled kind fails at parse time and is never silently constant."""
    with pytest.raises(ValidationError):
        GainTermConfig.model_validate(
            {"amplitude_error": 0.02, "time_model": {"kind": kind}}
        )


@pytest.mark.parametrize("kind", ["spline", "rfi_flagged", "chebyshev"])
def test_an_unknown_bandpass_kind_is_rejected(kind: str) -> None:
    """Including ``spline``, which is what the removed subclass was called."""
    with pytest.raises(ValidationError):
        BandpassTermConfig.model_validate({"model": {"kind": kind}})


def test_a_time_model_missing_its_own_parameter_is_rejected() -> None:
    """The discriminator does not excuse the variant from its own fields."""
    with pytest.raises(ValidationError):
        GainTermConfig.model_validate(
            {"amplitude_error": 0.02, "time_model": {"kind": "linear_drift"}}
        )


@pytest.mark.parametrize("period", [0.0, -1.0])
def test_a_non_positive_sinusoid_period_is_rejected(period: float) -> None:
    """A zero period is a division by zero, which the schema can see by itself."""
    with pytest.raises(ValidationError):
        GainTermConfig.model_validate(
            {
                "amplitude_error": 0.02,
                "time_model": {
                    "kind": "sinusoidal",
                    "depth": 0.1,
                    "period_hours": period,
                },
            }
        )


# ---------------------------------------------------------------------------
# Complex values and units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, 1.0 + 0.0j),
        (-0.05, -0.05 + 0.0j),
        ((0.5, 0.25), 0.5 + 0.25j),
        ([0.0, -1.0], -1.0j),
    ],
)
def test_a_complex_configuration_value_converts_once(
    value: Any,
    expected: complex,
) -> None:
    """``[re, im]`` and a bare real both convert, through exactly one function."""
    assert as_complex(value) == expected


def test_complex_coefficients_survive_the_schema() -> None:
    """A polynomial with complex coefficients is accepted and kept complex."""
    config = BandpassTermConfig.model_validate(
        {
            "model": {
                "kind": "polynomial",
                "coefficients": [[1.0, 0.0], [0.0, 0.2]],
            }
        }
    )

    assert [as_complex(value) for value in config.model.coefficients] == [
        1.0 + 0.0j,
        0.2j,
    ]


@pytest.mark.parametrize("literal", ["1+2j", "0.5", None])
def test_a_complex_value_written_as_a_string_is_rejected(literal: Any) -> None:
    """Section 21.3: never a Python complex literal, never a string."""
    with pytest.raises(ValidationError):
        BandpassTermConfig.model_validate(
            {"model": {"kind": "polynomial", "coefficients": [literal]}}
        )


def test_every_unit_bearing_field_carries_its_unit_in_its_name() -> None:
    """Section 21.3: no unit is implicit.

    Checked mechanically over the declared field names rather than by reading
    them, so a field added later without a suffix fails here instead of being
    noticed in review or not at all.
    """
    dimensionless = {
        "amplitude_error",
        "depth",
        "elevation_curve",
        "coefficients",
        "coefficients0",
        "coefficients1",
        "gains",
        "kind",
        "antenna",
        "feed",
        "model",
        "per_antenna",
        "time_model",
        "rate_per_hour",
        "period_hours",
        # Tier 7E.  ``amplitude`` is a dimensionless reflection coefficient;
        # ``d``, ``d0``, ``d1`` and ``d_terms`` are dimensionless leakages;
        # ``ixr_db`` carries its unit in the ``_db`` the field name ends in, and
        # decibels are the one unit the suffix table does not list because no
        # other field uses them.  The term letters are section keys, not values.
        "amplitude",
        "d",
        "d0",
        "d1",
        "d_term",
        "d_terms",
        "ixr_db",
        "G",
        "B",
        "Rc",
        "Kd",
        "X",
        "D",
    }
    suffixes = ("_rad", "_deg", "_hz", "_s", "_m", "_km")
    for model in (
        JonesConfig,
        GainTermConfig,
        BandpassTermConfig,
        BandpassPolynomialModel,
        BandpassTabulatedModel,
        StaticTimeModel,
        LinearDriftTimeModel,
        SinusoidalTimeModel,
        CableReflectionTermConfig,
        DelayTermConfig,
        CrosshandTermConfig,
        LeakageTermConfig,
        ExplicitLeakageModel,
        IXRLeakageModel,
        FrequencyPolynomialLeakageModel,
    ):
        for name in model.model_fields:
            assert name in dimensionless or name.endswith(suffixes), (
                model.__name__,
                name,
            )


# ---------------------------------------------------------------------------
# Structural well-formedness the schema can decide alone
# ---------------------------------------------------------------------------


def test_an_empty_coefficient_list_is_rejected() -> None:
    """A polynomial with no coefficients is not a polynomial."""
    with pytest.raises(ValidationError):
        BandpassTermConfig.model_validate(
            {"model": {"kind": "polynomial", "coefficients": []}}
        )


def test_fewer_than_four_nodes_is_rejected() -> None:
    """A cubic spline is not determined by three points."""
    with pytest.raises(ValidationError) as caught:
        BandpassTermConfig.model_validate(
            {
                "model": {
                    "kind": "tabulated",
                    "node_frequencies_hz": [1.0e8, 1.1e8, 1.2e8],
                    "gains": [1.0, 1.0, 1.0],
                }
            }
        )

    assert "at least 4 node frequencies" in str(caught.value)


def test_unordered_nodes_are_rejected() -> None:
    """Strictly increasing, so the spline and the R11 span are both well defined."""
    with pytest.raises(ValidationError) as caught:
        BandpassTermConfig.model_validate(
            {
                "model": {
                    "kind": "tabulated",
                    "node_frequencies_hz": [1.0e8, 1.2e8, 1.1e8, 1.3e8],
                    "gains": [1.0, 1.0, 1.0, 1.0],
                }
            }
        )

    assert "strictly increasing" in str(caught.value)


def test_a_gain_per_node_is_required() -> None:
    """One gain per node, checked where the two lists are both visible."""
    with pytest.raises(ValidationError) as caught:
        BandpassTermConfig.model_validate(
            {
                "model": {
                    "kind": "tabulated",
                    "node_frequencies_hz": [1.0e8, 1.1e8, 1.2e8, 1.3e8],
                    "gains": [1.0, 1.0],
                }
            }
        )

    assert "exactly one gain per node" in str(caught.value)


def test_a_gain_override_that_sets_nothing_is_rejected() -> None:
    """An override that overrides nothing is a typo, not a configuration."""
    with pytest.raises(ValidationError) as caught:
        GainTermConfig.model_validate(
            {"amplitude_error": 0.02, "per_antenna": [{"antenna": 0, "feed": 0}]}
        )

    assert "at least one of" in str(caught.value)


def test_an_empty_elevation_curve_is_rejected() -> None:
    """``elevation_curve: []`` is neither on nor off; omit the field instead."""
    with pytest.raises(ValidationError) as caught:
        GainTermConfig.model_validate({"amplitude_error": 0.02, "elevation_curve": []})

    assert "at least one coefficient" in str(caught.value)


def test_per_antenna_is_an_immutable_tuple() -> None:
    """Deep immutability, as for every other resolved input section."""
    config = GainTermConfig.model_validate(
        {
            "amplitude_error": 0.02,
            "per_antenna": [{"antenna": 0, "feed": 0, "amplitude_error": 0.1}],
        }
    )

    assert type(config.per_antenna) is tuple
    with pytest.raises(ValidationError):
        config.per_antenna = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tier 7F: the P block
# ---------------------------------------------------------------------------

_PARALLACTIC = """
jones:
  P:
    enabled: true
"""


def test_the_parallactic_block_is_a_bare_enabled_flag(tmp_path) -> None:
    """Section 21.3: ``P`` has no free parameter, so it declares none.

    The parallactic angle is fully determined by the instrument, the time grid
    and the directions.  Inventing a parameter to make ``P`` look like the other
    terms would be dishonest, and the plan says so; the schema is where that
    shows.
    """
    config = RadioSimConfig.model_validate(_document(tmp_path, _PARALLACTIC))

    assert config.jones is not None
    assert config.jones.P == ParallacticTermConfig(enabled=True)
    assert config.jones.configured_terms == ("P",)
    assert set(ParallacticTermConfig.model_fields) == {"enabled"}


def test_the_parallactic_block_requires_the_flag_and_forbids_anything_else(
    tmp_path,
) -> None:
    """Strictness, both ways: no default, and no extra key.

    ``enabled`` has no default because ``jones.P: {}`` would be a block that
    says nothing while looking like a decision.  ``minimum_elevation_deg`` is
    rejected because the accepted YAML's own comment said the directions it
    named were already masked -- a field documented as having no effect is the
    surface this tier exists to remove (Section 21.3, 7F correction).
    """
    with pytest.raises(ValidationError):
        RadioSimConfig.model_validate(_document(tmp_path, "jones:\n  P: {}\n"))

    with pytest.raises(ValidationError):
        RadioSimConfig.model_validate(
            _document(
                tmp_path,
                "jones:\n  P:\n    enabled: true\n    minimum_elevation_deg: 5.0\n",
            )
        )


def test_the_parallactic_block_is_frozen_and_strict() -> None:
    """The same ``StrictFrozenModel`` contract every other term block has."""
    assert ParallacticTermConfig.model_config["extra"] == "forbid"
    assert ParallacticTermConfig.model_config["frozen"] is True

    config = ParallacticTermConfig(enabled=True)
    with pytest.raises(ValidationError):
        config.enabled = False  # type: ignore[misc]


def test_enabled_false_parses_and_is_rejected_at_resolution(tmp_path) -> None:
    """The schema accepts the word; the resolver rejects the meaning.

    Section 21's "there is no ``enabled: false``" is a rule about runs, not
    about parsing, and R7 is the rejection that states it -- with a message that
    tells the user to remove the block.  Putting the refusal here instead would
    replace that sentence with a type error.
    """
    config = RadioSimConfig.model_validate(
        _document(tmp_path, "jones:\n  P:\n    enabled: false\n")
    )

    assert config.jones is not None
    assert config.jones.P is not None
    assert config.jones.P.enabled is False
    assert config.jones.configured_terms == ("P",)
