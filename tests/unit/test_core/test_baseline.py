"""Tier 2F canonical baseline model and generation contract tests."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import radiosim
import radiosim.core as core
import radiosim.core.instrument as instrument_module
from radiosim.core.baseline_resolution import (
    BaselineGenerationError,
    BaselineSelectionError,
    CoincidentAntennaError,
    EmptyBaselineSelectionError,
    generate_resolved_baselines,
)
from radiosim.core.instrument import (
    AntennaFieldSource,
    AntennaId,
    AntennaProvenance,
    BaselineSelectionCriteriaSnapshot,
    BaselineSelectionProvenance,
    ResolvedAntenna,
    ResolvedBaseline,
    ResolvedBaselineSelection,
    ResolvedEarthLocation,
    _create_resolved_instrument,
)
from radiosim.core.instrument_resolution import InstrumentResolutionError

BASELINE_MODEL_FIELDS = {
    ResolvedBaseline: (
        "ant1",
        "ant2",
        "vector_enu_m",
        "length_m",
        "is_autocorrelation",
        "azimuth_deg",
    ),
    BaselineSelectionCriteriaSnapshot: (
        "correlations",
        "length_mode",
        "length_targets_m",
        "length_tolerance_m",
        "length_ranges_m",
        "azimuth_ranges_deg",
    ),
    BaselineSelectionProvenance: (
        "schema_version",
        "instrument_sha256",
        "criteria",
        "generated_count",
        "after_correlation_count",
        "after_length_count",
        "after_azimuth_count",
        "azimuth_exempt_auto_count",
        "selected_ids",
    ),
    ResolvedBaselineSelection: ("baselines", "provenance"),
}


def _antenna(number: int, name: str, position: Any) -> ResolvedAntenna:
    return ResolvedAntenna(
        id=AntennaId(number, name),
        position_enu_m=position,
        diameter_m=12.0,
        mount_type=None,
        beam_id=None,
        provenance=AntennaProvenance(
            identity_source=AntennaFieldSource.LAYOUT_FILE,
            position_source=AntennaFieldSource.LAYOUT_FILE,
            diameter_source=AntennaFieldSource.CONFIG_DEFAULT,
            source_diameter_m=None,
            mount_source=None,
            beam_id_source=None,
            source_record=f"row:{number}",
        ),
    )


def _instrument(*records: tuple[int, str, tuple[float, float, float]]):
    if not records:
        records = ((0, "ANT0", (0.0, 0.0, 0.0)),)
    antennas = tuple(
        _antenna(number, name, position) for number, name, position in records
    )
    location = ResolvedEarthLocation(
        longitude_deg=0.0,
        latitude_deg=0.0,
        height_m=0.0,
        itrs_xyz_m=(1.0, 2.0, 3.0),
        source=AntennaFieldSource.EXPLICIT_CONFIG,
        reference="fixture:location",
    )
    return _create_resolved_instrument(
        name="Fixture Array",
        location=location,
        antennas=antennas,
        source_kind="fixture",
        source_reference="fixture:layout",
        source_format="radiosim",
        registry_policy=None,
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
        source_location_itrs_xyz_m=None,
        location_separation_m=None,
        pyuvdata_version=None,
        source_sha256=None,
    )


def _cross(
    *,
    vector: Any = (3.0, 4.0, 0.0),
    length: Any = 5.0,
    azimuth: Any = math.degrees(math.atan2(3.0, 4.0)),
) -> ResolvedBaseline:
    return ResolvedBaseline(
        ant1=AntennaId(1, "A1"),
        ant2=AntennaId(2, "A2"),
        vector_enu_m=vector,
        length_m=length,
        is_autocorrelation=False,
        azimuth_deg=azimuth,
    )


@pytest.mark.parametrize("model_type,expected", BASELINE_MODEL_FIELDS.items())
def test_baseline_models_have_exact_frozen_slotted_fields(model_type, expected):
    assert is_dataclass(model_type)
    assert tuple(item.name for item in fields(model_type)) == expected
    assert "__slots__" in model_type.__dict__
    assert "__dict__" not in model_type.__dict__


def test_resolved_baseline_owns_builtin_scalars_and_canonicalizes_negative_zero():
    baseline = ResolvedBaseline(
        ant1=AntennaId(np.int64(1), "A1"),
        ant2=AntennaId(np.int64(2), "A2"),
        vector_enu_m=np.array([-0.0, np.float64(5.0), -0.0]),
        length_m=np.float64(5.0),
        is_autocorrelation=False,
        azimuth_deg=np.float64(0.0),
    )

    assert baseline.vector_enu_m == (0.0, 5.0, 0.0)
    assert all(type(value) is float for value in baseline.vector_enu_m)
    assert all(
        math.copysign(1.0, value) == 1.0
        for value in (baseline.vector_enu_m[0], baseline.vector_enu_m[2])
    )
    assert type(baseline.length_m) is float
    assert type(baseline.azimuth_deg) is float
    assert isinstance(hash(baseline), int)
    with pytest.raises(FrozenInstanceError):
        baseline.length_m = 4.0


def test_resolved_baseline_copies_mutable_vector_input():
    vector = [3.0, 4.0, 0.0]
    baseline = _cross(vector=vector)
    vector[0] = 99.0

    assert baseline.vector_enu_m == (3.0, 4.0, 0.0)
    assert type(baseline.vector_enu_m) is tuple


def test_resolved_baseline_rejects_mutable_canonical_subclasses():
    def mutable_setattr(self, name, value):
        object.__setattr__(self, name, value)

    mutable_id_type = type(
        "MutableAntennaId",
        (AntennaId,),
        {"__setattr__": mutable_setattr},
    )
    mutable_id = mutable_id_type(1, "A1")
    mutable_id.name = "changed"

    with pytest.raises(TypeError):
        replace(_cross(), ant1=mutable_id)
    with pytest.raises(TypeError):
        replace(_cross(), ant2=object())


@pytest.mark.parametrize(
    "updates",
    [
        {"ant1": AntennaId(3, "A3")},
        {"ant2": AntennaId(1, "A1")},
        {"ant2": AntennaId(1, "different-name")},
        {"vector_enu_m": (3.0, 4.0)},
        {"vector_enu_m": (3.0, math.nan, 0.0)},
        {"length_m": -1.0},
        {"length_m": math.inf},
        {"length_m": 4.0},
        {"is_autocorrelation": True},
        {"is_autocorrelation": 0},
        {"azimuth_deg": None},
        {"azimuth_deg": 180.0},
        {"azimuth_deg": 90.0},
    ],
)
def test_resolved_cross_rejects_inconsistent_direct_states(updates):
    values = {
        "ant1": AntennaId(1, "A1"),
        "ant2": AntennaId(2, "A2"),
        "vector_enu_m": (3.0, 4.0, 0.0),
        "length_m": 5.0,
        "is_autocorrelation": False,
        "azimuth_deg": math.degrees(math.atan2(3.0, 4.0)),
    }
    values.update(updates)

    with pytest.raises((TypeError, ValueError)):
        ResolvedBaseline(**values)


@pytest.mark.parametrize(
    "updates",
    [
        {"ant2": AntennaId(1, "different-name")},
        {"vector_enu_m": (0.0, 0.0, 1.0)},
        {"length_m": 1.0},
        {"is_autocorrelation": False},
        {"azimuth_deg": 0.0},
    ],
)
def test_resolved_auto_requires_exact_zero_and_complete_same_identity(updates):
    antenna_id = AntennaId(1, "A1")
    values = {
        "ant1": antenna_id,
        "ant2": antenna_id,
        "vector_enu_m": (0.0, 0.0, 0.0),
        "length_m": 0.0,
        "is_autocorrelation": True,
        "azimuth_deg": None,
    }
    values.update(updates)

    with pytest.raises((TypeError, ValueError)):
        ResolvedBaseline(**values)


def test_cross_separation_must_be_strictly_above_coincidence_threshold():
    with pytest.raises(ValueError):
        ResolvedBaseline(
            AntennaId(1, "A1"),
            AntennaId(2, "A2"),
            (1e-9, 0.0, 0.0),
            1e-9,
            False,
            90.0,
        )


def test_one_antenna_generates_one_exact_auto():
    instrument = _instrument((7, "SEVEN", (4.0, -2.0, 3.0)))

    baselines = generate_resolved_baselines(instrument)

    assert len(baselines) == 1
    baseline = baselines[0]
    assert baseline.ant1 is instrument.antennas[0].id
    assert baseline.ant2 is instrument.antennas[0].id
    assert baseline.vector_enu_m == (0.0, 0.0, 0.0)
    assert baseline.length_m == 0.0
    assert baseline.is_autocorrelation is True
    assert baseline.azimuth_deg is None


@pytest.mark.parametrize("count", [2, 3])
def test_generation_count_formulas_and_exact_numeric_pair_order(count):
    records = tuple(
        (number, f"A{number}", (float(number), float(number * 2), 0.0))
        for number in reversed(range(count))
    )
    instrument = _instrument(*records)

    baselines = generate_resolved_baselines(instrument)
    pairs = tuple((item.ant1.number, item.ant2.number) for item in baselines)

    assert len(baselines) == count * (count + 1) // 2
    assert sum(item.is_autocorrelation for item in baselines) == count
    assert (
        sum(not item.is_autocorrelation for item in baselines)
        == count * (count - 1) // 2
    )
    assert pairs == tuple(sorted(pairs))
    assert all(first <= second for first, second in pairs)


def test_generation_uses_exact_signed_ant2_minus_ant1_enu_vector_and_3d_norm():
    instrument = _instrument(
        (7, "A7", (4.0, 2.0, -1.0)),
        (2, "A2", (1.0, -2.0, 3.0)),
    )

    baseline = generate_resolved_baselines(instrument)[1]

    assert (baseline.ant1.number, baseline.ant2.number) == (2, 7)
    assert baseline.vector_enu_m == (3.0, 4.0, -4.0)
    assert baseline.length_m == math.hypot(3.0, 4.0, -4.0)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ((0.0, 10.0, 0.0), 0.0),
        ((10.0, 0.0, 0.0), 90.0),
        ((0.0, -10.0, 0.0), 0.0),
        ((-10.0, 0.0, 0.0), 90.0),
        ((0.0, 0.0, 10.0), 0.0),
    ],
)
def test_generation_uses_north_zero_east_ninety_axial_convention(position, expected):
    baseline = generate_resolved_baselines(
        _instrument((0, "A0", (0.0, 0.0, 0.0)), (1, "A1", position))
    )[1]

    assert baseline.azimuth_deg == expected


def test_axial_azimuth_is_invariant_to_physical_direction_and_number_assignment():
    forward = _instrument(
        (1, "LOW", (0.0, 0.0, 0.0)),
        (9, "HIGH", (3.0, 4.0, 0.0)),
    )
    reverse = _instrument(
        (1, "LOW", (3.0, 4.0, 0.0)),
        (9, "HIGH", (0.0, 0.0, 0.0)),
    )

    forward_cross = generate_resolved_baselines(forward)[1]
    reverse_cross = generate_resolved_baselines(reverse)[1]

    assert forward_cross.vector_enu_m == tuple(
        -value for value in reverse_cross.vector_enu_m
    )
    assert forward_cross.azimuth_deg == pytest.approx(reverse_cross.azimuth_deg)


@pytest.mark.parametrize("separation", [0.0, 0.5e-9, 1e-9])
def test_generation_rejects_distinct_antennas_at_or_below_threshold(separation):
    instrument = _instrument(
        (2, "A2", (0.0, 0.0, 0.0)),
        (8, "A8", (separation, 0.0, 0.0)),
    )

    with pytest.raises(CoincidentAntennaError) as exc_info:
        generate_resolved_baselines(instrument)

    message = str(exc_info.value)
    assert "2/'A2'" in message
    assert "8/'A8'" in message
    assert "1e-09 m" in message


def test_generation_accepts_distinct_antennas_above_threshold():
    baseline = generate_resolved_baselines(
        _instrument(
            (2, "A2", (0.0, 0.0, 0.0)),
            (8, "A8", (1.000001e-9, 0.0, 0.0)),
        )
    )[1]

    assert baseline.length_m == pytest.approx(1.000001e-9)


def test_generation_rejects_subtraction_overflow_with_stable_pair_reference():
    instrument = _instrument(
        (2, "A2", (-1e308, 0.0, 0.0)),
        (8, "A8", (1e308, 0.0, 0.0)),
    )

    with pytest.raises(BaselineGenerationError, match=r"2, 8"):
        generate_resolved_baselines(instrument)


@pytest.mark.parametrize("invalid", [object(), {"antennas": []}])
def test_generation_requires_exact_canonical_instrument(invalid):
    with pytest.raises(TypeError):
        generate_resolved_baselines(invalid)


def test_generation_is_repeatable_hashable_and_does_not_mutate_input():
    instrument = _instrument(
        (0, "A0", (0.0, 0.0, 0.0)),
        (3, "A3", (3.0, 4.0, 0.0)),
        (9, "A9", (0.0, 0.0, 12.0)),
    )
    before = instrument.to_snapshot()

    first = generate_resolved_baselines(instrument)
    second = generate_resolved_baselines(instrument)

    assert first == second
    assert first is not second
    assert hash(first) == hash(second)
    assert instrument.to_snapshot() == before


def test_error_hierarchy_is_exact():
    assert BaselineGenerationError.__bases__ == (InstrumentResolutionError,)
    assert CoincidentAntennaError.__bases__ == (BaselineGenerationError,)
    assert BaselineSelectionError.__bases__ == (InstrumentResolutionError,)
    assert EmptyBaselineSelectionError.__bases__ == (BaselineSelectionError,)


def test_public_export_identity_is_narrow_and_legacy_binding_is_removed():
    model_names = tuple(model.__name__ for model in BASELINE_MODEL_FIELDS)
    for name in model_names:
        assert name in instrument_module.__all__
        assert name in core.__all__
        assert getattr(core, name) is getattr(instrument_module, name)
        assert name not in radiosim.__all__
        assert not hasattr(radiosim, name)

    assert not hasattr(core, "generate_baselines")
    assert "generate_baselines" not in core.__all__
    assert not hasattr(radiosim, "generate_baselines")
    assert "generate_baselines" not in radiosim.__all__
    assert "generate_resolved_baselines" not in core.__all__


def test_canonical_baseline_models_contain_no_legacy_opaque_fields():
    names = {item.name for item in fields(ResolvedBaseline)}
    assert names.isdisjoint({"D1D2", "BT1BT2", "A1A2", "Length", "BaselineVector"})

    baseline = _cross()
    assert not any(isinstance(value, np.ndarray) for value in baseline.vector_enu_m)
    assert not any("_" in antenna.name for antenna in (baseline.ant1, baseline.ant2))


def test_instrument_model_module_remains_standard_library_only():
    source = Path(instrument_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "enum",
        "hashlib",
        "json",
        "math",
        "numbers",
        "re",
        "types",
        "typing",
        "unicodedata",
    }
    assert "numpy" not in source
    assert "pydantic" not in source.lower()
    assert "astropy" not in source.lower()


def test_baseline_value_is_json_serializable_without_nonfinite_values():
    baseline = _cross()
    payload = {
        "ant1": {"number": baseline.ant1.number, "name": baseline.ant1.name},
        "ant2": {"number": baseline.ant2.number, "name": baseline.ant2.name},
        "vector_enu_m": list(baseline.vector_enu_m),
        "length_m": baseline.length_m,
        "is_autocorrelation": baseline.is_autocorrelation,
        "azimuth_deg": baseline.azimuth_deg,
    }

    json.dumps(payload, allow_nan=False)
