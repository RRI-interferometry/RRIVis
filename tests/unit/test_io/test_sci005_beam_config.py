"""Strict parse and typed rejection for the SCI-005 Stage-1 beam surface.

``docs/development/sci005_beam_physics_plan.md`` Section 3 defines one new
array-wide ``beams.aperture_physics`` block (normalization literal, central
blockage, support struts, real unit-RMS Zernike ``(n, m)`` modes) and one new
nested ``beams.surface_error.*.error_beam_diagnostic`` declaration. This module
is the red slice for that configuration surface.

**Error taxonomy.** Section 2 is normative and Section 3.5's first mechanic
states the Stage-1 consequence directly: "authored-kind failures surface as
``ConfigSchemaError`` carrying Pydantic's own issue codes, while value-domain,
identity, duplicate, and document-level cross-field failures surface as
``ConfigSemanticError`` carrying the frozen ``beam.aperture_physics.*`` and
``beam.ruze_power_diagnostic.*`` codes; the instrument-stage support-leg
geometry rejection uses the typed error named in Section 3.2 and carries no
``ConfigIssue`` code." Concretely:

* whether a value can be *read* as the declared kind of thing -- unknown field,
  missing key, wrong literal, wrong container shape, a boolean where an exact
  integer is required, a ``bool`` or ``int`` where an exact finite float is
  required, a non-finite float -- is a schema failure carrying Pydantic's own
  stable code;
* whether an authored value is *inside its physical domain*, whether two
  authored records collide, and whether an explicitly present block resolves to
  a real effect are semantic failures (Section 2: "Every explicitly present
  block must resolve to a real effect. Exact identity blocks are rejected, not
  accepted and discarded."; Section 3.5: "a blockage ratio, support width, or
  resolved support geometry is outside its physical domain");
* an excluded pupil profile, an excluded beam family, and an extended-precision
  diagnostic are unsupported-feature failures; and
* exactly one Stage-1 rejection is *not* a configuration issue at all -- the
  leg-wider-than-resolved-diameter geometry check, which Section 3.2 assigns to
  beam-assignment resolution because per-antenna diameters exist only after
  instrument resolution. Its section is at the end of this module.

Section 3 freezes five issue codes and three exact messages; those literals are
reproduced verbatim below. Section 3.5 freezes the two *namespaces* the
remaining semantic codes live in; the suffixes in :data:`SEMANTIC_CODES` are the
stable identifiers this red slice binds for the rejection families Section 3.5
enumerates but does not individually name.

Every rejection asserts the concrete exception type together with the issue
path and code, never a message substring alone (Section 2).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from radiosim.io.config_resolution import (
    ConfigResolutionError,
    ConfigSchemaError,
    ConfigSemanticError,
    ConfigurationSource,
    UnsupportedConfigError,
    resolve_config,
)
from tests.fixtures.configs import valid_config_mapping

# --- Section 3 frozen literals ------------------------------------------------

APERTURE_NORMALIZATION = "unmodified_ideal_aperture_v1"
ZERNIKE_CONVENTION = "radiosim.real_unit_rms_disk_surface_height.v1"
DIAGNOSTIC_KIND = "gaussian_covariance_power"

APERTURE_PATH = "beams.aperture_physics"
DEFAULT_DIAGNOSTIC_PATH = "beams.surface_error.default.error_beam_diagnostic"

#: Section 3.1's five frozen issue codes.
UNSUPPORTED_PUPIL_APERTURE = "beam.aperture_physics.unsupported_pupil_profile"
UNSUPPORTED_FAMILY_APERTURE = "beam.aperture_physics.unsupported_beam_family"
UNSUPPORTED_PUPIL_DIAGNOSTIC = "beam.ruze_power_diagnostic.unsupported_pupil_profile"
UNSUPPORTED_FAMILY_DIAGNOSTIC = "beam.ruze_power_diagnostic.unsupported_beam_family"
UNSUPPORTED_PRECISION_DIAGNOSTIC = "beam.ruze_power_diagnostic.unsupported_precision"

#: Section 3.1's exact unsupported-precision message.
UNSUPPORTED_PRECISION_MESSAGE = (
    "Ruze power diagnostics support only float32/complex64 and "
    "float64/complex128 beam precision."
)

#: The stable semantic codes this red slice binds for Section 3.5's families.
SEMANTIC_CODES: dict[str, str] = {
    "identity_block": "beam.aperture_physics.identity_block",
    "zernike_identity": "beam.aperture_physics.zernike_identity",
    "zernike_mode_domain": "beam.aperture_physics.zernike_mode_domain",
    "zernike_mode_reserved": "beam.aperture_physics.zernike_mode_reserved",
    "zernike_mode_duplicate": "beam.aperture_physics.zernike_mode_duplicate",
    "blockage_ratio_domain": "beam.aperture_physics.blockage_ratio_domain",
    "support_leg_width_domain": "beam.aperture_physics.support_leg_width_domain",
    "support_leg_angle_domain": "beam.aperture_physics.support_leg_angle_domain",
    "support_leg_duplicate": "beam.aperture_physics.support_leg_duplicate",
    "correlation_length_domain": (
        "beam.ruze_power_diagnostic.correlation_length_domain"
    ),
    "missing_surface_error": "beam.ruze_power_diagnostic.missing_surface_error",
}


def _unsupported_profile_message(
    feature: str,
    model_kind: str,
    taper_kind: str | None,
) -> str:
    """Section 3.1's exact unsupported-profile message template."""
    return (
        f"Stage-1 {feature} requires a canonical circular pupil; resolved model "
        f"{model_kind!r} with taper {taper_kind!r} has no supported v1 profile."
    )


def _unsupported_family_message(feature: str, model_kind: str) -> str:
    """Section 3.1's exact unsupported-family message template."""
    return f"Stage-1 {feature} does not support resolved beam family {model_kind!r}."


APERTURE_FEATURE = "aperture physics"
DIAGNOSTIC_FEATURE = "Ruze power diagnostic"


# --- configuration builders ---------------------------------------------------

UNIFORM_CIRCULAR: dict[str, Any] = {
    "kind": "circular_aperture",
    "taper": {"kind": "uniform"},
}
PARABOLIC_CIRCULAR: dict[str, Any] = {
    "kind": "circular_aperture",
    "taper": {"kind": "parabolic", "edge_taper_db": 12.0},
}


def _blockage(
    ratio: float = 0.15,
    legs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "central_diameter_ratio": ratio,
        "support_legs": [] if legs is None else legs,
    }


def _zernike(modes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "convention": ZERNIKE_CONVENTION,
        "modes": (
            [{"n": 2, "m": 0, "surface_height_coefficient_m": 0.0005}]
            if modes is None
            else modes
        ),
    }


def _aperture_physics(**children: Any) -> dict[str, Any]:
    block: dict[str, Any] = {"normalization": APERTURE_NORMALIZATION}
    block.update(children)
    return block


def _diagnostic(correlation_length_m: float = 0.25) -> dict[str, Any]:
    return {"kind": DIAGNOSTIC_KIND, "correlation_length_m": correlation_length_m}


def _analytic_beams(
    model: dict[str, Any] | None = None,
    *,
    aperture_physics: dict[str, Any] | None = None,
    surface_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    beams: dict[str, Any] = {
        "mode": "analytic",
        "model": deepcopy(UNIFORM_CIRCULAR if model is None else model),
    }
    if aperture_physics is not None:
        beams["aperture_physics"] = deepcopy(aperture_physics)
    if surface_error is not None:
        beams["surface_error"] = deepcopy(surface_error)
    return beams


def _resolve(
    tmp_path: Path,
    beams: dict[str, Any],
    *,
    beam_precision: str | None = None,
) -> Any:
    data = valid_config_mapping(tmp_path, beams=beams)
    if beam_precision is not None:
        # Replace rather than merge: an authored preset beside explicit leaves is
        # itself a semantic contradiction and would mask the case under test.
        data["execution"]["precision"] = {"jones": {"beam": beam_precision}}
    return resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )


def _rows(error: Any) -> list[tuple[str, str]]:
    return [(issue.path, issue.code) for issue in error.issues]


# --- accepted strict parse ----------------------------------------------------


def test_aperture_physics_resolves_blockage_support_legs_and_zernike_modes(
    tmp_path: Path,
) -> None:
    """Section 3.1--3.3: one parent, both children, and a stable ``(n, m)`` sort."""
    bundle = _resolve(
        tmp_path,
        _analytic_beams(
            aperture_physics=_aperture_physics(
                blockage=_blockage(
                    0.15,
                    [
                        {"position_angle_deg": 90.0, "width_m": 0.25},
                        {"position_angle_deg": 0.0, "width_m": 0.3},
                    ],
                ),
                zernike_surface=_zernike(
                    [
                        {"n": 4, "m": 0, "surface_height_coefficient_m": -0.0002},
                        {"n": 2, "m": -2, "surface_height_coefficient_m": 0.0},
                        {"n": 2, "m": 0, "surface_height_coefficient_m": 0.0005},
                    ]
                ),
            ),
        ),
    )

    aperture = bundle.runtime.beams.aperture_physics
    assert aperture.normalization == APERTURE_NORMALIZATION
    assert aperture.blockage.central_diameter_ratio == 0.15
    assert type(aperture.blockage.support_legs) is tuple
    assert [
        (leg.position_angle_deg, leg.width_m) for leg in aperture.blockage.support_legs
    ] == [(90.0, 0.25), (0.0, 0.3)]
    assert aperture.zernike_surface.convention == ZERNIKE_CONVENTION
    assert type(aperture.zernike_surface.modes) is tuple
    # Section 3.3: "Resolution sorts modes by ``(n, m)`` for a stable fingerprint
    # without changing the exact mathematical sum."  A zero coefficient beside a
    # non-zero sibling is explicitly permitted.
    assert [
        (mode.n, mode.m, mode.surface_height_coefficient_m)
        for mode in aperture.zernike_surface.modes
    ] == [(2, -2, 0.0), (2, 0, 0.0005), (4, 0, -0.0002)]


@pytest.mark.parametrize(
    "children",
    [
        {"blockage": _blockage(0.2)},
        {"blockage": _blockage(0.2, [{"position_angle_deg": 180.0, "width_m": 0.4}])},
        {"zernike_surface": _zernike()},
    ],
    ids=["blockage_only", "blockage_with_leg", "zernike_only"],
)
def test_one_effective_child_is_enough_for_the_parent_block(
    tmp_path: Path,
    children: dict[str, Any],
) -> None:
    """Section 3.1: "at least one effective ``blockage`` or ``zernike_surface``"."""
    bundle = _resolve(
        tmp_path,
        _analytic_beams(aperture_physics=_aperture_physics(**children)),
    )

    aperture = bundle.runtime.beams.aperture_physics
    assert aperture.normalization == APERTURE_NORMALIZATION
    assert (aperture.blockage is None) == ("blockage" not in children)
    assert (aperture.zernike_surface is None) == ("zernike_surface" not in children)


def test_ruze_power_diagnostic_declaration_resolves_on_a_supported_pupil(
    tmp_path: Path,
) -> None:
    """Section 3.4: the nested diagnostic keeps the coherent term's meaning."""
    bundle = _resolve(
        tmp_path,
        _analytic_beams(
            surface_error={
                "default": {
                    "rms_surface_error_m": 0.001,
                    "error_beam_diagnostic": _diagnostic(),
                },
                "per_antenna": [
                    {
                        "antenna": 1,
                        "rms_surface_error_m": 0.002,
                        "error_beam_diagnostic": _diagnostic(0.5),
                    }
                ],
            },
        ),
    )

    surface_error = bundle.runtime.beams.surface_error
    assert surface_error.default.rms_surface_error_m == 0.001
    assert surface_error.default.error_beam_diagnostic.kind == DIAGNOSTIC_KIND
    assert surface_error.default.error_beam_diagnostic.correlation_length_m == 0.25
    assert (
        surface_error.per_antenna[0].error_beam_diagnostic.correlation_length_m == 0.5
    )


def test_absent_stage1_blocks_leave_the_resolved_beam_untouched(
    tmp_path: Path,
) -> None:
    """Section 2: no absent block changes the resolved configuration."""
    bundle = _resolve(tmp_path, {"mode": "analytic"})

    beams = bundle.runtime.beams
    assert beams.aperture_physics is None
    assert beams.surface_error is None


@pytest.mark.parametrize(
    "model",
    [
        {"kind": "circular_aperture", "taper": {"kind": "gaussian"}},
        {"kind": "circular_aperture", "taper": {"kind": "cosine"}},
        {"kind": "rectangular_aperture", "north_length_m": 12.0, "east_length_m": 10.0},
        {
            "kind": "elliptical_aperture",
            "north_diameter_m": 14.0,
            "east_diameter_m": 12.0,
        },
        {
            "kind": "analytical_illumination",
            "illumination": {"kind": "corrugated_horn"},
            "taper_profile": {"kind": "gaussian"},
        },
        {
            "kind": "numerical_illumination",
            "illumination": {"kind": "open_waveguide"},
        },
    ],
    ids=[
        "circular_gaussian",
        "circular_cosine",
        "rectangular",
        "elliptical",
        "analytical_gaussian",
        "numerical",
    ],
)
def test_excluded_pupils_are_untouched_when_both_stage1_features_are_absent(
    tmp_path: Path,
    model: dict[str, Any],
) -> None:
    """Section 3.1: "no existing beam with both absent is re-resolved or changed"."""
    bundle = _resolve(tmp_path, _analytic_beams(model))

    assert bundle.runtime.beams.model.model.kind == model["kind"]
    assert bundle.runtime.beams.aperture_physics is None


# --- schema rejections --------------------------------------------------------

_LEGS = "beams.aperture_physics.blockage.support_legs"
_MODES = "beams.aperture_physics.zernike_surface.modes"


@pytest.mark.parametrize(
    ("beams", "path", "code"),
    [
        pytest.param(
            _analytic_beams(aperture_physics=_aperture_physics(enabled=True)),
            "beams.aperture_physics.enabled",
            "extra_forbidden",
            id="unknown_parent_field",
        ),
        pytest.param(
            _analytic_beams(aperture_physics={"blockage": _blockage()}),
            "beams.aperture_physics.normalization",
            "missing",
            id="missing_normalization",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics={
                    "normalization": "peak_normalized_v1",
                    "blockage": _blockage(),
                }
            ),
            "beams.aperture_physics.normalization",
            "literal_error",
            id="wrong_normalization_literal",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    blockage={
                        "central_diameter_ratio": 0.15,
                        "support_legs": [],
                        "n_support_legs": 4,
                    }
                )
            ),
            "beams.aperture_physics.blockage.n_support_legs",
            "extra_forbidden",
            id="unknown_blockage_field",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(blockage=_blockage("0.15"))
            ),
            "beams.aperture_physics.blockage.central_diameter_ratio",
            "float_type",
            id="blockage_ratio_text",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(blockage=_blockage(float("nan")))
            ),
            "beams.aperture_physics.blockage.central_diameter_ratio",
            "finite_number",
            id="blockage_ratio_not_finite",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    blockage=_blockage(
                        0.15, [{"position_angle_deg": 0.0, "width_m": "0.3"}]
                    )
                )
            ),
            f"{_LEGS}[0].width_m",
            "float_type",
            id="leg_width_text",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    blockage=_blockage(0.15, [{"position_angle_deg": 0.0}])
                )
            ),
            f"{_LEGS}[0].width_m",
            "missing",
            id="leg_width_missing",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    blockage=_blockage(
                        0.15,
                        [
                            {
                                "position_angle_deg": 0.0,
                                "width_m": 0.3,
                                "n_legs": 4,
                            }
                        ],
                    )
                )
            ),
            f"{_LEGS}[0].n_legs",
            "extra_forbidden",
            id="unknown_leg_field",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface={"convention": ZERNIKE_CONVENTION}
                )
            ),
            _MODES,
            "missing",
            id="modes_absent",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(zernike_surface=_zernike([]))
            ),
            _MODES,
            "too_short",
            id="modes_empty",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface={
                        "convention": ZERNIKE_CONVENTION,
                        "modes": {"2,0": 0.0005},
                    }
                )
            ),
            _MODES,
            "tuple_type",
            id="modes_mapping_shorthand",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [{"mode": [2, 0], "surface_height_coefficient_m": 0.0005}]
                    )
                )
            ),
            f"{_MODES}[0].mode",
            "extra_forbidden",
            id="pair_valued_mode_shorthand",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [
                            {
                                "n": 2,
                                "m": 0,
                                "surface_height_coefficient_m": 0.0005,
                                "noll": 4,
                            }
                        ]
                    )
                )
            ),
            f"{_MODES}[0].noll",
            "extra_forbidden",
            id="noll_index_is_not_accepted",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [{"n": 2, "surface_height_coefficient_m": 0.0005}]
                    )
                )
            ),
            f"{_MODES}[0].m",
            "missing",
            id="mode_missing_m",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [{"n": True, "m": 0, "surface_height_coefficient_m": 0.0005}]
                    )
                )
            ),
            f"{_MODES}[0].n",
            "int_type",
            id="mode_index_boolean",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [
                            {
                                "n": 2,
                                "m": 0,
                                "surface_height_coefficient_m": float("inf"),
                            }
                        ]
                    )
                )
            ),
            f"{_MODES}[0].surface_height_coefficient_m",
            "finite_number",
            id="mode_coefficient_not_finite",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface={
                        "convention": "radiosim.real_unit_rms_disk_surface_height.v2",
                        "modes": [
                            {"n": 2, "m": 0, "surface_height_coefficient_m": 0.0005}
                        ],
                    }
                )
            ),
            "beams.aperture_physics.zernike_surface.convention",
            "literal_error",
            id="wrong_zernike_convention",
        ),
        pytest.param(
            _analytic_beams(
                surface_error={
                    "default": {
                        "rms_surface_error_m": 0.001,
                        "error_beam_diagnostic": {
                            "kind": DIAGNOSTIC_KIND,
                            "correlation_length_m": 0.25,
                            "panel_size_m": 1.0,
                        },
                    }
                }
            ),
            f"{DEFAULT_DIAGNOSTIC_PATH}.panel_size_m",
            "extra_forbidden",
            id="unknown_diagnostic_field",
        ),
        pytest.param(
            _analytic_beams(
                surface_error={
                    "default": {
                        "rms_surface_error_m": 0.001,
                        "error_beam_diagnostic": {
                            "kind": "gaussian_error_beam",
                            "correlation_length_m": 0.25,
                        },
                    }
                }
            ),
            f"{DEFAULT_DIAGNOSTIC_PATH}.kind",
            "literal_error",
            id="wrong_diagnostic_kind",
        ),
        pytest.param(
            _analytic_beams(
                surface_error={
                    "default": {
                        "rms_surface_error_m": 0.001,
                        "error_beam_diagnostic": {"kind": DIAGNOSTIC_KIND},
                    }
                }
            ),
            f"{DEFAULT_DIAGNOSTIC_PATH}.correlation_length_m",
            "missing",
            id="missing_correlation_length",
        ),
    ],
)
def test_stage1_input_shape_type_and_unknown_field_failures_are_schema_errors(
    tmp_path: Path,
    beams: dict[str, Any],
    path: str,
    code: str,
) -> None:
    """Section 2: strict fields, no booleans as integers, no integers as floats."""
    with pytest.raises(ConfigSchemaError) as error:
        _resolve(tmp_path, beams)

    assert (path, code) in _rows(error.value)


def _float_field_document(field: str, value: Any) -> dict[str, Any]:
    """Build the smallest document that puts ``value`` in one Stage-1 float."""
    if field == "central_diameter_ratio":
        return _analytic_beams(
            aperture_physics=_aperture_physics(blockage=_blockage(value))
        )
    if field == "position_angle_deg":
        return _analytic_beams(
            aperture_physics=_aperture_physics(
                blockage=_blockage(
                    0.15, [{"position_angle_deg": value, "width_m": 0.3}]
                )
            )
        )
    if field == "width_m":
        return _analytic_beams(
            aperture_physics=_aperture_physics(
                blockage=_blockage(
                    0.15, [{"position_angle_deg": 0.0, "width_m": value}]
                )
            )
        )
    if field == "surface_height_coefficient_m":
        return _analytic_beams(
            aperture_physics=_aperture_physics(
                zernike_surface=_zernike(
                    [{"n": 2, "m": 0, "surface_height_coefficient_m": value}]
                )
            )
        )
    if field == "correlation_length_m":
        return _analytic_beams(
            surface_error={
                "default": {
                    "rms_surface_error_m": 0.001,
                    "error_beam_diagnostic": {
                        "kind": DIAGNOSTIC_KIND,
                        "correlation_length_m": value,
                    },
                }
            }
        )
    raise AssertionError(f"unknown Stage-1 float field {field!r}")


#: Every authored Stage-1 float, with the path its issue must carry.
STAGE1_FLOAT_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "central_diameter_ratio",
        "beams.aperture_physics.blockage.central_diameter_ratio",
    ),
    ("position_angle_deg", f"{_LEGS}[0].position_angle_deg"),
    ("width_m", f"{_LEGS}[0].width_m"),
    ("surface_height_coefficient_m", f"{_MODES}[0].surface_height_coefficient_m"),
    ("correlation_length_m", f"{DEFAULT_DIAGNOSTIC_PATH}.correlation_length_m"),
)


@pytest.mark.parametrize(("field", "path"), STAGE1_FLOAT_FIELDS)
@pytest.mark.parametrize(
    "value", [True, False, 1, 0], ids=["true", "false", "one", "zero"]
)
def test_every_stage1_float_field_rejects_bool_and_int_uniformly(
    tmp_path: Path,
    field: str,
    path: str,
    value: bool | int,
) -> None:
    """Section 3.5's second mechanic, in its corrected wording.

    "Section 2's rule that integers are not silently accepted as strict floats
    is not delivered by strict Pydantic floats, which silently coerce Python
    ints (bools, by contrast, are already rejected by strict float validation);
    every Stage-1 float field therefore rejects ``bool`` and ``int`` inputs
    explicitly and uniformly, reporting Pydantic's own ``float_type`` issue code
    through ``ConfigSchemaError``."

    Uniformity is the point: a field that happened to reject ``True`` only
    because strict validation already does, while quietly coercing ``0``, would
    satisfy neither the rule nor this test.
    """
    with pytest.raises(ConfigSchemaError) as error:
        _resolve(tmp_path, _float_field_document(field, value))

    assert (path, "float_type") in _rows(error.value)


# --- semantic rejections ------------------------------------------------------


@pytest.mark.parametrize(
    ("beams", "path", "code"),
    [
        pytest.param(
            _analytic_beams(aperture_physics=_aperture_physics()),
            APERTURE_PATH,
            SEMANTIC_CODES["identity_block"],
            id="parent_without_an_effective_child",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    blockage=_blockage(0.2),
                    zernike_surface=_zernike(
                        [
                            {"n": 2, "m": 0, "surface_height_coefficient_m": 0.0},
                            {"n": 4, "m": 0, "surface_height_coefficient_m": -0.0},
                        ]
                    ),
                )
            ),
            "beams.aperture_physics.zernike_surface",
            SEMANTIC_CODES["zernike_identity"],
            id="all_zero_zernike_coefficients",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(blockage=_blockage(0.0))
            ),
            "beams.aperture_physics.blockage.central_diameter_ratio",
            SEMANTIC_CODES["blockage_ratio_domain"],
            id="blockage_ratio_zero",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(blockage=_blockage(1.0))
            ),
            "beams.aperture_physics.blockage.central_diameter_ratio",
            SEMANTIC_CODES["blockage_ratio_domain"],
            id="blockage_ratio_one",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(blockage=_blockage(1.5))
            ),
            "beams.aperture_physics.blockage.central_diameter_ratio",
            SEMANTIC_CODES["blockage_ratio_domain"],
            id="blockage_ratio_above_one",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    blockage=_blockage(
                        0.15, [{"position_angle_deg": 0.0, "width_m": 0.0}]
                    )
                )
            ),
            f"{_LEGS}[0].width_m",
            SEMANTIC_CODES["support_leg_width_domain"],
            id="support_leg_width_zero",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    blockage=_blockage(
                        0.15, [{"position_angle_deg": 0.0, "width_m": -0.3}]
                    )
                )
            ),
            f"{_LEGS}[0].width_m",
            SEMANTIC_CODES["support_leg_width_domain"],
            id="support_leg_width_negative",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    blockage=_blockage(
                        0.15, [{"position_angle_deg": -180.0, "width_m": 0.3}]
                    )
                )
            ),
            f"{_LEGS}[0].position_angle_deg",
            SEMANTIC_CODES["support_leg_angle_domain"],
            id="support_leg_angle_at_the_open_end",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    blockage=_blockage(
                        0.15, [{"position_angle_deg": 180.5, "width_m": 0.3}]
                    )
                )
            ),
            f"{_LEGS}[0].position_angle_deg",
            SEMANTIC_CODES["support_leg_angle_domain"],
            id="support_leg_angle_above_the_interval",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    blockage=_blockage(
                        0.15,
                        [
                            {"position_angle_deg": 45.0, "width_m": 0.3},
                            {"position_angle_deg": 45.0, "width_m": 0.4},
                        ],
                    )
                )
            ),
            f"{_LEGS}[1].position_angle_deg",
            SEMANTIC_CODES["support_leg_duplicate"],
            id="duplicate_resolved_leg_angle",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [{"n": 33, "m": 1, "surface_height_coefficient_m": 0.0005}]
                    )
                )
            ),
            f"{_MODES}[0]",
            SEMANTIC_CODES["zernike_mode_domain"],
            id="radial_order_above_thirty_two",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [{"n": -2, "m": 0, "surface_height_coefficient_m": 0.0005}]
                    )
                )
            ),
            f"{_MODES}[0]",
            SEMANTIC_CODES["zernike_mode_domain"],
            id="negative_radial_order",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [{"n": 2, "m": 3, "surface_height_coefficient_m": 0.0005}]
                    )
                )
            ),
            f"{_MODES}[0]",
            SEMANTIC_CODES["zernike_mode_domain"],
            id="azimuthal_order_exceeds_radial_order",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [{"n": 2, "m": 1, "surface_height_coefficient_m": 0.0005}]
                    )
                )
            ),
            f"{_MODES}[0]",
            SEMANTIC_CODES["zernike_mode_domain"],
            id="odd_index_parity",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [{"n": 0, "m": 0, "surface_height_coefficient_m": 0.0005}]
                    )
                )
            ),
            f"{_MODES}[0]",
            SEMANTIC_CODES["zernike_mode_reserved"],
            id="piston_is_owned_by_delay",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [{"n": 1, "m": -1, "surface_height_coefficient_m": 0.0005}]
                    )
                )
            ),
            f"{_MODES}[0]",
            SEMANTIC_CODES["zernike_mode_reserved"],
            id="tip_is_owned_by_pointing",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [{"n": 1, "m": 1, "surface_height_coefficient_m": 0.0005}]
                    )
                )
            ),
            f"{_MODES}[0]",
            SEMANTIC_CODES["zernike_mode_reserved"],
            id="tilt_is_owned_by_pointing",
        ),
        pytest.param(
            _analytic_beams(
                aperture_physics=_aperture_physics(
                    zernike_surface=_zernike(
                        [
                            {"n": 2, "m": 0, "surface_height_coefficient_m": 0.0005},
                            {"n": 2, "m": 0, "surface_height_coefficient_m": -0.0005},
                        ]
                    )
                )
            ),
            f"{_MODES}[1]",
            SEMANTIC_CODES["zernike_mode_duplicate"],
            id="duplicate_index_pair",
        ),
        pytest.param(
            _analytic_beams(
                surface_error={
                    "default": {
                        "rms_surface_error_m": 0.001,
                        "error_beam_diagnostic": _diagnostic(0.0),
                    }
                }
            ),
            f"{DEFAULT_DIAGNOSTIC_PATH}.correlation_length_m",
            SEMANTIC_CODES["correlation_length_domain"],
            id="correlation_length_zero",
        ),
        pytest.param(
            _analytic_beams(
                surface_error={
                    "default": {
                        "rms_surface_error_m": 0.001,
                        "error_beam_diagnostic": _diagnostic(-0.25),
                    }
                }
            ),
            f"{DEFAULT_DIAGNOSTIC_PATH}.correlation_length_m",
            SEMANTIC_CODES["correlation_length_domain"],
            id="correlation_length_negative",
        ),
        pytest.param(
            _analytic_beams(
                surface_error={
                    "default": {
                        "rms_surface_error_m": 0.0,
                        "error_beam_diagnostic": _diagnostic(),
                    },
                    "per_antenna": [{"antenna": 1, "rms_surface_error_m": 0.002}],
                }
            ),
            DEFAULT_DIAGNOSTIC_PATH,
            SEMANTIC_CODES["missing_surface_error"],
            id="diagnostic_without_a_positive_surface_rms",
        ),
    ],
)
def test_stage1_identity_domain_and_cross_field_failures_are_semantic_errors(
    tmp_path: Path,
    beams: dict[str, Any],
    path: str,
    code: str,
) -> None:
    """Section 2 and Section 3.5's rejection families, one row per case."""
    with pytest.raises(ConfigSemanticError) as error:
        _resolve(tmp_path, beams)

    assert _rows(error.value) == [(path, code)]


# --- unsupported feature rejections -------------------------------------------


@pytest.mark.parametrize(
    ("model", "code", "message"),
    [
        pytest.param(
            {"kind": "circular_aperture", "taper": {"kind": "gaussian"}},
            UNSUPPORTED_PUPIL_APERTURE,
            _unsupported_profile_message(
                APERTURE_FEATURE, "circular_aperture", "gaussian"
            ),
            id="direct_gaussian_has_no_compact_disk_pupil",
        ),
        pytest.param(
            {"kind": "circular_aperture", "taper": {"kind": "cosine"}},
            UNSUPPORTED_PUPIL_APERTURE,
            _unsupported_profile_message(
                APERTURE_FEATURE, "circular_aperture", "cosine"
            ),
            id="direct_cosine_has_no_declared_radial_inverse",
        ),
        pytest.param(
            {
                "kind": "analytical_illumination",
                "illumination": {"kind": "corrugated_horn"},
                "taper_profile": {"kind": "gaussian"},
            },
            UNSUPPORTED_PUPIL_APERTURE,
            _unsupported_profile_message(
                APERTURE_FEATURE, "analytical_illumination", "gaussian"
            ),
            id="derived_gaussian_has_no_compact_disk_pupil",
        ),
        pytest.param(
            {
                "kind": "numerical_illumination",
                "illumination": {"kind": "open_waveguide"},
            },
            UNSUPPORTED_PUPIL_APERTURE,
            _unsupported_profile_message(
                APERTURE_FEATURE, "numerical_illumination", None
            ),
            id="numerical_rule_is_not_a_retained_compact_pupil",
        ),
        pytest.param(
            {
                "kind": "rectangular_aperture",
                "north_length_m": 12.0,
                "east_length_m": 10.0,
            },
            UNSUPPORTED_FAMILY_APERTURE,
            _unsupported_family_message(APERTURE_FEATURE, "rectangular_aperture"),
            id="rectangular_family",
        ),
        pytest.param(
            {
                "kind": "elliptical_aperture",
                "north_diameter_m": 14.0,
                "east_diameter_m": 12.0,
            },
            UNSUPPORTED_FAMILY_APERTURE,
            _unsupported_family_message(APERTURE_FEATURE, "elliptical_aperture"),
            id="elliptical_family",
        ),
    ],
)
def test_aperture_physics_rejects_every_excluded_pupil_and_family(
    tmp_path: Path,
    model: dict[str, Any],
    code: str,
    message: str,
) -> None:
    """Section 3.1's exact codes, path, and messages; family precedes profile."""
    with pytest.raises(UnsupportedConfigError) as error:
        _resolve(
            tmp_path,
            _analytic_beams(
                model,
                aperture_physics=_aperture_physics(blockage=_blockage(0.15)),
            ),
        )

    assert _rows(error.value) == [(APERTURE_PATH, code)]
    assert error.value.issues[0].message == message


def test_aperture_physics_on_a_fits_beam_is_rejected_as_a_family(
    tmp_path: Path,
) -> None:
    """Section 3.1: "A FITS file already contains its aperture physics"."""
    with pytest.raises(UnsupportedConfigError) as error:
        _resolve(
            tmp_path,
            {
                "mode": "shared_fits",
                "beam": {"kind": "fits", "path": str(tmp_path / "beam.fits")},
                "aperture_physics": _aperture_physics(blockage=_blockage(0.15)),
            },
        )

    assert _rows(error.value) == [(APERTURE_PATH, UNSUPPORTED_FAMILY_APERTURE)]
    assert error.value.issues[0].message == _unsupported_family_message(
        APERTURE_FEATURE, "fits"
    )


@pytest.mark.parametrize(
    ("model", "code", "message"),
    [
        pytest.param(
            {"kind": "circular_aperture", "taper": {"kind": "gaussian"}},
            UNSUPPORTED_PUPIL_DIAGNOSTIC,
            _unsupported_profile_message(
                DIAGNOSTIC_FEATURE, "circular_aperture", "gaussian"
            ),
            id="default_gaussian_beam_is_not_an_implicit_taper_change",
        ),
        pytest.param(
            {
                "kind": "rectangular_aperture",
                "north_length_m": 12.0,
                "east_length_m": 10.0,
            },
            UNSUPPORTED_FAMILY_DIAGNOSTIC,
            _unsupported_family_message(DIAGNOSTIC_FEATURE, "rectangular_aperture"),
            id="rectangular_family",
        ),
    ],
)
def test_diagnostic_only_rejections_use_the_authored_diagnostic_path(
    tmp_path: Path,
    model: dict[str, Any],
    code: str,
    message: str,
) -> None:
    """Section 3.1: a diagnostic-owned issue uses its exact authored path."""
    with pytest.raises(UnsupportedConfigError) as error:
        _resolve(
            tmp_path,
            _analytic_beams(
                model,
                surface_error={
                    "default": {
                        "rms_surface_error_m": 0.001,
                        "error_beam_diagnostic": _diagnostic(),
                    }
                },
            ),
        )

    assert _rows(error.value) == [(DEFAULT_DIAGNOSTIC_PATH, code)]
    assert error.value.issues[0].message == message


def test_diagnostic_paths_are_visited_default_then_ascending_per_antenna(
    tmp_path: Path,
) -> None:
    """Section 3.5 fixes the first rejection recorded in evidence."""
    with pytest.raises(UnsupportedConfigError) as error:
        _resolve(
            tmp_path,
            _analytic_beams(
                {"kind": "circular_aperture", "taper": {"kind": "gaussian"}},
                surface_error={
                    "default": {
                        "rms_surface_error_m": 0.001,
                        "error_beam_diagnostic": _diagnostic(),
                    },
                    "per_antenna": [
                        {
                            "antenna": 1,
                            "rms_surface_error_m": 0.002,
                            "error_beam_diagnostic": _diagnostic(0.4),
                        },
                        {
                            "antenna": 0,
                            "rms_surface_error_m": 0.003,
                            "error_beam_diagnostic": _diagnostic(0.5),
                        },
                    ],
                },
            ),
        )

    assert _rows(error.value) == [
        (DEFAULT_DIAGNOSTIC_PATH, UNSUPPORTED_PUPIL_DIAGNOSTIC),
        (
            "beams.surface_error.per_antenna[0].error_beam_diagnostic",
            UNSUPPORTED_PUPIL_DIAGNOSTIC,
        ),
        (
            "beams.surface_error.per_antenna[1].error_beam_diagnostic",
            UNSUPPORTED_PUPIL_DIAGNOSTIC,
        ),
    ]


def test_aperture_validation_runs_first_and_suppresses_the_duplicate_diagnostic(
    tmp_path: Path,
) -> None:
    """Section 3.1: "If both are present, aperture validation runs first"."""
    with pytest.raises(UnsupportedConfigError) as error:
        _resolve(
            tmp_path,
            _analytic_beams(
                {"kind": "circular_aperture", "taper": {"kind": "gaussian"}},
                aperture_physics=_aperture_physics(blockage=_blockage(0.15)),
                surface_error={
                    "default": {
                        "rms_surface_error_m": 0.001,
                        "error_beam_diagnostic": _diagnostic(),
                    }
                },
            ),
        )

    assert _rows(error.value) == [(APERTURE_PATH, UNSUPPORTED_PUPIL_APERTURE)]


def test_extended_precision_diagnostic_is_rejected_with_the_exact_message(
    tmp_path: Path,
) -> None:
    """Section 3.1: the diagnostic is narrower than the deterministic transform."""
    with pytest.raises(UnsupportedConfigError) as error:
        _resolve(
            tmp_path,
            _analytic_beams(
                PARABOLIC_CIRCULAR,
                surface_error={
                    "default": {
                        "rms_surface_error_m": 0.001,
                        "error_beam_diagnostic": _diagnostic(),
                    }
                },
            ),
            beam_precision="float128",
        )

    assert _rows(error.value) == [
        (DEFAULT_DIAGNOSTIC_PATH, UNSUPPORTED_PRECISION_DIAGNOSTIC)
    ]
    assert error.value.issues[0].message == UNSUPPORTED_PRECISION_MESSAGE


def test_the_same_beam_without_the_nested_diagnostic_keeps_extended_precision(
    tmp_path: Path,
) -> None:
    """Section 3.1: "The same beam without the nested diagnostic retains its
    existing extended-precision behavior"."""
    bundle = _resolve(
        tmp_path,
        _analytic_beams(
            PARABOLIC_CIRCULAR,
            surface_error={"default": {"rms_surface_error_m": 0.001}},
        ),
        beam_precision="float128",
    )

    assert bundle.runtime.execution.precision.jones.beam == "float128"
    assert bundle.runtime.beams.surface_error.default.error_beam_diagnostic is None


# --- Section 3.2: the one instrument-stage rejection --------------------------
#
# Every other Stage-1 rejection is a document-stage ``ConfigIssue``.  The
# leg-wider-than-resolved-diameter rejection is not, and Section 3.2 says why:
# "Document-stage validation does not own this check because per-antenna
# diameters exist only after instrument resolution".  The ownership ruling
# assigns it to ``core/beam/resolution.py``, raising the typed
# ``InvalidBeamGeometryError``.  The two tests below pin both halves of that
# boundary -- the document stage accepting the authored value, and beam-assignment
# resolution rejecting it -- because a check that silently migrated back to the
# document stage would pass a test that only looked at the second half.
#
# ``InvalidBeamGeometryError`` is imported inside each test rather than at module
# scope: today it does not exist, so the intended ``ImportError`` is visible per
# test instead of taking the whole module's collection down with it.

#: The over-wide leg: authored 14 m across, on a resolved 10 m dish.
OVER_WIDE_LEG_ANGLE_DEG = 45.0
OVER_WIDE_LEG_WIDTH_M = 14.0
SMALL_ANTENNA_DIAMETER_M = 10.0
LARGE_ANTENNA_DIAMETER_M = 20.0


def _heterogeneous_layout(data: dict[str, Any]) -> None:
    """Rewrite the fixture layout so the two antennas have unequal diameters.

    A leg wider than one antenna's dish but not the other's is the only way to
    show that the comparison is per assigned antenna rather than array-wide.
    """
    Path(data["instrument"]["source"]["path"]).write_text(
        "Name Number BeamID E N U Diameter\n"
        f"ANT0 0 0 0.0 0.0 0.0 {SMALL_ANTENNA_DIAMETER_M}\n"
        f"ANT1 1 0 14.0 0.0 0.0 {LARGE_ANTENNA_DIAMETER_M}\n",
        encoding="utf-8",
    )


def _resolve_document(tmp_path: Path, data: dict[str, Any]) -> Any:
    return resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )


def test_invalid_beam_geometry_error_is_an_append_only_beam_assignment_error() -> None:
    """Section 3.2 and Section 7.2: exactly one new ``BeamAssignmentError``.

    Section 7.2 grants ``core/beam/errors.py`` an append-only edit of "exactly
    one new class ``InvalidBeamGeometryError(BeamAssignmentError)`` with
    docstring and its ``__all__`` entry; no existing byte changes".  Section 3.5
    adds that this rejection "carries no ``ConfigIssue`` code", which is why it
    must not be a configuration error.
    """
    from radiosim.core.beam import errors as errors_module
    from radiosim.core.beam.errors import InvalidBeamGeometryError

    assert issubclass(InvalidBeamGeometryError, errors_module.BeamAssignmentError)
    assert issubclass(InvalidBeamGeometryError, errors_module.BeamError)
    assert not issubclass(InvalidBeamGeometryError, ConfigResolutionError)
    assert "InvalidBeamGeometryError" in errors_module.__all__
    # Append-only: every pre-existing assignment error is still its own class.
    for existing in (
        "UnknownBeamAntennaError",
        "DuplicateBeamAssignmentError",
        "IncompleteBeamAssignmentError",
        "InconsistentBeamAssignmentError",
    ):
        assert existing in errors_module.__all__
        assert getattr(errors_module, existing) is not InvalidBeamGeometryError


def test_an_over_wide_support_leg_is_rejected_at_beam_assignment_resolution(
    tmp_path: Path,
) -> None:
    """Section 3.2's ownership ruling, both halves.

    ``core/beam/resolution.py`` "compares each authored ``width_m`` against each
    assigned antenna's resolved aperture diameter and raises the typed
    ``InvalidBeamGeometryError`` ... whose message names the leg's position
    angle, the antenna, the authored width, and the resolved diameter."
    """
    from radiosim.core.beam.errors import InvalidBeamGeometryError
    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.instrument_resolution import resolve_instrument

    data = valid_config_mapping(
        tmp_path,
        beams=_analytic_beams(
            aperture_physics=_aperture_physics(
                blockage=_blockage(
                    0.15,
                    [
                        {
                            "position_angle_deg": OVER_WIDE_LEG_ANGLE_DEG,
                            "width_m": OVER_WIDE_LEG_WIDTH_M,
                        }
                    ],
                )
            )
        ),
    )
    _heterogeneous_layout(data)

    # First half: the document stage resolves the authored value untouched and
    # raises no ``ConfigIssue`` at all.
    bundle = _resolve_document(tmp_path, data)
    leg = bundle.runtime.beams.aperture_physics.blockage.support_legs[0]
    assert (leg.position_angle_deg, leg.width_m) == (
        OVER_WIDE_LEG_ANGLE_DEG,
        OVER_WIDE_LEG_WIDTH_M,
    )

    # Second half: beam-assignment resolution owns the rejection.
    instrument = resolve_instrument(bundle.runtime.instrument)
    with pytest.raises(InvalidBeamGeometryError) as error:
        resolve_beam_assignments(bundle.runtime.beams, instrument)

    message = str(error.value)
    assert str(OVER_WIDE_LEG_ANGLE_DEG) in message
    assert str(OVER_WIDE_LEG_WIDTH_M) in message
    assert str(SMALL_ANTENNA_DIAMETER_M) in message
    assert "ANT0" in message
    # The comparison is per assigned antenna: the 20 m dish is not offended by a
    # 14 m leg, so it is not the antenna this message names.
    assert str(LARGE_ANTENNA_DIAMETER_M) not in message
    # Section 3.5: this rejection carries no ``ConfigIssue`` code.
    assert not hasattr(error.value, "issues")


def test_a_leg_exactly_as_wide_as_the_aperture_is_not_wider_and_resolves(
    tmp_path: Path,
) -> None:
    """Section 3.2 rejects a leg *wider than* the resolved diameter.

    The closed-set boundary matters here as much as it does in the mask: a leg
    exactly as wide as the dish is degenerate but representable, and rejecting it
    would be a stricter rule than the one that was accepted.
    """
    from radiosim.core.beam.errors import InvalidBeamGeometryError
    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.instrument_resolution import resolve_instrument

    data = valid_config_mapping(
        tmp_path,
        beams=_analytic_beams(
            aperture_physics=_aperture_physics(
                blockage=_blockage(
                    0.15,
                    [
                        {
                            "position_angle_deg": OVER_WIDE_LEG_ANGLE_DEG,
                            "width_m": OVER_WIDE_LEG_WIDTH_M,
                        }
                    ],
                )
            )
        ),
    )
    # The shipped layout gives both antennas a 14 m dish, exactly the leg width.
    bundle = _resolve_document(tmp_path, data)
    instrument = resolve_instrument(bundle.runtime.instrument)

    try:
        state = resolve_beam_assignments(bundle.runtime.beams, instrument)
    except InvalidBeamGeometryError as exc:  # pragma: no cover - red assertion
        raise AssertionError(
            "a leg exactly as wide as the resolved aperture is not wider than "
            f"it, so it must resolve: {exc}"
        ) from exc

    assert len(state.assignments) == 2
