"""Generate and deeply validate the SCI-007 cross-validation evidence record.

The generator is intentionally available only from the optional ``crossval``
Pixi environment.  It imports the pinned pyuvsim/pyradiosky comparison lazily,
requires an approved clean source commit, installs Astropy's bundled IERS-A
table, and writes the dated record atomically without replacing an existing
artifact.

The validator uses only the Python standard library.  In addition to an
external raw-byte SHA-256 pin, it rejects duplicate JSON keys and checks the
complete schema, provenance, fixture, axes, equations, derived grids, metrics,
and scientific gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "radiosim-crossvalidation-1.2.0"
SLICE = "Post-Tier-8 WP-6 SCI-007"
ARTIFACT_PATH = Path("output/crossvalidation/2026-08-11-pyuvsim-1.4.0-sci007.json")
LOCKFILE_PATH = Path("pixi.lock")

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RFC3339_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")

_EXPECTED_LOCK_SHA256 = (
    "37db432e6ade2dd3e64222d5ccfe532be5671893b24ce29e717a3bbb12f38ade"
)
_EXPECTED_IERS_SHA256 = (
    "ff2d22108e982bd86e326e01d797fa8bd545d51483359dd98e6c08fa5737f667"
)
_EXPECTED_IERS_PACKAGE_RELATIVE_PATH = "astropy_iers_data/data/finals2000A.all"
_METRIC_PIN_ABSOLUTE_TOLERANCE = 5e-15
_EXPECTED_JD1 = [2460677.0, 2460677.0, 2460677.0]
_EXPECTED_JD2 = [
    -0.49999999999999994,
    -0.49861111111111106,
    -0.4972222222222222,
]
_EXPECTED_DUT1_SECONDS = [
    0.0463221,
    0.04632230777777777,
    0.046322515555555555,
]
_EXPECTED_XP_ARCSEC = [
    0.144124,
    0.14412235277777777,
    0.14412070555555556,
]
_EXPECTED_YP_ARCSEC = [
    0.305086,
    0.3050858652777778,
    0.3050857305555556,
]
_EXPECTED_PUBLIC_RADIANS = [
    [0.0009531231588648659, 0.001119916963098433, 0.0007644842652547723],
    [0.0009532125203191022, 0.0011199813948103987, 0.0007645894167778167],
    [0.0009532989596561237, 0.0011200433324138892, 0.000764691079294888],
]
_EXPECTED_EXACT_DEGREES = [
    [0.054345212925135854, 0.06450005592110125, 0.04297043726518762],
    [0.05434599062708946, 0.06450075625817411, 0.04297128951606063],
    [0.05434676529779634, 0.06450145458310087, 0.04297213754147768],
]

_EXPECTED_RELATIVE_METRIC_SCALARS = {
    "raw_linear_relative": {
        "value": 0.002052050642874229,
        "numerator_value": 0.0023454340841143435,
        "denominator_value": 1.1429708580823248,
    },
    "single_global_angle_relative": {
        "value": 0.00019606576512107846,
        "numerator_value": 0.0002240974558010066,
        "denominator_value": 1.1429708580823248,
    },
    "exact_source_time_relative": {
        "value": 2.400855498837282e-10,
        "numerator_value": 2.744107869637716e-10,
        "denominator_value": 1.1429708580823248,
    },
    "intensity_relative": {
        "value": 2.3139573996814273e-10,
        "numerator_value": 1.4019634164363787e-09,
        "denominator_value": 6.058726131385967,
    },
    "circular_relative": {
        "value": 4.0701816228520426e-11,
        "numerator_value": 2.4660115757860614e-10,
        "denominator_value": 6.058726131385967,
    },
    "unpolarized_relative": {
        "value": 2.8065456627916864e-14,
        "numerator_value": 9.515252560102877e-14,
        "denominator_value": 3.390378673061746,
    },
    "retired_q_control": {
        "value": 0.697576347665902,
        "numerator_value": 0.7973094366696302,
        "denominator_value": 1.1429708580823248,
    },
    "wrong_sign_control": {
        "value": 0.004103897953509379,
        "numerator_value": 0.004690635765404911,
        "denominator_value": 1.1429708580823248,
    },
    "unpolarized_no_fringe_control": {
        "value": 1.9544719611337873,
        "numerator_value": 6.626400054125158,
        "denominator_value": 3.390378673061746,
    },
    "linear_to_intensity_scale": {
        "value": 0.18864870820970148,
        "numerator_value": 1.1429708580823248,
        "denominator_value": 6.058726131385967,
    },
    "public_exact_angle_max_relative": {
        "value": 0.019580918743243865,
        "numerator_value": 1.4685792576241141e-05,
        "denominator_value": 0.0007500052867186469,
    },
}

_EXPECTED_SCALE_VALUES = {
    "intensity_scale": 6.058726131385967,
    "linear_scale": 1.1429708580823248,
}

_EXPECTED_FITTED_COMPLEX_RATIO = {
    "real": 1.000080971003292,
    "imaginary": 0.002024449363134283,
    "modulus": 1.0000830200328927,
    "phase_rad": 0.002024282689724037,
    "phase_deg": 0.11598285466257767,
    "half_phase_rotation_rad": 0.0010121413448620185,
    "half_phase_rotation_deg": 0.057991427331288835,
}

_EXPECTED_SOURCE_ADDITIVITY_SCALARS = {
    "radiosim": {
        "value": 0.0,
        "numerator_value": 0.0,
        "denominator_value": 3.1284242960457243,
    },
    "pyuvsim": {
        "value": 0.0,
        "numerator_value": 0.0,
        "denominator_value": 3.1284630458973512,
    },
}

_TOP_LEVEL_KEYS = {
    "schema",
    "recorded_utc",
    "slice",
    "gating",
    "identity",
    "reference",
    "runtime",
    "iers",
    "fixture",
    "axes",
    "equations",
    "correction",
    "predictions",
    "history",
    "tolerances",
    "metrics",
    "limits",
}

_REFERENCE = {
    "pyuvsim": {
        "package": "pyuvsim",
        "version": "1.4.0",
        "task": "pyuvsim.uvsim.UVTask",
        "entry_point": "pyuvsim.UVEngine.make_visibility",
        "beam_entry_point": "pyuvsim.UVEngine.apply_beam",
    },
    "pyradiosky": {
        "package": "pyradiosky",
        "version": "1.1.0",
        "visibility_path": [
            "SkyModel.update_positions",
            "SkyModel.coherency_calc",
            "SkyModel._calc_coherency_rotation",
            "SkyModel._calc_rotation_matrix",
            "SkyModel._calc_average_rotation_matrix",
        ],
        "k_extraction_path": [
            "SkyModel.update_positions",
            "SkyModel._calc_coherency_rotation",
            "SkyModel._calc_rotation_matrix",
            "SkyModel._calc_average_rotation_matrix",
        ],
        "coherency_operation": "K.T @ B_ICRS @ K",
    },
}

_RUNTIME = {
    "pixi_environment": "crossval",
    "solve_group": "py311",
    "packages": {
        "radiosim": "0.3.0",
        "python": "3.11.13",
        "numpy": "2.3.2",
        "astropy": "7.1.0",
        "pyuvdata": "3.2.1",
        "pyuvsim": "1.4.0",
        "pyradiosky": "1.1.0",
    },
    "host": {
        "platform": "macOS-26.6.1-arm64-arm-64bit",
        "system": "Darwin",
        "release": "25.6.0",
        "machine": "arm64",
        "scope": "osx-arm64 only",
    },
}

_EQUATIONS = {
    "B_NE": "ICRS catalogue coherency in a North/East tangent basis",
    "R": "R(a)=[[cos(a),sin(a)],[-sin(a),cos(a)]]",
    "S": "S=[[0,1],[1,0]]",
    "K": "pinned pyradiosky two-dimensional coherency-basis rotation",
    "alpha_PY": "K.T=R(alpha_PY)",
    "Delta": "Delta=wrap_pi(psi_RS+atan2(K[0,1],K[0,0]))",
    "L": "L=Q+iU",
    "wrap_interval": "[-pi, pi)",
    "fringe_mapping": "conj([XX,YX,XY,YY])",
    "v_mapping": "V_RS compared with -V_PY",
}

_CORRECTION = {
    "radiosim_to_pyuvsim": {"factor": "exp(-2j*Delta)"},
    "opposite_direction": {"factor": "exp(+2j*Delta)"},
    "granularity": "per source and time before source summation",
    "single_global_angle_allowed": False,
}

_HISTORY = {
    "pre_sci006_fit": {
        "value_deg": -0.057568764952046436,
        "status": "historical/superseded",
        "included_retired_q_compensation": True,
        "licensed_as_bound": False,
    },
    "cirs_probe": {
        "reported_range_deg": [0.041, 0.063],
        "reproduced_range_deg": [0.0412766, 0.0626888],
        "status": "corroborated/superseded",
        "licensed_as_bound": False,
    },
    "unreproduced_scalar": {
        "value_deg": 0.2,
        "status": "historical/unreproduced/superseded",
        "licensed_as_bound": False,
        "licensed_as_denominator": False,
    },
}

_NORMAL_SCOPE = "retained HERA-site three-source three-time fixture; default and py312"
_OPTIONAL_SCOPE = "retained fixture; crossval py311; non-gating"


def _bound(operator: str, value: float | str, units: str, scope: str) -> dict[str, Any]:
    return {
        "operator": operator,
        "value": value,
        "units": units,
        "scope": scope,
    }


_TOLERANCES = {
    "normal": {
        "public_min_abs_rad": _bound(">", 6e-4, "rad", _NORMAL_SCOPE),
        "public_max_abs_rad": _bound("<", 1.2e-3, "rad", _NORMAL_SCOPE),
        "public_spin2_effect": _bound("<", 2.4e-3, "dimensionless", _NORMAL_SCOPE),
    },
    "optional": {
        "public_exact_max_relative": _bound("<", 0.10, "relative", _OPTIONAL_SCOPE),
        "linear_mask_fraction": _bound(
            ">", 1e-12, "fraction of linear_scale", _OPTIONAL_SCOPE
        ),
        "raw_linear_relative_lower": _bound(">", 1e-3, "relative", _OPTIONAL_SCOPE),
        "raw_linear_relative_upper": _bound("<", 5e-3, "relative", _OPTIONAL_SCOPE),
        "single_global_angle_relative": _bound(">", 1e-4, "relative", _OPTIONAL_SCOPE),
        "single_global_angle_improves_raw": _bound(
            "<", "metrics.raw_linear_relative.value", "relative", _OPTIONAL_SCOPE
        ),
        "exact_source_time_relative": _bound("<", 5e-10, "relative", _OPTIONAL_SCOPE),
        "unpolarized_relative": _bound("<", 1e-11, "relative", _OPTIONAL_SCOPE),
        "intensity_relative": _bound("<", 1e-8, "relative", _OPTIONAL_SCOPE),
        "circular_relative": _bound("<", 1e-8, "relative", _OPTIONAL_SCOPE),
        "retired_q_control": _bound(">", 0.5, "relative", _OPTIONAL_SCOPE),
        "linear_to_intensity_scale": _bound(">", 0.1, "relative", _OPTIONAL_SCOPE),
        "unpolarized_no_fringe_control": _bound(">", 0.1, "relative", _OPTIONAL_SCOPE),
        "wrong_sign_control": _bound(
            ">", "metrics.raw_linear_relative.value", "relative", _OPTIONAL_SCOPE
        ),
    },
}

_LIMITS = {
    "production_code_changed": False,
    "production_frame_policy_changed": False,
    "fixture_scope": "retained HERA-site three-source three-time fixture only",
    "all_sky_claim": False,
    "environment": "optional crossval py311; non-gating",
    "private_api_pin": "pyradiosky==1.1.0",
    "host_scope": "osx-arm64 only",
    "sci007_status": "OPEN pending independent acceptance",
    "validator_metric_pin": {
        "absolute_tolerance": _METRIC_PIN_ABSOLUTE_TOLERANCE,
        "relative_tolerance": 0.0,
        "scope": "every numeric metrics leaf, in that leaf's recorded units",
    },
    "unlicensed_claims": [
        "production tangent-basis transport",
        "PrecisionConfig.ultra() tangent-basis transport",
        "validation against other simulators",
        "validation on other platforms",
        "generalized all-sky frame accuracy",
    ],
}

_TIME_AXIS = {
    "order": ["T0", "T1", "T2"],
    "shape": [3],
    "jd1": _EXPECTED_JD1,
    "jd2": _EXPECTED_JD2,
}
_BASELINE_AXIS = {
    "order": [[0, 1], [0, 2], [1, 2]],
    "shape": [3],
    "value_shape": [3, 2],
}
_FREQUENCY_AXIS = {
    "order": [120000000.0, 130000000.0, 140000000.0],
    "shape": [3],
    "units": "Hz",
}
_POLARIZATION_AXIS = {
    "order": ["XX", "XY", "YX", "YY"],
    "shape": [4],
}

_AXES = {
    "time": _TIME_AXIS,
    "source": {"order": ["S0", "S1", "S2"], "shape": [3]},
    "baseline": _BASELINE_AXIS,
    "frequency": _FREQUENCY_AXIS,
    "polarization": _POLARIZATION_AXIS,
    "aggregate_cube": {
        "order": ["time", "baseline", "frequency", "polarization"],
        "shape": [3, 3, 3, 4],
    },
    "source_cube": {
        "order": ["time", "source", "baseline", "frequency", "polarization"],
        "shape": [3, 3, 3, 3, 4],
    },
    "public_angle_grid": {"order": ["time", "source"], "shape": [3, 3]},
    "exact_angle_grid": {"order": ["time", "source"], "shape": [3, 3]},
    "unpolarized_control": {
        "time": _TIME_AXIS,
        "source": {"order": ["S0", "S1", "S2", "S3"], "shape": [4]},
        "baseline": _BASELINE_AXIS,
        "frequency": _FREQUENCY_AXIS,
        "polarization": _POLARIZATION_AXIS,
        "cube": {
            "order": ["time", "baseline", "frequency", "polarization"],
            "shape": [3, 3, 3, 4],
        },
    },
}

_PRIMARY_SOURCES = [
    {
        "name": "S0",
        "frame": "icrs",
        "ra_deg": 20.0,
        "dec_deg": -30.72,
        "iquv_jy": [3.0, 0.6, -0.4, 0.2],
    },
    {
        "name": "S1",
        "frame": "icrs",
        "ra_deg": 25.0,
        "dec_deg": -26.0,
        "iquv_jy": [1.5, -0.3, 0.5, -0.1],
    },
    {
        "name": "S2",
        "frame": "icrs",
        "ra_deg": 15.0,
        "dec_deg": -35.0,
        "iquv_jy": [2.25, 0.0, 0.0, 0.9],
    },
]

_UNPOLARIZED_SOURCES = [
    {
        "name": "S0",
        "frame": "icrs",
        "ra_deg": 20.0,
        "dec_deg": -30.72,
        "iquv_jy": [3.0, 0.0, 0.0, 0.0],
    },
    {
        "name": "S1",
        "frame": "icrs",
        "ra_deg": 25.0,
        "dec_deg": -26.0,
        "iquv_jy": [1.5, 0.0, 0.0, 0.0],
    },
    {
        "name": "S2",
        "frame": "icrs",
        "ra_deg": 15.0,
        "dec_deg": -35.0,
        "iquv_jy": [2.25, 0.0, 0.0, 0.0],
    },
    {
        "name": "S3",
        "frame": "icrs",
        "ra_deg": 22.0,
        "dec_deg": -31.5,
        "iquv_jy": [0.75, 0.0, 0.0, 0.0],
    },
]


class EvidenceError(ValueError):
    """Raised when SCI-007 evidence or generation provenance is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha(value: str, *, length: int, label: str) -> None:
    pattern = _HEX40 if length == 40 else _HEX64
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be a lowercase {length}-hex digest")


def _canonical_generation_command(approved_source_sha: str) -> str:
    return (
        "pixi run --environment crossval -- python "
        "tools/wp6_sci007_evidence.py generate "
        f"--approved-source-sha {approved_source_sha} --output {ARTIFACT_PATH}"
    )


def _identity(approved_source_sha: str) -> dict[str, Any]:
    return {
        "generating_source_sha": approved_source_sha,
        "clean_tree_at_generation": True,
        "artifact_absent_at_generation": True,
        "artifact_path": ARTIFACT_PATH.as_posix(),
        "lockfile": {
            "path": LOCKFILE_PATH.as_posix(),
            "sha256": _EXPECTED_LOCK_SHA256,
        },
        "generation_command": _canonical_generation_command(approved_source_sha),
        "evidence_relationship": (
            "This committed artifact is an evidence successor of "
            "generating_source_sha; the measurement was made with that commit "
            "at clean HEAD before this artifact existed."
        ),
    }


def _fixture() -> dict[str, Any]:
    authored_site = {
        "longitude_deg": 21.4283,
        "latitude_deg": -30.72152,
        "height_m": 1073.0,
        "geodetic_frame": "WGS84",
    }
    resolved_embedded_site = {
        "longitude_deg": 21.4283,
        "latitude_deg": -30.721519999999995,
        "height_m": 1073.0000000011523,
        "source": "embedded_dataset",
    }
    resolved_explicit_site = {
        "longitude_deg": 21.4283,
        "latitude_deg": -30.721519999999995,
        "height_m": 1073.0000000011523,
        "source": "explicit_config",
    }
    authored_antennas = [
        {
            "name": name,
            "number": number,
            "position_enu_m": enu,
            "diameter_m": 10.0,
        }
        for name, number, enu in (
            ("A000", 0, [0.0, 0.0, 0.0]),
            ("A001", 1, [50.0, 0.0, 0.0]),
            ("A002", 2, [0.0, 70.0, 0.0]),
        )
    ]
    resolved_primary_antennas = [
        {
            "name": "A000",
            "number": 0,
            "position_enu_m": [0.0, 0.0, 0.0],
            "diameter_m": 10.0,
            "instrument_mount_type": "alt-az",
        },
        {
            "name": "A001",
            "number": 1,
            "position_enu_m": [
                50.00000000007367,
                5.488679487023014e-11,
                9.236001072891127e-11,
            ],
            "diameter_m": 10.0,
            "instrument_mount_type": "alt-az",
        },
        {
            "name": "A002",
            "number": 2,
            "position_enu_m": [
                -3.111551230988278e-11,
                69.99999999981858,
                3.220245012585864e-11,
            ],
            "diameter_m": 10.0,
            "instrument_mount_type": "alt-az",
        },
    ]
    resolved_control_antennas = [
        {
            **antenna,
            "instrument_mount_type": None,
        }
        for antenna in authored_antennas
    ]
    times = {
        "scale": "utc",
        "start_iso": {
            "authored": "2025-01-01T00:00:00",
            "resolved": "2025-01-01T00:00:00.000",
        },
        "cadence_seconds": 120.0,
        "jd1": _EXPECTED_JD1,
        "jd2": _EXPECTED_JD2,
    }
    frequencies = [120000000.0, 130000000.0, 140000000.0]
    channel_widths = [1000000.0, 1000000.0, 1000000.0]
    beam = {
        "format": "BeamFITS",
        "beam_type": "efield",
        "response": "exact unit diagonal",
        "data_shape": [2, 2, 4, 5, 8],
        "native_dtype": "complex128",
        "data_normalization": "peak",
        "intrinsic_frequencies_hz": [
            100000000.0,
            120000000.0,
            140000000.0,
            160000000.0,
        ],
        "feed_array": ["x", "y"],
        "x_orientation": "east",
        "beamfits_mount_type": "fixed",
    }
    receptors = {
        "output_basis": "linear_xy",
        "native_basis": "linear",
        "feed_array": ["x", "y"],
        "feed_angle_rad": [1.5707963267948966, 0.0],
        "feed_rotation_rad": 0.0,
        "receptor_sha256": (
            "ed44122dfaf90f4155c15b029f8286450317d1985dad5f5b9287fd8335a7a721"
        ),
    }
    return {
        "primary": {
            "array": {
                "source_kind": "layout_file",
                "format": "uvfits",
                "site_source": "embedded_dataset",
                "antenna_position_source": "embedded_dataset",
            },
            "site": {"authored": authored_site, "resolved": resolved_embedded_site},
            "antennas": {
                "authored": authored_antennas,
                "resolved": resolved_primary_antennas,
            },
            "sources": _PRIMARY_SOURCES,
            "spectral_type": "full",
            "times": times,
            "frequencies_hz": frequencies,
            "channel_widths_hz": channel_widths,
            "correlations": ["XX", "XY", "YX", "YY"],
            "beam": beam,
            "instrument_mount": {
                "type": "alt-az",
                "resolved_per_antenna": ["alt-az", "alt-az", "alt-az"],
            },
            "receptors": receptors,
            "jones": {
                "p_enabled": True,
                "resolved_enabled_terms": ["H", "C", "E", "P"],
                "resolved_chain_order": ["H", "C", "E", "P"],
            },
            "pressure": {
                "radiosim_hpa": 0.0,
                "radiosim_mode": "effective pinned Astropy default",
                "public_oracle_hpa": 0.0,
                "public_oracle_mode": "explicit",
                "pyradiosky_hpa": 0.0,
                "pyradiosky_mode": "effective pinned Astropy default",
            },
            "aggregate_shape": [3, 3, 3, 4],
            "source_decomposed_shape": [3, 3, 3, 3, 4],
        },
        "controls": {
            "unpolarized": {
                "array": {
                    "source_kind": "layout_file",
                    "format": "radiosim",
                    "site_source": "explicit_config",
                    "antenna_position_source": "layout_file",
                },
                "site": {
                    "authored": authored_site,
                    "resolved": resolved_explicit_site,
                },
                "antennas": {
                    "authored": authored_antennas,
                    "resolved": resolved_control_antennas,
                },
                "sources": _UNPOLARIZED_SOURCES,
                "spectral_type": "full",
                "times": times,
                "frequencies_hz": frequencies,
                "channel_widths_hz": channel_widths,
                "correlations": ["XX", "XY", "YX", "YY"],
                "beam": beam,
                "instrument_mount": {
                    "comparison_semantics": "fixed/unspecified",
                    "resolved_per_antenna": [None, None, None],
                    "layout_mount_column_present": False,
                },
                "receptors": receptors,
                "jones": {
                    "p_enabled": False,
                    "resolved_optional_snapshot_empty": True,
                    "always_present_chain": ["H", "C", "E"],
                },
                "pressure": {
                    "radiosim_hpa": 0.0,
                    "radiosim_mode": "effective pinned Astropy default",
                    "pyradiosky_hpa": 0.0,
                    "pyradiosky_mode": "effective pinned Astropy default",
                },
                "shape": [3, 3, 3, 4],
            }
        },
    }


def _iers_record(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "policy": "bundled-IERS_A-with-auto_download-false",
        "auto_download": False,
        "table_class": "astropy.utils.iers.iers.IERS_A",
        "bundled_basename": "finals2000A.all",
        "bundled_package_relative_path": _EXPECTED_IERS_PACKAGE_RELATIVE_PATH,
        "table_sha256": _EXPECTED_IERS_SHA256,
        "package": "astropy-iers-data",
        "package_version": "0.2025.8.25.0.36.58",
        "samples": samples,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON number: {value}")


def _decode_json(data: bytes, source: Path) -> dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"could not read valid JSON from {source}: {exc}") from exc
    if type(value) is not dict:
        raise EvidenceError("artifact root must be a JSON object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"artifact is absent: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"could not read artifact {path}: {exc}") from exc
    return _decode_json(data, path)


def _require_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise EvidenceError(f"{path} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvidenceError(f"{path} key drift: missing={missing}, extra={extra}")
    return value


def _require_exact(actual: Any, expected: Any, path: str) -> None:
    if type(actual) is not type(expected):
        raise EvidenceError(
            f"{path} type changed: expected {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        )
    if type(expected) is dict:
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            raise EvidenceError(
                f"{path} key drift: missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key in expected:
            _require_exact(actual[key], expected[key], f"{path}.{key}")
        return
    if type(expected) in {list, tuple}:
        if len(actual) != len(expected):
            raise EvidenceError(
                f"{path} length changed: expected {len(expected)}, got {len(actual)}"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _require_exact(actual_item, expected_item, f"{path}[{index}]")
        return
    if actual != expected:
        raise EvidenceError(f"{path} changed: expected {expected!r}, got {actual!r}")


def _require_finite_tree(value: Any, path: str = "record") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise EvidenceError(f"{path} must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise EvidenceError(f"{path} has a non-string object key")
            _require_finite_tree(item, f"{path}.{key}")
        return
    raise EvidenceError(f"{path} has unsupported value type {type(value).__name__}")


def _require_float(value: Any, path: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise EvidenceError(f"{path} must be a finite float")
    return value


def _require_int(value: Any, path: str) -> int:
    if type(value) is not int:
        raise EvidenceError(f"{path} must be an int")
    return value


def _close(actual: float, expected: float, *, atol: float, path: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise EvidenceError(
            f"{path} changed: expected {expected!r} +/- {atol}, got {actual!r}"
        )


def _float_grid(value: Any, path: str) -> list[list[float]]:
    if type(value) is not list or len(value) != 3:
        raise EvidenceError(f"{path} must have shape [3,3]")
    grid: list[list[float]] = []
    for row_index, row in enumerate(value):
        if type(row) is not list or len(row) != 3:
            raise EvidenceError(f"{path} must have shape [3,3]")
        grid.append(
            [
                _require_float(item, f"{path}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            ]
        )
    return grid


def _flatten(grid: list[list[float]]) -> list[float]:
    return [item for row in grid for item in row]


def _validate_identity(value: Any, approved_source_sha: str) -> None:
    identity = _require_keys(
        value,
        {
            "generating_source_sha",
            "clean_tree_at_generation",
            "artifact_absent_at_generation",
            "artifact_path",
            "lockfile",
            "generation_command",
            "evidence_relationship",
        },
        "identity",
    )
    _require_exact(identity, _identity(approved_source_sha), "identity")
    _validate_sha(
        identity["generating_source_sha"],
        length=40,
        label="identity.generating_source_sha",
    )


def _validate_iers(value: Any) -> None:
    iers_record = _require_keys(
        value,
        {
            "policy",
            "auto_download",
            "table_class",
            "bundled_basename",
            "bundled_package_relative_path",
            "table_sha256",
            "package",
            "package_version",
            "samples",
        },
        "iers",
    )
    expected_header = _iers_record([])
    expected_header.pop("samples")
    actual_header = dict(iers_record)
    samples = actual_header.pop("samples")
    _require_exact(actual_header, expected_header, "iers metadata")
    _validate_sha(iers_record["table_sha256"], length=64, label="iers.table_sha256")
    if type(samples) is not list or len(samples) != 3:
        raise EvidenceError("iers.samples must be a list of length 3")
    for index, sample_value in enumerate(samples):
        sample = _require_keys(
            sample_value,
            {
                "time_index",
                "jd1",
                "jd2",
                "dut1_seconds",
                "xp_arcsec",
                "yp_arcsec",
                "dut1_status",
                "polar_motion_status",
            },
            f"iers.samples[{index}]",
        )
        _require_exact(
            _require_int(sample["time_index"], f"iers.samples[{index}].time_index"),
            index,
            f"iers.samples[{index}].time_index",
        )
        for key, expected, atol in (
            ("jd1", _EXPECTED_JD1[index], 0.0),
            ("jd2", _EXPECTED_JD2[index], 1e-16),
            ("dut1_seconds", _EXPECTED_DUT1_SECONDS[index], 1e-15),
            ("xp_arcsec", _EXPECTED_XP_ARCSEC[index], 1e-15),
            ("yp_arcsec", _EXPECTED_YP_ARCSEC[index], 1e-15),
        ):
            _close(
                _require_float(sample[key], f"iers.samples[{index}].{key}"),
                expected,
                atol=atol,
                path=f"iers.samples[{index}].{key}",
            )
        expected_status = {"code": 0, "name": "FROM_IERS_B"}
        _require_exact(
            sample["dut1_status"],
            expected_status,
            f"iers.samples[{index}].dut1_status",
        )
        _require_exact(
            sample["polar_motion_status"],
            expected_status,
            f"iers.samples[{index}].polar_motion_status",
        )


def _validate_predictions(value: Any) -> None:
    predictions = _require_keys(
        value,
        {"public", "exact", "public_minus_exact", "extrema"},
        "predictions",
    )
    public = _require_keys(
        predictions["public"], {"radians", "degrees"}, "predictions.public"
    )
    exact = _require_keys(
        predictions["exact"], {"radians", "degrees"}, "predictions.exact"
    )
    difference = _require_keys(
        predictions["public_minus_exact"],
        {"radians", "degrees", "absolute_rad", "relative"},
        "predictions.public_minus_exact",
    )
    extrema = _require_keys(
        predictions["extrema"],
        {
            "public_min_abs_rad",
            "public_max_abs_rad",
            "exact_min_abs_rad",
            "exact_max_abs_rad",
            "public_exact_max_relative",
            "public_spin2_effect_max",
        },
        "predictions.extrema",
    )
    public_rad = _float_grid(public["radians"], "predictions.public.radians")
    public_deg = _float_grid(public["degrees"], "predictions.public.degrees")
    exact_rad = _float_grid(exact["radians"], "predictions.exact.radians")
    exact_deg = _float_grid(exact["degrees"], "predictions.exact.degrees")
    difference_rad = _float_grid(
        difference["radians"], "predictions.public_minus_exact.radians"
    )
    difference_deg = _float_grid(
        difference["degrees"], "predictions.public_minus_exact.degrees"
    )
    absolute_rad = _float_grid(
        difference["absolute_rad"], "predictions.public_minus_exact.absolute_rad"
    )
    relative = _float_grid(
        difference["relative"], "predictions.public_minus_exact.relative"
    )

    for row in range(3):
        for column in range(3):
            suffix = f"[{row}][{column}]"
            _close(
                public_rad[row][column],
                _EXPECTED_PUBLIC_RADIANS[row][column],
                atol=0.0,
                path=f"predictions.public.radians{suffix}",
            )
            _close(
                exact_deg[row][column],
                _EXPECTED_EXACT_DEGREES[row][column],
                atol=0.0,
                path=f"predictions.exact.degrees{suffix}",
            )
            expected_public_deg = math.degrees(public_rad[row][column])
            expected_exact_deg = math.degrees(exact_rad[row][column])
            _close(
                public_deg[row][column],
                expected_public_deg,
                atol=5e-13,
                path=f"predictions.public.degrees{suffix}",
            )
            _close(
                exact_deg[row][column],
                expected_exact_deg,
                atol=5e-13,
                path=f"predictions.exact.degrees{suffix}",
            )
            expected_rad = public_rad[row][column] - exact_rad[row][column]
            expected_deg = public_deg[row][column] - exact_deg[row][column]
            _close(
                difference_rad[row][column],
                expected_rad,
                atol=5e-15,
                path=f"predictions.public_minus_exact.radians{suffix}",
            )
            _close(
                difference_deg[row][column],
                expected_deg,
                atol=5e-13,
                path=f"predictions.public_minus_exact.degrees{suffix}",
            )
            _close(
                absolute_rad[row][column],
                abs(expected_rad),
                atol=5e-15,
                path=f"predictions.public_minus_exact.absolute_rad{suffix}",
            )
            _close(
                relative[row][column],
                abs(expected_rad) / abs(exact_rad[row][column]),
                atol=5e-15,
                path=f"predictions.public_minus_exact.relative{suffix}",
            )

    public_flat = _flatten(public_rad)
    exact_flat = _flatten(exact_rad)
    expected_extrema = {
        "public_min_abs_rad": min(abs(item) for item in public_flat),
        "public_max_abs_rad": max(abs(item) for item in public_flat),
        "exact_min_abs_rad": min(abs(item) for item in exact_flat),
        "exact_max_abs_rad": max(abs(item) for item in exact_flat),
        "public_exact_max_relative": max(_flatten(relative)),
        "public_spin2_effect_max": max(
            abs(complex(math.cos(2.0 * item), math.sin(2.0 * item)) - 1.0)
            for item in public_flat
        ),
    }
    for key, expected in expected_extrema.items():
        _close(
            _require_float(extrema[key], f"predictions.extrema.{key}"),
            expected,
            atol=5e-15,
            path=f"predictions.extrema.{key}",
        )


_RELATIVE_METRIC_KEYS = {
    "value",
    "numerator_value",
    "numerator_units",
    "denominator_name",
    "denominator_value",
    "denominator_units",
    "definition",
}

_RELATIVE_METRIC_CONTRACT = {
    "raw_linear_relative": (
        "Jy",
        "linear_scale",
        "Jy",
        "max(abs(L_RS-L_PY))[valid]/linear_scale",
    ),
    "single_global_angle_relative": (
        "Jy",
        "linear_scale",
        "Jy",
        "max(abs(L_RS*exp(-1j*arg(fitted_ratio))-L_PY))[valid]/linear_scale",
    ),
    "exact_source_time_relative": (
        "Jy",
        "linear_scale",
        "Jy",
        ("max(abs(sum_source(L_RS_source*exp(-2j*Delta))-L_PY))[valid]/linear_scale"),
    ),
    "intensity_relative": (
        "Jy",
        "intensity_scale",
        "Jy",
        "max(abs(I_RS-I_PY))/intensity_scale",
    ),
    "circular_relative": (
        "Jy",
        "intensity_scale",
        "Jy",
        "max(abs(V_RS+V_PY))/intensity_scale",
    ),
    "unpolarized_relative": (
        "Jy",
        "unpolarized_cube_scale",
        "Jy",
        "max(abs(cube_RS-cube_PY))/unpolarized_cube_scale",
    ),
    "retired_q_control": (
        "Jy",
        "linear_scale",
        "Jy",
        "max(abs(L_RS-(-Q_PY+iU_PY)))/linear_scale",
    ),
    "wrong_sign_control": (
        "Jy",
        "linear_scale",
        "Jy",
        ("max(abs(sum_source(L_RS_source*exp(+2j*Delta))-L_PY))[valid]/linear_scale"),
    ),
    "unpolarized_no_fringe_control": (
        "Jy",
        "unpolarized_cube_scale",
        "Jy",
        ("max(abs(cube_RS-fringe_mapping(cube_PY_mapped)))/unpolarized_cube_scale"),
    ),
    "linear_to_intensity_scale": (
        "Jy",
        "intensity_scale",
        "Jy",
        "linear_scale/intensity_scale",
    ),
    "public_exact_angle_max_relative": (
        "rad",
        "exact_angle_at_max_relative_cell",
        "rad",
        "max(abs(Delta_public-Delta_exact)/abs(Delta_exact))",
    ),
}

_SOURCE_ADDITIVITY_CONTRACT = {
    "radiosim": (
        "Jy",
        "radiosim_cube_scale",
        "Jy",
        "max(abs(sum_source(cube_RS_source)-cube_RS))/cube_scale",
    ),
    "pyuvsim": (
        "Jy",
        "pyuvsim_cube_scale",
        "Jy",
        "max(abs(sum_source(cube_PY_source)-cube_PY))/cube_scale",
    ),
}


def _validate_relative_metric(value: Any, path: str) -> dict[str, Any]:
    metric = _require_keys(value, _RELATIVE_METRIC_KEYS, path)
    measured = _require_float(metric["value"], f"{path}.value")
    numerator = _require_float(metric["numerator_value"], f"{path}.numerator_value")
    denominator = _require_float(
        metric["denominator_value"], f"{path}.denominator_value"
    )
    if measured < 0.0:
        raise EvidenceError(f"{path}.value must be nonnegative")
    if numerator < 0.0:
        raise EvidenceError(f"{path}.numerator_value must be nonnegative")
    if denominator <= 0.0:
        raise EvidenceError(f"{path}.denominator_value must be positive")
    for key in (
        "numerator_units",
        "denominator_name",
        "denominator_units",
        "definition",
    ):
        if type(metric[key]) is not str or not metric[key]:
            raise EvidenceError(f"{path}.{key} must be a nonblank string")
    _close(measured, numerator / denominator, atol=5e-15, path=f"{path}.value")
    return metric


def _validate_relative_metric_contract(
    metric: dict[str, Any],
    contract: tuple[str, str, str, str],
    path: str,
) -> None:
    numerator_units, denominator_name, denominator_units, definition = contract
    _require_exact(
        metric["numerator_units"], numerator_units, f"{path}.numerator_units"
    )
    _require_exact(
        metric["denominator_name"], denominator_name, f"{path}.denominator_name"
    )
    _require_exact(
        metric["denominator_units"], denominator_units, f"{path}.denominator_units"
    )
    _require_exact(metric["definition"], definition, f"{path}.definition")


def _validate_metrics(value: Any, predictions: dict[str, Any]) -> None:
    metrics = _require_keys(
        value,
        {
            "raw_linear_relative",
            "single_global_angle_relative",
            "exact_source_time_relative",
            "intensity_relative",
            "circular_relative",
            "unpolarized_relative",
            "retired_q_control",
            "wrong_sign_control",
            "unpolarized_no_fringe_control",
            "intensity_scale",
            "linear_scale",
            "linear_to_intensity_scale",
            "fitted_complex_ratio",
            "linear_cells",
            "public_exact_angle_max_relative",
            "source_additivity",
        },
        "metrics",
    )
    relative_names = (
        "raw_linear_relative",
        "single_global_angle_relative",
        "exact_source_time_relative",
        "intensity_relative",
        "circular_relative",
        "unpolarized_relative",
        "retired_q_control",
        "wrong_sign_control",
        "unpolarized_no_fringe_control",
        "linear_to_intensity_scale",
        "public_exact_angle_max_relative",
    )
    relative = {
        name: _validate_relative_metric(metrics[name], f"metrics.{name}")
        for name in relative_names
    }
    for name, metric in relative.items():
        _validate_relative_metric_contract(
            metric,
            _RELATIVE_METRIC_CONTRACT[name],
            f"metrics.{name}",
        )
        for scalar_name, expected in _EXPECTED_RELATIVE_METRIC_SCALARS[name].items():
            _close(
                _require_float(metric[scalar_name], f"metrics.{name}.{scalar_name}"),
                expected,
                atol=_METRIC_PIN_ABSOLUTE_TOLERANCE,
                path=f"metrics.{name}.{scalar_name}",
            )
    for name in ("intensity_scale", "linear_scale"):
        scale = _require_keys(
            metrics[name], {"value", "units", "definition"}, f"metrics.{name}"
        )
        if _require_float(scale["value"], f"metrics.{name}.value") <= 0.0:
            raise EvidenceError(f"metrics.{name}.value must be positive")
        _close(
            _require_float(scale["value"], f"metrics.{name}.value"),
            _EXPECTED_SCALE_VALUES[name],
            atol=_METRIC_PIN_ABSOLUTE_TOLERANCE,
            path=f"metrics.{name}.value",
        )
        _require_exact(scale["units"], "Jy", f"metrics.{name}.units")
        expected_definition = (
            "max(abs(I_PY))" if name == "intensity_scale" else "max(abs(L_PY))"
        )
        _require_exact(
            scale["definition"],
            expected_definition,
            f"metrics.{name}.definition",
        )

    fitted = _require_keys(
        metrics["fitted_complex_ratio"],
        {
            "real",
            "imaginary",
            "modulus",
            "phase_rad",
            "phase_deg",
            "half_phase_rotation_rad",
            "half_phase_rotation_deg",
            "definition",
        },
        "metrics.fitted_complex_ratio",
    )
    fitted_numbers = {
        key: _require_float(fitted[key], f"metrics.fitted_complex_ratio.{key}")
        for key in (
            "real",
            "imaginary",
            "modulus",
            "phase_rad",
            "phase_deg",
            "half_phase_rotation_rad",
            "half_phase_rotation_deg",
        )
    }
    for key, expected in _EXPECTED_FITTED_COMPLEX_RATIO.items():
        _close(
            fitted_numbers[key],
            expected,
            atol=_METRIC_PIN_ABSOLUTE_TOLERANCE,
            path=f"metrics.fitted_complex_ratio.{key}",
        )
    _close(
        fitted_numbers["modulus"],
        abs(complex(fitted_numbers["real"], fitted_numbers["imaginary"])),
        atol=5e-15,
        path="metrics.fitted_complex_ratio.modulus",
    )
    expected_phase = math.atan2(fitted_numbers["imaginary"], fitted_numbers["real"])
    _close(
        fitted_numbers["phase_rad"],
        expected_phase,
        atol=5e-15,
        path="metrics.fitted_complex_ratio.phase_rad",
    )
    _close(
        fitted_numbers["phase_deg"],
        math.degrees(expected_phase),
        atol=5e-13,
        path="metrics.fitted_complex_ratio.phase_deg",
    )
    _close(
        fitted_numbers["half_phase_rotation_rad"],
        0.5 * expected_phase,
        atol=5e-15,
        path="metrics.fitted_complex_ratio.half_phase_rotation_rad",
    )
    _close(
        fitted_numbers["half_phase_rotation_deg"],
        0.5 * math.degrees(expected_phase),
        atol=5e-13,
        path="metrics.fitted_complex_ratio.half_phase_rotation_deg",
    )
    _require_exact(
        fitted["definition"],
        "vdot(L_PY,L_RS)/vdot(L_PY,L_PY) over valid cells",
        "metrics.fitted_complex_ratio.definition",
    )

    cells = _require_keys(
        metrics["linear_cells"],
        {"valid", "total", "mask_definition"},
        "metrics.linear_cells",
    )
    valid = _require_int(cells["valid"], "metrics.linear_cells.valid")
    total = _require_int(cells["total"], "metrics.linear_cells.total")
    if (valid, total) != (27, 27):
        raise EvidenceError("metrics.linear_cells must report exactly 27 valid of 27")
    _require_exact(
        cells["mask_definition"],
        "abs(L_reference) > linear_scale * 1e-12",
        "metrics.linear_cells.mask_definition",
    )

    additivity = _require_keys(
        metrics["source_additivity"],
        {"radiosim", "pyuvsim"},
        "metrics.source_additivity",
    )
    for name in ("radiosim", "pyuvsim"):
        path = f"metrics.source_additivity.{name}"
        metric = _validate_relative_metric(additivity[name], path)
        _validate_relative_metric_contract(
            metric,
            _SOURCE_ADDITIVITY_CONTRACT[name],
            path,
        )
        for scalar_name, expected in _EXPECTED_SOURCE_ADDITIVITY_SCALARS[name].items():
            _close(
                _require_float(metric[scalar_name], f"{path}.{scalar_name}"),
                expected,
                atol=_METRIC_PIN_ABSOLUTE_TOLERANCE,
                path=f"{path}.{scalar_name}",
            )
        if metric["numerator_value"] > 2e-15:
            raise EvidenceError(f"{path} absolute residual must be at most 2e-15 Jy")

    intensity_scale = metrics["intensity_scale"]["value"]
    linear_scale = metrics["linear_scale"]["value"]
    _require_exact(
        relative["intensity_relative"]["denominator_name"],
        "intensity_scale",
        "metrics.intensity_relative.denominator_name",
    )
    _require_exact(
        relative["circular_relative"]["denominator_name"],
        "intensity_scale",
        "metrics.circular_relative.denominator_name",
    )
    for name in (
        "raw_linear_relative",
        "single_global_angle_relative",
        "exact_source_time_relative",
        "retired_q_control",
        "wrong_sign_control",
    ):
        _require_exact(
            relative[name]["denominator_name"],
            "linear_scale",
            f"metrics.{name}.denominator_name",
        )
        _close(
            relative[name]["denominator_value"],
            linear_scale,
            atol=0.0,
            path=f"metrics.{name}.denominator_value",
        )
    _close(
        relative["intensity_relative"]["denominator_value"],
        intensity_scale,
        atol=0.0,
        path="metrics.intensity_relative.denominator_value",
    )
    _close(
        relative["circular_relative"]["denominator_value"],
        intensity_scale,
        atol=0.0,
        path="metrics.circular_relative.denominator_value",
    )
    _close(
        relative["linear_to_intensity_scale"]["numerator_value"],
        linear_scale,
        atol=0.0,
        path="metrics.linear_to_intensity_scale.numerator_value",
    )
    _close(
        relative["linear_to_intensity_scale"]["denominator_value"],
        intensity_scale,
        atol=0.0,
        path="metrics.linear_to_intensity_scale.denominator_value",
    )
    expected_angle_relative = predictions["extrema"]["public_exact_max_relative"]
    _close(
        relative["public_exact_angle_max_relative"]["value"],
        expected_angle_relative,
        atol=5e-15,
        path="metrics.public_exact_angle_max_relative.value",
    )

    raw = relative["raw_linear_relative"]["value"]
    if not 1e-3 < raw < 5e-3:
        raise EvidenceError("raw linear residual is outside (1e-3, 5e-3)")
    if relative["single_global_angle_relative"]["value"] <= 1e-4:
        raise EvidenceError("single-global-angle control must exceed 1e-4")
    if relative["single_global_angle_relative"]["value"] >= raw:
        raise EvidenceError("single-global-angle control must improve on raw")
    if relative["exact_source_time_relative"]["value"] >= 5e-10:
        raise EvidenceError("exact source-time residual must be below 5e-10")
    if relative["unpolarized_relative"]["value"] >= 1e-11:
        raise EvidenceError("unpolarized residual must be below 1e-11")
    if relative["intensity_relative"]["value"] >= 1e-8:
        raise EvidenceError("intensity residual must be below 1e-8")
    if relative["circular_relative"]["value"] >= 1e-8:
        raise EvidenceError("mapped circular residual must be below 1e-8")
    if relative["retired_q_control"]["value"] <= 0.5:
        raise EvidenceError("retired-Q control must exceed 0.5")
    if relative["linear_to_intensity_scale"]["value"] <= 0.1:
        raise EvidenceError("linear/intensity scale must exceed 0.1")
    if relative["unpolarized_no_fringe_control"]["value"] <= 0.1:
        raise EvidenceError("unpolarized no-fringe control must exceed 0.1")
    if relative["wrong_sign_control"]["value"] <= raw:
        raise EvidenceError("wrong-sign control must exceed the raw residual")


def validate_record(record: dict[str, Any], *, approved_source_sha: str) -> None:
    """Validate a parsed SCI-007 record against the complete 1.2.0 contract."""
    _validate_sha(approved_source_sha, length=40, label="approved source SHA")
    _require_finite_tree(record)
    _require_keys(record, _TOP_LEVEL_KEYS, "record")
    _require_exact(record["schema"], SCHEMA, "schema")
    _require_exact(record["slice"], SLICE, "slice")
    _require_exact(record["gating"], False, "gating")
    if type(record["recorded_utc"]) is not str or not _RFC3339_UTC.fullmatch(
        record["recorded_utc"]
    ):
        raise EvidenceError("recorded_utc must be second-resolution RFC3339 UTC")
    try:
        recorded = datetime.strptime(record["recorded_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise EvidenceError("recorded_utc is not a real UTC timestamp") from exc
    if recorded.date().isoformat() != "2026-08-11":
        raise EvidenceError(
            "recorded_utc date must be 2026-08-11 to match the canonical "
            "dated artifact path"
        )

    _validate_identity(record["identity"], approved_source_sha)
    _require_exact(record["reference"], _REFERENCE, "reference")
    _require_exact(record["runtime"], _RUNTIME, "runtime")
    _validate_iers(record["iers"])
    _require_exact(record["fixture"], _fixture(), "fixture")
    _require_exact(record["axes"], _AXES, "axes")
    _require_exact(record["equations"], _EQUATIONS, "equations")
    _require_exact(record["correction"], _CORRECTION, "correction")
    _validate_predictions(record["predictions"])
    _require_exact(record["history"], _HISTORY, "history")
    _require_exact(record["tolerances"], _TOLERANCES, "tolerances")
    _validate_metrics(record["metrics"], record["predictions"])
    _require_exact(record["limits"], _LIMITS, "limits")

    extrema = record["predictions"]["extrema"]
    if extrema["public_min_abs_rad"] <= 6e-4:
        raise EvidenceError("public minimum absolute angle must exceed 6e-4 rad")
    if extrema["public_max_abs_rad"] >= 1.2e-3:
        raise EvidenceError("public maximum absolute angle must be below 1.2e-3 rad")
    if extrema["public_spin2_effect_max"] >= 2.4e-3:
        raise EvidenceError("public spin-2 effect must be below 2.4e-3")
    if extrema["public_exact_max_relative"] >= 0.10:
        raise EvidenceError("public/exact angular disagreement must be below 0.10")


def validate_artifact(
    input_path: Path,
    *,
    approved_source_sha: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    """Validate artifact bytes and semantics, returning the parsed record."""
    _validate_sha(artifact_sha256, length=64, label="artifact SHA-256")
    try:
        raw = input_path.read_bytes()
    except FileNotFoundError as exc:
        raise EvidenceError(f"artifact is absent: {input_path}") from exc
    except OSError as exc:
        raise EvidenceError(f"could not read artifact {input_path}: {exc}") from exc
    measured_sha256 = hashlib.sha256(raw).hexdigest()
    if measured_sha256 != artifact_sha256:
        raise EvidenceError(
            "artifact SHA-256 mismatch: "
            f"expected {artifact_sha256}, measured {measured_sha256}"
        )
    record = _decode_json(raw, input_path)
    validate_record(record, approved_source_sha=approved_source_sha)
    return record


def _git(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError(f"git {' '.join(arguments)} failed: {exc}") from exc
    return completed.stdout.strip()


def _assert_generation_environment() -> None:
    """Require the exact locked Pixi project and crossval environment."""
    if os.environ.get("PIXI_ENVIRONMENT_NAME") != "crossval":
        raise EvidenceError(
            "generation requires PIXI_ENVIRONMENT_NAME=crossval; use the recorded "
            "pixi command"
        )
    expected_prefix = (REPO_ROOT / ".pixi/envs/crossval").resolve()
    measured_prefix = Path(sys.prefix).resolve()
    if measured_prefix != expected_prefix:
        raise EvidenceError(
            "generation is not running from the repository's locked crossval "
            f"environment: expected {expected_prefix}, got {measured_prefix}"
        )
    project_root_value = os.environ.get("PIXI_PROJECT_ROOT")
    if not project_root_value or Path(project_root_value).resolve() != REPO_ROOT:
        raise EvidenceError(
            "PIXI_PROJECT_ROOT must resolve to the generating repository root"
        )
    manifest_value = os.environ.get("PIXI_PROJECT_MANIFEST")
    expected_manifest = (REPO_ROOT / "pixi.toml").resolve()
    if not manifest_value or Path(manifest_value).resolve() != expected_manifest:
        raise EvidenceError(
            "PIXI_PROJECT_MANIFEST must resolve to the generating pixi.toml"
        )
    if Path.cwd().resolve() != REPO_ROOT:
        raise EvidenceError(
            f"generation must run from repository root {REPO_ROOT}, got {Path.cwd()}"
        )


def _preflight_generation(approved_source_sha: str, output: Path) -> Path:
    _validate_sha(approved_source_sha, length=40, label="approved source SHA")
    _assert_generation_environment()
    if output != ARTIFACT_PATH:
        raise EvidenceError(
            "--output must use the canonical repository-relative spelling "
            f"{ARTIFACT_PATH.as_posix()}"
        )
    resolved_output = REPO_ROOT / output
    resolved_output = resolved_output.resolve()
    expected_output = (REPO_ROOT / ARTIFACT_PATH).resolve()
    if resolved_output != expected_output:
        raise EvidenceError(
            f"output must be the canonical artifact path {ARTIFACT_PATH.as_posix()}"
        )
    if resolved_output.exists():
        raise EvidenceError(
            f"refusing to overwrite existing artifact: {resolved_output}"
        )
    if not resolved_output.parent.is_dir():
        raise EvidenceError(
            f"artifact parent directory is absent: {resolved_output.parent}"
        )
    head = _git("rev-parse", "HEAD")
    if head != approved_source_sha:
        raise EvidenceError(
            f"HEAD {head} does not equal approved source SHA {approved_source_sha}"
        )
    dirty = _git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise EvidenceError(
            "generation requires a clean tree before the artifact exists; git "
            f"reported:\n{dirty}"
        )
    lockfile = REPO_ROOT / LOCKFILE_PATH
    if not lockfile.is_file():
        raise EvidenceError(f"lockfile is absent: {lockfile}")
    lock_sha256 = _sha256(lockfile)
    if lock_sha256 != _EXPECTED_LOCK_SHA256:
        raise EvidenceError(
            f"pixi.lock drifted from the approved source contract: {lock_sha256}"
        )
    return resolved_output


def _postflight_generation(approved_source_sha: str, output: Path) -> None:
    """Re-authenticate source state after measurement and before publication."""
    head = _git("rev-parse", "HEAD")
    if head != approved_source_sha:
        raise EvidenceError(
            "HEAD changed during measurement: "
            f"expected {approved_source_sha}, measured {head}"
        )
    dirty = _git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise EvidenceError(
            f"the source tree changed during measurement; refusing to publish:\n{dirty}"
        )
    if output.exists():
        raise EvidenceError(f"artifact appeared during measurement: {output}")
    lock_sha256 = _sha256(REPO_ROOT / LOCKFILE_PATH)
    if lock_sha256 != _EXPECTED_LOCK_SHA256:
        raise EvidenceError(
            "pixi.lock changed during measurement: "
            f"expected {_EXPECTED_LOCK_SHA256}, measured {lock_sha256}"
        )


def _load_crossvalidation_module() -> ModuleType:
    module_path = REPO_ROOT / "tests/crossvalidation/test_pyuvsim_comparison.py"
    name = "_radiosim_wp6_crossvalidation"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise EvidenceError(f"cannot import cross-validation module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _assert_signature(owner: Any, name: str, expected: tuple[str, ...]) -> None:
    function = getattr(owner, name, None)
    if not callable(function):
        raise EvidenceError(f"required callable is absent: {owner.__name__}.{name}")
    actual = tuple(inspect.signature(function).parameters)
    if actual != expected:
        raise EvidenceError(
            f"signature drift for {owner.__name__}.{name}: "
            f"expected {expected}, got {actual}"
        )


def _assert_runtime_contract(module: ModuleType) -> None:
    import astropy
    import numpy
    import pyradiosky
    import pyuvdata
    import pyuvsim
    from pyradiosky import SkyModel
    from pyuvsim import UVEngine

    import radiosim

    measured_packages = {
        "radiosim": importlib.metadata.version("radiosim"),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "astropy": astropy.__version__,
        "pyuvdata": pyuvdata.__version__,
        "pyuvsim": pyuvsim.__version__,
        "pyradiosky": pyradiosky.__version__,
    }
    if measured_packages != _RUNTIME["packages"]:
        raise EvidenceError(
            f"runtime package drift: expected {_RUNTIME['packages']}, "
            f"got {measured_packages}"
        )
    resolved_radiosim = Path(radiosim.__file__).resolve().parent
    expected_radiosim = (REPO_ROOT / "src/radiosim").resolve()
    if resolved_radiosim != expected_radiosim:
        raise EvidenceError(
            "measurement imported RadioSim from the wrong checkout: "
            f"expected {expected_radiosim}, got {resolved_radiosim}"
        )
    measured_host = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "scope": "osx-arm64 only",
    }
    if measured_host != _RUNTIME["host"]:
        raise EvidenceError(
            f"generation host is outside the accepted scope: {measured_host}"
        )

    helper_signatures = {
        "_write_unit_beamfits": ("path",),
        "_run_sci007_comparison": ("tmp_path", "unit_beamfits"),
        "_run_unpolarized_comparison": ("tmp_path", "unit_beamfits"),
    }
    for name, expected in helper_signatures.items():
        _assert_signature(module, name, expected)
    helper_constants = {
        "LATITUDE_DEG": -30.72152,
        "LONGITUDE_DEG": 21.4283,
        "HEIGHT_M": 1073.0,
        "REFERENCE_PYUVSIM_VERSION": "1.4.0",
        "REFERENCE_PYRADIOSKY_VERSION": "1.1.0",
        "REFERENCE_ASTROPY_VERSION": "7.1.0",
        "ANTENNA_ENU_M": (
            (0.0, 0.0, 0.0),
            (50.0, 0.0, 0.0),
            (0.0, 70.0, 0.0),
        ),
        "ANTENNA_DIAMETER_M": 10.0,
        "FREQUENCIES_HZ": (120000000.0, 130000000.0, 140000000.0),
        "CHANNEL_WIDTH_HZ": 1000000.0,
        "START_TIME_ISO": "2025-01-01T00:00:00",
        "CADENCE_SECONDS": 120.0,
        "TIME_SAMPLES": 3,
        "POLARIZED_IQUV_BY_SOURCE": (
            (3.0, 0.6, -0.4, 0.2),
            (1.5, -0.3, 0.5, -0.1),
            (2.25, 0.0, 0.0, 0.9),
        ),
        "POLARIZED_RA_DEG": (20.0, 25.0, 15.0),
        "POLARIZED_DEC_DEG": (-30.72, -26.0, -35.0),
    }
    for name, expected in helper_constants.items():
        _require_exact(
            getattr(module, name, None),
            expected,
            f"crossvalidation.{name}",
        )
    for name, expected in {
        "update_positions": ("self", "time", "telescope_location"),
        "coherency_calc": ("self", "store_frame_coherency"),
        "_calc_coherency_rotation": ("self", "inds"),
        "_calc_rotation_matrix": ("self", "inds"),
        "_calc_average_rotation_matrix": ("self",),
    }.items():
        _assert_signature(SkyModel, name, expected)
    _assert_signature(UVEngine, "make_visibility", ("self",))
    _assert_signature(UVEngine, "apply_beam", ("self", "beam_interp_check"))


def _status_name(code: int, iers_module: Any) -> str:
    statuses = {
        int(iers_module.FROM_IERS_B): "FROM_IERS_B",
        int(iers_module.FROM_IERS_A): "FROM_IERS_A",
        int(iers_module.FROM_IERS_A_PREDICTION): "FROM_IERS_A_PREDICTION",
    }
    if code not in statuses:
        raise EvidenceError(f"unexpected IERS status code: {code}")
    return statuses[code]


def _measure_iers_samples(table: Any, jd1: Any, jd2: Any) -> list[dict[str, Any]]:
    import numpy as np
    from astropy import units
    from astropy.time import Time
    from astropy.utils import iers

    times = Time(jd1, jd2, format="jd", scale="utc")
    dut1, dut1_status = table.ut1_utc(times, return_status=True)
    xp, yp, polar_motion_status = table.pm_xy(times, return_status=True)
    samples = []
    for index in range(len(times)):
        dut1_code = int(np.asarray(dut1_status)[index])
        polar_motion_code = int(np.asarray(polar_motion_status)[index])
        samples.append(
            {
                "time_index": index,
                "jd1": float(np.asarray(times.jd1)[index]),
                "jd2": float(np.asarray(times.jd2)[index]),
                "dut1_seconds": float(dut1.to_value(units.s)[index]),
                "xp_arcsec": float(xp.to_value(units.arcsec)[index]),
                "yp_arcsec": float(yp.to_value(units.arcsec)[index]),
                "dut1_status": {
                    "code": dut1_code,
                    "name": _status_name(dut1_code, iers),
                },
                "polar_motion_status": {
                    "code": polar_motion_code,
                    "name": _status_name(polar_motion_code, iers),
                },
            }
        )
    return samples


def _assert_returned_iers_table(table: Any, runtime_path: Path, label: str) -> None:
    table_class = f"{type(table).__module__}.{type(table).__qualname__}"
    _require_exact(
        table_class,
        "astropy.utils.iers.iers.IERS_A",
        f"{label}.iers_table.class",
    )
    data_path_value = table.meta.get("data_path")
    if type(data_path_value) is not str or not data_path_value:
        raise EvidenceError(f"{label}.iers_table.meta['data_path'] is absent")
    data_path = Path(data_path_value).resolve()
    _require_exact(data_path, runtime_path, f"{label}.iers_table.data_path")
    _require_exact(
        _sha256(data_path),
        _EXPECTED_IERS_SHA256,
        f"{label}.iers_table.sha256",
    )


def _relative_metric(
    *,
    value: float,
    numerator_value: float,
    numerator_units: str,
    denominator_name: str,
    denominator_value: float,
    denominator_units: str,
    definition: str,
) -> dict[str, Any]:
    return {
        "value": float(value),
        "numerator_value": float(numerator_value),
        "numerator_units": numerator_units,
        "denominator_name": denominator_name,
        "denominator_value": float(denominator_value),
        "denominator_units": denominator_units,
        "definition": definition,
    }


def _prediction_record(public_angles: Any, exact_angles: Any) -> dict[str, Any]:
    import numpy as np

    public = np.asarray(public_angles, dtype=np.float64)
    exact = np.asarray(exact_angles, dtype=np.float64)
    if public.shape != (3, 3) or exact.shape != (3, 3):
        raise EvidenceError(
            f"angle shape drift: public={public.shape}, exact={exact.shape}"
        )
    difference = public - exact
    relative = np.abs(difference) / np.abs(exact)
    return {
        "public": {
            "radians": public.tolist(),
            "degrees": np.degrees(public).tolist(),
        },
        "exact": {
            "radians": exact.tolist(),
            "degrees": np.degrees(exact).tolist(),
        },
        "public_minus_exact": {
            "radians": difference.tolist(),
            "degrees": np.degrees(difference).tolist(),
            "absolute_rad": np.abs(difference).tolist(),
            "relative": relative.tolist(),
        },
        "extrema": {
            "public_min_abs_rad": float(np.min(np.abs(public))),
            "public_max_abs_rad": float(np.max(np.abs(public))),
            "exact_min_abs_rad": float(np.min(np.abs(exact))),
            "exact_max_abs_rad": float(np.max(np.abs(exact))),
            "public_exact_max_relative": float(np.max(relative)),
            "public_spin2_effect_max": float(np.max(np.abs(np.exp(2j * public) - 1.0))),
        },
    }


def _metric_record(
    primary: Any, unpolarized: Any, module: ModuleType
) -> dict[str, Any]:
    import numpy as np

    local_stokes = module._local_stokes
    fringe_mapping = module._apply_fringe_hermitian_mapping
    ours_stokes = local_stokes(primary.ours)
    theirs_stokes = local_stokes(primary.theirs)
    intensity_scale = float(np.max(np.abs(theirs_stokes[..., 0])))
    linear_scale = float(np.max(np.abs(primary.theirs_linear)))
    intensity_absolute = float(
        np.max(np.abs(ours_stokes[..., 0] - theirs_stokes[..., 0]))
    )
    circular_absolute = float(
        np.max(np.abs(ours_stokes[..., 3] + theirs_stokes[..., 3]))
    )
    valid = np.asarray(primary.valid_linear, dtype=np.bool_)
    reference = primary.theirs_linear[valid]
    measured = primary.ours_linear[valid]
    fitted_ratio = np.vdot(reference, measured) / np.vdot(reference, reference)
    global_corrected = primary.ours_linear * np.exp(-1j * np.angle(fitted_ratio))

    raw_absolute = float(
        np.max(np.abs(primary.ours_linear[valid] - primary.theirs_linear[valid]))
    )
    global_absolute = float(
        np.max(np.abs(global_corrected[valid] - primary.theirs_linear[valid]))
    )
    exact_absolute = float(
        np.max(
            np.abs(primary.exact_corrected_linear[valid] - primary.theirs_linear[valid])
        )
    )
    ours_source_stokes = local_stokes(primary.ours_by_source)
    ours_source_linear = ours_source_stokes[..., 1] + 1j * ours_source_stokes[..., 2]
    wrong_sign = np.sum(
        ours_source_linear * np.exp(2j * primary.exact_angles[:, :, None, None]),
        axis=1,
    )
    wrong_sign_absolute = float(
        np.max(np.abs(wrong_sign[valid] - primary.theirs_linear[valid]))
    )
    old_q_compensation = -theirs_stokes[..., 1] + 1j * theirs_stokes[..., 2]
    retired_q_absolute = float(np.max(np.abs(primary.ours_linear - old_q_compensation)))

    unpolarized_scale = float(np.max(np.abs(unpolarized.theirs)))
    unpolarized_absolute = float(np.max(np.abs(unpolarized.ours - unpolarized.theirs)))
    without_fringe = fringe_mapping(unpolarized.theirs)
    no_fringe_absolute = float(np.max(np.abs(unpolarized.ours - without_fringe)))

    ours_additivity_absolute = float(
        np.max(np.abs(np.sum(primary.ours_by_source, axis=1) - primary.ours))
    )
    theirs_additivity_absolute = float(
        np.max(np.abs(np.sum(primary.theirs_by_source, axis=1) - primary.theirs))
    )
    ours_cube_scale = float(np.max(np.abs(primary.ours)))
    theirs_cube_scale = float(np.max(np.abs(primary.theirs)))

    angle_difference = np.abs(primary.public_angles - primary.exact_angles)
    angle_relative = angle_difference / np.abs(primary.exact_angles)
    angle_flat_index = int(np.argmax(angle_relative))
    angle_index = np.unravel_index(angle_flat_index, angle_relative.shape)
    angle_numerator = float(angle_difference[angle_index])
    angle_denominator = float(abs(primary.exact_angles[angle_index]))

    record = {
        "raw_linear_relative": _relative_metric(
            value=raw_absolute / linear_scale,
            numerator_value=raw_absolute,
            numerator_units="Jy",
            denominator_name="linear_scale",
            denominator_value=linear_scale,
            denominator_units="Jy",
            definition="max(abs(L_RS-L_PY))[valid]/linear_scale",
        ),
        "single_global_angle_relative": _relative_metric(
            value=global_absolute / linear_scale,
            numerator_value=global_absolute,
            numerator_units="Jy",
            denominator_name="linear_scale",
            denominator_value=linear_scale,
            denominator_units="Jy",
            definition=(
                "max(abs(L_RS*exp(-1j*arg(fitted_ratio))-L_PY))[valid]/linear_scale"
            ),
        ),
        "exact_source_time_relative": _relative_metric(
            value=exact_absolute / linear_scale,
            numerator_value=exact_absolute,
            numerator_units="Jy",
            denominator_name="linear_scale",
            denominator_value=linear_scale,
            denominator_units="Jy",
            definition=(
                "max(abs(sum_source(L_RS_source*exp(-2j*Delta))-L_PY))"
                "[valid]/linear_scale"
            ),
        ),
        "intensity_relative": _relative_metric(
            value=intensity_absolute / intensity_scale,
            numerator_value=intensity_absolute,
            numerator_units="Jy",
            denominator_name="intensity_scale",
            denominator_value=intensity_scale,
            denominator_units="Jy",
            definition="max(abs(I_RS-I_PY))/intensity_scale",
        ),
        "circular_relative": _relative_metric(
            value=circular_absolute / intensity_scale,
            numerator_value=circular_absolute,
            numerator_units="Jy",
            denominator_name="intensity_scale",
            denominator_value=intensity_scale,
            denominator_units="Jy",
            definition="max(abs(V_RS+V_PY))/intensity_scale",
        ),
        "unpolarized_relative": _relative_metric(
            value=unpolarized_absolute / unpolarized_scale,
            numerator_value=unpolarized_absolute,
            numerator_units="Jy",
            denominator_name="unpolarized_cube_scale",
            denominator_value=unpolarized_scale,
            denominator_units="Jy",
            definition="max(abs(cube_RS-cube_PY))/unpolarized_cube_scale",
        ),
        "retired_q_control": _relative_metric(
            value=retired_q_absolute / linear_scale,
            numerator_value=retired_q_absolute,
            numerator_units="Jy",
            denominator_name="linear_scale",
            denominator_value=linear_scale,
            denominator_units="Jy",
            definition="max(abs(L_RS-(-Q_PY+iU_PY)))/linear_scale",
        ),
        "wrong_sign_control": _relative_metric(
            value=wrong_sign_absolute / linear_scale,
            numerator_value=wrong_sign_absolute,
            numerator_units="Jy",
            denominator_name="linear_scale",
            denominator_value=linear_scale,
            denominator_units="Jy",
            definition=(
                "max(abs(sum_source(L_RS_source*exp(+2j*Delta))-L_PY))"
                "[valid]/linear_scale"
            ),
        ),
        "unpolarized_no_fringe_control": _relative_metric(
            value=no_fringe_absolute / unpolarized_scale,
            numerator_value=no_fringe_absolute,
            numerator_units="Jy",
            denominator_name="unpolarized_cube_scale",
            denominator_value=unpolarized_scale,
            denominator_units="Jy",
            definition=(
                "max(abs(cube_RS-fringe_mapping(cube_PY_mapped)))"
                "/unpolarized_cube_scale"
            ),
        ),
        "intensity_scale": {
            "value": intensity_scale,
            "units": "Jy",
            "definition": "max(abs(I_PY))",
        },
        "linear_scale": {
            "value": linear_scale,
            "units": "Jy",
            "definition": "max(abs(L_PY))",
        },
        "linear_to_intensity_scale": _relative_metric(
            value=linear_scale / intensity_scale,
            numerator_value=linear_scale,
            numerator_units="Jy",
            denominator_name="intensity_scale",
            denominator_value=intensity_scale,
            denominator_units="Jy",
            definition="linear_scale/intensity_scale",
        ),
        "fitted_complex_ratio": {
            "real": float(fitted_ratio.real),
            "imaginary": float(fitted_ratio.imag),
            "modulus": float(abs(fitted_ratio)),
            "phase_rad": float(np.angle(fitted_ratio)),
            "phase_deg": float(np.degrees(np.angle(fitted_ratio))),
            "half_phase_rotation_rad": 0.5 * float(np.angle(fitted_ratio)),
            "half_phase_rotation_deg": 0.5 * float(np.degrees(np.angle(fitted_ratio))),
            "definition": "vdot(L_PY,L_RS)/vdot(L_PY,L_PY) over valid cells",
        },
        "linear_cells": {
            "valid": int(np.count_nonzero(valid)),
            "total": int(valid.size),
            "mask_definition": "abs(L_reference) > linear_scale * 1e-12",
        },
        "public_exact_angle_max_relative": _relative_metric(
            value=angle_numerator / angle_denominator,
            numerator_value=angle_numerator,
            numerator_units="rad",
            denominator_name="exact_angle_at_max_relative_cell",
            denominator_value=angle_denominator,
            denominator_units="rad",
            definition="max(abs(Delta_public-Delta_exact)/abs(Delta_exact))",
        ),
        "source_additivity": {
            "radiosim": _relative_metric(
                value=ours_additivity_absolute / ours_cube_scale,
                numerator_value=ours_additivity_absolute,
                numerator_units="Jy",
                denominator_name="radiosim_cube_scale",
                denominator_value=ours_cube_scale,
                denominator_units="Jy",
                definition="max(abs(sum_source(cube_RS_source)-cube_RS))/cube_scale",
            ),
            "pyuvsim": _relative_metric(
                value=theirs_additivity_absolute / theirs_cube_scale,
                numerator_value=theirs_additivity_absolute,
                numerator_units="Jy",
                denominator_name="pyuvsim_cube_scale",
                denominator_value=theirs_cube_scale,
                denominator_units="Jy",
                definition="max(abs(sum_source(cube_PY_source)-cube_PY))/cube_scale",
            ),
        },
    }

    expected_metrics = primary.metrics
    comparisons = {
        "measured_intensity_relative": record["intensity_relative"]["value"],
        "measured_circular_relative_after_explicit_v_mapping": record[
            "circular_relative"
        ]["value"],
        "measured_linear_relative_direct_q_u": record["raw_linear_relative"]["value"],
        "measured_linear_relative_single_global_angle": record[
            "single_global_angle_relative"
        ]["value"],
        "measured_linear_relative_exact_source_time": record[
            "exact_source_time_relative"
        ]["value"],
        "control_relative_with_wrong_exact_sign": record["wrong_sign_control"]["value"],
        "control_relative_with_retired_q_compensation": record["retired_q_control"][
            "value"
        ],
        "fitted_residual_frame_rotation_deg": record["fitted_complex_ratio"][
            "half_phase_rotation_deg"
        ],
        "fitted_linear_ratio_modulus": record["fitted_complex_ratio"]["modulus"],
        "linear_scale_over_intensity_scale": record["linear_to_intensity_scale"][
            "value"
        ],
        "public_exact_angle_max_relative": record["public_exact_angle_max_relative"][
            "value"
        ],
    }
    for key, measured_value in comparisons.items():
        if not math.isclose(
            measured_value,
            float(expected_metrics[key]),
            rel_tol=0.0,
            abs_tol=5e-15,
        ):
            raise EvidenceError(
                f"structured comparison metric drift for {key}: "
                f"{measured_value} != {expected_metrics[key]}"
            )
    if not math.isclose(
        record["unpolarized_relative"]["value"],
        float(unpolarized.metrics["measured_relative"]),
        rel_tol=0.0,
        abs_tol=5e-15,
    ):
        raise EvidenceError("unpolarized structured metric drift")
    return record


def _assert_fixture_results(
    primary: Any,
    unpolarized: Any,
    unit_beamfits: Path,
) -> None:
    import numpy as np
    from pyuvdata import UVBeam

    fixture = _fixture()

    def require_array(actual: Any, expected: Any, path: str) -> None:
        actual_array = np.asarray(actual)
        expected_array = np.asarray(expected)
        if actual_array.dtype.kind in {"f", "c"} and not np.all(
            np.isfinite(actual_array)
        ):
            raise EvidenceError(f"{path} contains non-finite values")
        if not np.array_equal(actual_array, expected_array):
            raise EvidenceError(
                f"{path} changed: expected {expected_array.tolist()!r}, "
                f"got {actual_array.tolist()!r}"
            )

    def assert_sky(comparison: Any, expected_sources: list[dict[str, Any]]) -> None:
        sky = comparison.sky
        names = [source["name"] for source in expected_sources]
        require_array(sky.name, names, "sky.name")
        require_array(
            sky.skycoord.icrs.ra.deg,
            [source["ra_deg"] for source in expected_sources],
            "sky.icrs.ra_deg",
        )
        require_array(
            sky.skycoord.icrs.dec.deg,
            [source["dec_deg"] for source in expected_sources],
            "sky.icrs.dec_deg",
        )
        require_array(
            sky.freq_array.to_value("Hz"),
            _FREQUENCY_AXIS["order"],
            "sky.freq_array_hz",
        )
        _require_exact(sky.spectral_type, "full", "sky.spectral_type")
        per_source = np.asarray(
            [source["iquv_jy"] for source in expected_sources], dtype=np.float64
        ).T
        expected_stokes = np.repeat(per_source[:, None, :], 3, axis=1)
        require_array(sky.stokes.to_value("Jy"), expected_stokes, "sky.full_stokes_jy")

    def assert_result(
        comparison: Any,
        expected_fixture: dict[str, Any],
        *,
        label: str,
    ) -> None:
        result = comparison.result
        grid = result.time_grid
        require_array(grid.utc_jd1, _EXPECTED_JD1, f"{label}.time_grid.jd1")
        require_array(grid.utc_jd2, _EXPECTED_JD2, f"{label}.time_grid.jd2")
        _require_exact(
            grid.start_time_iso,
            expected_fixture["times"]["start_iso"]["resolved"],
            f"{label}.time_grid.start_time_iso",
        )
        _require_exact(
            grid.cadence_seconds,
            120.0,
            f"{label}.time_grid.cadence_seconds",
        )
        require_array(
            result.frequencies_hz,
            _FREQUENCY_AXIS["order"],
            f"{label}.frequencies_hz",
        )
        require_array(
            result.channel_widths_hz,
            [1000000.0, 1000000.0, 1000000.0],
            f"{label}.channel_widths_hz",
        )
        _require_exact(
            list(result.correlations),
            _POLARIZATION_AXIS["order"],
            f"{label}.correlations",
        )
        _require_exact(
            str(result.polarization_basis),
            "linear_xy",
            f"{label}.polarization_basis",
        )
        pairs = [
            [int(baseline.ant1.number), int(baseline.ant2.number)]
            for baseline in result.selection.baselines
        ]
        _require_exact(pairs, _BASELINE_AXIS["order"], f"{label}.baseline_order")
        selection = result.selection.to_snapshot()
        _require_exact(
            selection["criteria"]["correlations"],
            "cross",
            f"{label}.baseline_selection.correlations",
        )

        instrument = result.instrument.to_snapshot()
        expected_array = expected_fixture["array"]
        _require_exact(
            instrument["source"]["kind"],
            expected_array["source_kind"],
            f"{label}.instrument.source.kind",
        )
        _require_exact(
            instrument["source"]["format"],
            expected_array["format"],
            f"{label}.instrument.source.format",
        )
        resolved_site = {
            "longitude_deg": instrument["location"]["longitude_deg"],
            "latitude_deg": instrument["location"]["latitude_deg"],
            "height_m": instrument["location"]["height_m"],
            "source": instrument["location"]["source"],
        }
        _require_exact(
            resolved_site,
            expected_fixture["site"]["resolved"],
            f"{label}.instrument.location",
        )
        resolved_antennas = [
            {
                "name": antenna["name"],
                "number": antenna["number"],
                "position_enu_m": antenna["position_enu_m"],
                "diameter_m": antenna["diameter_m"],
                "instrument_mount_type": antenna["mount_type"],
            }
            for antenna in instrument["antennas"]
        ]
        _require_exact(
            resolved_antennas,
            expected_fixture["antennas"]["resolved"],
            f"{label}.instrument.antennas",
        )

        receptor_snapshot = result.receptors.to_snapshot()
        expected_receptors = expected_fixture["receptors"]
        _require_exact(
            receptor_snapshot["output_basis"],
            expected_receptors["output_basis"],
            f"{label}.receptors.output_basis",
        )
        _require_exact(
            receptor_snapshot["receptor_sha256"],
            expected_receptors["receptor_sha256"],
            f"{label}.receptors.sha256",
        )
        receptor_rows = [
            {
                "antenna_name": row["antenna_name"],
                "antenna_number": row["antenna_number"],
                "basis": row["basis"],
                "feed_array": row["feed_array"],
                "feed_angle_rad": row["feed_angle_rad"],
                "feed_rotation_rad": row["feed_rotation_rad"],
            }
            for row in receptor_snapshot["receptors"]
        ]
        expected_rows = [
            {
                "antenna_name": f"A{number:03d}",
                "antenna_number": number,
                "basis": expected_receptors["native_basis"],
                "feed_array": expected_receptors["feed_array"],
                "feed_angle_rad": expected_receptors["feed_angle_rad"],
                "feed_rotation_rad": expected_receptors["feed_rotation_rad"],
            }
            for number in range(3)
        ]
        _require_exact(receptor_rows, expected_rows, f"{label}.receptors.rows")

        beam_snapshot = result.beam_state.to_snapshot()
        handlers = beam_snapshot["handlers"]
        if type(handlers) is not list or len(handlers) != 1:
            raise EvidenceError(f"{label}.beam must resolve exactly one handler")
        file_snapshot = handlers[0]["file"]
        expected_beam = expected_fixture["beam"]
        relevant_beam = {
            "beam_type": file_snapshot["beam_type"],
            "data_normalization": file_snapshot["data_normalization"],
            "data_shape": file_snapshot["data_shape"],
            "feed_array": file_snapshot["feed_array"],
            "frequency_count": file_snapshot["frequency_count"],
            "frequency_min_hz": file_snapshot["frequency_min_hz"],
            "frequency_max_hz": file_snapshot["frequency_max_hz"],
            "mount_type": file_snapshot["mount_type"],
            "native_dtype": file_snapshot["native_dtype"],
            "x_orientation": file_snapshot["x_orientation"],
        }
        expected_relevant_beam = {
            "beam_type": expected_beam["beam_type"],
            "data_normalization": expected_beam["data_normalization"],
            "data_shape": expected_beam["data_shape"],
            "feed_array": expected_beam["feed_array"],
            "frequency_count": len(expected_beam["intrinsic_frequencies_hz"]),
            "frequency_min_hz": min(expected_beam["intrinsic_frequencies_hz"]),
            "frequency_max_hz": max(expected_beam["intrinsic_frequencies_hz"]),
            "mount_type": expected_beam["beamfits_mount_type"],
            "native_dtype": expected_beam["native_dtype"],
            "x_orientation": expected_beam["x_orientation"],
        }
        _require_exact(
            relevant_beam,
            expected_relevant_beam,
            f"{label}.beam_state",
        )

    assert_sky(primary, _PRIMARY_SOURCES)
    assert_sky(unpolarized, _UNPOLARIZED_SOURCES)
    assert_result(primary, fixture["primary"], label="primary")
    assert_result(
        unpolarized,
        fixture["controls"]["unpolarized"],
        label="unpolarized",
    )

    primary_jones = primary.result.jones
    _require_exact(
        list(primary_jones["enabled_terms"]),
        ["H", "C", "E", "P"],
        "primary.jones.enabled_terms",
    )
    _require_exact(
        list(primary_jones["chain_order"]),
        ["H", "C", "E", "P"],
        "primary.jones.chain_order",
    )
    _require_exact(
        primary_jones["term_snapshots"]["P"]["enabled"],
        True,
        "primary.jones.P.enabled",
    )
    _require_exact(
        dict(primary_jones["mount_types"]),
        {"0": "alt-az", "1": "alt-az", "2": "alt-az"},
        "primary.jones.mount_types",
    )
    _require_exact(
        len(unpolarized.result.jones), 0, "unpolarized.jones.optional_term_count"
    )

    beam = UVBeam.from_file(str(unit_beamfits))
    require_array(
        beam.freq_array,
        fixture["primary"]["beam"]["intrinsic_frequencies_hz"],
        "unit_beam.freq_array",
    )
    require_array(beam.feed_array, ["x", "y"], "unit_beam.feed_array")
    _require_exact(beam.beam_type, "efield", "unit_beam.beam_type")
    _require_exact(beam.data_normalization, "peak", "unit_beam.data_normalization")
    _require_exact(beam.x_orientation, "east", "unit_beam.x_orientation")
    _require_exact(beam.mount_type, "fixed", "unit_beam.mount_type")
    data = np.asarray(beam.data_array)
    _require_exact(list(data.shape), [2, 2, 4, 5, 8], "unit_beam.data_shape")
    _require_exact(data.dtype.name, "complex128", "unit_beam.data_dtype")
    require_array(data[0, 0], np.ones_like(data[0, 0]), "unit_beam.xx")
    require_array(data[1, 1], np.ones_like(data[1, 1]), "unit_beam.yy")
    require_array(data[0, 1], np.zeros_like(data[0, 1]), "unit_beam.xy")
    require_array(data[1, 0], np.zeros_like(data[1, 0]), "unit_beam.yx")

    expected_shapes = {
        "primary.ours": (primary.ours, (3, 3, 3, 4)),
        "primary.ours_by_source": (primary.ours_by_source, (3, 3, 3, 3, 4)),
        "primary.theirs": (primary.theirs, (3, 3, 3, 4)),
        "primary.theirs_by_source": (primary.theirs_by_source, (3, 3, 3, 3, 4)),
        "primary.exact_rotations": (primary.exact_rotations, (3, 3, 2, 2)),
        "primary.public_angles": (primary.public_angles, (3, 3)),
        "primary.exact_angles": (primary.exact_angles, (3, 3)),
        "primary.ours_linear": (primary.ours_linear, (3, 3, 3)),
        "primary.theirs_linear": (primary.theirs_linear, (3, 3, 3)),
        "primary.theirs_source_linear": (
            primary.theirs_source_linear,
            (3, 3, 3, 3),
        ),
        "primary.exact_corrected_linear": (
            primary.exact_corrected_linear,
            (3, 3, 3),
        ),
        "primary.valid_linear": (primary.valid_linear, (3, 3, 3)),
        "unpolarized.ours": (unpolarized.ours, (3, 3, 3, 4)),
        "unpolarized.theirs": (unpolarized.theirs, (3, 3, 3, 4)),
    }
    for name, (array, expected) in expected_shapes.items():
        if np.asarray(array).shape != expected:
            raise EvidenceError(
                f"fixture shape drift for {name}: "
                f"expected {expected}, got {np.asarray(array).shape}"
            )
        if not np.all(np.isfinite(np.asarray(array))):
            raise EvidenceError(f"fixture returned non-finite values for {name}")
    if np.asarray(primary.valid_linear).dtype != np.dtype(np.bool_):
        raise EvidenceError("primary.valid_linear must be a boolean mask")


def _build_record(approved_source_sha: str) -> dict[str, Any]:
    import numpy as np
    from astropy.utils import iers

    module = _load_crossvalidation_module()
    _assert_runtime_contract(module)
    iers_path = Path(iers.IERS_A_FILE)
    if iers_path.name != "finals2000A.all":
        raise EvidenceError(f"unexpected bundled IERS basename: {iers_path.name}")
    if _sha256(iers_path) != _EXPECTED_IERS_SHA256:
        raise EvidenceError("bundled IERS table bytes drifted")
    if importlib.metadata.version("astropy-iers-data") != "0.2025.8.25.0.36.58":
        raise EvidenceError("astropy-iers-data version drifted")

    table = iers.IERS_A.open(iers.IERS_A_FILE)
    table_class = f"{type(table).__module__}.{type(table).__qualname__}"
    if table_class != "astropy.utils.iers.iers.IERS_A":
        raise EvidenceError(f"unexpected IERS table class: {table_class}")
    with (
        iers.conf.set_temp("auto_download", False),
        iers.earth_orientation_table.set(table),
        tempfile.TemporaryDirectory(prefix="wp6-sci007-") as raw,
    ):
        temporary = Path(raw)
        runtime_iers_path = Path(iers.IERS_A_FILE).resolve()
        if not runtime_iers_path.as_posix().endswith(
            _EXPECTED_IERS_PACKAGE_RELATIVE_PATH
        ):
            raise EvidenceError(
                "bundled IERS path is outside the pinned package location: "
                f"{runtime_iers_path}"
            )
        _assert_returned_iers_table(table, runtime_iers_path, "outer")
        if iers.earth_orientation_table.get() is not table:
            raise EvidenceError("outer bundled IERS_A table was not installed")
        unit_beamfits = module._write_unit_beamfits(temporary / "unit.beamfits")
        primary = module._run_sci007_comparison(temporary, unit_beamfits)
        if iers.earth_orientation_table.get() is not table:
            raise EvidenceError(
                "primary helper leaked or replaced the outer IERS table"
            )
        unpolarized = module._run_unpolarized_comparison(temporary, unit_beamfits)
        if iers.earth_orientation_table.get() is not table:
            raise EvidenceError(
                "unpolarized helper leaked or replaced the outer IERS table"
            )
        _assert_fixture_results(primary, unpolarized, unit_beamfits)
        _assert_returned_iers_table(primary.iers_table, runtime_iers_path, "primary")
        _assert_returned_iers_table(
            unpolarized.iers_table, runtime_iers_path, "unpolarized"
        )
        primary_grid = primary.result.time_grid
        primary_jd1 = np.asarray(primary_grid.utc_jd1, dtype=np.float64)
        primary_jd2 = np.asarray(primary_grid.utc_jd2, dtype=np.float64)
        control_grid = unpolarized.result.time_grid
        control_jd1 = np.asarray(control_grid.utc_jd1, dtype=np.float64)
        control_jd2 = np.asarray(control_grid.utc_jd2, dtype=np.float64)
        samples = _measure_iers_samples(primary.iers_table, primary_jd1, primary_jd2)
        control_samples = _measure_iers_samples(
            unpolarized.iers_table, control_jd1, control_jd2
        )
        _require_exact(
            control_samples,
            samples,
            "unpolarized IERS samples versus primary samples",
        )
        if iers.earth_orientation_table.get() is not table:
            raise EvidenceError("IERS sampling replaced the outer installed table")
    record = {
        "schema": SCHEMA,
        "recorded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slice": SLICE,
        "gating": False,
        "identity": _identity(approved_source_sha),
        "reference": _REFERENCE,
        "runtime": _RUNTIME,
        "iers": _iers_record(samples),
        "fixture": _fixture(),
        "axes": _AXES,
        "equations": _EQUATIONS,
        "correction": _CORRECTION,
        "predictions": _prediction_record(primary.public_angles, primary.exact_angles),
        "history": _HISTORY,
        "tolerances": _TOLERANCES,
        "metrics": _metric_record(primary, unpolarized, module),
        "limits": _LIMITS,
    }
    validate_record(record, approved_source_sha=approved_source_sha)
    return record


def _atomic_write_new(path: Path, data: bytes) -> None:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise EvidenceError(
                f"refusing to overwrite existing artifact: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def generate_artifact(
    *, approved_source_sha: str, output: Path
) -> tuple[dict[str, Any], str]:
    """Generate the SCI-007 record from an approved clean exact source."""
    resolved_output = _preflight_generation(approved_source_sha, output)
    record = _build_record(approved_source_sha)
    serialized = (
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    artifact_sha256 = hashlib.sha256(serialized).hexdigest()
    _postflight_generation(approved_source_sha, resolved_output)
    _atomic_write_new(resolved_output, serialized)
    return record, artifact_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="measure and atomically write clean SCI-007 evidence"
    )
    generate.add_argument("--approved-source-sha", required=True)
    generate.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser(
        "validate", help="validate pinned artifact bytes and complete semantics"
    )
    validate.add_argument("--approved-source-sha", required=True)
    validate.add_argument("--artifact-sha256", required=True)
    validate.add_argument("--input", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "generate":
            record, artifact_sha256 = generate_artifact(
                approved_source_sha=args.approved_source_sha,
                output=args.output,
            )
            summary = {
                "artifact_path": record["identity"]["artifact_path"],
                "artifact_sha256": artifact_sha256,
                "generating_source_sha": record["identity"]["generating_source_sha"],
                "passed": True,
            }
        else:
            record = validate_artifact(
                args.input,
                approved_source_sha=args.approved_source_sha,
                artifact_sha256=args.artifact_sha256,
            )
            summary = {
                "artifact_path": record["identity"]["artifact_path"],
                "artifact_sha256": args.artifact_sha256,
                "generating_source_sha": record["identity"]["generating_source_sha"],
                "passed": True,
            }
    except (EvidenceError, AssertionError, OSError) as exc:
        print(f"SCI-007 evidence error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
