"""SCI-004 phase-M3 characterization pins for the new m-mode families.

``docs/development/sci004_mmode_design.md`` Section 11, as narrowed by the
accepted 2026-08-24 accepted-capability-characterization-envelope correction,
names the four families this phase adds:

.. code-block:: text

    mmode_single_scalar_mode
    mmode_point_stokes_i
    mmode_point_full_stokes
    mmode_circular_receptor

"The set names exactly the capability accepted M2 licenses through the public
solve path."  Each family "records the raw cube, ``scientific_sha256``, solver
snapshot, ERA/UTC grid, harmonic index table, and input identity", and "the
family record's grid and input-identity digests use the namespaced domains
``radiosim.sci004.characterization-time.v1`` and
``radiosim.sci004.characterization-input.v1``, computed from the retained
``SimulationResult`` exactly as the strict validator re-derives them; Section
14.0's solver-internal domains do not apply to a result-derived record."

**Why this file was re-cut.** The superseded red slice
``62a7d3d90dcbf0488e8b7c875ae5f95acba007b6`` authored seven families, three of
which the public solve path cannot produce: measured through the public API at
the accepted dimensions, the two HEALPix families returned identically zero
cubes that passed the Section 7.3 gate vacuously under one shared
``scientific_sha256``, the hybrid family silently dropped its diffuse half
while its gate passed, and a ``beams.squint`` fixture failed after ``108.8 s``
with a ``BeamEvaluationError``.  The former ``mmode_nonscalar_east_x``
reproduced ``mmode_point_full_stokes`` byte for byte, because the shipped
default receptor set already *is* east-X.  The correction narrowed the set,
deferred the diffuse, hybrid and non-scalar-beam cases to a future red-sliced
phase, and ruled the two Section 8 rejections this file's rejection oracles
pin.  Its root-cause finding also lives here: the superseded green control
proved only that a configuration *resolves*, which is why a fixture that could
never run passed it, so this file's control runs one family to completion.

**Why the family oracles fail fast.** No production surface assembles a family
record, so each family oracle's first statement imports the absent
``radiosim.core.result.mmode_characterization_record`` and fails there.  Paying
a full m-mode solve to observe an absence that is already decidable would make
the red slice slower without making it truer; the end-to-end control pays that
cost once, deliberately.

**Why the pins are observation sets, not bare digests.** Section 11: "The
initial harvest binds exactly the platform/Python cells this phase's acceptance
actually runs on; every other cell and every newly observed NumPy/OpenBLAS
dispatch class enters afterwards by the standing admission discipline, exactly
as the accepted AVX-512 admissions did."  A digest is therefore never a pin by
itself, and this module requires the pin surface to *be* an observation set.

**Why the family dimensions are the accepted ones.** Section 7.3's every-run
two-tier gate is unforgiving, and its qualified truncation is the accepted
fixture's: the phase-M2 integration fixture records that ``lmax = 16`` "is
pinned by the accepted evidence, not chosen for convenience".  A measurement
taken while authoring the superseded slice confirms that reading -- the same
geometry at ``lmax = mmax = 8``, ``quadrature_nside = 4`` and ``33`` sidereal
samples fails the tier-1a horizon-free shell at ``3.480803e-06`` Jy against its
``2.261455e-08`` Jy limit -- so the family fixtures reuse the accepted ``49``
samples, ``lmax = mmax = 16`` and ``quadrature_nside = 8``.  The three point
families were measured to qualify at those dimensions with four to five orders
of margin on tier 1a and convergence factors between ``5.182`` and ``6.105``
against the ``2.0`` floor; Section 11 requires ``mmode_circular_receptor`` to be
qualified by the same protocol at ``S3`` before it is pinned.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

#: Section 11's four new characterized families, in the order the memo prints
#: them. Production must declare exactly this tuple.
SECTION_11_FAMILIES: tuple[str, ...] = (
    "mmode_single_scalar_mode",
    "mmode_point_stokes_i",
    "mmode_point_full_stokes",
    "mmode_circular_receptor",
)

#: Section 11's exact pre-fix and replacement record orders.
FAMILY_RECORD_V1_KEYS: tuple[str, ...] = (
    "family_id",
    "raw_cube_sha256",
    "scientific_sha256",
    "solver_snapshot",
    "era_utc_grid_sha256",
    "harmonic_index_table_sha256",
    "input_identity_sha256",
)
FAMILY_RECORD_V2_KEYS: tuple[str, ...] = (
    "family_id",
    "raw_cube_sha256",
    "scientific_sha256",
    "solver_snapshot",
    "characterization_time_manifest",
    "era_utc_grid_sha256",
    "harmonic_index_table_sha256",
    "characterization_input_manifest",
    "input_identity_sha256",
)

#: The exact scalar-family layout document, including its final LF.
FINGERPRINT_RED_LAYOUT_BYTES = (
    b"Name Number BeamID E N U Diameter\n"
    b"ANT0 0 0 0.0 0.0 0.0 2.5\n"
    b"ANT1 1 0 4.0 0.0 0.0 2.5\n"
)

#: Section 11's two namespaced characterization domains.
CHARACTERIZATION_TIME_DOMAIN = "radiosim.sci004.characterization-time.v1"
CHARACTERIZATION_INPUT_DOMAIN = "radiosim.sci004.characterization-input.v1"

#: The accepted phase-M2 fixture's qualified truncation and geometry.
FAMILY_SIDEREAL_SAMPLES = 49
FAMILY_LMAX = 16
FAMILY_MMAX = 16
FAMILY_QUADRATURE_NSIDE = 8
FAMILY_BASELINE_EAST_M = 4.0
FAMILY_DISH_DIAMETER_M = 2.5
FAMILY_SOURCE_DEC_DEG = -75.0
FAMILY_FREQUENCY_HZ = 50e6
FAMILY_CHANNEL_WIDTH_HZ = 1e6
FAMILY_WORKING_MEMORY_BYTES = 1 << 30

MMODE_CONVENTION = "radiosim.mmode-forward.v1"
MMODE_FRAME_MODEL = "radiosim.frozen-cirs-rigid-era.v1"
MMODE_HARMONIC_CONVENTION = "radiosim.shaw-polarized-harmonics.v1"
TANGENT_FRAME_SCHEMA = "radiosim.sky-tangent-polarization.v1"

#: Section 5.1's exact six-key canonical tangent-polarization block. A polarized
#: m-mode input is rejected without it, so every full-Stokes family declares it.
CANONICAL_TANGENT_FRAME: dict[str, str] = {
    "schema_version": TANGENT_FRAME_SCHEMA,
    "coordinate_frame": "icrs",
    "axes": "north_east",
    "position_angle": "north_through_east",
    "linear_complex": "q_plus_i_u",
    "stokes_v": "iau_incoming_r_minus_l",
}

#: Section 8's exact ``mmode_public_components`` and ``mmode_public_beam``
#: codes and messages, transcribed from the issue table the accepted
#: correction added. The rejection oracles compare against these literally.
MMODE_PUBLIC_COMPONENTS_CODE = "mmode_public_components"
MMODE_PUBLIC_COMPONENTS_MESSAGE = (
    "execution.simulator='mmode' supports point-source components only in this "
    "phase; a HEALPix-bearing sky requires a future accepted phase."
)
MMODE_PUBLIC_BEAM_CODE = "mmode_public_beam"
MMODE_PUBLIC_BEAM_MESSAGE = (
    "execution.simulator='mmode' supports the scalar beam response only in this "
    "phase; a non-scalar resolved beam system requires a future accepted phase."
)


def _point_source(
    *,
    seed: int,
    num_sources: int,
    polarized: bool,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "kind": "test_sources",
        "representation": "point_sources",
        "num_sources": num_sources,
        "distribution": "uniform",
        "seed": seed,
        "dec_deg": FAMILY_SOURCE_DEC_DEG,
        "dec_range_deg": 0.0,
        "spectral_index": 0.0,
        "polarization_fraction": 0.2 if polarized else 0.0,
        "stokes_v_fraction": 0.1 if polarized else 0.0,
    }
    if polarized:
        source["tangent_polarization_frame"] = CANONICAL_TANGENT_FRAME
    return source


#: Each family's sky payload. The geometry, truncation and time grid are
#: shared, so a family is exactly its input identity.
_FAMILY_SKY: dict[str, dict[str, Any]] = {
    "mmode_single_scalar_mode": {
        "sources": [_point_source(seed=1, num_sources=1, polarized=False)]
    },
    "mmode_point_stokes_i": {
        "sources": [_point_source(seed=2, num_sources=3, polarized=False)]
    },
    "mmode_point_full_stokes": {
        "sources": [_point_source(seed=4, num_sources=3, polarized=True)]
    },
    "mmode_circular_receptor": {
        "sources": [_point_source(seed=4, num_sources=3, polarized=True)]
    },
}

#: ``mmode_circular_receptor`` is "the full-Stokes point fixture under the
#: accepted circular receptor basis": the sky is the full-Stokes one, and the
#: receptor declaration is what distinguishes the family. Every other family
#: takes the shipped default, which is already east-X -- the reason the
#: superseded ``mmode_nonscalar_east_x`` characterized nothing.
_FAMILY_RECEPTORS: dict[str, dict[str, Any]] = {
    "mmode_circular_receptor": {"default": {"basis": "circular"}}
}


def family_mapping(tmp_path: Path, family_id: str) -> dict[str, Any]:
    """Return one family's complete configuration mapping."""
    if family_id not in SECTION_11_FAMILIES:
        raise KeyError(family_id)
    tmp_path.mkdir(parents=True, exist_ok=True)
    layout = tmp_path / f"{family_id}-antennas.txt"
    layout.write_text(
        "Name Number BeamID E N U Diameter\n"
        f"ANT0 0 0 0.0 0.0 0.0 {FAMILY_DISH_DIAMETER_M}\n"
        f"ANT1 1 0 {FAMILY_BASELINE_EAST_M} 0.0 0.0 {FAMILY_DISH_DIAMETER_M}\n",
        encoding="utf-8",
    )
    mapping: dict[str, Any] = {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": str(layout),
                "format": "radiosim",
                "telescope_name": "SCI-004 M3 family array",
            },
            "location": {
                "longitude_deg": 21.4,
                "latitude_deg": -30.7,
                "height_m": 1000.0,
            },
            "default_diameter_m": FAMILY_DISH_DIAMETER_M,
        },
        "baseline_selection": {"correlations": "all"},
        "beams": {
            "mode": "analytic",
            "model": {
                "kind": "circular_aperture",
                "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
            },
        },
        "obs_time": {
            "mode": "full_sidereal",
            "start_time": "2025-01-01T00:00:00",
            "sidereal_samples": FAMILY_SIDEREAL_SAMPLES,
            "integration_fraction": 1.0,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": [FAMILY_FREQUENCY_HZ],
            "channel_widths_hz": [FAMILY_CHANNEL_WIDTH_HZ],
        },
        "sky_model": {"flux_unit": "Jy", **_FAMILY_SKY[family_id]},
        "visibility": {"sky_representation": "point_sources"},
        "execution": {
            "backend": "numpy",
            "offline": True,
            "simulator": "mmode",
            "mmode": {
                "convention": MMODE_CONVENTION,
                "frame_model": MMODE_FRAME_MODEL,
                "harmonic_convention": MMODE_HARMONIC_CONVENTION,
                "lmax": FAMILY_LMAX,
                "mmax": FAMILY_MMAX,
                "quadrature_nside": FAMILY_QUADRATURE_NSIDE,
                "working_memory_bytes": FAMILY_WORKING_MEMORY_BYTES,
            },
        },
    }
    receptors = _FAMILY_RECEPTORS.get(family_id)
    if receptors is not None:
        mapping["receptors"] = receptors
    return mapping


def _family_result_and_phase_input_manifest(
    root: Path,
    family_id: str = "mmode_single_scalar_mode",
    *,
    retain_phase_input_manifest: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Run one family and independently retain its path-free phase preimage."""
    from radiosim.api.simulator import Simulator
    from radiosim.core.result import MMODE_CHARACTERIZATION_INPUT_DOMAIN

    mapping = family_mapping(root, family_id)
    layout = Path(str(mapping["instrument"]["source"]["path"]))
    assert layout.read_bytes() == FINGERPRINT_RED_LAYOUT_BYTES
    simulator = Simulator.from_mapping(mapping, base_dir=root)
    result = simulator.run(progress=False)
    if not retain_phase_input_manifest and MMODE_CHARACTERIZATION_INPUT_DOMAIN == (
        "radiosim.sci004.characterization-input.v1"
    ):
        return result, {}

    from radiosim.core.mmode.solver import build_m1_evidence

    bundle = build_m1_evidence(simulator.build_solve_request())
    manifest = dict(bundle["input_identity_manifest"])
    assert bundle["input_identity_sha256"]
    return result, manifest


def _characterization_record_for_active_domain(
    result: Any,
    phase_input_identity_manifest: dict[str, Any],
    *,
    family_id: str = "mmode_single_scalar_mode",
) -> dict[str, Any]:
    """Call exactly the production signature selected by its domain literal."""
    from radiosim.core.result import (
        MMODE_CHARACTERIZATION_INPUT_DOMAIN,
        mmode_characterization_record,
    )

    if MMODE_CHARACTERIZATION_INPUT_DOMAIN == (
        "radiosim.sci004.characterization-input.v1"
    ):
        return mmode_characterization_record(result, family_id=family_id)
    if MMODE_CHARACTERIZATION_INPUT_DOMAIN == (
        "radiosim.sci004.characterization-input.v2"
    ):
        return mmode_characterization_record(
            result,
            family_id=family_id,
            phase_input_identity_manifest=phase_input_identity_manifest,
        )
    raise AssertionError(
        "unknown SCI-004 characterization input domain "
        f"{MMODE_CHARACTERIZATION_INPUT_DOMAIN!r}"
    )


def _relocated_family_records(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct the scalar family independently under two filesystem roots."""
    from radiosim.core.mmode.types import array_digest

    observations: list[dict[str, Any]] = []
    for label in ("ROOT-A", "ROOT-B"):
        result, manifest = _family_result_and_phase_input_manifest(
            root / label,
            retain_phase_input_manifest=False,
        )
        cube = np.asarray(result.visibilities)
        observations.append(
            {
                "record": _characterization_record_for_active_domain(result, manifest),
                "scientific_sha256": result.scientific_sha256,
                "raw_cube_sha256": array_digest(
                    "radiosim.mmode-visibility-cube.v1",
                    "visibility_cube",
                    ["time", "baseline", "frequency", "correlation"],
                    "Jy",
                    cube,
                    dtype=(
                        "complex64-be" if cube.dtype.itemsize == 8 else "complex128-be"
                    ),
                ),
            }
        )
    return observations[0], observations[1]


def _semantic_layout_mutation(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Change one antenna coordinate at the same layout pathname."""
    result, manifest = _family_result_and_phase_input_manifest(
        root,
        retain_phase_input_manifest=False,
    )
    baseline = _characterization_record_for_active_domain(result, manifest)

    layout = root / "mmode_single_scalar_mode-antennas.txt"
    assert layout.read_bytes() == FINGERPRINT_RED_LAYOUT_BYTES
    layout.write_bytes(
        FINGERPRINT_RED_LAYOUT_BYTES.replace(
            b"ANT1 1 0 4.0 0.0 0.0 2.5\n",
            b"ANT1 1 0 5.0 0.0 0.0 2.5\n",
        )
    )

    from radiosim.api.simulator import Simulator
    from radiosim.core.result import MMODE_CHARACTERIZATION_INPUT_DOMAIN

    mapping = family_mapping(root, "mmode_single_scalar_mode")
    layout.write_bytes(
        FINGERPRINT_RED_LAYOUT_BYTES.replace(
            b"ANT1 1 0 4.0 0.0 0.0 2.5\n",
            b"ANT1 1 0 5.0 0.0 0.0 2.5\n",
        )
    )
    simulator = Simulator.from_mapping(mapping, base_dir=root)
    mutated_result = simulator.run(progress=False)
    mutated_manifest: dict[str, Any] = {}
    if MMODE_CHARACTERIZATION_INPUT_DOMAIN == (
        "radiosim.sci004.characterization-input.v2"
    ):
        from radiosim.core.mmode.solver import build_m1_evidence

        mutated_manifest = dict(
            build_m1_evidence(simulator.build_solve_request())[
                "input_identity_manifest"
            ]
        )
    mutated = _characterization_record_for_active_domain(
        mutated_result,
        mutated_manifest,
    )
    return baseline, mutated


def healpix_bearing_mapping(tmp_path: Path) -> dict[str, Any]:
    """Return a public m-mode configuration whose sky carries a HEALPix payload.

    This is the input Section 8's ``mmode_public_components`` must refuse. The
    superseded slice measured what happens without the guard: the run completes
    in about ``107 s`` and publishes an identically zero cube whose
    ``component_element_counts`` is ``[0]``, which the Section 7.3 gate passes
    vacuously under its exact-zero corner.
    """
    mapping = family_mapping(tmp_path, "mmode_point_stokes_i")
    mapping["visibility"] = {"sky_representation": "healpix_map"}
    mapping["sky_model"] = {
        "flux_unit": "Jy",
        "sources": [
            {
                "kind": "test_sources",
                "representation": "healpix_map",
                "num_sources": 3,
                "distribution": "uniform",
                "seed": 3,
                "nside": 16,
                "dec_deg": FAMILY_SOURCE_DEC_DEG,
                "dec_range_deg": 0.0,
                "spectral_index": 0.0,
                "polarization_fraction": 0.0,
                "stokes_v_fraction": 0.0,
            }
        ],
    }
    return mapping


def non_scalar_beam_mapping(tmp_path: Path) -> dict[str, Any]:
    """Return a public m-mode configuration whose resolved beam is non-scalar.

    ``beams.squint`` is the one analytic non-scalar ``E`` the accepted SCI-005
    Stage-2 subset provides, so it is the input Section 8's ``mmode_public_beam``
    must refuse.  Without the guard the superseded slice measured the run
    failing after ``108.8 s`` deep inside beam evaluation, with an untyped
    ``BeamEvaluationError`` naming a missing ``boresight_parallactic_rad``
    rather than the ruled typed rejection before any work.
    """
    mapping = family_mapping(tmp_path, "mmode_point_stokes_i")
    mapping["beams"] = {
        **mapping["beams"],
        "squint": {
            "default": {
                "convention": "cotton_uson_exact_v1",
                "reference_frequency_hz": FAMILY_FREQUENCY_HZ,
                "per_feed_offset_deg_at_reference": 0.5,
                "mechanical_feed_position_angle_deg": 90.0,
                "positive_native_feed": "x",
            }
        },
    }
    return mapping


#: The exact retained bytes of the family inventory this slice pins.
FAMILY_INVENTORY_BYTES = ("\n".join(SECTION_11_FAMILIES) + "\n").encode("utf-8")

#: The exact retained bytes of the two Section 8 rejection contracts.
REJECTION_CONTRACT_BYTES = (
    f"{MMODE_PUBLIC_COMPONENTS_CODE}: {MMODE_PUBLIC_COMPONENTS_MESSAGE}\n"
    f"{MMODE_PUBLIC_BEAM_CODE}: {MMODE_PUBLIC_BEAM_MESSAGE}\n"
).encode()

_PHASE3_FAMILY_GREEN_CONTROL = (
    "tests/characterization/test_sci004_mmode.py::"
    "test_one_point_family_runs_to_completion_through_the_public_path"
)

#: The accepted corrections granted ``R3`` ``tests/performance/test_sci004_mmode.py``
#: scoped to "the pinned fixture-product, workload-row-key-set, and claims-array
#: literals", so the case rows that *describe* those oracles are declared here,
#: in a file ``R3`` owns unscoped, rather than by adding a phase-M3 case table to
#: a file whose grant covers literals alone.  Both nodes are accepted M2 oracles
#: that later design changes falsified: each pins a superseded ``Section 11``
#: literal by value through an import of ``radiosim.benchmarks``, so both are
#: genuinely red at this observation tree -- where that module still carries the
#: superseded trio, the thirty-three-key row and the five-literal claim array --
#: and both flip only when ``S3`` lands the aligned constants.  Their exclusion
#: control is the PERF-001 surface oracle in the same file, which passes in the
#: same invocation and proves the module imports and the accepted PERF-001
#: inventory is intact.
_PHASE3_PERFORMANCE_NODE = (
    "tests/performance/test_sci004_mmode.py::"
    "test_the_official_v1_inventory_is_the_exact_nine_row_product"
)
_PHASE3_ROW_SCHEMA_NODE = (
    "tests/performance/test_sci004_mmode.py::"
    "test_every_workload_row_carries_its_exact_fields_and_claim_array"
)
_PHASE3_PERFORMANCE_CONTROL = (
    "tests/performance/test_sci004_mmode.py::"
    "test_the_perf001_record_surface_and_non_gating_markers_hold_today"
)

#: The exact retained bytes of the amended Section 11 performance product: the
#: three point-family groups crossed with the three backends.
PERFORMANCE_PRODUCT_BYTES = (
    b"fixtures: mmode_single_scalar_mode, mmode_point_stokes_i, "
    b"mmode_point_full_stokes\n"
    b"backends: numpy, jax, dask\n"
    b"workload_count: 9\n"
    b"sky_representation: point\n"
)

#: The exact retained bytes of the honest-backend-axis row schema: the two row
#: fields the correction added, in their ruled positions, and the sixth claim
#: literal.  ``dense_execution`` is the same literal on every row because the
#: dense path is backend-invariant; ``kernel_backend_block`` is where the two
#: stages Section 9 admits carry their real JAX or Dask measurement.
WORKLOAD_ROW_SCHEMA_BYTES = (
    b"row_key_count: 35\n"
    b"row_key_tail: direct_comparison, backend_comparison, dense_execution, "
    b"kernel_backend_block, claims_not_licensed\n"
    b"dense_execution: numpy_host_v1\n"
    b"kernel_backend_block_status: not_applicable|measured\n"
    b"kernel_stages: per_m_contraction, synthesis\n"
    b"claims_not_licensed: general_speedup, gpu_or_accelerator_support, "
    b"mmode_end_to_end_backend_execution, perf001_evidence_or_closure, "
    b"performance_regression_gate, unmeasured_workloads\n"
)

_FAMILY_IMPORT_PATTERN = (
    r"cannot import name "
    r"'(MMODE_CHARACTERIZATION_FAMILIES|mmode_characterization_record)' "
    r"from 'radiosim\.core\.result'"
)


def _phase3_case(
    case_id: str,
    requirement_id: str,
    function: str,
    *,
    expected_failure_kind: str = "missing-symbol",
    expected_failure_pattern: str = _FAMILY_IMPORT_PATTERN,
    fixture_bytes: bytes = FAMILY_INVENTORY_BYTES,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": f"tests/characterization/test_sci004_mmode.py::{function}",
        "expected_failure_kind": expected_failure_kind,
        "expected_failure_pattern": expected_failure_pattern,
        "fixture_defect_excluded_by": _PHASE3_FAMILY_GREEN_CONTROL,
        "fixture_bytes": fixture_bytes,
    }


SCI004_PHASE3_RED_CASES: tuple[dict[str, object], ...] = (
    _phase3_case(
        "m3.characterization.family-inventory",
        "sci004.section-11.four-new-characterized-families",
        "test_production_declares_exactly_the_four_section_11_families",
    ),
    *(
        _phase3_case(
            f"m3.characterization.family-record.{family_id}",
            "sci004.section-11.family-records-its-six-parts",
            (f"test_every_new_family_records_its_six_section_11_parts[{family_id}]"),
        )
        for family_id in SECTION_11_FAMILIES
    ),
    _phase3_case(
        "m3.characterization.dispatch-class-observation-set",
        "sci004.section-11.ci001-observation-set-not-a-bare-digest",
        "test_a_family_pin_is_a_ci001_observation_set_not_a_bare_digest",
    ),
    _phase3_case(
        "m3.rejection.public-components",
        "sci004.section-8.mmode-public-components-rejection",
        "test_a_healpix_bearing_sky_is_rejected_before_any_solver_work",
        expected_failure_kind="assertion",
        expected_failure_pattern=r"DID NOT RAISE",
        fixture_bytes=REJECTION_CONTRACT_BYTES,
    ),
    _phase3_case(
        "m3.rejection.public-beam",
        "sci004.section-8.mmode-public-beam-rejection",
        "test_a_non_scalar_resolved_beam_system_is_rejected_before_any_solver_work",
        expected_failure_kind="exception",
        expected_failure_pattern=r"boresight_parallactic_rad",
        fixture_bytes=REJECTION_CONTRACT_BYTES,
    ),
    {
        "case_id": "m3.performance.workload-row-schema",
        "requirement_id": (
            "sci004.section-11.workload-row-carries-dense-execution-and-kernel-block"
        ),
        "test_nodeid": _PHASE3_ROW_SCHEMA_NODE,
        "expected_failure_kind": "assertion",
        # Both tuples truncate to the same ``workload_id`` prefix in pytest's
        # abbreviated first line -- the distinguishing "At index 32 diff:
        # 'claims_not_licensed' != 'dense_execution'" sits on a later line the
        # junit message drops -- so the pattern pins what the first line
        # actually carries: the tracked row-key tuple and the amended one
        # disagree, both beginning at ``workload_id``.  The key assertion runs
        # before the claims assertion, so the key set is what this record
        # observed; the claims literal is red for the same reason and lands in
        # the same node.
        "expected_failure_pattern": (
            r"AssertionError: assert \('workload_id.*\) == \('workload_id"
        ),
        "fixture_defect_excluded_by": _PHASE3_PERFORMANCE_CONTROL,
        "fixture_bytes": WORKLOAD_ROW_SCHEMA_BYTES,
    },
    {
        "case_id": "m3.performance.point-family-product",
        "requirement_id": "sci004.section-11.performance-product-is-three-point-groups",
        "test_nodeid": _PHASE3_PERFORMANCE_NODE,
        "expected_failure_kind": "assertion",
        # The observed first line is pytest's abbreviated tuple comparison, so
        # the pattern pins both sides of it by their surviving prefixes: the
        # inventory the benchmark surface still builds starts at
        # ``mmode_poin``\ ted full Stokes, while the amended product this oracle
        # now pins starts at ``mmode_sing``\ le scalar mode.
        "expected_failure_pattern": (
            r"AssertionError: assert \(\('mmode_poin.*\) == \(\('mmode_sing"
        ),
        "fixture_defect_excluded_by": _PHASE3_PERFORMANCE_CONTROL,
        "fixture_bytes": PERFORMANCE_PRODUCT_BYTES,
    },
)

SCI004_PHASE3_RED_GREEN_CONTROLS: tuple[str, ...] = (
    _PHASE3_FAMILY_GREEN_CONTROL,
    _PHASE3_PERFORMANCE_CONTROL,
)


# --- green controls -----------------------------------------------------------


@pytest.mark.parametrize("family_id", SECTION_11_FAMILIES)
def test_every_family_configuration_resolves_to_the_mmode_strategy(
    tmp_path: Path,
    family_id: str,
) -> None:
    """The four family fixtures are valid m-mode inputs at this tip.

    This is necessary and, as the superseded slice proved, nowhere near
    sufficient: three fixtures that could never *run* passed exactly this
    check.  The control the red record cites is the end-to-end one below.
    """
    from radiosim.api.simulator import Simulator

    simulator = Simulator.from_mapping(
        family_mapping(tmp_path, family_id), base_dir=tmp_path
    )

    assert simulator.config.execution.simulator == "mmode"
    assert simulator.config.execution.mmode is not None
    assert simulator.config.execution.mmode.lmax == FAMILY_LMAX
    assert simulator.config.execution.mmode.mmax == FAMILY_MMAX
    assert len(simulator.config.observation.time_grid) == FAMILY_SIDEREAL_SAMPLES


def test_one_point_family_runs_to_completion_through_the_public_path(
    tmp_path: Path,
) -> None:
    """One family runs end to end, so a fixture that cannot run cannot pass.

    This is the red record's fixture-defect exclusion, and it is deliberately
    the expensive kind.  The superseded slice's control stopped at
    configuration resolution, which is why two families that publish an
    identically zero cube and one that silently drops half its sky all passed
    it.  Running a family proves the harness end to end: the solver produces a
    contributing point component, the Section 7.3 gate passes on real numbers
    rather than on the exact-zero corner, and the published cube is finite and
    non-zero.
    """
    from radiosim.api.simulator import Simulator

    result = Simulator.from_mapping(
        family_mapping(tmp_path, "mmode_point_stokes_i"), base_dir=tmp_path
    ).run(progress=False)
    cube = np.asarray(result.visibilities)
    gate = result.solver.direct_gate.as_mapping()

    snapshot = result.solver.as_mapping()

    assert result.solver.solver == "mmode"
    assert result.solver.sky_representation == "point_sources"
    # Read the solved components from the Section 10 snapshot, which is accepted
    # M1/M2 surface. The typed attribute form is phase-M3 production, so a
    # control that used it would fail for the very reason the red oracles exist.
    assert tuple(snapshot["components"]) == ("point",)
    assert all(int(count) > 0 for count in snapshot["component_element_counts"])
    assert cube.shape == (
        FAMILY_SIDEREAL_SAMPLES,
        len(result.selection.baselines),
        1,
        4,
    )
    assert np.all(np.isfinite(cube))
    assert float(np.max(np.abs(cube))) > 0.0
    assert gate["pass"] is True
    # Not the exact-zero corner: a vacuous pin is the defect this control exists
    # to make impossible.
    assert gate["deficit_max_jy"] > 0.0
    assert gate["convergence_factor"] >= 2.0
    assert (
        gate["deficit_max_quarter_jy"]
        > gate["deficit_max_half_jy"]
        > gate["deficit_max_jy"]
    )


# --- Section 11 family oracles ------------------------------------------------


def test_production_declares_exactly_the_four_section_11_families() -> None:
    """Section 11 names four new characterized families, and only four."""
    from radiosim.core.result import (  # noqa: F401
        MMODE_CHARACTERIZATION_FAMILIES,
        mmode_characterization_record,
    )

    assert tuple(MMODE_CHARACTERIZATION_FAMILIES) == SECTION_11_FAMILIES
    assert len(set(MMODE_CHARACTERIZATION_FAMILIES)) == 4


@pytest.mark.parametrize("family_id", SECTION_11_FAMILIES)
def test_every_new_family_records_its_six_section_11_parts(
    tmp_path: Path,
    family_id: str,
) -> None:
    """Section 11: "Each family records the raw cube, ``scientific_sha256``,
    solver snapshot, ERA/UTC grid, harmonic index table, and input identity."

    The record's digests are required to join the result they came from, so a
    family pin cannot quietly describe a different run: the scientific
    fingerprint is the result's own, the snapshot is the exact Section 10
    twenty-key m-mode arm, and the two derived digests carry the namespaced
    characterization domains the accepted correction fixed.
    """
    from radiosim.core.result import (
        MMODE_CHARACTERIZATION_INPUT_DOMAIN,
        MMODE_SOLVER_SNAPSHOT_KEYS,
    )

    result, phase_manifest = _family_result_and_phase_input_manifest(
        tmp_path,
        family_id,
        retain_phase_input_manifest=False,
    )
    record = _characterization_record_for_active_domain(
        result,
        phase_manifest,
        family_id=family_id,
    )

    if MMODE_CHARACTERIZATION_INPUT_DOMAIN == (
        "radiosim.sci004.characterization-input.v1"
    ):
        assert tuple(record) == FAMILY_RECORD_V1_KEYS
        assert "characterization_time_manifest" not in record
        assert "characterization_input_manifest" not in record
    elif MMODE_CHARACTERIZATION_INPUT_DOMAIN == (
        "radiosim.sci004.characterization-input.v2"
    ):
        assert tuple(record) == FAMILY_RECORD_V2_KEYS
        assert (
            record["characterization_input_manifest"]["phase_input_identity_manifest"]
            == phase_manifest
        )
        assert record["characterization_time_manifest"]
    else:
        raise AssertionError(
            "unknown SCI-004 characterization input domain "
            f"{MMODE_CHARACTERIZATION_INPUT_DOMAIN!r}"
        )
    assert record["family_id"] == family_id
    assert record["scientific_sha256"] == result.scientific_sha256
    snapshot = record["solver_snapshot"]
    assert tuple(snapshot) == MMODE_SOLVER_SNAPSHOT_KEYS
    assert snapshot["solver"] == "mmode"
    assert snapshot["lmax"] == FAMILY_LMAX
    assert snapshot["mmax"] == FAMILY_MMAX
    assert snapshot["sidereal_samples"] == FAMILY_SIDEREAL_SAMPLES
    for field in (
        "raw_cube_sha256",
        "era_utc_grid_sha256",
        "harmonic_index_table_sha256",
        "input_identity_sha256",
    ):
        value = record[field]
        assert isinstance(value, str) and len(value) == 64, field
        assert all(character in "0123456789abcdef" for character in value), field
    # A family record is deterministic: the same result yields the same pin.
    assert (
        _characterization_record_for_active_domain(
            result,
            phase_manifest,
            family_id=family_id,
        )
        == record
    )


def test_characterization_input_preimage_is_retained_and_reconstructible(
    tmp_path: Path,
) -> None:
    """The v2 row retains the complete path-free preimage beside its digest."""
    from radiosim.core.result import MMODE_CHARACTERIZATION_INPUT_DOMAIN

    result, phase_manifest = _family_result_and_phase_input_manifest(tmp_path)
    record = _characterization_record_for_active_domain(result, phase_manifest)
    assert (
        MMODE_CHARACTERIZATION_INPUT_DOMAIN
        == "radiosim.sci004.characterization-input.v2"
        and "characterization_input_manifest" in record
    ), "characterization input manifest is absent from the family record"

    manifest = record["characterization_input_manifest"]
    instrument = {
        "schema_version": "radiosim.sci004.characterization-instrument.v1",
        "site_manifest": phase_manifest["site_manifest"],
        "antenna_rows": phase_manifest["antenna_rows"],
        "baseline_rows": phase_manifest["baseline_rows"],
    }
    receptors = {
        "schema_version": "radiosim.sci004.characterization-receptors.v1",
        "receptor_rows": phase_manifest["receptor_rows"],
    }
    loaded_beam = {
        "schema_version": "radiosim.sci004.characterization-loaded-beam.v1",
        "beam_rows": phase_manifest["beam_rows"],
    }
    phase_payload = json.dumps(
        phase_manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    expected = {
        "schema_version": "radiosim.sci004.characterization-input.v2",
        "family_id": "mmode_single_scalar_mode",
        "fixture_id": "mmode_single_scalar_mode",
        "phase_input_identity_manifest": phase_manifest,
        "phase_input_identity_sha256": hashlib.sha256(
            b"radiosim.mmode-input-identity.v1\x00"
            + len(phase_payload).to_bytes(8, "big")
            + phase_payload
        ).hexdigest(),
        "instrument_manifest": instrument,
        "instrument_sha256": "",
        "receptor_manifest": receptors,
        "receptor_sha256": "",
        "loaded_beam_manifest": loaded_beam,
        "beam_loaded_fingerprint": "",
        "correlations": list(result.correlations),
        "polarization_basis": str(result.polarization_basis),
        "frequencies_hz_f64be": [
            row["center_hz_f64be"] for row in phase_manifest["frequency_rows"]
        ],
    }
    for manifest_key, digest_key, domain in (
        (
            "instrument_manifest",
            "instrument_sha256",
            "radiosim.sci004.characterization-instrument.v1",
        ),
        (
            "receptor_manifest",
            "receptor_sha256",
            "radiosim.sci004.characterization-receptors.v1",
        ),
        (
            "loaded_beam_manifest",
            "beam_loaded_fingerprint",
            "radiosim.sci004.characterization-loaded-beam.v1",
        ),
    ):
        subpayload = json.dumps(
            expected[manifest_key],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        expected[digest_key] = hashlib.sha256(
            domain.encode("ascii")
            + b"\x00"
            + len(subpayload).to_bytes(8, "big")
            + subpayload
        ).hexdigest()
    assert manifest == expected
    payload = json.dumps(
        expected,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert (
        record["input_identity_sha256"]
        == hashlib.sha256(
            b"radiosim.sci004.characterization-input.v2\x00"
            + len(payload).to_bytes(8, "big")
            + payload
        ).hexdigest()
    )
    forbidden_suffixes = (
        "path",
        "_path",
        "_paths",
        "_file",
        "_files",
        "_dir",
        "_directory",
        "_root",
        "_uri",
    )
    pending: list[Any] = [manifest]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            assert not any(
                str(key).lower() == "path"
                or str(key).lower().endswith(forbidden_suffixes[1:])
                for key in value
            )
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def test_characterization_input_identity_is_equal_under_distinct_layout_roots(
    tmp_path: Path,
) -> None:
    """Filesystem relocation is not a scientific characterization input."""
    first, second = _relocated_family_records(tmp_path)

    assert first["scientific_sha256"] == second["scientific_sha256"]
    assert first["raw_cube_sha256"] == second["raw_cube_sha256"]
    assert (
        first["record"]["era_utc_grid_sha256"]
        == second["record"]["era_utc_grid_sha256"]
    )
    assert (
        first["record"]["input_identity_sha256"]
        == second["record"]["input_identity_sha256"]
    ), "characterization input identity changed under filesystem relocation"


def test_distinct_layout_roots_preserve_scientific_and_cube_identities(
    tmp_path: Path,
) -> None:
    """The relocation fixture preserves independently derived result science."""
    first, second = _relocated_family_records(tmp_path)

    assert first["scientific_sha256"] == second["scientific_sha256"]
    assert first["raw_cube_sha256"] == second["raw_cube_sha256"]


def test_characterization_input_identity_changes_for_semantic_instrument_content(
    tmp_path: Path,
) -> None:
    """A real antenna-coordinate change remains characterization-significant."""
    baseline, mutated = _semantic_layout_mutation(tmp_path)

    assert baseline["input_identity_sha256"] != mutated["input_identity_sha256"]


def test_a_family_pin_is_a_ci001_observation_set_not_a_bare_digest() -> None:
    """Section 11: "The initial harvest binds exactly the platform/Python cells
    this phase's acceptance actually runs on; every other cell and every newly
    observed NumPy/OpenBLAS dispatch class enters afterwards by the standing
    admission discipline."

    A single-machine digest is therefore not a pin, and the surface that holds
    the pins must say so in its own shape: an observation set keyed by
    environment cell, never one string.  The two namespaced domains are checked
    here too, because a record whose derived digests silently used a
    solver-internal domain would join nothing the validator re-derives.
    """
    from radiosim.core.result import (
        MMODE_CHARACTERIZATION_FAMILIES,
        MMODE_CHARACTERIZATION_INPUT_DOMAIN,
        MMODE_CHARACTERIZATION_TIME_DOMAIN,
        mmode_characterization_observation_set,
    )

    assert MMODE_CHARACTERIZATION_TIME_DOMAIN == CHARACTERIZATION_TIME_DOMAIN
    assert MMODE_CHARACTERIZATION_INPUT_DOMAIN == CHARACTERIZATION_INPUT_DOMAIN
    for family_id in MMODE_CHARACTERIZATION_FAMILIES:
        observations = mmode_characterization_observation_set(family_id)
        assert isinstance(observations, dict)
        assert observations, family_id
        for cell, digests in observations.items():
            assert isinstance(cell, str) and cell
            assert isinstance(digests, tuple)
            assert digests, cell
            for digest in digests:
                assert isinstance(digest, str) and len(digest) == 64


# --- Section 8 public-path rejection oracles ----------------------------------


def test_a_healpix_bearing_sky_is_rejected_before_any_solver_work(
    tmp_path: Path,
) -> None:
    """Section 8: ``mmode_public_components``, with its exact message.

    Section 11's deferral paragraph is the reason this rejection exists rather
    than a wiring: "the accepted harmonic machinery reaches the public path only
    through a future red-sliced phase, the public path rejects a HEALPix-bearing
    payload and a non-scalar resolved beam system with the Section 8 typed
    issues before any work".  Today no guard exists, so the run completes and
    publishes the vacuous zero cube that made the superseded HEALPix families
    pass their gate while characterizing nothing.
    """
    from radiosim.api.simulator import Simulator
    from radiosim.io.config_resolution import UnsupportedConfigError

    with pytest.raises(UnsupportedConfigError) as raised:
        Simulator.from_mapping(
            healpix_bearing_mapping(tmp_path), base_dir=tmp_path
        ).run(progress=False)

    codes = [issue.code for issue in raised.value.issues]
    messages = [issue.message for issue in raised.value.issues]
    assert MMODE_PUBLIC_COMPONENTS_CODE in codes
    assert MMODE_PUBLIC_COMPONENTS_MESSAGE in messages


def test_a_non_scalar_resolved_beam_system_is_rejected_before_any_solver_work(
    tmp_path: Path,
) -> None:
    """Section 8: ``mmode_public_beam``, with its exact message.

    Without the guard the failure is an untyped ``BeamEvaluationError`` raised
    deep inside beam evaluation after the whole frame and transfer stage has
    run -- the opposite of "before any solver work" -- which is what this oracle
    records today.
    """
    from radiosim.api.simulator import Simulator
    from radiosim.io.config_resolution import UnsupportedConfigError

    with pytest.raises(UnsupportedConfigError) as raised:
        Simulator.from_mapping(
            non_scalar_beam_mapping(tmp_path), base_dir=tmp_path
        ).run(progress=False)

    codes = [issue.code for issue in raised.value.issues]
    messages = [issue.message for issue in raised.value.issues]
    assert MMODE_PUBLIC_BEAM_CODE in codes
    assert MMODE_PUBLIC_BEAM_MESSAGE in messages
