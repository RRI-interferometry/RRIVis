r"""SCI-004 phase-M2 red oracles for the m-mode memory and block schedule.

``docs/development/sci004_mmode_design.md`` Section 9 makes the scheduler part of
the scientific contract rather than an optimization:

    The complete baseline transfer is never materialized.  The deterministic
    scheduler orders frequency, signed-``m``, and baseline blocks, choosing the
    largest block that fits ``working_memory_bytes`` under a
    component-by-component estimate.  It streams/discards each ``B`` block after
    contraction and retains only sky coefficients, auditable per-antenna
    coefficients subject to the same budget, the ``v_m`` cube, and one output
    synthesis block.  It does not inspect free RAM or change block order after an
    allocation failure.

``get_memory_estimate()`` reports seven components separately -- canonical sky
coefficients; quadrature directions/weights and sampled Jones fields; the
optional per-antenna harmonic cache; the largest baseline-transfer block;
retained m-mode visibilities; time-domain output and synthesis assembly; and
backend/native allocations not included in the host estimate -- together with the
logical and scheduled dimensions and a **one-block minimum**.  A budget smaller
than that minimum is rejected *before allocation*.

Section 11 fixes the retained schedule shape.  ``resolved_block_dimensions`` has
exactly ``frequency_block_max``, ``signed_m_block_max``, ``baseline_block_max``,
``packed_value_block_max``, ``scheduled_block_count``, ``schedule_rows`` and
``schedule_sha256``; each row has exactly ``block_index``, ``frequency_start``,
``frequency_stop``, ``signed_m_start``, ``signed_m_stop``, ``baseline_start``,
``baseline_stop`` and ``packed_value_count``; rows are in actual canonical
frequency/signed-``m``/baseline execution order; ``block_index`` is contiguous
from zero; ranges are half-open; each maximum is recomputed from the rows; and
``schedule_sha256`` is
``D("radiosim.sci004.block-schedule.v1", J(schedule_rows))``.  "Missing,
duplicate, reordered, overlapping, or uncovered work is invalid."

Section 9 licenses no speed, scaling, or memory *advantage*: "No speed, scaling,
or memory advantage is claimed without the retained record in Section 11", and
``PERF-001`` governs every performance statement.  Nothing in this module times
anything.

The Section 13.4 owners are ``radiosim.core.mmode.solver`` and
``radiosim.simulator.mmode``; neither the estimate nor the scheduler exists at
``A1``; imports are function-local so each node yields its own Section 14.1
outcome.
"""

from __future__ import annotations

from typing import Any

#: Section 9's seven separately reported estimate components, in its own order.
MEMORY_COMPONENTS: tuple[str, ...] = (
    "canonical_sky_coefficients",
    "quadrature_directions_weights_and_jones",
    "per_antenna_harmonic_cache",
    "largest_baseline_transfer_block",
    "retained_mmode_visibilities",
    "time_domain_output_and_synthesis",
    "backend_native_allocations",
)

#: Section 11's exact ``resolved_block_dimensions`` key set.
BLOCK_DIMENSION_KEYS: tuple[str, ...] = (
    "frequency_block_max",
    "signed_m_block_max",
    "baseline_block_max",
    "packed_value_block_max",
    "scheduled_block_count",
    "schedule_rows",
    "schedule_sha256",
)

#: Section 11's exact schedule-row key set, in order.
SCHEDULE_ROW_KEYS: tuple[str, ...] = (
    "block_index",
    "frequency_start",
    "frequency_stop",
    "signed_m_start",
    "signed_m_stop",
    "baseline_start",
    "baseline_stop",
    "packed_value_count",
)

#: Section 14's domain for the schedule digest.
SCHEDULE_DIGEST_DOMAIN = "radiosim.sci004.block-schedule.v1"

N_BASELINE = 6
N_FREQUENCY = 4
LMAX = 8
MMAX = 6
QUADRATURE_NSIDE = 8

#: Three budgets Section 12.2 family 8 requires: "at least three memory budgets".
BUDGETS: tuple[int, ...] = (1 << 20, 1 << 24, 1 << 30)

_ESTIMATE_FIXTURE = f"""\
n_baselines: {N_BASELINE}
n_frequencies: {N_FREQUENCY}
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
working_memory_bytes: {BUDGETS[-1]}
components:
  - canonical_sky_coefficients
  - quadrature_directions_weights_and_jones
  - per_antenna_harmonic_cache
  - largest_baseline_transfer_block
  - retained_mmode_visibilities
  - time_domain_output_and_synthesis
  - backend_native_allocations
""".encode()

_MINIMUM_FIXTURE = f"""\
n_baselines: {N_BASELINE}
n_frequencies: {N_FREQUENCY}
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
working_memory_bytes: 1
rejected_before_allocation: true
""".encode()

_SCHEDULE_FIXTURE = f"""\
n_baselines: {N_BASELINE}
n_frequencies: {N_FREQUENCY}
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
working_memory_bytes: {BUDGETS[1]}
block_order: [frequency, signed_m, baseline]
""".encode()

_BUDGETS_FIXTURE = f"""\
n_baselines: {N_BASELINE}
n_frequencies: {N_FREQUENCY}
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
working_memory_bytes_set: [{BUDGETS[0]}, {BUDGETS[1]}, {BUDGETS[2]}]
""".encode()

_DIGEST_FIXTURE = f"""\
schedule_digest_domain: {SCHEDULE_DIGEST_DOMAIN}
n_baselines: {N_BASELINE}
n_frequencies: {N_FREQUENCY}
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
working_memory_bytes: {BUDGETS[1]}
""".encode()

_OVERRIDE_FIXTURE = b"""\
capability_cases:
  - {case_kind: method, simulator: mmode, method: get_memory_estimate, overridden: true}
"""

_REGISTRY_ORACLE = (
    "tests/unit/test_simulator/test_sci004_memory.py::"
    "test_the_mmode_registry_entry_and_digest_vocabulary_hold_today"
)

_SOLVER_IMPORT_PATTERN = (
    r"ImportError: cannot import name '\w+' from 'radiosim\.core\.mmode\.solver'"
)


def _local(function: str) -> str:
    return f"tests/unit/test_simulator/test_sci004_memory.py::{function}"


def _case(
    case_id: str,
    requirement_id: str,
    function: str,
    kind: str,
    pattern: str,
    fixture: bytes,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": _local(function),
        "expected_failure_kind": kind,
        "expected_failure_pattern": pattern,
        "fixture_defect_excluded_by": _REGISTRY_ORACLE,
        "fixture_bytes": fixture,
    }


SCI004_PHASE2_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m2.memory.seven-components",
        "sci004.section-9.memory-estimate-reports-seven-components",
        "test_the_memory_estimate_reports_the_seven_section_9_components",
        "missing-symbol",
        _SOLVER_IMPORT_PATTERN,
        _ESTIMATE_FIXTURE,
    ),
    _case(
        "m2.memory.one-block-minimum",
        "sci004.section-9.budget-below-the-one-block-minimum-is-rejected",
        "test_a_budget_below_the_one_block_minimum_is_rejected_before_allocation",
        "missing-symbol",
        _SOLVER_IMPORT_PATTERN,
        _MINIMUM_FIXTURE,
    ),
    _case(
        "m2.memory.deterministic-schedule",
        "sci004.section-11.block-schedule-covers-every-cell-exactly-once",
        "test_the_block_schedule_is_deterministic_and_covers_every_cell_once",
        "missing-symbol",
        _SOLVER_IMPORT_PATTERN,
        _SCHEDULE_FIXTURE,
    ),
    _case(
        "m2.memory.three-budgets",
        "sci004.section-12.2.at-least-three-memory-budgets",
        "test_three_memory_budgets_change_the_schedule_but_not_the_covered_work",
        "missing-symbol",
        _SOLVER_IMPORT_PATTERN,
        _BUDGETS_FIXTURE,
    ),
    _case(
        "m2.memory.schedule-digest-domain",
        "sci004.section-14.0.schedule-digest-uses-its-exact-domain",
        "test_the_schedule_digest_uses_its_exact_section_14_domain",
        "missing-symbol",
        _SOLVER_IMPORT_PATTERN,
        _DIGEST_FIXTURE,
    ),
    _case(
        "m2.memory.simulator-override",
        "sci004.section-9.mmode-overrides-get-memory-estimate",
        "test_the_simulator_overrides_get_memory_estimate_with_the_mmode_components",
        "assertion",
        (
            r"AssertionError: accepted phase M2 overrides "
            r"MModeSimulator\.get_memory_estimate"
        ),
        _OVERRIDE_FIXTURE,
    ),
)

SCI004_PHASE2_RED_GREEN_CONTROLS: tuple[str, ...] = (_REGISTRY_ORACLE,)


# --- green control ------------------------------------------------------------


def test_the_mmode_registry_entry_and_digest_vocabulary_hold_today() -> None:
    """The registry entry and Section 14 digest primitives are sound at ``A1``.

    Every red node below reaches the m-mode strategy through the registry and
    rebuilds a Section 14 digest in its own body; both already work, so a red
    failure is the absence of the estimate and scheduler rather than a broken
    import path or a mis-specified digest domain.
    """
    from radiosim.core.mmode.types import canonical_json, domain_digest
    from radiosim.simulator import get_simulator

    simulator = get_simulator("mmode")
    assert simulator.name == "mmode"
    assert simulator.supports_gpu is False

    rows = [{"block_index": 0, "packed_value_count": 3}]
    digest = domain_digest(SCHEDULE_DIGEST_DOMAIN, canonical_json(rows))
    assert len(digest) == 64
    assert digest == digest.lower()
    # A different domain is a different identity, which is the whole point of
    # Section 14.0's domain separation.
    assert digest != domain_digest(
        "radiosim.mmode-visibility-cube.v1", canonical_json(rows)
    )


# --- Section 9 / 11 red oracles -----------------------------------------------


def test_the_memory_estimate_reports_the_seven_section_9_components() -> None:
    """Section 9: seven components, plus logical/scheduled dimensions."""
    from radiosim.core.mmode.solver import estimate_mmode_memory

    estimate = estimate_mmode_memory(
        n_baselines=N_BASELINE,
        n_frequencies=N_FREQUENCY,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
        working_memory_bytes=BUDGETS[-1],
    )

    assert tuple(estimate.components) == MEMORY_COMPONENTS
    for name in MEMORY_COMPONENTS:
        value = estimate.components[name]
        assert isinstance(value, int) and not isinstance(value, bool), name
        assert value >= 0, name

    assert estimate.logical_dimensions["n_baselines"] == N_BASELINE
    assert estimate.logical_dimensions["n_frequencies"] == N_FREQUENCY
    assert estimate.one_block_minimum_bytes > 0
    assert estimate.one_block_minimum_bytes <= estimate.total_bytes
    assert estimate.total_bytes <= BUDGETS[-1]
    # The complete baseline transfer is never materialized: the retained
    # component is the *largest block*, which must be smaller than the whole.
    assert (
        estimate.components["largest_baseline_transfer_block"]
        < estimate.complete_baseline_transfer_bytes
    )


def test_a_budget_below_the_one_block_minimum_is_rejected_before_allocation() -> None:
    """Section 9: "A budget smaller than that minimum is rejected before allocation"."""
    from radiosim.core.mmode.solver import estimate_mmode_memory

    raised = None
    try:
        estimate_mmode_memory(
            n_baselines=N_BASELINE,
            n_frequencies=N_FREQUENCY,
            lmax=LMAX,
            mmax=MMAX,
            quadrature_nside=QUADRATURE_NSIDE,
            working_memory_bytes=1,
        )
    except ValueError as error:  # pragma: no cover - the red path
        raised = error
    assert raised is not None, "a budget below the one-block minimum is rejected"


def test_the_block_schedule_is_deterministic_and_covers_every_cell_once() -> None:
    """Section 11: contiguous, half-open, non-overlapping, complete coverage."""
    from radiosim.core.mmode.solver import schedule_mmode_blocks

    schedule = schedule_mmode_blocks(
        n_baselines=N_BASELINE,
        n_frequencies=N_FREQUENCY,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
        working_memory_bytes=BUDGETS[1],
    )
    mapping = schedule.as_mapping()

    assert tuple(mapping) == BLOCK_DIMENSION_KEYS
    rows = mapping["schedule_rows"]
    assert isinstance(rows, list) and rows
    assert mapping["scheduled_block_count"] == len(rows)

    covered: set[tuple[int, int, int]] = set()
    for index, row in enumerate(rows):
        assert tuple(row) == SCHEDULE_ROW_KEYS, index
        assert row["block_index"] == index
        for start, stop in (
            (row["frequency_start"], row["frequency_stop"]),
            (row["signed_m_start"], row["signed_m_stop"]),
            (row["baseline_start"], row["baseline_stop"]),
        ):
            assert isinstance(start, int) and isinstance(stop, int)
            assert 0 <= start < stop
        assert row["packed_value_count"] > 0
        for frequency in range(row["frequency_start"], row["frequency_stop"]):
            for signed_m in range(row["signed_m_start"], row["signed_m_stop"]):
                for baseline in range(row["baseline_start"], row["baseline_stop"]):
                    cell = (frequency, signed_m, baseline)
                    assert cell not in covered, cell
                    covered.add(cell)

    assert len(covered) == N_FREQUENCY * (2 * MMAX + 1) * N_BASELINE
    # Each maximum is recomputed from the rows, never carried independently.
    assert mapping["frequency_block_max"] == max(
        row["frequency_stop"] - row["frequency_start"] for row in rows
    )
    assert mapping["signed_m_block_max"] == max(
        row["signed_m_stop"] - row["signed_m_start"] for row in rows
    )
    assert mapping["baseline_block_max"] == max(
        row["baseline_stop"] - row["baseline_start"] for row in rows
    )
    assert mapping["packed_value_block_max"] == max(
        row["packed_value_count"] for row in rows
    )

    # Determinism: the same configuration and budget give the same rows.
    again = schedule_mmode_blocks(
        n_baselines=N_BASELINE,
        n_frequencies=N_FREQUENCY,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
        working_memory_bytes=BUDGETS[1],
    )
    assert again.as_mapping() == mapping


def test_three_memory_budgets_change_the_schedule_but_not_the_covered_work() -> None:
    """Section 12.2 family 8: at least three budgets, one deterministic order."""
    from radiosim.core.mmode.solver import schedule_mmode_blocks

    schedules = [
        schedule_mmode_blocks(
            n_baselines=N_BASELINE,
            n_frequencies=N_FREQUENCY,
            lmax=LMAX,
            mmax=MMAX,
            quadrature_nside=QUADRATURE_NSIDE,
            working_memory_bytes=budget,
        ).as_mapping()
        for budget in BUDGETS
    ]

    expected_cells = N_FREQUENCY * (2 * MMAX + 1) * N_BASELINE
    for mapping in schedules:
        covered = sum(
            (row["frequency_stop"] - row["frequency_start"])
            * (row["signed_m_stop"] - row["signed_m_start"])
            * (row["baseline_stop"] - row["baseline_start"])
            for row in mapping["schedule_rows"]
        )
        assert covered == expected_cells

    # A larger budget takes larger blocks, so the block count is non-increasing.
    counts = [mapping["scheduled_block_count"] for mapping in schedules]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1], "three budgets must not all schedule identically"


def test_the_schedule_digest_uses_its_exact_section_14_domain() -> None:
    """Section 14.0: ``D("radiosim.sci004.block-schedule.v1", J(schedule_rows))``."""
    from radiosim.core.mmode.solver import schedule_mmode_blocks
    from radiosim.core.mmode.types import canonical_json, domain_digest

    mapping = schedule_mmode_blocks(
        n_baselines=N_BASELINE,
        n_frequencies=N_FREQUENCY,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
        working_memory_bytes=BUDGETS[1],
    ).as_mapping()

    expected = domain_digest(
        SCHEDULE_DIGEST_DOMAIN, canonical_json(mapping["schedule_rows"])
    )
    assert mapping["schedule_sha256"] == expected

    # Section 11: schedule order is semantic and may not be sorted, so a
    # reordered row array is a different identity rather than the same one.
    reordered = list(reversed(mapping["schedule_rows"]))
    if len(reordered) > 1:
        assert (
            domain_digest(SCHEDULE_DIGEST_DOMAIN, canonical_json(reordered)) != expected
        )


def test_the_simulator_overrides_get_memory_estimate_with_the_mmode_components() -> (
    None
):
    """Section 9: the m-mode estimate is the solver's own, not the base default.

    The base ``VisibilitySimulator.get_memory_estimate`` reports a direct-RIME
    shape with ``output_bytes``/``working_bytes``/``total_bytes``; Section 9's
    m-mode estimate reports seven named components and a one-block minimum, so
    inheriting the permissive default would misreport the scheduler entirely --
    the same failure mode Section 9 already forbids for ``supports_polarization``.
    """
    from radiosim.simulator import MModeSimulator

    assert "get_memory_estimate" in vars(MModeSimulator), (
        "accepted phase M2 overrides MModeSimulator.get_memory_estimate"
    )
