"""Tier 5C contract for the single canonical correlation-coordinate table.

``Tier5ReceptorFeedPlan.md`` Section 20.1 requires exactly one module-level
table, populated from the Section 14.2 rows.  Tier 5C creates it; Tier 5E and
Tier 5F retire the four independent copies (defect D4) in favour of it.  Every
literal below is transcribed from Section 14.2, not imported from RadioSim.
"""

from __future__ import annotations

import pytest

from radiosim.core.polarization_basis import (
    AIPS_CODES_CANONICAL,
    AIPS_CODES_FILE_ORDER,
    CORRELATION_LABELS,
    POLARIZATION_BASES,
    PYUVDATA_FEEDS,
    PYUVDATA_POLARIZATIONS,
    basis_for_correlations,
    parallel_hand_indices,
)

SECTION_14_2 = {
    "linear_xy": {
        "labels": ("XX", "XY", "YX", "YY"),
        "canonical": (-5, -7, -8, -6),
        "file_order": (-5, -6, -7, -8),
        "feeds": ("x", "y"),
        "polarizations": ("xx", "xy", "yx", "yy"),
    },
    "circular_rl": {
        "labels": ("RR", "RL", "LR", "LL"),
        "canonical": (-1, -3, -4, -2),
        "file_order": (-1, -2, -3, -4),
        "feeds": ("r", "l"),
        "polarizations": ("rr", "rl", "lr", "ll"),
    },
}


def test_exactly_two_bases_are_accepted() -> None:
    assert POLARIZATION_BASES == ("linear_xy", "circular_rl")


@pytest.mark.parametrize("basis", ("linear_xy", "circular_rl"))
def test_every_table_reproduces_section_14_2(basis: str) -> None:
    expected = SECTION_14_2[basis]

    assert CORRELATION_LABELS[basis] == expected["labels"]
    assert AIPS_CODES_CANONICAL[basis] == expected["canonical"]
    assert AIPS_CODES_FILE_ORDER[basis] == expected["file_order"]
    assert PYUVDATA_FEEDS[basis] == expected["feeds"]
    assert PYUVDATA_POLARIZATIONS[basis] == expected["polarizations"]


def test_the_linear_row_preserves_the_existing_production_constants() -> None:
    """Section 14.2: the linear row must reproduce today's constants exactly.

    NARROWED BY: Tier 5E.  ``radiosim.io.hdf5`` no longer defines
    ``CORRELATIONS`` or ``AIPS_CODES`` -- it imports this table instead
    (Section 20.1), so the two clauses that compared against those constants
    have no constant left to compare against and were removed rather than
    weakened.

    NARROWED AGAIN BY: Tier 5F, for the same reason and by the same authority:
    ``io/standard_visibility.py``'s ``CANONICAL_CORRELATIONS``,
    ``CANONICAL_CODES``, and ``FILE_CODES`` are gone, so the three surviving
    clauses now assert the pre-Tier-5 literal values directly.  Those values
    are the pinned contract; the removed constants were only one expression of
    them, and `tests/characterization/test_tier5_current_behavior.py` records
    that all four sites now read this table.
    """
    assert CORRELATION_LABELS["linear_xy"] == ("XX", "XY", "YX", "YY")
    assert AIPS_CODES_CANONICAL["linear_xy"] == (-5, -7, -8, -6)
    assert AIPS_CODES_FILE_ORDER["linear_xy"] == (-5, -6, -7, -8)


def test_tables_are_read_only_mappings() -> None:
    for table in (
        CORRELATION_LABELS,
        AIPS_CODES_CANONICAL,
        AIPS_CODES_FILE_ORDER,
        PYUVDATA_FEEDS,
        PYUVDATA_POLARIZATIONS,
    ):
        with pytest.raises(TypeError):
            table["linear_xy"] = ("A", "B", "C", "D")  # type: ignore[index]


@pytest.mark.parametrize("basis", ("linear_xy", "circular_rl"))
def test_basis_for_correlations_round_trips_the_labels(basis: str) -> None:
    assert basis_for_correlations(CORRELATION_LABELS[basis]) == basis


@pytest.mark.parametrize("basis", ("linear_xy", "circular_rl"))
def test_parallel_hand_indices_are_zero_and_three_in_both_bases(basis: str) -> None:
    """Section 14.1: indices 0 and 3 are the parallel hands in both bases."""
    assert parallel_hand_indices(CORRELATION_LABELS[basis]) == (0, 3)


def test_reordered_or_unknown_correlations_are_rejected_by_name() -> None:
    rejected = (
        ("XX", "YY", "XY", "YX"),
        ("RR", "LL", "RL", "LR"),
        ("XX", "XY", "YX"),
        ("XX", "XY", "YX", "RR"),
        (),
    )
    for correlations in rejected:
        with pytest.raises(ValueError) as excinfo:
            basis_for_correlations(correlations)
        message = str(excinfo.value)
        assert "('XX', 'XY', 'YX', 'YY')" in message
        assert "('RR', 'RL', 'LR', 'LL')" in message

        with pytest.raises(ValueError):
            parallel_hand_indices(correlations)


def test_non_tuple_input_is_rejected() -> None:
    for value in (None, "XXXYYXYY", ["XX", "XY", "YX", "YY"]):
        with pytest.raises(TypeError):
            basis_for_correlations(value)  # type: ignore[arg-type]


def test_the_basis_names_agree_with_the_resolved_receptor_set() -> None:
    """Tier 5B already resolves an ``output_basis``; the tokens must match."""
    from radiosim.core.receptor import _OUTPUT_BASIS_BY_NATIVE

    assert set(_OUTPUT_BASIS_BY_NATIVE.values()) == set(POLARIZATION_BASES)


def test_the_table_is_exported_from_radiosim_core() -> None:
    """Section 24 lists both names as public ``radiosim.core`` additions."""
    import radiosim.core as core

    assert core.CORRELATION_LABELS is CORRELATION_LABELS
    assert "PolarizationBasis" in core.__all__
    assert "CORRELATION_LABELS" in core.__all__
