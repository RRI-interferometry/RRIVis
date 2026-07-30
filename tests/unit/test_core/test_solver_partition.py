"""The deterministic solver time-axis partition (Tier 6E).

``Tier6HybridRuntimePlan.md`` Section 11.3 requires the solver's worker
decomposition to be "an explicit deterministic partition function
(``n_times``, ``workers``) -> tuple of ``(start, stop)`` pairs, which is
unit-tested independently of any pool".  Section 27 W6 is the row this module
satisfies: the partition covers ``[0, n_times)`` exactly once for every swept
``(n_times, workers)`` pair, and ``workers > n_times`` clamps.

``SolverPartitionError`` (Section 20) is an internal invariant guard, never
user-triggerable: the schema rejects non-positive worker counts long before a
partition is built, and the resolver clamps ``workers`` to the time-sample
count.  It is exercised here directly through the validator so the invariant is
proven non-vacuous rather than assumed.
"""

from __future__ import annotations

import pytest

from radiosim.core.solver_partition import (
    SolverPartitionError,
    partition_time_axis,
    validate_time_partition,
)


class TestPartitionCoverage:
    """Section 27 W6 -- exact, contiguous, ordered coverage."""

    def test_the_partition_covers_every_time_index_exactly_once(self) -> None:
        """The swept range Section 27 W6 asks for, in one assertion loop."""
        for n_times in range(1, 33):
            for workers in (1, 2, 3, 4, 5, 7, 8, 16, 64):
                blocks = partition_time_axis(n_times, workers)
                covered = [
                    index for start, stop in blocks for index in range(start, stop)
                ]
                assert covered == list(range(n_times)), (n_times, workers, blocks)

    def test_every_block_is_contiguous_nonempty_and_in_time_order(self) -> None:
        for n_times in range(1, 33):
            for workers in (1, 2, 3, 4, 5, 7, 8, 16, 64):
                blocks = partition_time_axis(n_times, workers)
                assert all(stop > start for start, stop in blocks)
                assert blocks[0][0] == 0
                assert blocks[-1][1] == n_times
                assert all(
                    previous[1] == following[0]
                    for previous, following in zip(blocks, blocks[1:], strict=False)
                )

    def test_the_block_count_is_the_clamped_worker_count(self) -> None:
        """``workers`` greater than the time-sample count clamps (Section 11.3)."""
        for n_times in range(1, 33):
            for workers in (1, 2, 3, 4, 5, 7, 8, 16, 64):
                blocks = partition_time_axis(n_times, workers)
                assert len(blocks) == min(workers, n_times), (n_times, workers)

    def test_block_sizes_differ_by_at_most_one(self) -> None:
        """A balanced partition keeps every worker's share within one time step."""
        for n_times in range(1, 33):
            for workers in (1, 2, 3, 4, 5, 7, 8, 16, 64):
                sizes = [
                    stop - start
                    for start, stop in partition_time_axis(n_times, workers)
                ]
                assert max(sizes) - min(sizes) <= 1, (n_times, workers, sizes)

    def test_the_partition_is_a_pure_deterministic_tuple(self) -> None:
        first = partition_time_axis(17, 4)
        second = partition_time_axis(17, 4)

        assert first == second
        assert type(first) is tuple
        assert all(type(block) is tuple and len(block) == 2 for block in first)

    def test_an_empty_time_axis_partitions_into_nothing(self) -> None:
        assert partition_time_axis(0, 4) == ()
        assert partition_time_axis(0, 1) == ()

    def test_one_worker_is_exactly_one_block(self) -> None:
        assert partition_time_axis(6, 1) == ((0, 6),)


class TestPartitionRejections:
    """Argument rejection is typed, and never a silent coercion."""

    @pytest.mark.parametrize("workers", [0, -1, -8])
    def test_a_non_positive_worker_count_is_rejected(self, workers: int) -> None:
        with pytest.raises(ValueError, match="workers must be a positive integer"):
            partition_time_axis(4, workers)

    def test_a_negative_time_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_times must be a non-negative integer"):
            partition_time_axis(-1, 2)

    @pytest.mark.parametrize("value", [1.0, True, "4", None])
    def test_non_integer_arguments_are_rejected(self, value: object) -> None:
        with pytest.raises(TypeError):
            partition_time_axis(value, 2)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            partition_time_axis(4, value)  # type: ignore[arg-type]


class TestPartitionInvariantGuard:
    """Section 20 -- ``SolverPartitionError`` is a real, non-vacuous guard."""

    def test_the_error_is_a_runtime_error_subclass(self) -> None:
        assert issubclass(SolverPartitionError, RuntimeError)

    def test_a_valid_partition_validates_silently(self) -> None:
        blocks = partition_time_axis(11, 3)

        assert validate_time_partition(blocks, 11) is blocks

    @pytest.mark.parametrize(
        "blocks",
        [
            ((0, 2), (3, 5)),  # gap
            ((0, 3), (2, 5)),  # overlap
            ((0, 4),),  # short of n_times
            ((0, 6),),  # past n_times
            ((0, 0), (0, 5)),  # empty block
            ((2, 5), (0, 2)),  # out of order
        ],
    )
    def test_a_partition_that_does_not_cover_the_axis_once_is_rejected(
        self,
        blocks: tuple[tuple[int, int], ...],
    ) -> None:
        with pytest.raises(SolverPartitionError, match="time partition"):
            validate_time_partition(blocks, 5)

    def test_the_error_is_exported_from_the_core_package(self) -> None:
        import radiosim.core as core

        assert core.SolverPartitionError is SolverPartitionError
        assert "SolverPartitionError" in core.__all__
