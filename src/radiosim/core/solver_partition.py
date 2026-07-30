# radiosim/core/solver_partition.py
"""Deterministic time-axis partitioning and worker execution for the solvers.

``Tier6HybridRuntimePlan.md`` Section 11.3 makes the solver's worker policy
parallelize **the time axis only**: each worker computes a contiguous block of
time indices and returns its own per-time ``(B, F, 2, 2)`` blocks, and the
orchestrator concatenates them in time order.  Because the Section 13.3
accumulation restructure already makes each time step produce an independent
block, and because no reduction is repartitioned, the result is bit-identical to
the serial path for any worker count -- a structural property, not a tolerance.

:func:`partition_time_axis` is deliberately separated from any pool so that
Section 27 W6 can test it as a pure function; :func:`execute_time_blocks` is the
one place where a pool is created, so both solvers share exactly one
concurrency implementation.  Beyond the standard library this module imports
only the already-eager resolved-configuration dataclass, so importing
:mod:`radiosim.core` costs nothing new.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from radiosim.core.runtime_config import ResolvedSolverExecutionConfig

__all__ = [
    "SERIAL_SOLVER_EXECUTION",
    "SolverPartitionError",
    "execute_time_blocks",
    "partition_time_axis",
    "require_solver_execution",
    "validate_time_partition",
]

TimeBlock = tuple[int, int]
TimePartition = tuple[TimeBlock, ...]


class SolverPartitionError(RuntimeError):
    """A time partition does not cover ``[0, n_times)`` exactly once.

    This is an internal invariant (``Tier6HybridRuntimePlan.md`` Section 20),
    never user-triggerable: non-positive worker counts are rejected by the
    configuration schema and ``workers`` is clamped to the time-sample count
    during resolution, so a partition built by :func:`partition_time_axis` is
    always valid.  The guard exists so that a future change which breaks the
    decomposition fails loudly instead of silently dropping or duplicating time
    samples.
    """


def _require_count(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value < minimum:
        adjective = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {adjective} integer")
    return value


def partition_time_axis(n_times: int, workers: int) -> TimePartition:
    """Split ``[0, n_times)`` into contiguous blocks, one per effective worker.

    Parameters
    ----------
    n_times : int
        Number of time samples on the canonical grid.  Zero yields an empty
        partition, matching the solver's degenerate-axis guard.
    workers : int
        Requested worker count.  A value greater than ``n_times`` clamps to
        ``n_times``, so every returned block is non-empty; the clamp is
        observable as ``len(partition_time_axis(...))`` and is recorded in the
        resolved configuration by :mod:`radiosim.io.config_resolution`.

    Returns
    -------
    tuple of (int, int)
        ``(start, stop)`` half-open blocks in strictly increasing time order.
        Block sizes differ by at most one, so no worker is handed a share that
        is more than one time step larger than another's.
    """
    n_times = _require_count(n_times, name="n_times", minimum=0)
    workers = _require_count(workers, name="workers", minimum=1)
    if n_times == 0:
        return ()

    block_count = min(workers, n_times)
    base, remainder = divmod(n_times, block_count)
    blocks: list[TimeBlock] = []
    start = 0
    for index in range(block_count):
        stop = start + base + (1 if index < remainder else 0)
        blocks.append((start, stop))
        start = stop
    return validate_time_partition(tuple(blocks), n_times)


def validate_time_partition(blocks: TimePartition, n_times: int) -> TimePartition:
    """Return ``blocks`` unchanged, or raise :class:`SolverPartitionError`.

    The partition must consist of non-empty, contiguous, strictly ordered
    half-open blocks whose union is exactly ``[0, n_times)``.
    """
    expected = 0
    for block in blocks:
        if type(block) is not tuple or len(block) != 2:
            raise SolverPartitionError(
                "the solver time partition must contain (start, stop) pairs; "
                f"got {block!r}"
            )
        start, stop = block
        if type(start) is not int or type(stop) is not int:
            raise SolverPartitionError(
                "the solver time partition must contain exact integer bounds; "
                f"got {block!r}"
            )
        if start != expected or stop <= start:
            raise SolverPartitionError(
                "the solver time partition must cover [0, "
                f"{n_times}) exactly once with contiguous non-empty blocks; "
                f"block {block!r} does not continue from index {expected}"
            )
        expected = stop
    if expected != n_times:
        raise SolverPartitionError(
            "the solver time partition must cover [0, "
            f"{n_times}) exactly once; the blocks end at index {expected}"
        )
    return blocks


SERIAL_SOLVER_EXECUTION = ResolvedSolverExecutionConfig(workers=1, executor="thread")
"""The default solver policy: one worker, i.e. the exact serial path."""


def require_solver_execution(value: object) -> ResolvedSolverExecutionConfig:
    if type(value) is not ResolvedSolverExecutionConfig:
        raise TypeError(
            "solver_execution must be an exact ResolvedSolverExecutionConfig"
        )
    if value.executor != "thread":
        raise ValueError("the solver executor must be 'thread'")
    return value


def execute_time_blocks(
    compute_block: Callable[[int, int], Sequence[Any]],
    *,
    n_times: int,
    solver_execution: ResolvedSolverExecutionConfig,
    thread_name_prefix: str,
) -> list[Any]:
    """Run ``compute_block`` over the time partition and reassemble in order.

    ``compute_block(start, stop)`` must return the per-time output blocks for
    the half-open time range ``[start, stop)``, in time order.  With one
    effective worker the callable runs inline on the calling thread -- the exact
    serial path, with no pool and no thread hop.  With ``N > 1`` each partition
    block is submitted to a :class:`~concurrent.futures.ThreadPoolExecutor` of
    exactly ``N`` threads and the results are concatenated in submission order,
    which is time order, so the assembled sequence does not depend on completion
    order.

    Failure semantics (Section 20): the first failing block in *time* order
    raises, blocks that have not started are cancelled, and nothing partial is
    returned -- the caller never sees a half-filled cube.
    """
    solver_execution = require_solver_execution(solver_execution)
    blocks = partition_time_axis(n_times, solver_execution.workers)
    if not blocks:
        return []
    if len(blocks) == 1:
        start, stop = blocks[0]
        return _checked_block_result(compute_block(start, stop), start, stop)

    assembled: list[Any] = []
    with ThreadPoolExecutor(
        max_workers=len(blocks),
        thread_name_prefix=thread_name_prefix,
    ) as pool:
        futures: list[tuple[TimeBlock, Future[Sequence[Any]]]] = [
            (block, pool.submit(compute_block, block[0], block[1])) for block in blocks
        ]
        try:
            for (start, stop), future in futures:
                assembled.extend(_checked_block_result(future.result(), start, stop))
        except BaseException:
            for _, pending in futures:
                pending.cancel()
            raise
    return assembled


def _checked_block_result(
    produced: Sequence[Any],
    start: int,
    stop: int,
) -> list[Any]:
    """Guard the one invariant ordered reassembly depends on."""
    blocks = list(produced)
    if len(blocks) != stop - start:
        raise SolverPartitionError(
            f"the time block [{start}, {stop}) produced {len(blocks)} output "
            f"blocks instead of {stop - start}"
        )
    return blocks
