"""Parallel sky-model loading machinery.

Split out of ``operations/factories.py`` (spec item F1) so the factory
module stays focused on single-model construction. This module owns the
executor-selection policy (:func:`recommend_executor_for_loaders`) and the
concurrent loader driver (:func:`load_models_parallel`), plus the failure
container types they raise.

The driver takes its pool size and executor policy from the caller and takes no
default of its own, so no call site can silently inherit a concurrency decision
(``Tier6HybridRuntimePlan.md`` Section 11.2). What it actually did is reported
back in a :class:`LoaderExecutionRecord`.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import pickle
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..registry import loader_registry
from ..support.precision import require_precision

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..containers.model import SkyModel

logger = logging.getLogger(__name__)

#: Prefix that carries one JSON-encoded :class:`LoaderExecutionRecord` through
#: ``SimulationResult.history``, mirroring the established
#: ``RADIOSIM_PROJECTION_JSON=`` convention in :mod:`radiosim.io`.
LOADER_EXECUTION_HISTORY_PREFIX = "RADIOSIM_SKY_LOADER_JSON="

_RECORD_FIELDS = frozenset(
    {"requested_executor", "actual_executor", "max_workers", "degraded_reason"}
)


@dataclass(frozen=True)
class SkyLoadError:
    """Context for one failed sky-loader request."""

    loader_name: str
    kwargs: dict[str, Any]
    exception: Exception
    traceback_text: str


class SkyLoadAggregateError(RuntimeError):
    """Raised when strict parallel sky loading has one or more failures."""

    def __init__(self, failures: list[SkyLoadError]) -> None:
        self.failures = failures
        details = "\n".join(
            f"- {failure.loader_name}: {failure.exception}" for failure in failures
        )
        super().__init__(
            f"load_parallel: {len(failures)} loader(s) failed (strict=True):\n{details}"
        )


class WorkerPolicyError(ValueError):
    """Raised when a requested worker policy cannot be honoured as written."""


@dataclass(frozen=True)
class LoaderExecutionRecord:
    """What the loader driver was asked for and what it actually ran.

    ``requested_executor`` is the policy as configured (including ``"auto"``);
    ``actual_executor`` is the concrete pool class used. ``degraded_reason`` is
    non-``None`` only when an ``"auto"`` request could not use the executor the
    registry recommended -- an *explicit* request is rejected instead of degraded
    (Section 11.2), so a degradation is never silent.
    """

    requested_executor: Literal["auto", "thread", "process"]
    actual_executor: Literal["thread", "process"]
    max_workers: int
    degraded_reason: str | None = None

    def __post_init__(self) -> None:
        if self.requested_executor not in {"auto", "thread", "process"}:
            raise ValueError(
                "requested_executor must be 'auto', 'thread', or 'process'"
            )
        if self.actual_executor not in {"thread", "process"}:
            raise ValueError("actual_executor must be 'thread' or 'process'")
        if type(self.max_workers) is not int or self.max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if self.degraded_reason is not None and type(self.degraded_reason) is not str:
            raise TypeError("degraded_reason must be a string or None")

    def to_snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of every recorded field."""
        return {
            "requested_executor": self.requested_executor,
            "actual_executor": self.actual_executor,
            "max_workers": self.max_workers,
            "degraded_reason": self.degraded_reason,
        }

    def to_history_line(self) -> str:
        """Encode the record as one canonical result-history line."""
        return LOADER_EXECUTION_HISTORY_PREFIX + json.dumps(
            self.to_snapshot(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_history(cls, history: Iterable[str]) -> LoaderExecutionRecord | None:
        """Decode the record from a result history, or ``None`` if absent.

        A malformed or incomplete line decodes to ``None`` rather than raising:
        history is free-form provenance text, so a reader must never fail on it.
        """
        for line in history:
            if not isinstance(line, str):
                continue
            if not line.startswith(LOADER_EXECUTION_HISTORY_PREFIX):
                continue
            try:
                payload = json.loads(line[len(LOADER_EXECUTION_HISTORY_PREFIX) :])
            except (ValueError, TypeError):
                return None
            if not isinstance(payload, dict) or set(payload) != _RECORD_FIELDS:
                return None
            try:
                return cls(**payload)
            except (TypeError, ValueError):
                return None
        return None


# Loader categories whose work holds the GIL during heavy CPU operations
# (PCA reconstruction, pygdsm spectral interpolation, Stokes cube
# allocation, FITS / skyh5 file parsing, synthetic map generation). A
# thread pool would run these effectively serially, so a process pool is
# preferred whenever any requested loader falls into one of these
# categories. ``"catalog"`` loaders (Vizier / CASDA TAP) are I/O-bound and
# release the GIL, so they stay on the thread pool. This is the single
# source of truth for the executor decision (spec items D5 / F7) — there is
# no separate hardcoded loader-name list.
_GIL_BOUND_CATEGORIES: frozenset[str] = frozenset({"diffuse", "synthetic", "file"})


def _run_one_loader(method_name: str, kw: dict, offline: bool) -> SkyModel:
    """Worker entry point reused by both thread and process executors.

    The resolved offline policy is installed *before* the loader callable is
    resolved, so a spawned process starts from the run's policy instead of a
    fresh module default (``Tier6HybridRuntimePlan.md`` Section 16.1).
    """
    from radiosim.utils.network import set_offline_policy

    set_offline_policy(offline)
    return loader_registry.resolve_callable(method_name)(**kw)


def recommend_executor_for_loaders(
    loaders: list[tuple[str, dict[str, Any]]],
) -> Literal["thread", "process"]:
    """Recommend an executor class for :func:`load_models_parallel`.

    Returns ``"process"`` when any requested loader belongs to a GIL-bound
    category (diffuse map regrid / PCA, file parsing, synthetic generation)
    — those run effectively serially under a thread pool because they hold
    the GIL. Returns ``"thread"`` otherwise: catalog loaders (Vizier /
    CASDA TAP) are I/O-bound and benefit from concurrent threads.

    The category is read from the loader registry, so adding a new loader
    needs no change here — its declared ``category`` drives the decision.

    :func:`load_models_parallel` consults this whenever ``executor="auto"``, so
    users do not have to think about executor choice; an explicit ``"thread"`` or
    ``"process"`` policy overrides it.
    """
    for name, _kwargs in loaders:
        try:
            canonical = loader_registry.resolve_name(name)
            definition = loader_registry.definition(canonical)
        except ValueError:
            # Unknown loader; let load_models_parallel surface the error.
            continue
        if definition.category in _GIL_BOUND_CATEGORIES:
            return "process"
    return "thread"


def _pickle_probe(
    loaders: list[tuple[str, dict[str, Any]]],
    precision: PrecisionConfig | None,
) -> tuple[str, str] | None:
    """Return ``(loader_name, reason)`` for the first request that cannot pickle.

    ``None`` means every request can survive ``ProcessPoolExecutor`` IPC.
    """
    for method_name, kwargs in loaders:
        kw = dict(kwargs)
        if precision is not None and "precision" not in kw:
            kw["precision"] = precision
        try:
            pickle.dumps((method_name, kw))
        except (pickle.PicklingError, TypeError, AttributeError) as exc:
            return method_name, str(exc)
    return None


def _explicit_process_rejection(loader_name: str, reason: str) -> str:
    """Return the verbatim Section 18.3 message for an explicit process request."""
    return (
        "execution.sky_loading.executor=process was requested explicitly, but "
        f"loader arguments for {loader_name} cannot be pickled: {reason}. Use "
        "execution.sky_loading.executor=auto to allow a thread fallback, or "
        "thread to force it."
    )


def _resolve_executor_policy(
    loaders: list[tuple[str, dict[str, Any]]],
    precision: PrecisionConfig | None,
    requested: Literal["auto", "thread", "process"],
    max_workers: int,
) -> LoaderExecutionRecord:
    """Decide the concrete executor and record the decision.

    Runs before any pool is created and before any loader is submitted, so an
    unhonourable policy fails with no network or filesystem side effect.
    """
    if requested not in {"auto", "thread", "process"}:
        raise WorkerPolicyError(
            "execution.sky_loading.executor must be 'auto', 'thread', or 'process'"
        )
    if type(max_workers) is not int or max_workers < 1:
        raise WorkerPolicyError(
            "execution.sky_loading.max_workers must be a positive integer"
        )

    actual: Literal["thread", "process"] = (
        recommend_executor_for_loaders(loaders) if requested == "auto" else requested
    )
    degraded_reason: str | None = None

    if actual == "process":
        failure = _pickle_probe(loaders, precision)
        if failure is not None:
            loader_name, reason = failure
            if requested == "process":
                raise WorkerPolicyError(
                    _explicit_process_rejection(loader_name, reason)
                )
            actual = "thread"
            degraded_reason = (
                f"loader arguments for {loader_name} cannot be pickled: {reason}"
            )
            logger.warning(
                "load_models_parallel: executor='auto' recommended a process pool "
                "but loader kwargs failed the pickle check (%s). Falling back to "
                "thread pool.",
                degraded_reason,
            )

    return LoaderExecutionRecord(
        requested_executor=requested,
        actual_executor=actual,
        max_workers=max_workers,
        degraded_reason=degraded_reason,
    )


def load_models_parallel(
    loaders: list[tuple[str, dict[str, Any]]],
    max_workers: int,
    precision: PrecisionConfig | None = None,
    strict: bool = True,
    executor: Literal["auto", "thread", "process"] = "auto",
) -> tuple[list[SkyModel], LoaderExecutionRecord]:
    """Load multiple sky models concurrently under an explicit worker policy.

    Each loader is a (method_name, kwargs) tuple identifying a registered
    loader function. ``max_workers`` has no default: the resolved policy is the
    only source of a pool size (Section 11.2).

    Executor policy
    ---------------
    ``executor="thread"`` uses :class:`ThreadPoolExecutor`, which is appropriate
    for I/O-bound loaders that release the GIL (Vizier / TAP queries, FITS file
    reads). Several built-in loaders are CPU-bound and hold the GIL — notably
    ``diffuse_sky`` (pygdsm regrid + log-poly scaling), ``pyradiosky_file``
    parsing, and large ``healpy.ud_grade`` calls — and run effectively serially
    under a thread pool. ``executor="process"`` dispatches via
    :class:`ProcessPoolExecutor` instead; the loader name and kwargs must be
    picklable.

    ``executor="auto"`` asks :func:`recommend_executor_for_loaders`, which reads
    the registry category of every request. When ``auto`` lands on a process pool
    whose kwargs cannot be pickled the call degrades to threads and the reason is
    recorded in the returned :class:`LoaderExecutionRecord`. When ``"process"``
    was requested *explicitly* the same failure raises
    :class:`WorkerPolicyError`: an explicit request must not silently become
    something else.

    Determinism
    -----------
    Results are placed by request index, never by completion order, so the
    returned list is identical for any pool size or executor.

    Parameters
    ----------
    loaders
        ``(loader_name, kwargs)`` tuples.
    max_workers
        Worker pool size (capped to ``len(loaders)``). Required.
    precision
        Default precision injected into any loader missing one.
    strict
        Raise :class:`SkyLoadAggregateError` if any loader failed.
    executor
        ``"auto"`` (default), ``"thread"``, or ``"process"``. See above.

    Returns
    -------
    tuple
        ``(models, record)`` — the successfully loaded models in request order,
        and the executed worker policy.
    """
    from radiosim.utils.network import offline_policy

    precision = require_precision(precision)
    record = _resolve_executor_policy(loaders, precision, executor, max_workers)
    pool_size = min(len(loaders), record.max_workers) if loaders else 1
    results: list[SkyModel | None] = [None] * len(loaders)
    failures: list[SkyLoadError] = []
    offline = offline_policy()

    pool_cls: type[concurrent.futures.Executor] = (
        concurrent.futures.ProcessPoolExecutor
        if record.actual_executor == "process"
        else concurrent.futures.ThreadPoolExecutor
    )

    with pool_cls(max_workers=pool_size) as pool:
        future_to_loader: dict[concurrent.futures.Future, tuple[int, str]] = {}
        for index, (method_name, kwargs) in enumerate(loaders):
            kw = dict(kwargs)
            if precision is not None and "precision" not in kw:
                kw["precision"] = precision
            f = pool.submit(_run_one_loader, method_name, kw, offline)
            future_to_loader[f] = (index, method_name)

        for future in concurrent.futures.as_completed(future_to_loader):
            index, name = future_to_loader[future]
            try:
                sky = future.result()
                if sky.formats:
                    results[index] = sky
                    n_elements = (
                        sky.n_healpix_pixels
                        if sky.healpix is not None
                        else sky.n_point_sources
                    )
                    logger.info(
                        f"Parallel load complete: {name} ({n_elements:,} sky elements)"
                    )
                else:
                    logger.info(f"Parallel load: {name} returned empty model")
            except Exception as e:
                failures.append(
                    SkyLoadError(
                        loader_name=name,
                        kwargs=loaders[index][1],
                        exception=e,
                        traceback_text=traceback.format_exc(),
                    )
                )
                logger.warning(f"Parallel load failed for {name}: {e}")

    loaded = [sky for sky in results if sky is not None]

    logger.info(
        "load_parallel: %d/%d loaders succeeded (executor=%s, pool=%d)",
        len(loaded),
        len(loaders),
        record.actual_executor,
        pool_size,
    )

    if failures and strict:
        raise SkyLoadAggregateError(failures)

    return loaded, record
