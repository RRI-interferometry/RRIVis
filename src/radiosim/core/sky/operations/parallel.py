"""Parallel sky-model loading machinery.

Split out of ``operations/factories.py`` (spec item F1) so the factory
module stays focused on single-model construction. This module owns the
executor-selection policy (:func:`recommend_executor_for_loaders`) and the
concurrent loader driver (:func:`load_models_parallel`), plus the failure
container types they raise.
"""

from __future__ import annotations

import concurrent.futures
import logging
import pickle
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..registry import loader_registry
from ..support.precision import require_precision

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..containers.model import SkyModel

logger = logging.getLogger(__name__)


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


def _run_one_loader(method_name: str, kw: dict) -> SkyModel:
    """Worker entry point reused by both thread and process executors."""
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

    The simulator wires this into :func:`load_models_parallel` so users do
    not have to think about executor choice; explicit overrides are still
    honoured.
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


def _kwargs_picklable(
    loaders: list[tuple[str, dict[str, Any]]],
    precision: PrecisionConfig | None,
) -> bool:
    """Return True if every loader request can survive ProcessPoolExecutor IPC."""
    for method_name, kwargs in loaders:
        kw = dict(kwargs)
        if precision is not None and "precision" not in kw:
            kw["precision"] = precision
        try:
            pickle.dumps((method_name, kw))
        except (pickle.PicklingError, TypeError, AttributeError):
            return False
    return True


def load_models_parallel(
    loaders: list[tuple[str, dict[str, Any]]],
    max_workers: int = 8,
    precision: PrecisionConfig | None = None,
    strict: bool = True,
    executor: Literal["thread", "process"] = "thread",
) -> list[SkyModel]:
    """Load multiple sky models in parallel.

    Each loader is a (method_name, kwargs) tuple identifying a registered
    loader function.

    Performance notes
    -----------------
    The default ``executor="thread"`` uses :class:`ThreadPoolExecutor`,
    which is appropriate for I/O-bound loaders that release the GIL
    (Vizier / TAP queries, FITS file reads). Several built-in loaders are
    CPU-bound and hold the GIL — notably ``diffuse_sky`` (pygdsm regrid +
    log-poly scaling), ``pyradiosky_file`` parsing, and large
    ``healpy.ud_grade`` calls. Those will run effectively serially under
    the thread pool.

    Pass ``executor="process"`` to dispatch via
    :class:`ProcessPoolExecutor` instead. The loader name and kwargs must
    be picklable; this is true for the current loader registry (kwargs
    are dicts of primitives, numpy arrays, and ``PrecisionConfig``).  If
    a kwarg fails the pickle check, the call falls back to threads with a
    warning rather than raising.

    Parameters
    ----------
    loaders
        ``(loader_name, kwargs)`` tuples.
    max_workers
        Worker pool size (capped to ``len(loaders)``).
    precision
        Default precision injected into any loader missing one.
    strict
        Raise :class:`SkyLoadAggregateError` if any loader failed.
    executor
        ``"thread"`` (default) or ``"process"``. See above.
    """
    precision = require_precision(precision)
    n = min(len(loaders), max_workers)
    results: list[SkyModel | None] = [None] * len(loaders)
    failures: list[SkyLoadError] = []

    pool_cls: type[concurrent.futures.Executor]
    if executor == "process":
        if not _kwargs_picklable(loaders, precision):
            logger.warning(
                "load_models_parallel: requested executor='process' but loader "
                "kwargs failed the pickle check. Falling back to thread pool."
            )
            pool_cls = concurrent.futures.ThreadPoolExecutor
        else:
            pool_cls = concurrent.futures.ProcessPoolExecutor
    else:
        pool_cls = concurrent.futures.ThreadPoolExecutor

    with pool_cls(max_workers=n) as pool:
        future_to_loader: dict[concurrent.futures.Future, tuple[int, str]] = {}
        for index, (method_name, kwargs) in enumerate(loaders):
            kw = dict(kwargs)
            if precision is not None and "precision" not in kw:
                kw["precision"] = precision
            f = pool.submit(_run_one_loader, method_name, kw)
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

    logger.info(f"load_parallel: {len(loaded)}/{len(loaders)} loaders succeeded")

    if failures and strict:
        raise SkyLoadAggregateError(failures)

    return loaded
