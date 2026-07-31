"""NumPy array backend with optional Dask arrays and an optional Dask client.

This backend executes **NumPy** operations. When ``use_dask_arrays=True`` it
wraps them in Dask arrays, which delegate back to the same NumPy kernels, so a
Dask run is bit-identical to a NumPy run of the same workload.

It was called ``NumbaBackend`` before Tier 6H. That name was wrong in a way
worth recording: the class imported ``numba.jit``/``numba.prange``, advertised
"JIT and parallel loops", and exposed a ``jit_compile()`` helper, but it never
compiled a single kernel of its own, ``prange`` was never called, and its
``mode="gpu"`` path validated a CUDA device and then ran NumPy anyway. The
rename removes the claim; **it adds no capability, and none was lost.** Numba
itself remains a declared dependency because PySM needs it, not because
RadioSim computes with it (``Tier6HybridRuntimePlan.md`` Sections 14.1, 14.2).

Usage:
    >>> from radiosim.backends import get_backend
    >>> backend = get_backend("dask")
    >>> backend.name
    'dask-cpu'

With precision control:
    >>> from radiosim.core.precision import PrecisionConfig
    >>> backend = get_backend("dask", precision="fast")

The strict high-level resolver rejects incompatible Dask/float128 requests
before constructing this lower-level backend.
"""

from typing import TYPE_CHECKING, Any, Union

import numpy as np

from radiosim.backends.base import ArrayBackend, BackendNotAvailableError

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

try:
    import dask
    import dask.array as da
    from dask.distributed import Client, LocalCluster

    DASK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without Dask installed
    dask = None
    da = None
    Client = None
    LocalCluster = None
    DASK_AVAILABLE = False


#: The modes this backend accepts. ``"gpu"`` was removed in Tier 6H: it
#: validated a CUDA device and then executed NumPy, which made every
#: ``actual_backend`` provenance value it produced misleading.
DASK_MODES = ("cpu", "distributed")


class DaskBackend(ArrayBackend):
    """NumPy array backend with optional Dask arrays and client management.

    This backend provides:

    - NumPy array operations (always)
    - optional Dask arrays and Dask client/cluster management
    - precision control (float32/float64 only)

    It compiles nothing. Its arrays are NumPy arrays, or Dask arrays whose
    chunks are NumPy arrays, so results are bit-identical to the NumPy backend.

    Modes:

    - ``'cpu'``: local NumPy operations, with an optional local Dask cluster
      when ``n_workers`` is given
    - ``'distributed'``: Dask client; set ``use_dask_arrays=True`` for Dask
      arrays

    Example:
        >>> # CPU mode (default)
        >>> backend = DaskBackend(mode="cpu")

        >>> # Distributed mode
        >>> backend = DaskBackend(mode="distributed", n_workers=8, use_dask_arrays=True)

        >>> # With precision control
        >>> from radiosim.core.precision import PrecisionConfig
        >>> backend = DaskBackend(precision="fast")
    """

    def __init__(
        self,
        mode: str = "cpu",
        n_workers: int | None = None,
        threads_per_worker: int = 1,
        scheduler_address: str | None = None,
        use_dask_arrays: bool = False,
        precision: Union["PrecisionConfig", str] | None = None,
    ):
        """Initialize the NumPy/Dask backend.

        Parameters
        ----------
        mode : str
            Execution mode, ``'cpu'`` or ``'distributed'``.
        n_workers : int, optional
            Number of Dask workers (default: auto-detect).
        threads_per_worker : int
            Threads per Dask worker.
        scheduler_address : str, optional
            Dask scheduler address for a remote cluster.
        use_dask_arrays : bool
            Use Dask arrays for lazy evaluation.
        precision : PrecisionConfig, str, or None
            Precision configuration. Can be:
            - None: Use standard float64 precision
            - str: Preset name ("standard", "fast", "precise", "ultra")
            - PrecisionConfig: Full configuration object
            Note: float128 is not supported and falls back to float64.

        Raises
        ------
        BackendNotAvailableError
            If a Dask-requiring mode is selected without Dask installed.
        ValueError
            If ``mode`` is not one of :data:`DASK_MODES`.
        """
        if mode == "gpu":
            raise ValueError(
                'DaskBackend mode="gpu": removed before v1.0; the mode validated '
                "a CUDA device and then executed NumPy. Use mode='cpu' or "
                "mode='distributed'."
            )
        if mode not in DASK_MODES:
            raise ValueError(
                f"DaskBackend mode must be one of {list(DASK_MODES)}, got {mode!r}"
            )

        self.mode = mode
        self.use_dask_arrays = use_dask_arrays and DASK_AVAILABLE
        self._xp = np  # NumPy-compatible interface
        self.dask_client = None

        if mode == "distributed":
            if not DASK_AVAILABLE:
                raise BackendNotAvailableError(
                    "Dask not available. Install with:\n  pip install dask[complete]"
                )

            if scheduler_address:
                # Connect to remote cluster
                self.dask_client = Client(scheduler_address)
            else:
                # Create local cluster
                cluster = LocalCluster(
                    n_workers=n_workers,
                    threads_per_worker=threads_per_worker,
                    processes=True,
                )
                self.dask_client = Client(cluster)

        elif mode == "cpu":
            # CPU mode with optional Dask for parallelism
            if DASK_AVAILABLE and n_workers:
                cluster = LocalCluster(
                    n_workers=n_workers,
                    threads_per_worker=threads_per_worker,
                    processes=False,  # Use threads for shared memory
                )
                self.dask_client = Client(cluster)

        # Resolve and set precision (with float128 fallback warning)
        if precision is not None:
            from radiosim.core.precision import resolve_precision

            self.precision = resolve_precision(precision)

    @property
    def name(self) -> str:
        """Backend name."""
        if self.mode == "distributed":
            return "dask-distributed"
        return "dask-cpu"

    @property
    def xp(self) -> Any:
        """NumPy-compatible array namespace."""
        return self._xp

    @property
    def device_kind(self) -> str:
        """Execution device kind. Always ``'cpu'``: this backend runs NumPy."""
        return "cpu"

    def is_available(self) -> bool:
        """Check whether this backend can be used.

        NumPy is always available, so the CPU mode always is; the distributed
        mode additionally needs Dask.
        """
        if self.mode == "distributed":
            return DASK_AVAILABLE
        return True

    # =========================================================================
    # Array Creation and Conversion
    # =========================================================================

    def asarray(self, arr: Any, dtype: Any | None = None) -> Any:
        """Convert to array (optionally Dask array).

        Args:
            arr: Input array-like
            dtype: Optional data type

        Returns:
            NumPy or Dask array
        """
        arr = np.asarray(arr, dtype=dtype)

        if self.use_dask_arrays and DASK_AVAILABLE:
            return da.from_array(arr, chunks="auto")
        return arr

    def to_numpy(self, arr: Any) -> np.ndarray:
        """Convert to NumPy array.

        For Dask arrays, this triggers computation.

        Args:
            arr: Input array

        Returns:
            NumPy array
        """
        if DASK_AVAILABLE and isinstance(arr, da.Array):
            return arr.compute()
        return np.asarray(arr)

    # =========================================================================
    # Mathematical Operations
    # =========================================================================

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication.

        Uses optimized BLAS via NumPy, or Dask for distributed.

        Args:
            a: First matrix
            b: Second matrix

        Returns:
            Matrix product
        """
        if DASK_AVAILABLE and (isinstance(a, da.Array) or isinstance(b, da.Array)):
            return da.matmul(a, b)
        return np.matmul(a, b)

    def conjugate_transpose(self, a: Any) -> Any:
        """Hermitian conjugate.

        Args:
            a: Input matrix

        Returns:
            Conjugate transpose
        """
        if DASK_AVAILABLE and isinstance(a, da.Array):
            return da.conj(da.swapaxes(a, -2, -1))
        return np.conj(np.swapaxes(a, -2, -1))

    def exp(self, x: Any) -> Any:
        """Exponential function.

        Args:
            x: Input array

        Returns:
            exp(x)
        """
        if DASK_AVAILABLE and isinstance(x, da.Array):
            return da.exp(x)
        return np.exp(x)

    def sin(self, x: Any) -> Any:
        """Sine function.

        Args:
            x: Input array

        Returns:
            sin(x)
        """
        if DASK_AVAILABLE and isinstance(x, da.Array):
            return da.sin(x)
        return np.sin(x)

    def cos(self, x: Any) -> Any:
        """Cosine function.

        Args:
            x: Input array

        Returns:
            cos(x)
        """
        if DASK_AVAILABLE and isinstance(x, da.Array):
            return da.cos(x)
        return np.cos(x)

    # =========================================================================
    # Memory Management
    # =========================================================================

    def free_memory(self, arr: Any) -> None:
        """Free array memory.

        Args:
            arr: Array to free
        """
        if DASK_AVAILABLE and isinstance(arr, da.Array):
            # Dask handles cleanup automatically
            pass
        del arr

    def memory_info(self) -> dict[str, Any]:
        """Get memory information.

        Returns:
            Dictionary with memory stats
        """
        info = {
            "backend": "dask",
            "mode": self.mode,
        }

        try:
            import psutil

            mem = psutil.virtual_memory()
            info.update(
                {
                    "total_bytes": mem.total,
                    "available_bytes": mem.available,
                    "used_bytes": mem.used,
                    "percent_used": mem.percent,
                }
            )
        except ImportError:
            info["note"] = "Install psutil for detailed memory info"

        return info

    def get_device_info(self) -> dict[str, Any]:
        """Get device information.

        Returns:
            Dictionary with device details
        """
        import platform

        info: dict[str, Any] = {
            "backend": "dask",
            "mode": self.mode,
            "dask_version": dask.__version__ if DASK_AVAILABLE else None,
            "architecture": platform.machine(),
            "device": "CPU",
        }

        try:
            import psutil

            info["cores_physical"] = psutil.cpu_count(logical=False)
            info["cores_logical"] = psutil.cpu_count(logical=True)
        except ImportError:
            pass

        if self.dask_client:
            try:
                scheduler_info = self.dask_client.scheduler_info()
                info["dask_workers"] = len(scheduler_info.get("workers", {}))
                info["dask_threads"] = sum(
                    w.get("nthreads", 1)
                    for w in scheduler_info.get("workers", {}).values()
                )
            except Exception:
                pass

        return info

    def synchronize(self, arr: Any = None) -> Any:
        """Wait for pending work and return the materialized array.

        NumPy arrays are already materialized. A Dask array is only a task
        graph, so an explicit array argument is computed here; without one there
        is nothing meaningful to wait for.
        """
        if arr is None:
            return None
        if DASK_AVAILABLE and isinstance(arr, da.Array):
            return arr.compute()
        return arr

    # =========================================================================
    # Removed surface
    # =========================================================================

    def __getattr__(self, item: str) -> Any:
        """Give the Tier 6H removals an actionable error instead of a bare miss."""
        if item in {"jit_compile", "jit", "prange"}:
            raise AttributeError(
                f"DaskBackend.{item}: removed before v1.0; the backend formerly "
                "named 'numba' never compiled any kernel and had no caller for "
                "this helper. Use execution.backend=jax, whose ArrayBackend."
                "supports_compilation is True and whose ArrayBackend.compile "
                "is jax.jit."
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {item!r}"
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def shutdown(self):
        """Shutdown Dask client if active."""
        # ``getattr`` with a default, not ``self.dask_client``: ``__del__`` runs
        # even for an instance whose ``__init__`` rejected its arguments before
        # the attribute existed.
        client = getattr(self, "dask_client", None)
        if client:
            try:
                client.close()
            except Exception:
                pass
            self.dask_client = None

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


def is_dask_available() -> bool:
    """Check if Dask is available for distributed computing.

    Returns:
        True if Dask is installed
    """
    return DASK_AVAILABLE
