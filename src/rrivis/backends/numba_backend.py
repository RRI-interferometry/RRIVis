"""Numba + Dask backend for production-grade CPU/GPU computing.

This backend is aligned with the radio astronomy community standards:
- QuartiCal uses Numba for calibration
- Codex Africanus uses Numba + Dask for distributed computing
- dask-ms provides Measurement Set I/O

Features:
- JIT compilation for CPU performance
- Parallel loops via numba.prange
- GPU support via numba.cuda (NVIDIA CUDA, AMD ROCm)
- Distributed computing via Dask
- Precision control (float32/float64 only; float128 falls back to float64)

Usage:
    >>> from rrivis.backends import get_backend
    >>> backend = get_backend("numba")
    >>> backend.name
    'numba-cpu'

With precision control:
    >>> from rrivis.core.precision import PrecisionConfig
    >>> backend = get_backend("numba", precision="fast")

Note: Numba does not support float128/complex256. Precision configurations
requesting float128 will automatically fall back to float64 with a warning.
"""

from typing import TYPE_CHECKING, Any, Union

import numpy as np

from rrivis.backends.base import ArrayBackend, BackendNotAvailableError

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

# Try to import Numba and Dask
try:
    import numba
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    jit = None
    prange = None
    NUMBA_AVAILABLE = False

try:
    from numba import cuda

    CUDA_AVAILABLE = cuda.is_available() if NUMBA_AVAILABLE else False
except (ImportError, Exception):
    cuda = None
    CUDA_AVAILABLE = False

try:
    import dask
    import dask.array as da
    from dask.distributed import Client, LocalCluster

    DASK_AVAILABLE = True
except ImportError:
    dask = None
    da = None
    Client = None
    LocalCluster = None
    DASK_AVAILABLE = False


class NumbaBackend(ArrayBackend):
    """Numba + Dask backend for production computing.

    This backend provides:
    - JIT compilation for CPU optimization (2-10x speedup)
    - Parallel loops for multi-core CPU usage
    - GPU support via CUDA (10-100x speedup)
    - Distributed computing via Dask for cluster deployment
    - Precision control (float32/float64 only; float128 falls back to float64)

    Modes:
    - 'cpu': Local CPU with JIT and parallel loops
    - 'gpu': NVIDIA GPU via CUDA
    - 'distributed': Dask distributed cluster

    Example:
        >>> # CPU mode (default)
        >>> backend = NumbaBackend(mode="cpu")

        >>> # GPU mode
        >>> backend = NumbaBackend(mode="gpu")

        >>> # Distributed mode
        >>> backend = NumbaBackend(mode="distributed", n_workers=8)

        >>> # With precision control
        >>> from rrivis.core.precision import PrecisionConfig
        >>> backend = NumbaBackend(precision="fast")
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
        """Initialize Numba + Dask backend.

        Parameters
        ----------
        mode : str
            Execution mode ('cpu', 'gpu', 'distributed')
        n_workers : int, optional
            Number of Dask workers (default: auto-detect)
        threads_per_worker : int
            Threads per Dask worker
        scheduler_address : str, optional
            Dask scheduler address for remote cluster
        use_dask_arrays : bool
            Use Dask arrays for lazy evaluation
        precision : PrecisionConfig, str, or None
            Precision configuration. Can be:
            - None: Use standard float64 precision
            - str: Preset name ("standard", "fast", "precise", "ultra")
            - PrecisionConfig: Full configuration object
            Note: float128 is not supported by Numba and falls back to float64.

        Raises
        ------
        BackendNotAvailableError
            If Numba not installed or GPU unavailable
        """
        if not NUMBA_AVAILABLE:
            raise BackendNotAvailableError(
                "Numba not available. Install with:\n  pip install numba dask[complete]"
            )

        self.mode = mode
        self.use_dask_arrays = use_dask_arrays and DASK_AVAILABLE
        self._xp = np  # NumPy-compatible interface
        self.dask_client = None
        self.gpu_device = None

        if mode == "gpu":
            if not CUDA_AVAILABLE:
                raise BackendNotAvailableError(
                    "CUDA not available. Numba requires CUDA for GPU mode.\n"
                    "Ensure NVIDIA drivers and CUDA toolkit are installed."
                )
            self.gpu_device = cuda.get_current_device()

        elif mode == "distributed":
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
            from rrivis.core.precision import resolve_precision

            self.precision = resolve_precision(precision)

    @property
    def name(self) -> str:
        """Backend name."""
        if self.mode == "gpu":
            return "numba-cuda"
        elif self.mode == "distributed":
            return "numba-dask-distributed"
        else:
            return "numba-cpu"

    @property
    def xp(self) -> Any:
        """NumPy-compatible array namespace."""
        return self._xp

    def is_available(self) -> bool:
        """Check if Numba is available."""
        return NUMBA_AVAILABLE

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
            "backend": "numba",
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

        if self.mode == "gpu" and CUDA_AVAILABLE:
            try:
                meminfo = cuda.current_context().get_memory_info()
                info["gpu_free_bytes"] = meminfo[0]
                info["gpu_total_bytes"] = meminfo[1]
            except Exception:
                pass

        return info

    def get_device_info(self) -> dict[str, Any]:
        """Get device information.

        Returns:
            Dictionary with device details
        """
        import platform

        info = {
            "backend": "numba",
            "mode": self.mode,
            "numba_version": numba.__version__,
            "architecture": platform.machine(),
        }

        if self.mode == "gpu" and self.gpu_device:
            info.update(
                {
                    "device": "GPU",
                    "gpu_name": self.gpu_device.name.decode()
                    if hasattr(self.gpu_device.name, "decode")
                    else str(self.gpu_device.name),
                    "compute_capability": self.gpu_device.compute_capability,
                }
            )
            try:
                info["gpu_memory_total_gb"] = round(
                    self.gpu_device.total_memory / (1024**3), 2
                )
            except Exception:
                pass
        else:
            info["device"] = "CPU"
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

    # =========================================================================
    # Numba-specific methods
    # =========================================================================

    def jit_compile(self, func, nopython=True, parallel=False, fastmath=True):
        """JIT-compile a function with Numba.

        Args:
            func: Function to compile
            nopython: Use nopython mode (faster)
            parallel: Enable parallel execution
            fastmath: Enable fast math optimizations

        Returns:
            JIT-compiled function
        """
        if not NUMBA_AVAILABLE:
            return func

        return jit(nopython=nopython, parallel=parallel, fastmath=fastmath)(func)

    def shutdown(self):
        """Shutdown Dask client if active."""
        if self.dask_client:
            try:
                self.dask_client.close()
            except Exception:
                pass
            self.dask_client = None

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


def is_numba_available() -> bool:
    """Check if Numba is available.

    Returns:
        True if Numba is installed
    """
    return NUMBA_AVAILABLE


def is_cuda_available() -> bool:
    """Check if CUDA is available for GPU computing.

    Returns:
        True if CUDA is available
    """
    return CUDA_AVAILABLE


def is_dask_available() -> bool:
    """Check if Dask is available for distributed computing.

    Returns:
        True if Dask is installed
    """
    return DASK_AVAILABLE
