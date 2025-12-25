"""NumPy-based CPU backend.

This is the default backend that is always available. It supports both
x86-64 and ARM architectures (including Apple Silicon).

Usage:
    >>> from rrivis.backends import get_backend
    >>> backend = get_backend("numpy")
    >>> backend.name
    'numpy-cpu'

With precision control:
    >>> from rrivis.core.precision import PrecisionConfig
    >>> backend = get_backend("numpy", precision=PrecisionConfig.precise())
"""

from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING
import platform
import numpy as np

from rrivis.backends.base import ArrayBackend

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig


class NumPyBackend(ArrayBackend):
    """NumPy-based CPU backend.

    This backend uses NumPy for all array operations. It is always available
    and serves as the baseline implementation and fallback for other backends.

    Features:
    - Works on any platform with Python (x86, ARM, etc.)
    - Single-threaded by default (NumPy uses BLAS for some ops)
    - No additional dependencies required
    - Supports all NumPy dtypes including float128/complex256 (platform-dependent)
    - Full precision control support

    Example:
        >>> backend = NumPyBackend()
        >>> arr = backend.asarray([1, 2, 3], dtype=np.float64)
        >>> result = backend.exp(arr)

        # With precision control:
        >>> from rrivis.core.precision import PrecisionConfig
        >>> backend = NumPyBackend(precision=PrecisionConfig.fast())
    """

    def __init__(
        self,
        precision: Optional[Union["PrecisionConfig", str]] = None,
    ):
        """Initialize NumPy backend.

        Parameters
        ----------
        precision : PrecisionConfig, str, or None
            Precision configuration. Can be:
            - None: Use standard float64 precision
            - str: Preset name ("standard", "fast", "precise", "ultra")
            - PrecisionConfig: Full configuration object
        """
        self._xp = np

        # Resolve and set precision
        if precision is not None:
            from rrivis.core.precision import resolve_precision
            self.precision = resolve_precision(precision)

    @property
    def name(self) -> str:
        """Backend name."""
        return "numpy-cpu"

    @property
    def xp(self) -> Any:
        """NumPy module."""
        return self._xp

    def is_available(self) -> bool:
        """NumPy is always available."""
        return True

    # =========================================================================
    # Array Creation and Conversion
    # =========================================================================

    def asarray(
        self,
        arr: Any,
        dtype: Optional[Any] = None
    ) -> np.ndarray:
        """Convert to NumPy array.

        Args:
            arr: Input array-like
            dtype: Optional data type

        Returns:
            NumPy array
        """
        return np.asarray(arr, dtype=dtype)

    def to_numpy(self, arr: Any) -> np.ndarray:
        """Convert to NumPy array (no-op for NumPy backend).

        Args:
            arr: Input array

        Returns:
            NumPy array
        """
        return np.asarray(arr)

    # =========================================================================
    # Mathematical Operations
    # =========================================================================

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        """Matrix multiplication using NumPy.

        Uses optimized BLAS routines when available.

        Args:
            a: First matrix
            b: Second matrix

        Returns:
            Matrix product
        """
        return np.matmul(a, b)

    def conjugate_transpose(self, a: Any) -> np.ndarray:
        """Hermitian conjugate.

        Args:
            a: Input matrix

        Returns:
            Conjugate transpose
        """
        # Use swapaxes for proper handling of batched matrices
        return np.conj(np.swapaxes(a, -2, -1))

    def exp(self, x: Any) -> np.ndarray:
        """Exponential function.

        Args:
            x: Input array

        Returns:
            exp(x)
        """
        return np.exp(x)

    def sin(self, x: Any) -> np.ndarray:
        """Sine function.

        Args:
            x: Input array

        Returns:
            sin(x)
        """
        return np.sin(x)

    def cos(self, x: Any) -> np.ndarray:
        """Cosine function.

        Args:
            x: Input array

        Returns:
            cos(x)
        """
        return np.cos(x)

    # =========================================================================
    # Memory Management
    # =========================================================================

    def free_memory(self, arr: Any) -> None:
        """Free array memory.

        For NumPy, we simply delete the reference and let Python's
        garbage collector handle the rest.

        Args:
            arr: Array to free
        """
        del arr

    def memory_info(self) -> Dict[str, Any]:
        """Get system memory information.

        Returns:
            Dictionary with memory statistics
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_bytes": mem.total,
                "available_bytes": mem.available,
                "used_bytes": mem.used,
                "percent_used": mem.percent,
            }
        except ImportError:
            # psutil not available, return basic info
            return {
                "note": "Install psutil for detailed memory info",
                "total_bytes": None,
                "available_bytes": None,
            }

    def get_device_info(self) -> Dict[str, Any]:
        """Get CPU and system information.

        Returns:
            Dictionary with device details
        """
        info = {
            "backend": "numpy",
            "device": "CPU",
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        }

        # Try to get more detailed CPU info
        try:
            import psutil
            info["cores_physical"] = psutil.cpu_count(logical=False)
            info["cores_logical"] = psutil.cpu_count(logical=True)
            info["memory_total_gb"] = round(
                psutil.virtual_memory().total / (1024**3), 2
            )
        except ImportError:
            pass

        return info

    # =========================================================================
    # Additional NumPy-specific methods
    # =========================================================================

    def einsum(self, subscripts: str, *operands: Any) -> np.ndarray:
        """Einstein summation convention.

        Useful for complex tensor operations in RIME calculations.

        Args:
            subscripts: Specifies the subscripts for summation
            *operands: Input arrays

        Returns:
            Result of einsum operation
        """
        return np.einsum(subscripts, *operands)

    def broadcast_arrays(self, *args: Any) -> Tuple[np.ndarray, ...]:
        """Broadcast arrays to common shape.

        Args:
            *args: Arrays to broadcast

        Returns:
            Tuple of broadcasted arrays
        """
        return np.broadcast_arrays(*args)
