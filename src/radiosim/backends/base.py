"""Abstract base class for lower-level array computation backends.

This module defines the common array interface. Capability is backend- and
runtime-specific; implementing this interface does not prove that every
high-level Simulator kernel executes on the selected device.

Available implementations:
- NumPyBackend: CPU baseline, always available
- NumbaBackend: NumPy/Dask operations plus a Numba JIT helper
- JAXBackend: JAX arrays on devices exposed by the installed JAX runtime
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig


class BackendNotAvailableError(Exception):
    """Raised when a requested backend is not available on this system."""

    pass


class ArrayBackend(ABC):
    """Abstract base class for array computation backends.

    All backends must implement this interface to ensure consistent behavior
    across NumPy, Numba+Dask, and JAX implementations.

    The abstraction provides a shared array surface across NumPy, optional
    Dask arrays and Numba compilation helpers, and JAX arrays. Callers must
    verify which scientific kernels use that surface before making device or
    performance claims.

    Attributes
    ----------
    precision : PrecisionConfig, optional
        Precision configuration for controlling dtypes throughout computation.

    Example usage:
        >>> from radiosim.backends import get_backend
        >>> backend = get_backend("auto")  # Auto-detect best available
        >>> xp = backend.xp  # Get array namespace (numpy-like API)
        >>> arr = backend.asarray([1, 2, 3])
        >>> result = backend.exp(arr)

        # With precision control:
        >>> from radiosim.core.precision import PrecisionConfig
        >>> backend = get_backend("numpy", precision=PrecisionConfig.fast())
    """

    _precision: Optional["PrecisionConfig"] = None

    @property
    def precision(self) -> Optional["PrecisionConfig"]:
        """Get the precision configuration for this backend."""
        return self._precision

    @precision.setter
    def precision(self, value: Optional["PrecisionConfig"]) -> None:
        """Set the precision configuration for this backend."""
        self._precision = value
        if value is not None:
            # Validate precision for this backend
            warnings_list = value.validate_for_backend(self.backend_type)
            for warning in warnings_list:
                import warnings as warn_module

                warn_module.warn(warning, UserWarning, stacklevel=2)

    @property
    def backend_type(self) -> str:
        """Get the backend type name (numpy, jax, numba)."""
        # Extract base type from full name like "numpy-cpu", "jax-gpu-cuda"
        return self.name.split("-")[0]

    def get_real_dtype(
        self,
        component: str | None = None,
        sub_component: str | None = None,
    ) -> Any:
        """Get real dtype based on precision config.

        Parameters
        ----------
        component : str, optional
            Component name (coordinates, jones, accumulation, output)
        sub_component : str, optional
            Sub-component for coordinates or jones

        Returns
        -------
        dtype
            NumPy-compatible real dtype
        """
        if self._precision is None:
            return np.float64

        return self._precision.get_real_dtype(
            component or "default", sub_component, self.backend_type
        )

    def get_complex_dtype(
        self,
        component: str | None = None,
        sub_component: str | None = None,
    ) -> Any:
        """Get complex dtype based on precision config.

        Parameters
        ----------
        component : str, optional
            Component name (jones, accumulation, output)
        sub_component : str, optional
            Sub-component for jones

        Returns
        -------
        dtype
            NumPy-compatible complex dtype
        """
        if self._precision is None:
            return np.complex128

        return self._precision.get_complex_dtype(
            component or "default", sub_component, self.backend_type
        )

    @property
    def default_real_dtype(self) -> Any:
        """Default real dtype based on precision config."""
        return self.get_real_dtype()

    @property
    def default_complex_dtype(self) -> Any:
        """Default complex dtype based on precision config."""
        return self.get_complex_dtype()

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier string.

        Returns:
            Human-readable name like 'numpy-cpu', 'numba-cuda', 'jax-gpu-cuda'
        """
        pass

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Array namespace (numpy-like module).

        Returns:
            Module with numpy-compatible API (numpy, jax.numpy, etc.)
            Use this for array creation and operations.

        Example:
            >>> xp = backend.xp
            >>> arr = xp.zeros((10, 10), dtype=complex)
            >>> result = xp.sum(arr)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system.

        Returns:
            True if backend can be used, False otherwise
        """
        pass

    # =========================================================================
    # Array Creation and Conversion
    # =========================================================================

    @abstractmethod
    def asarray(self, arr: Any, dtype: Any | None = None) -> Any:
        """Convert input to backend array.

        Args:
            arr: Input array-like (list, numpy array, etc.)
            dtype: Optional data type (e.g., np.complex128)

        Returns:
            Array on the backend device (CPU/GPU/TPU)
        """
        pass

    @abstractmethod
    def to_numpy(self, arr: Any) -> np.ndarray:
        """Convert backend array to NumPy array.

        This may involve copying data from device (GPU/TPU) to host (CPU).

        Args:
            arr: Backend array

        Returns:
            NumPy array on CPU
        """
        pass

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Any:
        """Create array of zeros.

        Args:
            shape: Array shape
            dtype: Data type (if None, uses precision config default)

        Returns:
            Zero-filled array on backend device
        """
        if dtype is None:
            dtype = self.default_real_dtype
        return self.xp.zeros(shape, dtype=dtype)

    def zeros_complex(self, shape: tuple[int, ...], dtype: Any | None = None) -> Any:
        """Create complex array of zeros.

        Args:
            shape: Array shape
            dtype: Complex data type (if None, uses precision config default)

        Returns:
            Zero-filled complex array on backend device
        """
        if dtype is None:
            dtype = self.default_complex_dtype
        return self.xp.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Any:
        """Create array of ones.

        Args:
            shape: Array shape
            dtype: Data type (if None, uses precision config default)

        Returns:
            One-filled array on backend device
        """
        if dtype is None:
            dtype = self.default_real_dtype
        return self.xp.ones(shape, dtype=dtype)

    def eye(self, n: int, dtype: Any | None = None) -> Any:
        """Create identity matrix.

        Args:
            n: Matrix dimension (n x n)
            dtype: Data type (if None, uses precision config default)

        Returns:
            Identity matrix on backend device
        """
        if dtype is None:
            dtype = self.default_real_dtype
        return self.xp.eye(n, dtype=dtype)

    def eye_complex(self, n: int, dtype: Any | None = None) -> Any:
        """Create complex identity matrix (useful for Jones matrices).

        Args:
            n: Matrix dimension (n x n)
            dtype: Complex data type (if None, uses precision config default)

        Returns:
            Complex identity matrix on backend device
        """
        if dtype is None:
            dtype = self.default_complex_dtype
        return self.xp.eye(n, dtype=dtype)

    def batch_eye(
        self,
        batch_shape: tuple[int, ...],
        n: int,
        dtype: Any | None = None,
    ) -> Any:
        """Create a batch of identity matrices without item assignment.

        This helper avoids in-place diagonal writes, which are illegal for
        immutable array backends such as JAX.
        """
        identity = self.eye(n, dtype=dtype)
        return self.xp.broadcast_to(identity, (*batch_shape, n, n)).copy()

    def diagonal_matrix(self, diagonal: Any, dtype: Any | None = None) -> Any:
        """Create batched diagonal matrices from trailing-axis diagonals."""
        diagonal_array = self.asarray(diagonal, dtype=dtype)
        n = int(diagonal_array.shape[-1])
        eye = self.eye(n, dtype=diagonal_array.dtype)
        return diagonal_array[..., :, None] * eye

    def set_at(self, arr: Any, index: Any, value: Any) -> Any:
        """Return ``arr`` with ``value`` set at ``index``.

        NumPy/Numba arrays are updated in place. Immutable backends such as
        JAX use their functional ``.at[index].set(...)`` update API.

        This is a general-purpose helper and is deliberately *not* used to
        accumulate solver output: a per-cell functional update copies the whole
        cube on an immutable backend. The solvers assemble blocks with
        :meth:`stack` instead (``Tier6HybridRuntimePlan.md`` Section 13.3).
        """
        if hasattr(arr, "at"):
            return arr.at[index].set(value)
        arr[index] = value
        return arr

    def stack(self, arrays: Sequence[Any], axis: int = 0) -> Any:
        """Assemble equal-shaped arrays along a new axis.

        This is the solvers' one accumulation primitive. It is functional on
        every backend -- the inputs are never written to -- so a whole block of
        results is materialized in a single operation instead of ``O(T*B*F)``
        per-cell updates.

        Args:
            arrays: Sequence of arrays that share one shape and dtype.
            axis: Position of the new axis in the result.

        Returns:
            One array whose shape is the common input shape with ``axis``
            inserted, and whose dtype is the common input dtype.
        """
        return self.xp.stack(arrays, axis=axis)

    def bincount(
        self,
        x: Any,
        weights: Any | None = None,
        minlength: int = 0,
    ) -> Any:
        """Count/bin values using the backend namespace."""
        x_backend = self.asarray(x)
        weights_backend = None if weights is None else self.asarray(weights)
        if self.xp is np:
            return np.bincount(
                np.asarray(x_backend),
                weights=(
                    None if weights_backend is None else np.asarray(weights_backend)
                ),
                minlength=minlength,
            )
        try:
            return self.xp.bincount(
                x_backend,
                weights=weights_backend,
                length=minlength,
            )
        except TypeError:
            return self.xp.bincount(
                x_backend,
                weights=weights_backend,
                minlength=minlength,
            )

    # =========================================================================
    # Mathematical Operations
    # =========================================================================

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication.

        Args:
            a: First matrix, shape (..., M, K)
            b: Second matrix, shape (..., K, N)

        Returns:
            Product matrix, shape (..., M, N)
        """
        pass

    @abstractmethod
    def conjugate_transpose(self, a: Any) -> Any:
        """Hermitian conjugate (conjugate transpose).

        Args:
            a: Input matrix, shape (..., M, N)

        Returns:
            Conjugate transpose, shape (..., N, M)
        """
        pass

    def conj(self, x: Any) -> Any:
        """Complex conjugate.

        Args:
            x: Input array

        Returns:
            Complex conjugate of input
        """
        return self.xp.conj(x)

    @abstractmethod
    def exp(self, x: Any) -> Any:
        """Exponential function.

        Args:
            x: Input array

        Returns:
            Element-wise exp(x)
        """
        pass

    @abstractmethod
    def sin(self, x: Any) -> Any:
        """Sine function.

        Args:
            x: Input array (radians)

        Returns:
            Element-wise sin(x)
        """
        pass

    @abstractmethod
    def cos(self, x: Any) -> Any:
        """Cosine function.

        Args:
            x: Input array (radians)

        Returns:
            Element-wise cos(x)
        """
        pass

    def sqrt(self, x: Any) -> Any:
        """Square root.

        Args:
            x: Input array

        Returns:
            Element-wise sqrt(x)
        """
        return self.xp.sqrt(x)

    def abs(self, x: Any) -> Any:
        """Absolute value.

        Args:
            x: Input array

        Returns:
            Element-wise |x|
        """
        return self.xp.abs(x)

    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum of array elements.

        Args:
            x: Input array
            axis: Axis along which to sum (None = all elements)

        Returns:
            Sum of elements
        """
        return self.xp.sum(x, axis=axis)

    # =========================================================================
    # Complex Number Operations
    # =========================================================================

    def complex_multiply(self, a: Any, b: Any) -> Any:
        """Complex multiplication (element-wise).

        Default implementation uses xp.multiply.
        Backends may override for optimization.

        Args:
            a: First array
            b: Second array

        Returns:
            Element-wise product a * b
        """
        return self.xp.multiply(a, b)

    def real(self, x: Any) -> Any:
        """Real part of complex array.

        Args:
            x: Complex array

        Returns:
            Real part
        """
        return self.xp.real(x)

    def imag(self, x: Any) -> Any:
        """Imaginary part of complex array.

        Args:
            x: Complex array

        Returns:
            Imaginary part
        """
        return self.xp.imag(x)

    # =========================================================================
    # Memory Management
    # =========================================================================

    @abstractmethod
    def free_memory(self, arr: Any) -> None:
        """Free array memory.

        Important for GPU backends to avoid memory leaks.
        CPU backends may simply delete the reference.

        Args:
            arr: Array to free
        """
        pass

    @abstractmethod
    def memory_info(self) -> dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Dictionary with memory info (keys vary by backend):
            - 'total_bytes': Total memory
            - 'available_bytes': Available memory
            - 'used_bytes': Used memory
            - For GPU: 'gpu_total_bytes', 'gpu_free_bytes'
        """
        pass

    @abstractmethod
    def get_device_info(self) -> dict[str, Any]:
        """Get device information.

        Returns:
            Dictionary with device info:
            - 'backend': Backend name
            - 'device': Device type ('CPU', 'GPU', 'TPU')
            - 'architecture': CPU architecture (x86_64, arm64)
            - For GPU: 'vendor', 'gpu_name', 'compute_capability'
        """
        pass

    # =========================================================================
    # Synchronization
    # =========================================================================

    def synchronize(self) -> None:
        """Wait for all pending operations to complete.

        Important for GPU/TPU backends where operations are asynchronous.
        CPU backends can implement as no-op (default).
        """
        return

    # =========================================================================
    # Backend Info
    # =========================================================================

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def get_config(self) -> dict[str, Any]:
        """Get backend configuration.

        Returns:
            Dictionary with backend configuration for logging/reproducibility
        """
        config = {
            "name": self.name,
            "available": self.is_available(),
            "device_info": self.get_device_info(),
        }
        if self._precision is not None:
            config["precision"] = self._precision.model_dump()
        return config
