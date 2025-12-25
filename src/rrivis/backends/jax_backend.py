"""JAX-based GPU/TPU backend.

This backend supports automatic differentiation and universal hardware acceleration:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm)
- Apple Silicon (Metal)
- Google TPUs
- Intel GPUs (OneAPI, experimental)

JAX automatically detects and uses the best available hardware.

Usage:
    >>> from rrivis.backends import get_backend
    >>> backend = get_backend("jax")  # Auto-detect GPU/TPU
    >>> backend.name
    'jax-gpu-cuda'  # or 'jax-gpu-metal', 'jax-tpu', etc.

With precision control:
    >>> from rrivis.core.precision import PrecisionConfig
    >>> backend = get_backend("jax", precision="fast")  # float32 where safe

Note: JAX does not support float128/complex256. Precision configurations
requesting float128 will automatically fall back to float64 with a warning.
"""

from typing import Any, Dict, Optional, Union, TYPE_CHECKING
import numpy as np

from rrivis.backends.base import ArrayBackend, BackendNotAvailableError

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


class JAXBackend(ArrayBackend):
    """JAX-based universal GPU/TPU backend.

    This backend uses JAX for hardware-accelerated array operations.
    JAX provides:
    - Automatic differentiation (for Bayesian inference workflows)
    - JIT compilation (for performance)
    - Universal hardware support (NVIDIA, AMD, Apple, Google TPU)

    Features:
    - Automatic device detection
    - XLA compilation for optimal performance
    - Gradient computation for optimization
    - Multi-device support (pmap)
    - Precision control (float32/float64 only; float128 falls back to float64)

    Example:
        >>> backend = JAXBackend(device="gpu")
        >>> arr = backend.asarray([1, 2, 3], dtype=jnp.float32)
        >>> result = backend.exp(arr)  # Runs on GPU

        # With precision control:
        >>> from rrivis.core.precision import PrecisionConfig
        >>> backend = JAXBackend(precision="fast")  # float32 where safe
    """

    def __init__(
        self,
        device: str = "gpu",
        precision: Optional[Union["PrecisionConfig", str]] = None,
    ):
        """Initialize JAX backend.

        Parameters
        ----------
        device : str
            Device type - 'cpu', 'gpu', or 'tpu'
            JAX auto-detects specific hardware (CUDA/ROCm/Metal)
        precision : PrecisionConfig, str, or None
            Precision configuration. Can be:
            - None: Use standard float64 precision
            - str: Preset name ("standard", "fast", "precise", "ultra")
            - PrecisionConfig: Full configuration object
            Note: float128 is not supported by JAX and falls back to float64.

        Raises
        ------
        BackendNotAvailableError
            If JAX is not installed or device unavailable
        """
        if not JAX_AVAILABLE:
            raise BackendNotAvailableError(
                "JAX not installed. Install with:\n"
                "  pip install rrivis[gpu]        # Generic GPU\n"
                "  pip install rrivis[gpu-cuda]   # NVIDIA CUDA\n"
                "  pip install rrivis[gpu-rocm]   # AMD ROCm\n"
                "  pip install rrivis[tpu]        # Google TPU"
            )

        self._device_type = device
        self._xp = jnp

        # Get available devices
        try:
            self.devices = jax.devices(device)
        except RuntimeError:
            # Device type not available, try to get any device
            self.devices = []

        if not self.devices:
            # Fall back to CPU if requested device not available
            if device != "cpu":
                try:
                    self.devices = jax.devices("cpu")
                    self._device_type = "cpu"
                except RuntimeError:
                    raise BackendNotAvailableError(
                        f"No {device} devices available and CPU fallback failed."
                    )

        if self.devices:
            self.device = self.devices[0]
        else:
            raise BackendNotAvailableError(
                f"No devices available for JAX backend."
            )

        # Resolve and set precision (with float128 fallback warning)
        if precision is not None:
            from rrivis.core.precision import resolve_precision
            self.precision = resolve_precision(precision)

    @property
    def name(self) -> str:
        """Backend name including device info."""
        platform = self.device.platform
        try:
            backend_name = jax.default_backend()
        except Exception:
            backend_name = "unknown"
        return f"jax-{platform}-{backend_name}"

    @property
    def xp(self) -> Any:
        """JAX numpy module."""
        return self._xp

    def is_available(self) -> bool:
        """Check if JAX is available with devices."""
        return JAX_AVAILABLE and len(self.devices) > 0

    # =========================================================================
    # Array Creation and Conversion
    # =========================================================================

    def asarray(
        self,
        arr: Any,
        dtype: Optional[Any] = None
    ) -> Any:
        """Convert to JAX array on target device.

        Args:
            arr: Input array-like
            dtype: Optional data type

        Returns:
            JAX array on device
        """
        with jax.default_device(self.device):
            return jnp.asarray(arr, dtype=dtype)

    def to_numpy(self, arr: Any) -> np.ndarray:
        """Convert JAX array to NumPy (copies from device).

        Args:
            arr: JAX array

        Returns:
            NumPy array on CPU
        """
        # JAX arrays can be converted directly
        return np.asarray(arr)

    # =========================================================================
    # Mathematical Operations
    # =========================================================================

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication (GPU-accelerated).

        Uses optimized libraries:
        - cuBLAS (NVIDIA)
        - rocBLAS (AMD)
        - Metal Performance Shaders (Apple)

        Args:
            a: First matrix
            b: Second matrix

        Returns:
            Matrix product
        """
        return jnp.matmul(a, b)

    def conjugate_transpose(self, a: Any) -> Any:
        """Hermitian conjugate.

        Args:
            a: Input matrix

        Returns:
            Conjugate transpose
        """
        return jnp.conj(jnp.swapaxes(a, -2, -1))

    def exp(self, x: Any) -> Any:
        """Exponential function.

        Args:
            x: Input array

        Returns:
            exp(x)
        """
        return jnp.exp(x)

    def sin(self, x: Any) -> Any:
        """Sine function.

        Args:
            x: Input array

        Returns:
            sin(x)
        """
        return jnp.sin(x)

    def cos(self, x: Any) -> Any:
        """Cosine function.

        Args:
            x: Input array

        Returns:
            cos(x)
        """
        return jnp.cos(x)

    # =========================================================================
    # Memory Management
    # =========================================================================

    def free_memory(self, arr: Any) -> None:
        """Free device memory.

        JAX handles memory via XLA runtime. Deleting reference
        allows garbage collection.

        Args:
            arr: Array to free
        """
        del arr

    def memory_info(self) -> Dict[str, Any]:
        """Get device memory information.

        Returns:
            Dictionary with memory info
        """
        info = {
            "backend": "jax",
            "platform": self.device.platform,
            "note": "JAX manages memory automatically via XLA",
        }

        # Try to get more detailed info for GPU
        if self.device.platform == "gpu":
            try:
                # This works for some JAX installations
                info["device_kind"] = self.device.device_kind
            except Exception:
                pass

        return info

    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information.

        Returns:
            Dictionary with device details
        """
        platform = self.device.platform
        try:
            backend = jax.default_backend()
        except Exception:
            backend = "unknown"

        info = {
            "backend": "jax",
            "device": platform.upper(),
            "platform": backend,
            "device_id": self.device.id,
            "num_devices": len(self.devices),
        }

        # Add device kind if available
        try:
            info["device_kind"] = self.device.device_kind
        except Exception:
            pass

        # Add platform-specific vendor info
        if platform == "gpu":
            if backend == "cuda":
                info["vendor"] = "NVIDIA"
            elif backend == "rocm":
                info["vendor"] = "AMD"
            elif backend == "metal":
                info["vendor"] = "Apple"
            elif backend == "oneapi":
                info["vendor"] = "Intel"
        elif platform == "tpu":
            info["vendor"] = "Google"

        return info

    def synchronize(self) -> None:
        """Wait for all device operations to complete.

        Important for timing and ensuring results are ready.
        """
        # Block until all operations complete
        jax.block_until_ready(jnp.array(0))

    # =========================================================================
    # JAX-specific methods
    # =========================================================================

    def jit(self, func):
        """JIT-compile a function for performance.

        Args:
            func: Function to compile

        Returns:
            JIT-compiled function
        """
        return jax.jit(func)

    def grad(self, func):
        """Get gradient function (for auto-diff).

        Args:
            func: Function to differentiate

        Returns:
            Gradient function
        """
        return jax.grad(func)

    def vmap(self, func, in_axes=0, out_axes=0):
        """Vectorize a function over batch dimension.

        Args:
            func: Function to vectorize
            in_axes: Input axes to vectorize over
            out_axes: Output axes

        Returns:
            Vectorized function
        """
        return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)


def is_jax_available() -> bool:
    """Check if JAX is available.

    Returns:
        True if JAX is installed and functional
    """
    return JAX_AVAILABLE
