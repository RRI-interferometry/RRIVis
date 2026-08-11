"""Lower-level JAX array backend.

Available devices depend on the installed JAX build and local runtime. Selecting
this backend controls its array operations; it does not establish that every
high-level Simulator kernel remains on that device.

Usage:
    >>> from radiosim.backends import get_backend
    >>> backend = get_backend("jax", device="cpu")
    >>> backend.name
    'jax-cpu-cpu'

With precision control:
    >>> from radiosim.core.precision import PrecisionConfig
    >>> backend = get_backend("jax", precision="fast")  # runtime-default device

Note: JAX does not support float128/complex256. Precision configurations
requesting float128 will automatically fall back to float64 with a warning.
"""

import importlib.util
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from radiosim.backends.base import ArrayBackend, BackendNotAvailableError

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

# JAX is imported lazily. Tier 6H made a CPU-only JAX a declared dependency of
# every standard pixi gate so backend parity is measured rather than skipped
# (``Tier6HybridRuntimePlan.md`` Sections 28, 32.8); importing it eagerly here
# would then put roughly a second of XLA start-up into the import graph of every
# caller that merely touches ``radiosim.backends``, including point-source runs
# that never select it. ``jax`` is treated like the other heavy optional
# dependencies (``healpy``, ``pyuvdata``): detected by spec, imported on first
# construction of this backend.
jax: Any = None
jnp: Any = None


def _load_jax() -> bool:
    """Import JAX on first use, caching the modules in this module's globals."""
    global jax, jnp
    if jax is not None:
        return True
    try:
        import jax as _jax
        import jax.numpy as _jnp
    except ImportError:
        return False
    jax = _jax
    jnp = _jnp
    return True


class JAXBackend(ArrayBackend):
    """JAX array backend for a requested available device.

    This backend uses JAX for its own array operations. Device availability and
    behavior come from the installed JAX runtime.
    JAX provides:
    - Automatic differentiation (for Bayesian inference workflows)
    - JIT compilation
    - Vectorization

    Features:
    - Automatic device detection
    - XLA compilation
    - Gradient computation for optimization
    - Precision control (float32/float64 only; float128 falls back to float64)

    Example:
        >>> import jax.numpy as jnp
        >>> from radiosim.backends.jax_backend import JAXBackend
        >>> backend = JAXBackend(device="cpu")
        >>> arr = backend.asarray([1, 2, 3], dtype=jnp.float32)
        >>> backend.exp(arr).shape
        (3,)

        With precision control:

        >>> fast = JAXBackend(device="cpu", precision="fast")  # float32 where safe
    """

    def __init__(
        self,
        device: str | None = None,
        precision: Union["PrecisionConfig", str] | None = None,
    ):
        """Initialize JAX backend.

        Parameters
        ----------
        device : str or None
            ``None`` uses JAX's runtime-default device. Explicit ``'cpu'``,
            ``'gpu'``, or ``'tpu'`` values are strict requirements and never
            fall back to another device.
        precision : PrecisionConfig, str, or None
            Precision configuration. Can be:
            - None: Use the standard precision preset
            - str: Preset name ("standard", "fast", "precise", "ultra")
            - PrecisionConfig: Full configuration object
            Note: float128 is not supported by JAX and falls back to float64.

        Raises
        ------
        BackendNotAvailableError
            If JAX is not installed or device unavailable
        """
        try:
            loaded = _load_jax()
        except Exception as exc:
            raise BackendNotAvailableError(
                "The installed JAX runtime could not be initialized."
            ) from exc
        if not loaded:
            raise BackendNotAvailableError(
                "JAX not installed. Standard pixi gates declare CPU-only\n"
                "jax/jaxlib, so `pixi install` is the supported fix. Outside pixi:\n"
                "  pip install radiosim[jax]\n"
                "There is no device-named extra: RadioSim has measured no\n"
                "accelerator, so a CUDA, ROCm or TPU build is the vendor's JAX\n"
                "wheel, installed directly."
            )

        # RadioSim's standard compute/output paths use float64/complex128. JAX
        # disables x64 by default, so enable it before creating backend arrays
        # to honor the dtype contract instead of silently truncating requested
        # dtypes.
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception as exc:
            raise BackendNotAvailableError(
                "The installed JAX runtime could not enable x64 support."
            ) from exc

        if device not in {None, "cpu", "gpu", "tpu"}:
            raise BackendNotAvailableError(
                "Unknown JAX device requirement "
                f"{device!r}; expected one of cpu, gpu, tpu, or None."
            )

        self._device_type = device
        self._xp = jnp

        # Query no platform when the caller requested JAX's runtime default.
        # Named devices are requirements, not preferences.
        try:
            self.devices = jax.devices() if device is None else jax.devices(device)
        except Exception as exc:
            requirement = "runtime-default" if device is None else device
            raise BackendNotAvailableError(
                f"The JAX {requirement} device requirement is unavailable."
            ) from exc

        if self.devices:
            actual_platforms = sorted(
                {str(candidate.platform) for candidate in self.devices}
            )
            unsupported = set(actual_platforms) - {"cpu", "gpu", "tpu"}
            if unsupported:
                raise BackendNotAvailableError(
                    "The JAX runtime reported unsupported device platform(s) "
                    f"{sorted(unsupported)}; expected cpu, gpu, or tpu."
                )
            if device is not None and any(
                str(candidate.platform) != device for candidate in self.devices
            ):
                raise BackendNotAvailableError(
                    f"The JAX {device} device requirement resolved to "
                    f"{actual_platforms}, so the runtime result was rejected."
                )
            self.device = self.devices[0]
        else:
            requirement = "runtime-default" if device is None else device
            raise BackendNotAvailableError(
                f"No JAX {requirement} devices are available."
            )

        # Resolve and set precision (with float128 fallback warning)
        if precision is not None:
            from radiosim.core.precision import resolve_precision

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

    @property
    def device_kind(self) -> str:
        """Device kind actually in use: ``'cpu'``, ``'gpu'``, or ``'tpu'``.

        Read from the accepted live JAX device rather than from the request.
        Construction rejects an unsupported or mismatched platform, and an
        explicit device requirement never falls back.
        """
        return str(self.device.platform)

    @property
    def supports_compilation(self) -> bool:
        """JAX compiles. See :meth:`compile`."""
        return True

    def compile(self, func: Any) -> Any:
        """Return an XLA-compiled form of ``func`` via :func:`jax.jit`.

        This is the *only* compilation entry point in RadioSim, and the solvers
        apply it to exactly one function: the per-(time, frequency)
        baseline-batched contraction (``Tier6HybridRuntimePlan.md``
        Section 13.6). The uncompiled function remains the reference.
        """
        return jax.jit(func)

    def is_available(self) -> bool:
        """Check if JAX is available with devices."""
        return jax is not None and len(self.devices) > 0

    # =========================================================================
    # Array Creation and Conversion
    # =========================================================================

    def asarray(self, arr: Any, dtype: Any | None = None) -> Any:
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
        """Matrix multiplication on the selected JAX device.

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

    def memory_info(self) -> dict[str, Any]:
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

    def get_device_info(self) -> dict[str, Any]:
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

    def synchronize(self, arr: Any = None) -> Any:
        """Block until ``arr`` is materialized on the device, and return it.

        JAX dispatch is asynchronous, so a timing measurement that does not
        block on the array it produced measures dispatch, not computation.
        Before Tier 6H this method blocked on a freshly constructed throwaway
        constant (``jax.block_until_ready(jnp.array(0))``), which completes
        immediately and orders nothing, making every JAX timing number
        meaningless (``Tier6HybridRuntimePlan.md`` Section 13.6, defect D13).

        Args:
            arr: Array to block on. Omitting it keeps the previous best-effort
                behavior, which does **not** order the caller's own work.

        Returns:
            ``arr`` once ready, or ``None`` when no array was given.
        """
        if arr is None:
            jax.block_until_ready(jnp.array(0))
            return None
        return jax.block_until_ready(arr)

    # =========================================================================
    # JAX-specific methods
    # =========================================================================

    def jit(self, func):
        """Deprecated alias of :meth:`compile`, kept for JAX-specific callers.

        Args:
            func: Function to compile

        Returns:
            JIT-compiled function
        """
        return self.compile(func)

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
    """Check if JAX is available, without importing it.

    Returns:
        True if JAX is installed
    """
    if jax is not None:
        return True
    try:
        return importlib.util.find_spec("jax") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False
