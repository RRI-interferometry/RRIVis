"""Lower-level array backend management for RadioSim.

This module selects array implementations and optional devices. Backend or
device selection alone does not establish end-to-end acceleration or numerical
parity for every high-level simulation kernel.

Backends:
- numpy: CPU baseline, always available
- numba: NumPy/Dask operations plus an explicit Numba JIT helper
- jax: JAX array operations on a device supported by the installed JAX runtime

Usage:
    >>> from radiosim.backends import get_backend, list_backends
    >>>
    >>> # List available backends
    >>> print(list_backends())
    {'numpy': True, 'numba': True, 'jax': False}
    >>>
    >>> # Get backend (auto-detect best available)
    >>> backend = get_backend("auto")
    >>> print(backend.name)
    'numpy-cpu'
    >>>
    >>> # Use backend for computation
    >>> xp = backend.xp
    >>> arr = backend.asarray([1, 2, 3])
    >>> result = backend.exp(arr)

With precision control:
    >>> from radiosim.core.precision import PrecisionConfig
    >>> backend = get_backend("numpy", precision="fast")  # Use fast preset
    >>> backend = get_backend("numpy", precision=PrecisionConfig.precise())
"""

from typing import TYPE_CHECKING, Union

from radiosim.backends.base import ArrayBackend, BackendNotAvailableError
from radiosim.backends.numpy_backend import NumPyBackend

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

# Try to import optional backends
try:
    from radiosim.backends.jax_backend import JAXBackend, is_jax_available

    JAX_AVAILABLE = is_jax_available()
except ImportError:
    JAXBackend = None
    JAX_AVAILABLE = False

    def is_jax_available():
        return False


try:
    from radiosim.backends.numba_backend import (
        NumbaBackend,
        is_cuda_available,
        is_dask_available,
        is_numba_available,
    )

    NUMBA_AVAILABLE = is_numba_available()
except ImportError:
    NumbaBackend = None
    NUMBA_AVAILABLE = False

    def is_numba_available():
        return False

    def is_cuda_available():
        return False

    def is_dask_available():
        return False


def _require_supported_precision(
    precision: Union["PrecisionConfig", str] | None,
    backend_name: str,
) -> "PrecisionConfig":
    """Resolve precision and reject any backend dtype downgrade."""
    from radiosim.core.precision import resolve_precision

    resolved = resolve_precision(precision)
    issues = resolved.validate_for_backend(backend_name)
    if issues:
        raise BackendNotAvailableError(
            f"The {backend_name} backend cannot honor the requested precision: "
            + "; ".join(issues)
        )
    return resolved


def get_backend(
    name: str = "auto", precision: Union["PrecisionConfig", str] | None = None, **kwargs
) -> ArrayBackend:
    """Get computation backend.

    Parameters
    ----------
    name : str
        Backend name:
        - "auto": Select an available backend/device strategy
        - "numpy" or "cpu": NumPy CPU backend (always available)
        - "numba": NumPy/Dask operations with a Numba JIT helper
        - "jax": JAX arrays on the requested available device
        - "gpu": Request a lower-level GPU-capable backend wrapper
        - "tpu": Request the JAX TPU device

        The Numba ``mode="gpu"`` path currently detects and reports a CUDA
        device, but its common array operations remain NumPy/Dask operations.
        None of these selections proves end-to-end Simulator device execution.
    precision : PrecisionConfig, str, or None
        Precision configuration. Can be:
        - None: Use standard float64 precision
        - str: Preset name ("standard", "fast", "precise", "ultra")
        - PrecisionConfig: Full configuration object
    **kwargs
        Backend-specific options:
        - For numba: mode, n_workers, scheduler_address
        - For jax: device

    Returns
    -------
    ArrayBackend
        Configured backend instance

    Raises
    ------
    BackendNotAvailableError
        If requested backend is unavailable
    ValueError
        If backend name is unknown

    Examples
    --------
    >>> # Auto-detect (recommended)
    >>> backend = get_backend("auto")

    >>> # Force CPU (NumPy)
    >>> backend = get_backend("numpy")

    >>> # With precision control
    >>> backend = get_backend("numpy", precision="fast")

    >>> # Numba with parallel CPU
    >>> backend = get_backend("numba", mode="cpu", n_workers=4)

    >>> # JAX on an explicitly requested device, when available
    >>> from radiosim.core.precision import PrecisionConfig
    >>> backend = get_backend("jax", device="cpu", precision=PrecisionConfig.fast())
    """
    name = name.lower()

    if name == "auto":
        from radiosim.core.precision import resolve_precision

        resolved_precision = resolve_precision(precision)
        if resolved_precision.validate_for_backend("jax"):
            numpy_issues = resolved_precision.validate_for_backend("numpy")
            if numpy_issues:
                raise BackendNotAvailableError(
                    "No installed backend can honor the requested precision: "
                    + "; ".join(numpy_issues)
                )
            return NumPyBackend(precision=resolved_precision)

        # Auto-detect best available backend
        # Priority: GPU (JAX) > GPU (Numba CUDA) > CPU (Numba) > CPU (NumPy)
        if JAX_AVAILABLE:
            try:
                import jax

                # Check for GPU or TPU
                if jax.devices("tpu"):
                    return JAXBackend(device="tpu", precision=resolved_precision)
                elif jax.devices("gpu"):
                    return JAXBackend(device="gpu", precision=resolved_precision)
            except Exception:
                pass

        if NUMBA_AVAILABLE:
            if is_cuda_available():
                try:
                    return NumbaBackend(mode="gpu", precision=resolved_precision)
                except BackendNotAvailableError:
                    pass
            # Fall back to Numba CPU
            try:
                return NumbaBackend(mode="cpu", precision=resolved_precision)
            except BackendNotAvailableError:
                pass

        # Default to NumPy (always available)
        return NumPyBackend(precision=resolved_precision)

    elif name in ("numpy", "cpu"):
        resolved_precision = _require_supported_precision(precision, "numpy")
        return NumPyBackend(precision=resolved_precision)

    elif name == "numba":
        resolved_precision = _require_supported_precision(precision, "numba")
        if not NUMBA_AVAILABLE:
            raise BackendNotAvailableError(
                "Numba not available. Install with: pip install numba dask[complete]"
            )
        mode = kwargs.get("mode", "cpu")
        n_workers = kwargs.get("n_workers")
        scheduler_address = kwargs.get("scheduler_address")
        return NumbaBackend(
            mode=mode,
            n_workers=n_workers,
            scheduler_address=scheduler_address,
            precision=resolved_precision,
        )

    elif name == "jax":
        resolved_precision = _require_supported_precision(precision, "jax")
        if not JAX_AVAILABLE:
            raise BackendNotAvailableError(
                "JAX not available. Install with:\n"
                "  pip install radiosim[gpu]        # Generic\n"
                "  pip install radiosim[gpu-cuda]   # NVIDIA\n"
                "  pip install radiosim[gpu-rocm]   # AMD"
            )
        device = kwargs.get("device", "gpu")
        return JAXBackend(device=device, precision=resolved_precision)

    elif name == "gpu":
        resolved_precision = _require_supported_precision(precision, "jax")
        # Best GPU backend
        if JAX_AVAILABLE:
            try:
                return JAXBackend(device="gpu", precision=resolved_precision)
            except BackendNotAvailableError:
                pass

        if NUMBA_AVAILABLE and is_cuda_available():
            try:
                return NumbaBackend(mode="gpu", precision=resolved_precision)
            except BackendNotAvailableError:
                pass

        raise BackendNotAvailableError(
            "No GPU backend available. Install JAX or Numba with CUDA support."
        )

    elif name == "tpu":
        resolved_precision = _require_supported_precision(precision, "jax")
        if not JAX_AVAILABLE:
            raise BackendNotAvailableError(
                "JAX required for TPU. Install with: pip install radiosim[tpu]"
            )
        return JAXBackend(device="tpu", precision=resolved_precision)

    else:
        available = ["auto", "numpy", "cpu", "numba", "jax", "gpu", "tpu"]
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")


def list_backends() -> dict[str, bool]:
    """List available backends.

    Returns:
        Dictionary mapping backend name to availability

    Examples:
        >>> backends = list_backends()
        >>> print(backends)
        {'numpy': True, 'numba': True, 'jax': False, 'cuda': False, 'tpu': False}
    """
    backends = {
        "numpy": True,  # Always available
        "numba": NUMBA_AVAILABLE,
        "jax": JAX_AVAILABLE,
        "cuda": is_cuda_available() if NUMBA_AVAILABLE else False,
        "dask": is_dask_available() if NUMBA_AVAILABLE else False,
    }

    # Check for GPU/TPU via JAX
    if JAX_AVAILABLE:
        try:
            import jax

            backends["jax_gpu"] = len(jax.devices("gpu")) > 0
            backends["jax_tpu"] = len(jax.devices("tpu")) > 0
        except Exception:
            backends["jax_gpu"] = False
            backends["jax_tpu"] = False
    else:
        backends["jax_gpu"] = False
        backends["jax_tpu"] = False

    return backends


def get_backend_info() -> dict[str, dict]:
    """Get detailed information about all available backends.

    Returns:
        Dictionary with backend details

    Examples:
        >>> info = get_backend_info()
        >>> print(info["numpy"]["device"])
        'CPU'
    """
    info = {}

    # NumPy (always available)
    try:
        backend = NumPyBackend()
        info["numpy"] = backend.get_device_info()
    except Exception as e:
        info["numpy"] = {"error": str(e)}

    # Numba
    if NUMBA_AVAILABLE:
        try:
            backend = NumbaBackend(mode="cpu")
            info["numba"] = backend.get_device_info()
        except Exception as e:
            info["numba"] = {"error": str(e)}

    # JAX
    if JAX_AVAILABLE:
        try:
            backend = JAXBackend(device="cpu")
            info["jax"] = backend.get_device_info()
        except Exception as e:
            info["jax"] = {"error": str(e)}

    return info


__all__ = [
    # Base classes
    "ArrayBackend",
    "BackendNotAvailableError",
    # Backend implementations
    "NumPyBackend",
    "NumbaBackend",
    "JAXBackend",
    # Factory functions
    "get_backend",
    "list_backends",
    "get_backend_info",
    # Availability checks
    "is_jax_available",
    "is_numba_available",
    "is_cuda_available",
    "is_dask_available",
]
