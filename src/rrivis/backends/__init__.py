"""Hardware backend management for RRIvis.

This module provides abstraction over different computational backends
for CPU/GPU/TPU acceleration of visibility calculations.

Backends:
- numpy: CPU (x86 + ARM), always available, baseline implementation
- numba: CPU/GPU with JIT compilation + Dask distributed (production)
- jax: GPU/TPU with auto-differentiation (research)

Usage:
    >>> from rrivis.backends import get_backend, list_backends
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
    >>> from rrivis.core.precision import PrecisionConfig
    >>> backend = get_backend("numpy", precision="fast")  # Use fast preset
    >>> backend = get_backend("numpy", precision=PrecisionConfig.precise())
"""

from typing import Dict, Optional, Union, TYPE_CHECKING

from rrivis.backends.base import ArrayBackend, BackendNotAvailableError
from rrivis.backends.numpy_backend import NumPyBackend

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

# Try to import optional backends
try:
    from rrivis.backends.jax_backend import JAXBackend, is_jax_available
    JAX_AVAILABLE = is_jax_available()
except ImportError:
    JAXBackend = None
    JAX_AVAILABLE = False

    def is_jax_available():
        return False

try:
    from rrivis.backends.numba_backend import (
        NumbaBackend,
        is_numba_available,
        is_cuda_available,
        is_dask_available,
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


def get_backend(
    name: str = "auto",
    precision: Optional[Union["PrecisionConfig", str]] = None,
    **kwargs
) -> ArrayBackend:
    """Get computation backend.

    Parameters
    ----------
    name : str
        Backend name:
        - "auto": Auto-detect best available (recommended)
        - "numpy" or "cpu": NumPy CPU backend (always available)
        - "numba": Numba JIT backend (CPU/GPU)
        - "jax": JAX GPU/TPU backend
        - "gpu": Best GPU backend (JAX or Numba CUDA)
        - "tpu": JAX TPU backend
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

    >>> # Numba GPU (CUDA)
    >>> backend = get_backend("numba", mode="gpu")

    >>> # JAX GPU (vendor-agnostic) with precision
    >>> from rrivis.core.precision import PrecisionConfig
    >>> backend = get_backend("jax", precision=PrecisionConfig.fast())

    >>> # JAX TPU
    >>> backend = get_backend("jax", device="tpu")
    """
    name = name.lower()

    if name == "auto":
        # Auto-detect best available backend
        # Priority: GPU (JAX) > GPU (Numba CUDA) > CPU (Numba) > CPU (NumPy)
        if JAX_AVAILABLE:
            try:
                import jax
                # Check for GPU or TPU
                if jax.devices("tpu"):
                    return JAXBackend(device="tpu", precision=precision)
                elif jax.devices("gpu"):
                    return JAXBackend(device="gpu", precision=precision)
            except Exception:
                pass

        if NUMBA_AVAILABLE:
            if is_cuda_available():
                try:
                    return NumbaBackend(mode="gpu", precision=precision)
                except BackendNotAvailableError:
                    pass
            # Fall back to Numba CPU
            try:
                return NumbaBackend(mode="cpu", precision=precision)
            except BackendNotAvailableError:
                pass

        # Default to NumPy (always available)
        return NumPyBackend(precision=precision)

    elif name in ("numpy", "cpu"):
        return NumPyBackend(precision=precision)

    elif name == "numba":
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
            precision=precision,
        )

    elif name == "jax":
        if not JAX_AVAILABLE:
            raise BackendNotAvailableError(
                "JAX not available. Install with:\n"
                "  pip install rrivis[gpu]        # Generic\n"
                "  pip install rrivis[gpu-cuda]   # NVIDIA\n"
                "  pip install rrivis[gpu-rocm]   # AMD"
            )
        device = kwargs.get("device", "gpu")
        return JAXBackend(device=device, precision=precision)

    elif name == "gpu":
        # Best GPU backend
        if JAX_AVAILABLE:
            try:
                return JAXBackend(device="gpu", precision=precision)
            except BackendNotAvailableError:
                pass

        if NUMBA_AVAILABLE and is_cuda_available():
            try:
                return NumbaBackend(mode="gpu", precision=precision)
            except BackendNotAvailableError:
                pass

        raise BackendNotAvailableError(
            "No GPU backend available. Install JAX or Numba with CUDA support."
        )

    elif name == "tpu":
        if not JAX_AVAILABLE:
            raise BackendNotAvailableError(
                "JAX required for TPU. Install with: pip install rrivis[tpu]"
            )
        return JAXBackend(device="tpu", precision=precision)

    else:
        available = ["auto", "numpy", "cpu", "numba", "jax", "gpu", "tpu"]
        raise ValueError(
            f"Unknown backend '{name}'. Available: {available}"
        )


def list_backends() -> Dict[str, bool]:
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


def get_backend_info() -> Dict[str, Dict]:
    """Get detailed information about all available backends.

    Returns:
        Dictionary with backend details

    Examples:
        >>> info = get_backend_info()
        >>> print(info['numpy']['device'])
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
