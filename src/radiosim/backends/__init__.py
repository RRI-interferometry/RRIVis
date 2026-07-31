"""Lower-level array backend management for RadioSim.

This module selects array implementations and optional devices. Backend or
device selection alone does not establish end-to-end acceleration or numerical
parity for every high-level simulation kernel.

Backends:
- numpy: CPU baseline, always available
- dask: NumPy operations with optional Dask arrays and client (renamed from
  ``numba`` in Tier 6H; it never compiled anything)
- jax: JAX array operations on a device supported by the installed JAX runtime

Usage:
    >>> from radiosim.backends import get_backend, list_backends
    >>>
    >>> # List available backends
    >>> print(list_backends())
    {'numpy': True, 'dask': True, 'jax': True}
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

# Optional backends. ``jax_backend`` itself imports JAX lazily, so importing it
# here costs nothing beyond a module-spec lookup.
try:
    from radiosim.backends.jax_backend import JAXBackend, is_jax_available

    JAX_AVAILABLE = is_jax_available()
except ImportError:  # pragma: no cover - the module has no import-time deps
    JAXBackend = None
    JAX_AVAILABLE = False

    def is_jax_available():
        return False


try:
    from radiosim.backends.dask_backend import DaskBackend, is_dask_available

    DASK_AVAILABLE = is_dask_available()
except ImportError:  # pragma: no cover - exercised only without Dask installed
    DaskBackend = None
    DASK_AVAILABLE = False

    def is_dask_available():
        return False


def _has_non_cpu_jax_device() -> bool:
    """Whether the installed JAX runtime exposes a real accelerator.

    ``auto`` selects JAX only when this is true. RadioSim's JAX dependency is a
    CPU-only build, so on every environment this repository locks the answer is
    ``False`` and ``auto`` resolves to NumPy -- which is exactly what executes
    (``Tier6HybridRuntimePlan.md`` Section 14.1, defect D9).
    """
    if not JAX_AVAILABLE:
        return False
    try:
        import jax

        return bool(jax.devices("tpu")) or bool(jax.devices("gpu"))
    except Exception:
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
        - "auto": JAX when the installed runtime exposes a non-CPU device,
          otherwise NumPy. The Dask backend is never auto-selected, because it
          delegates to NumPy and exists for explicit opt-in only.
        - "numpy" or "cpu": NumPy CPU backend (always available)
        - "dask": NumPy operations with optional Dask arrays and client
        - "jax": JAX arrays on the requested available device
        - "gpu": Request the JAX GPU device
        - "tpu": Request the JAX TPU device

        ``"numba"`` was removed before v1.0: that backend never compiled a
        kernel. Use ``"dask"`` or ``"numpy"``. None of these selections proves
        end-to-end Simulator device execution.
    precision : PrecisionConfig, str, or None
        Precision configuration. Can be:
        - None: Use standard float64 precision
        - str: Preset name ("standard", "fast", "precise", "ultra")
        - PrecisionConfig: Full configuration object
    **kwargs
        Backend-specific options:
        - For dask: mode, n_workers, scheduler_address
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

    >>> # NumPy operations with a local Dask cluster
    >>> backend = get_backend("dask", mode="cpu", n_workers=4)

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

        # Auto precedence (Section 14.1): JAX only when a non-CPU JAX device
        # exists, otherwise NumPy. The Dask backend is deliberately absent from
        # this chain -- selecting it automatically would report "dask" for a run
        # that executes plain NumPy.
        if _has_non_cpu_jax_device():
            import jax

            try:
                if jax.devices("tpu"):
                    return JAXBackend(device="tpu", precision=resolved_precision)
                return JAXBackend(device="gpu", precision=resolved_precision)
            except Exception:
                pass

        # Default to NumPy (always available, and what actually executes here)
        return NumPyBackend(precision=resolved_precision)

    elif name in ("numpy", "cpu"):
        resolved_precision = _require_supported_precision(precision, "numpy")
        return NumPyBackend(precision=resolved_precision)

    elif name == "numba":
        raise ValueError(
            "Unknown backend 'numba': removed before v1.0; the backend never "
            "compiled any kernel. Use get_backend('dask') for the NumPy/Dask "
            "backend or get_backend('numpy')."
        )

    elif name == "dask":
        resolved_precision = _require_supported_precision(precision, "dask")
        if not DASK_AVAILABLE and kwargs.get("mode") == "distributed":
            raise BackendNotAvailableError(
                "Dask not available. Install with: pip install dask[complete]"
            )
        mode = kwargs.get("mode", "cpu")
        n_workers = kwargs.get("n_workers")
        scheduler_address = kwargs.get("scheduler_address")
        return DaskBackend(
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
        if JAX_AVAILABLE:
            try:
                return JAXBackend(device="gpu", precision=resolved_precision)
            except BackendNotAvailableError:
                pass

        raise BackendNotAvailableError(
            "No GPU backend available. Install a GPU-capable JAX build; the JAX "
            "declared by every pixi environment is CPU-only by design."
        )

    elif name == "tpu":
        resolved_precision = _require_supported_precision(precision, "jax")
        if not JAX_AVAILABLE:
            raise BackendNotAvailableError(
                "JAX required for TPU. Install with: pip install radiosim[tpu]"
            )
        return JAXBackend(device="tpu", precision=resolved_precision)

    else:
        available = ["auto", "numpy", "cpu", "dask", "jax", "gpu", "tpu"]
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")


def list_backends() -> dict[str, bool]:
    """List available backends.

    Returns:
        Dictionary mapping backend name to availability

    Examples:
        >>> backends = list_backends()
        >>> print(backends)
        {'numpy': True, 'dask': True, 'jax': True, 'jax_gpu': False,
         'jax_tpu': False}
    """
    backends = {
        "numpy": True,  # Always available
        "dask": DASK_AVAILABLE,
        "jax": JAX_AVAILABLE,
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

    # Dask
    if DaskBackend is not None:
        try:
            backend = DaskBackend(mode="cpu")
            info["dask"] = backend.get_device_info()
        except Exception as e:
            info["dask"] = {"error": str(e)}

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
    "DaskBackend",
    "JAXBackend",
    # Factory functions
    "get_backend",
    "list_backends",
    "get_backend_info",
    # Availability checks
    "is_jax_available",
    "is_dask_available",
]
