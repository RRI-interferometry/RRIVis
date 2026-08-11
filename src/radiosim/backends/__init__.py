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
>>> # Discover the reported backend keys; NumPy is always available
>>> availability = list_backends()
>>> sorted(availability)
['dask', 'jax', 'jax_gpu', 'jax_tpu', 'numpy']
>>> availability["numpy"]
True
>>>
>>> # Select a backend explicitly.  ``"auto"`` is deterministic NumPy and
>>> # does not probe optional runtimes.
>>> backend = get_backend("numpy")
>>> backend.name
'numpy-cpu'
>>>
>>> # Use backend for computation
>>> xp = backend.xp
>>> arr = backend.asarray([1, 2, 3])
>>> result = backend.exp(arr)

With precision control:

>>> from radiosim.core.precision import PrecisionConfig
>>> get_backend("numpy", precision="fast").name  # Use fast preset
'numpy-cpu'
>>> get_backend("numpy", precision=PrecisionConfig.standard()).name
'numpy-cpu'

``PrecisionConfig.precise()`` and ``PrecisionConfig.ultra()`` request
``float128``, which the NumPy backend rejects with
``BackendNotAvailableError`` on platforms that do not provide it (Apple
Silicon among them), so they are not shown as executed examples here.
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

        - "auto": deterministic NumPy selection. It never imports or probes
          JAX and never selects Dask.
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

        - None: Use the standard precision preset
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
    >>> # Deterministic automatic selection
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
        # PERF-001 P-c deliberately separates deterministic selection from
        # explicit discovery. Precision resolution and NumPy validation do not
        # touch an optional backend module or runtime.
        resolved_precision = _require_supported_precision(precision, "numpy")
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
                "  pip install radiosim[jax]\n"
                "The device-named extras were removed before 0.3.0: RadioSim "
                "has measured no accelerator, so supply a vendor JAX wheel "
                "yourself if you have one."
            )
        # An omitted device delegates selection to the JAX runtime. Explicit
        # cpu/gpu/tpu values are strict requirements in JAXBackend.
        device = kwargs.get("device")
        return JAXBackend(device=device, precision=resolved_precision)

    elif name == "gpu":
        resolved_precision = _require_supported_precision(precision, "jax")
        if not JAX_AVAILABLE:
            raise BackendNotAvailableError(
                "No GPU backend available. Install a GPU-capable JAX build."
            )
        return JAXBackend(device="gpu", precision=resolved_precision)

    elif name == "tpu":
        resolved_precision = _require_supported_precision(precision, "jax")
        if not JAX_AVAILABLE:
            raise BackendNotAvailableError(
                "JAX required for TPU. Install with: pip install radiosim[jax] "
                "plus the TPU JAX wheel for your runtime; RadioSim ships no "
                "device-named extra, because it has measured no accelerator."
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
        "jax": False,
    }

    # Explicit discovery may initialize JAX. Probe accelerator plugins
    # independently because one failed plugin says nothing about the other.
    if JAX_AVAILABLE:
        try:
            import jax
        except Exception:
            backends["jax_gpu"] = False
            backends["jax_tpu"] = False
        else:
            try:
                backends["jax"] = bool(jax.devices())
            except Exception:
                backends["jax"] = False
            for key, device_kind in (("jax_gpu", "gpu"), ("jax_tpu", "tpu")):
                try:
                    backends[key] = bool(jax.devices(device_kind))
                except Exception:
                    backends[key] = False
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
        >>> info["numpy"]["device"]
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
            backend = JAXBackend(device=None)
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
