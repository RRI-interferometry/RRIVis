"""Lazy access to healpy with actionable import errors.

Healpy is a core RadioSim dependency (required for HEALPix sky operations).
Callers that only touch point-source paths should not import it at module load;
functions that need HEALPix geometry import it on first use via :func:`_healpy`.
"""

from __future__ import annotations

from typing import Any

_HEALPY_MODULE: Any = None

HEALPY_IMPORT_ERROR_MESSAGE = (
    "RadioSim requires healpy for HEALPix sky operations. "
    "Reinstall project dependencies with `pixi install` "
    "(or `pip install healpy`)."
)


def _healpy() -> Any:
    """Return the healpy module, importing it on first use."""
    global _HEALPY_MODULE
    if _HEALPY_MODULE is None:
        try:
            import healpy as hp
        except ImportError as exc:
            raise ImportError(HEALPY_IMPORT_ERROR_MESSAGE) from exc
        _HEALPY_MODULE = hp
    return _HEALPY_MODULE


class _LazyHealpyModule:
    """Deferred healpy accessor — attribute lookups import healpy on demand."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_healpy(), name)

    def __dir__(self) -> list[str]:
        return dir(_healpy())


lazy_healpy = _LazyHealpyModule()


def healpy_rotator(*args: Any, **kwargs: Any) -> Any:
    """Construct a :class:`healpy.rotator.Rotator` on demand."""
    from healpy.rotator import Rotator

    return Rotator(*args, **kwargs)
