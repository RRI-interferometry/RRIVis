"""Loader registry for SkyModel factory methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_LOADERS: dict[str, Callable[..., Any]] = {}


def register_loader(name: str) -> Callable:
    """Decorator: register a sky model loader function by name."""

    def decorator(func: Callable) -> Callable:
        _LOADERS[name] = func
        return func

    return decorator


def get_loader(name: str) -> Callable:
    """Get a registered loader by name."""
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown sky model loader '{name}'. Available: {sorted(_LOADERS.keys())}"
        )
    return _LOADERS[name]


def list_loaders() -> list[str]:
    """List registered loader names."""
    return sorted(_LOADERS.keys())
