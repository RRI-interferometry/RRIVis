"""High-level sky-model recipes built from RadioSim primitives.

Recipes compose loaders, combine, and provenance tagging into ready-made
sky models for common simulation scenarios.
"""

from __future__ import annotations

from .realistic_foreground import realistic_foreground_sky

__all__ = [
    "realistic_foreground_sky",
]
