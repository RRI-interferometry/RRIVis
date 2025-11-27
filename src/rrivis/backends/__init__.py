"""Hardware backend management for RRIvis.

This module provides abstraction over different computational backends
(NumPy, JAX, Numba) for CPU/GPU/TPU acceleration.

Note: Full backend implementation planned for Phase 2.
Currently defaults to NumPy backend.
"""

from typing import Dict


def get_backend(name: str = "auto"):
    """
    Get computation backend.

    Args:
        name: Backend name
            - "auto": Auto-detect best available (recommended)
            - "numpy": NumPy CPU backend (x86 + ARM)
            - "jax": JAX GPU/TPU backend
            - "numba": Numba JIT backend

    Returns:
        Backend instance (currently returns None, uses NumPy by default)

    Note:
        Full backend implementation coming in Phase 2.
        Currently all operations use NumPy.
    """
    # Phase 2 will implement full backend abstraction
    # For now, return None to indicate default NumPy behavior
    return None


def list_backends() -> Dict[str, bool]:
    """
    List available backends.

    Returns:
        Dictionary mapping backend name to availability
    """
    return {
        "numpy": True,  # Always available
        "jax": False,   # Phase 2
        "numba": False, # Phase 2
    }


__all__ = [
    "get_backend",
    "list_backends",
]
