"""Simulator abstraction for RRIvis.

This module provides the simulator interface for different
visibility calculation algorithms.

Note: Full simulator abstraction planned for Phase 3.
Currently uses direct RIME implementation.
"""

from typing import List


def get_simulator(name: str = "rime"):
    """
    Get visibility simulator.

    Args:
        name: Simulator name
            - "rime": Direct RIME calculation (default)

    Returns:
        Simulator instance (currently returns None, uses direct RIME)

    Note:
        Full simulator abstraction coming in Phase 3.
    """
    return None


def list_simulators() -> List[str]:
    """
    List available simulators.

    Returns:
        List of available simulator names
    """
    return ["rime"]


__all__ = [
    "get_simulator",
    "list_simulators",
]
