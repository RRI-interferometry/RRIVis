"""Visibility simulator implementations for RRIvis.

This module provides the simulator abstraction layer, allowing different
visibility calculation algorithms to be swapped without changing the
user-facing API.

Current Implementations
-----------------------
- **rime**: Direct RIME summation (accurate reference implementation)
    - O(N_src × N_bl × N_freq) complexity
    - Full polarization support
    - GPU acceleration via JAX backend

Future Implementations (v0.3.0+)
--------------------------------
- **fft**: FFT-based NUFFT simulator (10-100× faster for large source counts)
- **matvis**: Matrix-based GPU simulator (HERA standard)

Quick Start
-----------
>>> from rrivis.simulator import get_simulator, list_simulators
>>>
>>> # List available simulators
>>> print(list_simulators())
{'rime': 'Direct RIME summation (accurate reference implementation)'}
>>>
>>> # Get a simulator instance
>>> sim = get_simulator("rime")
>>> print(sim.name, sim.complexity)
rime O(N_src × N_bl × N_freq)
>>>
>>> # Calculate visibilities
>>> visibilities = sim.calculate_visibilities(
...     antennas=antennas,
...     baselines=baselines,
...     sources=sources,
...     frequencies=freqs,
...     backend=backend,
...     location=location,
...     obstime=obstime,
...     wavelengths=wavelengths,
...     duration_seconds=1.0,
...     time_step_seconds=1.0,
... )

API Reference
-------------
get_simulator(name)
    Factory function to get simulator by name.

list_simulators()
    List all available simulators with descriptions.

VisibilitySimulator
    Abstract base class defining the simulator interface.

RIMESimulator
    Direct RIME implementation (default).

See Also
--------
rrivis.backends : Backend abstraction for CPU/GPU/TPU
rrivis.core.visibility : Core visibility calculation
rrivis.core.jones : Jones matrix framework
"""

from rrivis.simulator.base import VisibilitySimulator
from rrivis.simulator.rime import RIMESimulator

# Registry of available simulators
# Maps simulator name -> simulator class
_SIMULATORS: dict[str, type[VisibilitySimulator]] = {
    "rime": RIMESimulator,
}

# Default simulator to use
_DEFAULT_SIMULATOR = "rime"


def get_simulator(name: str = "rime") -> VisibilitySimulator:
    """
    Get a visibility simulator instance by name.

    This is the primary factory function for obtaining simulator instances.
    Use this instead of instantiating simulator classes directly to ensure
    proper initialization and future compatibility.

    Parameters
    ----------
    name : str, optional
        Simulator name. Available options:
            - "rime": Direct RIME summation (accurate, default)

        Future options (v0.3.0+):
            - "fft": FFT-based NUFFT (fast for many sources)
            - "matvis": Matrix-based GPU (HERA standard)

        Default is "rime".

    Returns
    -------
    VisibilitySimulator
        Simulator instance ready to calculate visibilities.

    Raises
    ------
    ValueError
        If the requested simulator name is not available.

    Examples
    --------
    >>> from rrivis.simulator import get_simulator
    >>>
    >>> # Get default (RIME) simulator
    >>> sim = get_simulator()
    >>> print(sim.name)
    rime
    >>>
    >>> # Explicitly request RIME
    >>> sim = get_simulator("rime")
    >>>
    >>> # Check properties
    >>> print(sim.complexity)
    O(N_src × N_bl × N_freq)
    >>> print(sim.supports_gpu)
    True

    See Also
    --------
    list_simulators : List all available simulators
    VisibilitySimulator : Abstract base class
    """
    if name not in _SIMULATORS:
        available = list(_SIMULATORS.keys())
        raise ValueError(
            f"Unknown simulator '{name}'. "
            f"Available simulators: {available}. "
            f"Use list_simulators() to see descriptions."
        )

    return _SIMULATORS[name]()


def list_simulators() -> dict[str, str]:
    """
    List all available simulators with their descriptions.

    Returns a dictionary mapping simulator names to their human-readable
    descriptions. Use this to discover available simulators and their
    characteristics.

    Returns
    -------
    dict
        Dictionary mapping simulator name (str) to description (str).

    Examples
    --------
    >>> from rrivis.simulator import list_simulators
    >>>
    >>> sims = list_simulators()
    >>> for name, desc in sims.items():
    ...     print(f"{name}: {desc}")
    rime: Direct RIME summation (accurate reference implementation)
    >>>
    >>> # Check if a specific simulator is available
    >>> if "fft" in list_simulators():
    ...     sim = get_simulator("fft")

    See Also
    --------
    get_simulator : Get a simulator instance by name
    """
    return {name: cls().description for name, cls in _SIMULATORS.items()}


def get_simulator_names() -> list[str]:
    """
    Get list of available simulator names.

    Convenience function returning just the names without descriptions.
    Useful for programmatic iteration.

    Returns
    -------
    list
        List of available simulator names.

    Examples
    --------
    >>> from rrivis.simulator import get_simulator_names
    >>> names = get_simulator_names()
    >>> print(names)
    ['rime']
    """
    return list(_SIMULATORS.keys())


def get_default_simulator() -> str:
    """
    Get the name of the default simulator.

    Returns
    -------
    str
        Default simulator name ("rime").

    Examples
    --------
    >>> from rrivis.simulator import get_default_simulator, get_simulator
    >>> default = get_default_simulator()
    >>> sim = get_simulator(default)
    """
    return _DEFAULT_SIMULATOR


# Public API
__all__ = [
    # Base class
    "VisibilitySimulator",
    # Implementations
    "RIMESimulator",
    # Factory functions
    "get_simulator",
    "list_simulators",
    "get_simulator_names",
    "get_default_simulator",
]
