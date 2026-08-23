"""Visibility simulator implementations for RadioSim.

This module provides the simulator abstraction layer, allowing different
visibility calculation algorithms to be swapped without changing the
user-facing API.

Current Implementations
-----------------------
- **rime**: Direct RIME summation (accurate reference implementation)
    - O(N_src × N_bl × N_freq) complexity
    - Full polarization support
    - Array work routed through the selected NumPy/JAX/Dask backend, one
      compiled kernel under JAX, and no measured accelerator run:
      ``RIMESimulator.supports_gpu`` is ``False`` and every measured JAX run is
      slower than NumPy (records: ``output/benchmarks/reference/``,
      register row ``PERF-001``)
- **mmode**: m-mode full-sidereal harmonic forward model, SCI-004
    - A second *complete* forward model, not an optimization of the direct sum
    - Full Stokes: ``supports_polarization`` is ``True``, and a payload with
      non-zero ``Q``, ``U`` or ``V`` takes the polarized execution path
    - ``supports_gpu`` is ``False``; the recorded transform execution policy
      ``host_harmonics_backend_native_dense_v1`` splits host-side harmonics from
      backend-native dense work and is **not** an accelerator claim. A
      polarized capability is not a speed claim

Registry
--------
The registry holds ``rime`` and ``mmode``, and its keys are exactly the accepted
values of ``execution.simulator``.  An FFT/NUFFT solver and a matrix-based
solver are candidates for a future registration, not shipped code and not a
promise attached to a version number; nothing here measures or claims what they
would cost.

Quick Start
-----------
>>> from radiosim.simulator import get_simulator, list_simulators
>>>
>>> # List available simulators
>>> for name in sorted(list_simulators()):
...     print(name)
mmode
rime
>>>
>>> # Get a simulator instance
>>> sim = get_simulator("rime")
>>> print(sim.name, sim.complexity)
rime O(N_src × N_bl × N_freq)

The solver inputs are built by :meth:`radiosim.api.Simulator.setup`, so the
call itself is illustrative rather than executed:

.. code-block:: python

    visibilities = sim.calculate_visibilities(
        instrument=instrument,
        beam_system=beam_system,
        source_arrays=source_arrays,
        frequencies=frequencies,
        backend=backend,
        location=location,
        time_grid=time_grid,
    )
    # Backend-native receptor cube: (time, baseline, frequency, 2, 2)
    assert visibilities.shape == (1, 15, 2, 2, 2)

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
radiosim.backends : Backend abstraction for CPU/GPU/TPU
radiosim.core.visibility : Core visibility calculation
radiosim.core.jones : Jones matrix framework
"""

from radiosim.simulator.base import (
    SkySolveOutcome,
    SkySolveRequest,
    VisibilitySimulator,
)
from radiosim.simulator.mmode import MModeSimulator
from radiosim.simulator.rime import RIMESimulator

# Registry of available simulators
# Maps simulator name -> simulator class.  ``docs/development/
# sci004_mmode_design.md`` Section 2 keeps one standing invariant exact:
# ``accepted values of execution.simulator == simulator registry keys``.  A new
# algorithm therefore arrives as a registry entry the single selector already
# honours, never as a second unread configuration field.
_SIMULATORS: dict[str, type[VisibilitySimulator]] = {
    "rime": RIMESimulator,
    "mmode": MModeSimulator,
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
        Simulator name. The accepted values are exactly the keys of the
        registry, which holds ``"rime"`` -- direct RIME summation, and the
        default -- and ``"mmode"``, the SCI-004 m-mode full-sidereal harmonic
        forward model. Any other name raises ``ValueError`` naming the
        registered set, so this docstring cannot drift into advertising a
        solver that is not registered.

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
    >>> from radiosim.simulator import get_simulator
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
    False

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
    >>> from radiosim.simulator import list_simulators
    >>>
    >>> sims = list_simulators()
    >>> print(sims["rime"])
    Direct RIME summation (accurate reference implementation)
    >>> print(sims["mmode"])
    m-mode full-sidereal harmonic forward model (full Stokes)
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
    >>> from radiosim.simulator import get_simulator_names
    >>> names = get_simulator_names()
    >>> print(sorted(names))
    ['mmode', 'rime']
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
    >>> from radiosim.simulator import get_default_simulator, get_simulator
    >>> default = get_default_simulator()
    >>> sim = get_simulator(default)
    """
    return _DEFAULT_SIMULATOR


# Public API
__all__ = [
    # Base class
    "VisibilitySimulator",
    # The whole-SkyModel strategy boundary
    "SkySolveOutcome",
    "SkySolveRequest",
    # Implementations
    "MModeSimulator",
    "RIMESimulator",
    # Factory functions
    "get_simulator",
    "list_simulators",
    "get_simulator_names",
    "get_default_simulator",
]
