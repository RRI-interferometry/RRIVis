"""
RadioSim: Radio Astronomy Visibility Simulator

A modern Python package for simulating radio interferometer visibilities
with strict, source-aware configuration and selectable computation backends.

Basic usage:
    >>> import radiosim
    >>> print(radiosim.__version__)
    '0.2.0'

    >>> # High-level API
    >>> sim = radiosim.Simulator.from_yaml("config.yaml")
    >>> results = sim.run()
    >>> sim.plot()
    >>> sim.save("output/")

    >>> # Programmatic API
    >>> sim = radiosim.Simulator.from_mapping(config_data, base_dir=project_dir)
    >>> results = sim.run()

For more information, see https://github.com/RRI-interferometry/RadioSim
"""

from importlib import import_module
from typing import TYPE_CHECKING

from radiosim.__about__ import (
    __author__,
    __description__,
    __email__,
    __license__,
    __version__,
    __version_info__,
)

if TYPE_CHECKING:
    from radiosim.api.simulator import Simulator
    from radiosim.backends import get_backend, list_backends
    from radiosim.core import (
        AntennaFieldSource,
        AntennaId,
        AntennaProvenance,
        InstrumentProvenance,
        LoadedSimulationResult,
        ObservationTimeGrid,
        PhaseCenter,
        ResolvedAntenna,
        ResolvedEarthLocation,
        ResolvedInstrument,
        SimulationResult,
        calculate_visibility,
    )
    from radiosim.simulator import (
        RIMESimulator,
        VisibilitySimulator,
        get_simulator,
        list_simulators,
    )
    from radiosim.utils.device import get_device_resources
    from radiosim.utils.network import is_online


_LAZY_EXPORTS = {
    "Simulator": ("radiosim.api.simulator", "Simulator"),
    "SimulationResult": ("radiosim.core.result", "SimulationResult"),
    "LoadedSimulationResult": (
        "radiosim.core.result",
        "LoadedSimulationResult",
    ),
    "ObservationTimeGrid": (
        "radiosim.core.time_grid",
        "ObservationTimeGrid",
    ),
    "PhaseCenter": ("radiosim.core.phase_center", "PhaseCenter"),
    "get_backend": ("radiosim.backends", "get_backend"),
    "list_backends": ("radiosim.backends", "list_backends"),
    "AntennaId": ("radiosim.core", "AntennaId"),
    "AntennaFieldSource": ("radiosim.core", "AntennaFieldSource"),
    "ResolvedEarthLocation": ("radiosim.core", "ResolvedEarthLocation"),
    "AntennaProvenance": ("radiosim.core", "AntennaProvenance"),
    "ResolvedAntenna": ("radiosim.core", "ResolvedAntenna"),
    "InstrumentProvenance": ("radiosim.core", "InstrumentProvenance"),
    "ResolvedInstrument": ("radiosim.core", "ResolvedInstrument"),
    "calculate_visibility": ("radiosim.core", "calculate_visibility"),
    "get_simulator": ("radiosim.simulator", "get_simulator"),
    "list_simulators": ("radiosim.simulator", "list_simulators"),
    "VisibilitySimulator": ("radiosim.simulator", "VisibilitySimulator"),
    "RIMESimulator": ("radiosim.simulator", "RIMESimulator"),
    "is_online": ("radiosim.utils.network", "is_online"),
    "get_device_resources": ("radiosim.utils.device", "get_device_resources"),
}


def __getattr__(name: str) -> object:
    """Load public API objects only when they are first accessed."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public API objects in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    # Metadata
    "__version__",
    "__version_info__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    # High-level API
    "Simulator",
    "SimulationResult",
    "LoadedSimulationResult",
    "ObservationTimeGrid",
    "PhaseCenter",
    # Backend selection
    "get_backend",
    "list_backends",
    # Simulator selection
    "get_simulator",
    "list_simulators",
    "VisibilitySimulator",
    "RIMESimulator",
    # Canonical instrument models
    "AntennaId",
    "AntennaFieldSource",
    "ResolvedEarthLocation",
    "AntennaProvenance",
    "ResolvedAntenna",
    "InstrumentProvenance",
    "ResolvedInstrument",
    # Network & device utilities
    "is_online",
    "get_device_resources",
    # Core functions
    "calculate_visibility",
]
