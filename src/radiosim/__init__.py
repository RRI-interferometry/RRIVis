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

from radiosim.__about__ import (
    __author__,
    __description__,
    __email__,
    __license__,
    __version__,
    __version_info__,
)

# High-level API
from radiosim.api.simulator import Simulator

# Backend selection
from radiosim.backends import get_backend, list_backends

# Core functions and canonical instrument models
from radiosim.core import (
    AntennaFieldSource,
    AntennaId,
    AntennaProvenance,
    InstrumentProvenance,
    ResolvedAntenna,
    ResolvedEarthLocation,
    ResolvedInstrument,
    calculate_visibility,
)

# Simulator selection
from radiosim.simulator import (
    RIMESimulator,
    VisibilitySimulator,
    get_simulator,
    list_simulators,
)

# Network & device utilities
from radiosim.utils.device import get_device_resources
from radiosim.utils.network import is_online

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
