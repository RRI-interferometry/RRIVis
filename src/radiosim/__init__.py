"""
RadioSim: Radio Astronomy Visibility Simulator

A modern Python package for simulating radio interferometer visibilities
with full polarization support and GPU acceleration.

Basic usage:
    >>> import radiosim
    >>> print(radiosim.__version__)
    '0.2.0'

    >>> # High-level API
    >>> sim = radiosim.Simulator.from_config("config.yaml")
    >>> results = sim.run()
    >>> sim.plot()
    >>> sim.save("output/")

    >>> # Programmatic API
    >>> sim = radiosim.Simulator(
    ...     config={
    ...         "antenna_layout": {
    ...             "antenna_positions_file": "HERA65.csv",
    ...             "antenna_file_format": "radiosim",
    ...             "all_antenna_diameter": 14.0,
    ...         },
    ...         "obs_frequency": {
    ...             "frequencies_hz": [100e6, 150e6, 200e6],
    ...             "frequency_unit": "MHz",
    ...         },
    ...         "sky_model": {"sources": [{"kind": "gleam"}]},
    ...         "visibility": {"sky_representation": "point_sources"},
    ...     },
    ... )
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

# Core functions (for advanced users)
# Note: These imports may fail until import updates are complete
# They will be enabled once all modules are updated
try:
    from radiosim.core import (
        calculate_visibility,
        generate_baselines,
        read_antenna_positions,
    )

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False
    calculate_visibility = None
    read_antenna_positions = None
    generate_baselines = None

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
    # Network & device utilities
    "is_online",
    "get_device_resources",
    # Core functions (when available)
    "calculate_visibility",
    "read_antenna_positions",
    "generate_baselines",
]
