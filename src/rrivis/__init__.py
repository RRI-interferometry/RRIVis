"""
RRIvis: Radio Astronomy Visibility Simulator

A modern Python package for simulating radio interferometer visibilities
with full polarization support and GPU acceleration.

Basic usage:
    >>> import rrivis
    >>> print(rrivis.__version__)
    '0.2.0'

    >>> # High-level API
    >>> sim = rrivis.Simulator.from_config("config.yaml")
    >>> results = sim.run()
    >>> sim.plot()
    >>> sim.save("output/")

    >>> # Programmatic API
    >>> sim = rrivis.Simulator(
    ...     antenna_layout="HERA65.csv",
    ...     frequencies=[100, 150, 200],
    ...     sky_model="gleam",
    ... )
    >>> results = sim.run()

For more information, see https://github.com/kartikmandar/RRIvis
"""

from rrivis.__about__ import (
    __author__,
    __description__,
    __email__,
    __license__,
    __version__,
    __version_info__,
)

# High-level API
from rrivis.api.simulator import Simulator

# Backend selection
from rrivis.backends import get_backend, list_backends

# Simulator selection
from rrivis.simulator import (
    RIMESimulator,
    VisibilitySimulator,
    get_simulator,
    list_simulators,
)

# Core functions (for advanced users)
# Note: These imports may fail until import updates are complete
# They will be enabled once all modules are updated
try:
    from rrivis.core import (
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
    # Core functions (when available)
    "calculate_visibility",
    "read_antenna_positions",
    "generate_baselines",
]
