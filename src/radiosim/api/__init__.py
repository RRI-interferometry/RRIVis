"""High-level API for RadioSim.

This module provides the main user-facing API for running simulations
programmatically in Python scripts and Jupyter notebooks.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radiosim.api.simulator import Simulator
    from radiosim.core.phase_center import PhaseCenter
    from radiosim.core.result import LoadedSimulationResult, SimulationResult
    from radiosim.core.time_grid import ObservationTimeGrid
    from radiosim.io.result_format import ResultFormat

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
    "ResultFormat": ("radiosim.io.result_format", "ResultFormat"),
}


def __getattr__(name: str) -> object:
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
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "Simulator",
    "SimulationResult",
    "LoadedSimulationResult",
    "ObservationTimeGrid",
    "PhaseCenter",
    "ResultFormat",
]
