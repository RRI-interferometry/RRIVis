"""Utility modules for RadioSim.

This module provides common utilities for logging, validation,
network connectivity detection, and other helper functions.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radiosim.utils.cosmology import (
        F_21CM_HZ,
        add_redshift_secondary_axis,
        frequency_to_redshift_21cm,
        redshift_to_frequency_21cm,
    )
    from radiosim.utils.device import DeviceResources, get_device_resources
    from radiosim.utils.logging import get_logger, setup_logging
    from radiosim.utils.network import (
        NetworkStatus,
        check_all_services,
        check_service,
        is_online,
    )


_LAZY_EXPORTS = {
    "F_21CM_HZ": ("radiosim.utils.cosmology", "F_21CM_HZ"),
    "add_redshift_secondary_axis": (
        "radiosim.utils.cosmology",
        "add_redshift_secondary_axis",
    ),
    "frequency_to_redshift_21cm": (
        "radiosim.utils.cosmology",
        "frequency_to_redshift_21cm",
    ),
    "redshift_to_frequency_21cm": (
        "radiosim.utils.cosmology",
        "redshift_to_frequency_21cm",
    ),
    "DeviceResources": ("radiosim.utils.device", "DeviceResources"),
    "get_device_resources": ("radiosim.utils.device", "get_device_resources"),
    "get_logger": ("radiosim.utils.logging", "get_logger"),
    "setup_logging": ("radiosim.utils.logging", "setup_logging"),
    "NetworkStatus": ("radiosim.utils.network", "NetworkStatus"),
    "check_all_services": ("radiosim.utils.network", "check_all_services"),
    "check_service": ("radiosim.utils.network", "check_service"),
    "is_online": ("radiosim.utils.network", "is_online"),
}

__all__ = [
    "setup_logging",
    "get_logger",
    "NetworkStatus",
    "is_online",
    "check_service",
    "check_all_services",
    "DeviceResources",
    "get_device_resources",
    "F_21CM_HZ",
    "frequency_to_redshift_21cm",
    "redshift_to_frequency_21cm",
    "add_redshift_secondary_axis",
]


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
