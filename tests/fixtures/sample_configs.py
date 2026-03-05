# tests/fixtures/sample_configs.py
"""
Sample configuration fixtures for RRIvis tests.

These fixtures can be used to generate test configuration files.
"""

from pathlib import Path
from typing import Any

import yaml


def get_minimal_config() -> dict[str, Any]:
    """
    Minimal valid configuration for basic tests.

    Returns
    -------
    dict
        Minimal configuration dictionary.
    """
    return {
        "antenna_layout": {
            "antenna_positions_file": "test_antennas.txt",
            "antenna_file_format": "rrivis",
            "all_antenna_type": "parabolic",
            "all_antenna_diameter": 14.0,
        },
        "obs_frequency": {
            "starting_frequency": 100,
            "frequency_bandwidth": 50,
            "frequency_interval": 10,
            "frequency_unit": "MHz",
        },
        "location": {
            "lat": -26.7033,
            "lon": 116.6708,
            "height": 377.0,
        },
        "obs_time": {
            "start_time": "2023-06-21T12:00:00",
        },
        "sky_model": {
            "test_sources": {
                "use_test_sources": True,
                "num_sources": 10,
            },
            "gleam": {
                "use_gleam": False,
            },
            "gsm_healpix": {
                "use_gsm": False,
            },
        },
    }


def get_full_config() -> dict[str, Any]:
    """
    Full configuration with all options specified.

    Returns
    -------
    dict
        Complete configuration dictionary with all options.
    """
    return {
        "antenna_layout": {
            "antenna_positions_file": "HERA65.csv",
            "antenna_file_format": "rrivis",
            "all_antenna_type": "parabolic",
            "all_antenna_diameter": 14.0,
        },
        "obs_frequency": {
            "starting_frequency": 100,
            "frequency_bandwidth": 100,
            "frequency_interval": 5,
            "frequency_unit": "MHz",
            "n_channels": 21,
        },
        "location": {
            "lat": -26.7033,
            "lon": 116.6708,
            "height": 377.0,
        },
        "obs_time": {
            "start_time": "2023-06-21T00:00:00",
            "end_time": "2023-06-21T06:00:00",
            "time_interval": 60,  # seconds
        },
        "sky_model": {
            "test_sources": {
                "use_test_sources": False,
                "num_sources": 10,
            },
            "gleam": {
                "use_gleam": True,
                "flux_limit": 1.0,
                "max_sources": 1000,
            },
            "gsm_healpix": {
                "use_gsm": False,
                "nside": 64,
            },
        },
        "beam": {
            "beam_type": "gaussian",
            "beam_file": None,
        },
        "backend": "auto",
        "jones": {
            "K": {"enabled": True},
            "E": {"enabled": True},
            "G": {"enabled": False, "sigma": 0.01},
            "B": {"enabled": False},
            "D": {"enabled": False},
            "P": {"enabled": False},
            "Z": {"enabled": False},
            "T": {"enabled": False},
        },
        "output": {
            "directory": "output/",
            "format": "hdf5",
            "save_uvw": True,
            "save_flags": True,
        },
    }


def get_jones_enabled_config() -> dict[str, Any]:
    """
    Configuration with Jones matrix effects enabled.

    Returns
    -------
    dict
        Configuration with various Jones terms enabled.
    """
    config = get_minimal_config()
    config["jones"] = {
        "K": {"enabled": True},
        "E": {"enabled": True, "beam_type": "gaussian"},
        "G": {"enabled": True, "sigma": 0.05},
        "B": {"enabled": True},
        "D": {"enabled": True, "d_xy": 0.01, "d_yx": 0.01},
        "P": {"enabled": True},
        "Z": {"enabled": False},  # Requires ionosphere model
        "T": {"enabled": False},  # Requires atmosphere model
    }
    config["backend"] = "numpy"
    return config


def get_gleam_config() -> dict[str, Any]:
    """
    Configuration for GLEAM catalog sky model.

    Returns
    -------
    dict
        Configuration using GLEAM catalog.
    """
    config = get_minimal_config()
    config["sky_model"] = {
        "test_sources": {"use_test_sources": False},
        "gleam": {
            "use_gleam": True,
            "flux_limit": 0.5,
            "max_sources": 500,
        },
        "gsm_healpix": {"use_gsm": False},
    }
    return config


def get_gsm_config() -> dict[str, Any]:
    """
    Configuration for Global Sky Model (GSM).

    Returns
    -------
    dict
        Configuration using GSM.
    """
    config = get_minimal_config()
    config["sky_model"] = {
        "test_sources": {"use_test_sources": False},
        "gleam": {"use_gleam": False},
        "gsm_healpix": {
            "use_gsm": True,
            "nside": 64,
            "freq_unit": "MHz",
        },
    }
    return config


def get_gpu_config() -> dict[str, Any]:
    """
    Configuration for GPU-accelerated computation.

    Returns
    -------
    dict
        Configuration with GPU backend.
    """
    config = get_minimal_config()
    config["backend"] = "jax"
    config["jones"] = {
        "K": {"enabled": True},
        "E": {"enabled": True},
    }
    return config


def get_numba_config() -> dict[str, Any]:
    """
    Configuration for Numba backend.

    Returns
    -------
    dict
        Configuration with Numba backend.
    """
    config = get_minimal_config()
    config["backend"] = "numba"
    return config


def write_config_to_file(config: dict[str, Any], path: Path) -> Path:
    """
    Write a configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    path : Path
        Output file path.

    Returns
    -------
    Path
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return path


def create_test_antenna_file(path: Path, antennas: dict = None) -> Path:
    """
    Create a test antenna positions file.

    Parameters
    ----------
    path : Path
        Output file path.
    antennas : dict, optional
        Antenna dictionary. If None, uses simple 3-antenna layout.

    Returns
    -------
    Path
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if antennas is None:
        antennas = {
            0: {"Name": "Ant0", "Position": (0.0, 0.0, 0.0)},
            1: {"Name": "Ant1", "Position": (100.0, 0.0, 0.0)},
            2: {"Name": "Ant2", "Position": (50.0, 86.6, 0.0)},
        }

    with open(path, "w") as f:
        f.write("# name number beamid east north up\n")
        for ant_num, ant_data in antennas.items():
            pos = ant_data["Position"]
            name = ant_data.get("Name", f"Ant{ant_num}")
            beam_id = ant_data.get("BeamID", 0)
            f.write(
                f"{name} {ant_num} {beam_id} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n"
            )

    return path


# Convenience dictionary for quick access
CONFIGS = {
    "minimal": get_minimal_config,
    "full": get_full_config,
    "jones": get_jones_enabled_config,
    "gleam": get_gleam_config,
    "gsm": get_gsm_config,
    "gpu": get_gpu_config,
    "numba": get_numba_config,
}
