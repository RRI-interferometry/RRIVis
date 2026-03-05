# rrivis/io/writers.py
"""Data output writers for RRIvis.

This module provides functions for writing simulation results to various formats:
- HDF5 for visibility data
- YAML for configuration files
"""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


def save_visibilities_hdf5(
    output_path: str | Path,
    visibilities: dict[tuple, list],
    frequencies: np.ndarray,
    time_points_mjd: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save visibility data to HDF5 format.

    Parameters
    ----------
    output_path : str or Path
        Path to output HDF5 file
    visibilities : dict
        Dictionary mapping baseline tuples (ant_i, ant_j) to visibility arrays
    frequencies : np.ndarray
        Array of observation frequencies in Hz
    time_points_mjd : np.ndarray
        Array of time points in MJD
    metadata : dict, optional
        Additional metadata to store as HDF5 attributes

    Returns
    -------
    str
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5file:
        # Create a group for each baseline
        for key, vis in visibilities.items():
            baseline_group = h5file.create_group(f"baseline_{key}")

            # Save complex visibility
            vis_array = np.stack(vis)  # Convert to 2D NumPy array
            baseline_group.create_dataset(
                "complex_visibility",
                data=vis_array.astype(np.complex128),
                dtype="complex128",
            )

        # Save frequencies and time points
        h5file.create_dataset("frequencies", data=frequencies)
        h5file.create_dataset("time_points_mjd", data=time_points_mjd)

        # Save metadata as attributes (flatten complex objects)
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    # Convert Pydantic models/dicts to JSON-serializable format
                    if hasattr(value, "model_dump"):
                        value = value.model_dump()
                    elif isinstance(value, dict):
                        # Recursively convert nested values
                        value = {
                            k: (v.model_dump() if hasattr(v, "model_dump") else v)
                            for k, v in value.items()
                        }
                    # Try to save as string representation for complex types
                    try:
                        if isinstance(value, (str, int, float, bool)):
                            h5file.attrs[key] = value
                        elif isinstance(value, (list, np.ndarray)):
                            h5file.attrs[key] = str(value)
                        else:
                            h5file.attrs[key] = str(value)
                    except (TypeError, ValueError):
                        pass

    return str(output_path)


def save_config_yaml(
    config: dict[str, Any],
    output_path: str | Path,
) -> str:
    """Save configuration to YAML format with proper formatting.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save
    output_path : str or Path
        Path to output YAML file

    Returns
    -------
    str
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom YAML dumper for better readability
    class FormattedDumper(yaml.Dumper):
        def write_line_break(self, data=None):
            # Add a blank line between top-level sections
            if self.indent == 0:
                self.stream.write("\n")
            super().write_line_break(data)

    with open(output_path, "w") as f:
        yaml.dump(
            config, f, Dumper=FormattedDumper, default_flow_style=False, sort_keys=True
        )

    return str(output_path)


def load_visibilities_hdf5(
    input_path: str | Path,
) -> dict[str, Any]:
    """Load visibility data from HDF5 format.

    Parameters
    ----------
    input_path : str or Path
        Path to input HDF5 file

    Returns
    -------
    dict
        Dictionary containing:
        - 'visibilities': dict mapping baseline tuples to visibility arrays
        - 'frequencies': frequency array
        - 'time_points_mjd': time points array
        - 'metadata': dict of metadata attributes
    """
    input_path = Path(input_path)

    result = {
        "visibilities": {},
        "frequencies": None,
        "time_points_mjd": None,
        "metadata": {},
    }

    with h5py.File(input_path, "r") as h5file:
        # Load visibilities from baseline groups
        for key in h5file.keys():
            if key.startswith("baseline_"):
                # Parse baseline tuple from group name
                baseline_str = key.replace("baseline_", "")
                # Handle both (i, j) and "i_j" formats
                if baseline_str.startswith("("):
                    baseline = eval(baseline_str)
                else:
                    parts = baseline_str.split("_")
                    baseline = (int(parts[0]), int(parts[1]))

                result["visibilities"][baseline] = h5file[key]["complex_visibility"][:]

        # Load frequencies and time points
        if "frequencies" in h5file:
            result["frequencies"] = h5file["frequencies"][:]
        if "time_points_mjd" in h5file:
            result["time_points_mjd"] = h5file["time_points_mjd"][:]

        # Load metadata attributes
        for attr_name, attr_value in h5file.attrs.items():
            result["metadata"][attr_name] = attr_value

    return result
