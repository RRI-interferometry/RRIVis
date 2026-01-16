# rrivis/core/baseline.py
"""Baseline generation utilities for interferometric arrays."""

import logging
import sys

import numpy as np


logger = logging.getLogger(__name__)


def generate_baselines(antennas, beams_per_antenna, beam_response_per_antenna, verbose=False):
    """
    Generate baselines based on antenna positions and metadata, using Numbers as keys.
    Includes detailed metadata for each baseline:
      - Baseline vector
      - Absolute length (magnitude of baseline vector)
      - D1D2: Antenna diameters
      - BT1BT2: Beam type information from beams_per_antenna
      - A1A2: Beam response information from beam_response_per_antenna

    Parameters:
    antennas (dict): Dictionary of antenna metadata and positions.
    beams_per_antenna (dict): Dictionary mapping antenna numbers to beam types.
    beam_response_per_antenna (dict): Dictionary mapping antenna numbers to beam responses.

    Returns:
    dict: Dictionary of baselines with metadata:
          - BaselineVector: Vector between antenna positions.
          - Length: Absolute length of the baseline vector.
          - D1D2: Antenna diameters (D1, D2).
          - BT1BT2: Beam type information (BT1, BT2) from beams_per_antenna.
          - A1A2: Beam response information (A1, A2) from beam_response_per_antenna.

    Raises:
    ValueError: If the antennas dictionary is empty or has invalid data.
    KeyError: If required keys are missing in the antenna metadata.
    TypeError: If the position data type is invalid.
    """
    # Check if antennas dictionary is provided and not empty
    if not antennas:
        raise ValueError(
            "The antennas dictionary is empty. Please provide valid antenna data."
        )

    baselines = {}

    try:
        # Extract antenna metadata and positions
        antenna_metadata = {
            ant["Number"]: {
                "Position": np.array(ant["Position"], dtype=float),
                "Diameter": ant.get("diameter", None),
            }
            for ant in antennas.values()
        }
    except KeyError as e:
        raise KeyError(f"Missing required key in antenna metadata: {e}")
    except (ValueError, TypeError) as e:
        raise TypeError(f"Invalid position data in antenna metadata: {e}")

    # Ensure antenna_metadata is not empty
    if not antenna_metadata:
        raise ValueError("No valid antenna metadata found in the provided data.")

    # Sort antenna numbers to ensure all combinations are considered
    sorted_antenna_numbers = sorted(antenna_metadata.keys())

    # Generate baselines
    try:
        for i, ant1 in enumerate(sorted_antenna_numbers):
            for ant2 in sorted_antenna_numbers[i:]:  # Ensure ant1 <= ant2
                pos1 = antenna_metadata[ant1]["Position"]
                pos2 = antenna_metadata[ant2]["Position"]

                # Compute baseline vector and its magnitude (length)
                baseline_vector = pos2 - pos1
                baseline_length = np.linalg.norm(baseline_vector)

                # Get beam type and response from the provided configs
                beamtype_info = f"{beams_per_antenna[ant1]}_{beams_per_antenna[ant2]}"
                beamresponse_info = f"{beam_response_per_antenna[ant1]}_{beam_response_per_antenna[ant2]}"

                # Metadata for baseline
                diameter_info = f"{antenna_metadata[ant1]['Diameter']}_{antenna_metadata[ant2]['Diameter']}"

                # Store baseline with metadata
                baselines[(ant1, ant2)] = {
                    "BaselineVector": baseline_vector,
                    "Length": baseline_length,
                    "D1D2": diameter_info,
                    "BT1BT2": beamtype_info,
                    "A1A2": beamresponse_info,
                }
    except Exception as e:
        raise ValueError(f"Error while generating baselines: {e}")

    # Debug output (only when verbose=True)
    if verbose:
        logger.debug("Generated baselines:")
        for key, value in baselines.items():
            logger.debug(f"Baseline {key}: {value}")

        # Calculate total memory usage in MB
        total_memory_bytes = sys.getsizeof(baselines) + sum(
            sys.getsizeof(key) + sys.getsizeof(value) for key, value in baselines.items()
        )
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        logger.debug(f"Total memory used by baselines: {total_memory_mb:.4f} MB")

    return baselines
