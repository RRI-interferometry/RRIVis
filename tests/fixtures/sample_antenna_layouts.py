# tests/fixtures/sample_antenna_layouts.py
"""
Sample antenna layout fixtures for RRIvis tests.

These fixtures can be imported directly for use in tests or examples.
"""

import numpy as np


def get_simple_two_antenna_layout():
    """
    Simple 2-antenna layout for basic tests.

    Returns
    -------
    dict
        Dictionary with antenna data keyed by antenna number.
    """
    return {
        0: {
            "Name": "Ant0",
            "Number": 0,
            "BeamID": 0,
            "Position": (0.0, 0.0, 0.0),
            "diameter": 14.0,
        },
        1: {
            "Name": "Ant1",
            "Number": 1,
            "BeamID": 0,
            "Position": (100.0, 0.0, 0.0),
            "diameter": 14.0,
        },
    }


def get_three_antenna_layout():
    """
    Three-antenna layout (HERA-like subset).

    Returns
    -------
    dict
        Dictionary with antenna data keyed by antenna number.
    """
    return {
        136: {
            "Name": "HH136",
            "Number": 136,
            "BeamID": 0,
            "Position": (-156.5976, 2.9439, -0.1819),
            "diameter": 14.0,
        },
        140: {
            "Name": "HH140",
            "Number": 140,
            "BeamID": 0,
            "Position": (-98.1662, 3.1671, -0.3008),
            "diameter": 14.0,
        },
        121: {
            "Name": "HH121",
            "Number": 121,
            "BeamID": 0,
            "Position": (-90.8139, -9.4618, -0.1707),
            "diameter": 14.0,
        },
    }


def get_linear_array_layout(n_antennas=10, spacing=14.6):
    """
    Linear array layout.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    spacing : float
        Spacing between antennas in meters.

    Returns
    -------
    dict
        Dictionary with antenna data keyed by antenna number.
    """
    antennas = {}
    for i in range(n_antennas):
        antennas[i] = {
            "Name": f"Ant{i}",
            "Number": i,
            "BeamID": 0,
            "Position": (i * spacing, 0.0, 0.0),
            "diameter": 14.0,
        }
    return antennas


def get_hexagonal_layout(n_rings=2, spacing=14.6):
    """
    Hexagonal close-packed array layout (like HERA).

    Parameters
    ----------
    n_rings : int
        Number of rings around the center antenna.
    spacing : float
        Distance between adjacent antennas in meters.

    Returns
    -------
    dict
        Dictionary with antenna data keyed by antenna number.
    """
    antennas = {}
    ant_num = 0

    # Center antenna
    antennas[ant_num] = {
        "Name": f"Ant{ant_num}",
        "Number": ant_num,
        "BeamID": 0,
        "Position": (0.0, 0.0, 0.0),
        "diameter": 14.0,
    }
    ant_num += 1

    # Add rings
    for ring in range(1, n_rings + 1):
        # 6 sides of hexagon, each with 'ring' antennas
        for side in range(6):
            angle_start = np.pi / 3 * side
            for pos in range(ring):
                # Calculate position along the side
                angle1 = angle_start
                angle2 = angle_start + np.pi / 3

                # Start corner
                x1 = ring * spacing * np.cos(angle1)
                y1 = ring * spacing * np.sin(angle1)

                # End corner
                x2 = ring * spacing * np.cos(angle2)
                y2 = ring * spacing * np.sin(angle2)

                # Interpolate
                t = pos / ring if ring > 0 else 0
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)

                antennas[ant_num] = {
                    "Name": f"Ant{ant_num}",
                    "Number": ant_num,
                    "BeamID": 0,
                    "Position": (x, y, 0.0),
                    "diameter": 14.0,
                }
                ant_num += 1

    return antennas


def get_random_layout(n_antennas=20, extent=500, seed=42):
    """
    Random antenna layout within a square area.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    extent : float
        Half-size of the square area in meters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with antenna data keyed by antenna number.
    """
    np.random.seed(seed)
    antennas = {}

    for i in range(n_antennas):
        x = np.random.uniform(-extent, extent)
        y = np.random.uniform(-extent, extent)
        z = np.random.uniform(-1, 1)  # Small height variation

        antennas[i] = {
            "Name": f"Ant{i}",
            "Number": i,
            "BeamID": 0,
            "Position": (x, y, z),
            "diameter": 14.0,
        }

    return antennas


def get_baselines_for_antennas(antennas, include_autos=False):
    """
    Generate baselines for a given antenna layout.

    Parameters
    ----------
    antennas : dict
        Antenna dictionary from any of the layout functions.
    include_autos : bool
        Whether to include auto-correlations.

    Returns
    -------
    dict
        Dictionary of baselines keyed by (ant1, ant2) tuples.
    """
    baselines = {}
    ant_nums = sorted(antennas.keys())

    for i, ant1 in enumerate(ant_nums):
        start = i if include_autos else i + 1
        for ant2 in ant_nums[start:]:
            pos1 = np.array(antennas[ant1]["Position"])
            pos2 = np.array(antennas[ant2]["Position"])
            baseline_vector = pos2 - pos1

            baselines[(ant1, ant2)] = {
                "BaselineVector": baseline_vector,
            }

    return baselines


# Convenience dictionaries for quick access
LAYOUTS = {
    "simple": get_simple_two_antenna_layout,
    "three": get_three_antenna_layout,
    "linear": get_linear_array_layout,
    "hexagonal": get_hexagonal_layout,
    "random": get_random_layout,
}
