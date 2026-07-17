"""Scientific input-data validation utilities for RadioSim."""

import numpy as np


def validate_antenna_positions(
    positions: dict | np.ndarray,
    n_antennas: int | None = None,
) -> bool:
    """
    Validate antenna position data.

    Args:
        positions: Antenna positions (dict or array)
        n_antennas: Expected number of antennas (optional)

    Returns:
        True if valid

    Raises:
        ValueError: If positions are invalid
    """
    if isinstance(positions, dict):
        if not positions:
            raise ValueError("Antenna positions dictionary is empty")
        # Check that each antenna has E, N, U coordinates
        for name, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            required = {"E", "N", "U"}
            if not required.issubset(pos.keys()):
                raise ValueError(
                    f"Antenna {name} missing coordinates: {required - set(pos.keys())}"
                )

    elif isinstance(positions, np.ndarray):
        if positions.ndim != 2 or positions.shape[1] < 3:
            raise ValueError(f"Expected (N, 3+) array, got shape {positions.shape}")
        if n_antennas is not None and positions.shape[0] != n_antennas:
            raise ValueError(
                f"Expected {n_antennas} antennas, got {positions.shape[0]}"
            )

    return True


def validate_frequencies(
    frequencies: list[float] | np.ndarray,
    min_freq: float = 1.0,
    max_freq: float = 1000.0,
) -> bool:
    """
    Validate frequency array.

    Args:
        frequencies: Frequencies in MHz
        min_freq: Minimum allowed frequency (MHz)
        max_freq: Maximum allowed frequency (MHz)

    Returns:
        True if valid

    Raises:
        ValueError: If frequencies are invalid
    """
    freqs = np.asarray(frequencies)

    if freqs.size == 0:
        raise ValueError("Frequency array is empty")

    if np.any(freqs < min_freq) or np.any(freqs > max_freq):
        raise ValueError(f"Frequencies must be in range [{min_freq}, {max_freq}] MHz")

    if np.any(np.diff(freqs) < 0):
        raise ValueError("Frequencies must be monotonically increasing")

    return True


def validate_stokes_parameters(
    stokes_i: float,
    stokes_q: float = 0.0,
    stokes_u: float = 0.0,
    stokes_v: float = 0.0,
) -> bool:
    """
    Validate Stokes parameters for physical consistency.

    Args:
        stokes_i: Stokes I (total intensity)
        stokes_q: Stokes Q (linear polarization)
        stokes_u: Stokes U (linear polarization)
        stokes_v: Stokes V (circular polarization)

    Returns:
        True if valid

    Raises:
        ValueError: If Stokes parameters are unphysical
    """
    # Total intensity must be non-negative
    if stokes_i < 0:
        raise ValueError(f"Stokes I must be non-negative, got {stokes_i}")

    # Polarization constraint: I^2 >= Q^2 + U^2 + V^2
    polarization_intensity = stokes_q**2 + stokes_u**2 + stokes_v**2
    if polarization_intensity > stokes_i**2 + 1e-10:  # Small tolerance
        raise ValueError(
            f"Unphysical polarization: I^2={stokes_i**2} < Q^2+U^2+V^2={polarization_intensity}"
        )

    return True
