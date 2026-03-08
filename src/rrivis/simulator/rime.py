"""Direct RIME visibility simulator (reference implementation).

This module implements the RIMESimulator, which computes visibilities using
direct summation over sources following the Radio Interferometer Measurement
Equation (RIME). This is the proven, accurate implementation from RRIvis v0.1.x.

The RIME computes:
    V_pq = Σ_sources J_p @ C_source @ J_q^H

Where:
    - V_pq: 2×2 visibility matrix for baseline (p, q)
    - J_p: Jones matrix chain for antenna p
    - C_source: 2×2 coherency matrix from Stokes parameters
    - ^H: Hermitian conjugate
"""

from typing import Any

import numpy as np

from rrivis.simulator.base import VisibilitySimulator


class RIMESimulator(VisibilitySimulator):
    """
    Direct Radio Interferometer Measurement Equation (RIME) simulator.

    This simulator computes visibilities using direct summation over sources
    and baselines, implementing the full polarized RIME with Jones matrices.
    It serves as the reference implementation for accuracy validation.

    Algorithm
    ---------
    For each baseline (p, q) and frequency ν:
        V_pq(ν) = Σ_s J_p(s, ν) @ C_s(ν) @ J_q(s, ν)^H

    Where the Jones chain J = B @ G @ D @ P @ E @ T @ Z @ K includes:
        - K: Geometric phase delay (fringe rotation)
        - E: Primary beam response (direction-dependent)
        - G: Electronic gains (time-variable)
        - B: Bandpass (frequency-dependent)
        - D: Polarization leakage
        - P: Parallactic angle rotation
        - Z: Ionospheric effects (Faraday rotation, TEC)
        - T: Tropospheric effects

    Complexity
    ----------
    - Time: O(N_sources × N_baselines × N_frequencies)
    - Memory: O(N_baselines × N_frequencies) for output
              + O(N_sources × N_frequencies) for working arrays

    Performance Characteristics
    ---------------------------
    - Accurate for all problem sizes and source distributions
    - Optimal for small to medium source counts (< 10,000)
    - Handles arbitrary source positions (no gridding required)
    - Full polarization support (2×2 Jones matrices)
    - GPU acceleration via JAX backend (10-50× speedup)
    - Numba JIT compilation for CPU optimization

    Use Cases
    ---------
    - Reference calculations requiring high accuracy
    - Small to medium simulations (< 10,000 sources)
    - Arbitrary source positions (point sources, catalog sources)
    - Full polarization studies
    - Validation of faster approximate methods

    Limitations
    -----------
    - Slower than FFT methods for large source counts (> 10,000)
    - Memory scales with N_sources × N_frequencies
    - Not optimal for dense diffuse emission (use FFT for that)

    Examples
    --------
    >>> from rrivis.simulator import get_simulator
    >>> from rrivis.backends import get_backend
    >>>
    >>> # Create RIME simulator
    >>> sim = get_simulator("rime")
    >>> print(sim.name, sim.complexity)
    rime O(N_src × N_bl × N_freq)
    >>>
    >>> # Calculate visibilities with GPU acceleration
    >>> backend = get_backend("jax")  # or "numpy", "numba"
    >>> visibilities = sim.calculate_visibilities(
    ...     antennas=antennas,
    ...     baselines=baselines,
    ...     sources=sources,
    ...     frequencies=freqs,
    ...     backend=backend,
    ...     location=location,
    ...     obstime=obstime,
    ...     wavelengths=wavelengths,
    ...     hpbw_per_antenna=hpbw,
    ... )
    >>>
    >>> # Access correlation products
    >>> vis_xx = visibilities[(0, 1)]["XX"]
    >>> vis_i = visibilities[(0, 1)]["I"]

    See Also
    --------
    rrivis.core.visibility.calculate_visibility : Core implementation
    rrivis.core.jones : Jones matrix framework
    rrivis.backends : Backend abstraction for CPU/GPU

    References
    ----------
    .. [1] Smirnov, O. M. (2011). "Revisiting the radio interferometer
           measurement equation." A&A, 527, A106.
    .. [2] Hamaker, J. P., Bregman, J. D., & Sault, R. J. (1996).
           "Understanding radio polarimetry." A&AS, 117, 137.
    """

    @property
    def name(self) -> str:
        """Simulator identifier."""
        return "rime"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Direct RIME summation (accurate reference implementation)"

    @property
    def complexity(self) -> str:
        """Algorithm complexity."""
        return "O(N_src × N_bl × N_freq)"

    @property
    def supports_polarization(self) -> bool:
        """Full polarization support."""
        return True

    @property
    def supports_gpu(self) -> bool:
        """GPU acceleration supported via JAX backend."""
        return True

    def calculate_visibilities(
        self,
        antennas: dict[Any, dict],
        baselines: dict[tuple[Any, Any], dict],
        sources: list[dict],
        frequencies: np.ndarray,
        backend: Any,
        **kwargs,
    ) -> dict[tuple[Any, Any], dict]:
        """
        Calculate visibilities using direct RIME summation.

        Delegates to the core visibility calculation function which contains
        the proven implementation from RRIvis v0.1.x with full polarization
        support and backend abstraction.

        Parameters
        ----------
        antennas : dict
            Dictionary of antenna positions and properties.
            Keys: antenna identifiers
            Values: dicts with "Position", "Name", etc.

        baselines : dict
            Dictionary of baselines.
            Keys: (ant1, ant2) tuples
            Values: dicts with "BaselineVector"

        sources : list
            List of source dictionaries with:
                - "coords": SkyCoord object
                - "flux": flux density in Jy
                - "spectral_index": spectral index
                - Optional: "stokes_q", "stokes_u", "stokes_v"

        frequencies : ndarray
            Frequency array in Hz.

        backend : ArrayBackend
            Computation backend (numpy, jax, or numba).

        **kwargs : dict
            Required:
                - location: EarthLocation for observer
                - obstime: Time for observation
                - wavelengths: Quantity array of wavelengths
                - hpbw_per_antenna: dict of HPBW per antenna

            Optional:
                - beam_manager: BeamManager for FITS beams
                - return_correlations: bool (default True)
                - jones_config: dict of Jones term configs

        Returns
        -------
        dict
            Visibilities for each baseline.
            Keys: (ant1, ant2) tuples
            Values: dict with "XX", "XY", "YX", "YY", "I" arrays

        Raises
        ------
        ValueError
            If required kwargs are missing.
        ImportError
            If core.visibility module cannot be imported.

        Notes
        -----
        This method is a thin wrapper around
        `rrivis.core.visibility.calculate_visibility()`. All the heavy
        computation is done in the core module, which has been extensively
        tested and validated against other simulators.
        """
        # Extract required parameters from kwargs FIRST (before importing)
        location = kwargs.get("location")
        obstime = kwargs.get("obstime")
        wavelengths = kwargs.get("wavelengths")
        hpbw_per_antenna = kwargs.get("hpbw_per_antenna")

        # Validate required parameters
        missing = []
        if location is None:
            missing.append("location")
        if obstime is None:
            missing.append("obstime")
        if wavelengths is None:
            missing.append("wavelengths")
        if hpbw_per_antenna is None:
            missing.append("hpbw_per_antenna")

        if missing:
            raise ValueError(
                f"RIMESimulator requires the following kwargs: {missing}. "
                "These are needed for coordinate transforms and beam calculations."
            )

        # Import here to avoid circular imports (after validation)
        from rrivis.core.visibility import calculate_visibility

        # Extract optional parameters with defaults
        beam_manager = kwargs.get("beam_manager", None)
        return_correlations = kwargs.get("return_correlations", True)
        jones_config = kwargs.get("jones_config", None)

        # Extract time-stepping parameters (required)
        duration_seconds = kwargs.get("duration_seconds", 1.0)
        time_step_seconds = kwargs.get("time_step_seconds", 1.0)

        # Delegate to core implementation
        return calculate_visibility(
            antennas=antennas,
            baselines=baselines,
            sources=sources,
            location=location,
            obstime=obstime,
            wavelengths=wavelengths,
            freqs=frequencies,
            hpbw_per_antenna=hpbw_per_antenna,
            duration_seconds=duration_seconds,
            time_step_seconds=time_step_seconds,
            beam_manager=beam_manager,
            return_correlations=return_correlations,
            backend=backend,
            jones_config=jones_config,
        )

    def get_memory_estimate(
        self,
        n_antennas: int,
        n_baselines: int,
        n_sources: int,
        n_frequencies: int,
        n_times: int = 1,
        polarized: bool = True,
    ) -> dict[str, Any]:
        """
        Estimate memory requirements for RIME simulation.

        The RIME algorithm requires memory for:
        1. Output visibilities: N_bl × N_freq × N_times × (4 if polarized)
        2. Source arrays: N_src × N_freq for flux, direction cosines
        3. Beam patterns: N_ant × N_freq × (4 if polarized)
        4. Jones matrices: N_ant × N_src × N_freq × (4 if polarized)

        Parameters
        ----------
        n_antennas : int
            Number of antennas.
        n_baselines : int
            Number of baselines.
        n_sources : int
            Number of sky sources.
        n_frequencies : int
            Number of frequency channels.
        n_times : int, optional
            Number of time steps (default 1).
        polarized : bool, optional
            Whether using full polarization (default True).

        Returns
        -------
        dict
            Memory estimates with human-readable sizes.
        """
        bytes_per_complex = 16  # complex128
        pol_factor = 4 if polarized else 1

        # Output: baselines × freq × time × polarization
        output_bytes = (
            n_baselines * n_frequencies * n_times * pol_factor * bytes_per_complex
        )

        # Working memory for RIME:
        # - Source flux arrays: n_src × n_freq × complex
        # - Direction cosines (l, m, n): 3 × n_src × 8 bytes (float64)
        # - Stokes parameters: 4 × n_src × 8 bytes
        # - Beam patterns per antenna: n_ant × n_freq × pol_factor × complex
        # - Per-source Jones matrices (peak): n_src × pol_factor × complex
        source_arrays = n_sources * n_frequencies * bytes_per_complex
        direction_cosines = 3 * n_sources * 8
        stokes_params = 4 * n_sources * 8
        beam_arrays = n_antennas * n_frequencies * pol_factor * bytes_per_complex
        jones_working = n_sources * pol_factor * bytes_per_complex * 2  # Two antennas

        working_bytes = (
            source_arrays
            + direction_cosines
            + stokes_params
            + beam_arrays
            + jones_working
        )
        total_bytes = output_bytes + working_bytes

        def format_bytes(b: int) -> str:
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if b < 1024:
                    return f"{b:.1f} {unit}"
                b /= 1024
            return f"{b:.1f} PB"

        warning = None
        if total_bytes > 16 * 1024**3:
            warning = (
                "Very high memory usage. Consider: "
                "(1) reducing source count, "
                "(2) using fewer frequency channels, "
                "(3) using an FFT-based simulator for large source counts."
            )
        elif total_bytes > 4 * 1024**3:
            warning = "High memory usage. Ensure sufficient RAM is available."

        return {
            "output_bytes": output_bytes,
            "working_bytes": working_bytes,
            "total_bytes": total_bytes,
            "output_human": format_bytes(output_bytes),
            "working_human": format_bytes(working_bytes),
            "total_human": format_bytes(total_bytes),
            "warning": warning,
            "algorithm": "rime",
            "details": {
                "n_antennas": n_antennas,
                "n_baselines": n_baselines,
                "n_sources": n_sources,
                "n_frequencies": n_frequencies,
                "n_times": n_times,
                "polarized": polarized,
            },
            "breakdown": {
                "source_arrays": format_bytes(source_arrays),
                "direction_cosines": format_bytes(direction_cosines),
                "beam_patterns": format_bytes(beam_arrays),
            },
        }
