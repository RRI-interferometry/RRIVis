"""Abstract base class for visibility simulators.

This module defines the interface that all visibility simulators must implement,
allowing for different algorithms (direct RIME, FFT-based, matrix-based) to be
swapped without changing the user-facing API.

The design follows the Strategy pattern, enabling runtime selection of simulation
algorithms based on problem characteristics (source count, array density, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VisibilitySimulator(ABC):
    """
    Abstract base class for visibility simulators.

    This interface allows swapping visibility calculation algorithms without
    changing the user-facing API. Implementations can optimize for different
    scenarios (accuracy vs speed, few vs many sources, etc.).

    Current Implementations:
        - RIMESimulator: Direct RIME summation, O(N_src × N_bl × N_freq), accurate

    Future Implementations (v0.3.0+):
        - FFTSimulator: FFT-based NUFFT, O(N log N), fast for many sources
        - MatVisSimulator: Matrix-based GPU-optimized, HERA standard

    Examples
    --------
    >>> from rrivis.simulator import get_simulator
    >>> sim = get_simulator("rime")
    >>> print(sim.name, sim.complexity)
    rime O(N_src × N_bl × N_freq)
    >>> visibilities = sim.calculate_visibilities(
    ...     antennas=antennas,
    ...     baselines=baselines,
    ...     sources=sources,
    ...     frequencies=freqs,
    ...     backend=backend,
    ...     **kwargs,
    ... )
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short identifier for the simulator.

        Returns
        -------
        str
            Simulator name (e.g., 'rime', 'fft', 'matvis').
            Used for registry lookup and logging.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of the simulator.

        Returns
        -------
        str
            Description including algorithm type and key characteristics.
        """
        pass

    @property
    def complexity(self) -> str:
        """
        Algorithm complexity in Big-O notation.

        Returns
        -------
        str
            Complexity string (e.g., 'O(N_src × N_bl × N_freq)').
            Default returns 'Unknown' if not overridden.
        """
        return "Unknown"

    @property
    def supports_polarization(self) -> bool:
        """
        Whether the simulator supports full polarization.

        Returns
        -------
        bool
            True if simulator computes full 2×2 Jones/coherency matrices.
            Default is True.
        """
        return True

    @property
    def supports_gpu(self) -> bool:
        """
        Whether the simulator supports GPU acceleration.

        Returns
        -------
        bool
            True if simulator can use GPU backends (JAX, etc.).
            Default is True.
        """
        return True

    @abstractmethod
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
        Calculate visibilities for all baselines.

        This is the core computation method that each simulator must implement.
        The specific algorithm (direct summation, FFT, matrix multiplication)
        depends on the implementation.

        Parameters
        ----------
        antennas : dict
            Dictionary of antenna positions and properties.
            Keys: antenna identifiers (int or str)
            Values: dicts with at minimum:
                - "Position": [x, y, z] in meters (ECEF or local ENU)
                - "Name": antenna name string
                Optional: "Diameter", "BeamID", etc.

        baselines : dict
            Dictionary of baselines between antenna pairs.
            Keys: (ant1, ant2) tuples
            Values: dicts with at minimum:
                - "BaselineVector": [u, v, w] in meters

        sources : list
            List of source dictionaries, each containing:
                - "coords": astropy.SkyCoord object
                - "flux": flux density in Jy (Stokes I)
                - "spectral_index": spectral index α (S ∝ ν^α)
                Optional:
                - "stokes_q", "stokes_u", "stokes_v": polarization (default 0)

        frequencies : ndarray
            Frequency array in Hz. Shape: (N_freq,)

        backend : ArrayBackend
            Computation backend instance from rrivis.backends.
            Provides array operations (numpy-like API) and device management.
            Use get_backend("numpy"), get_backend("jax"), etc.

        **kwargs : dict
            Algorithm-specific parameters. Common options include:
                - location: astropy.EarthLocation for observer position
                - obstime: astropy.Time for observation time
                - wavelengths: astropy.Quantity array (derived from frequencies)
                - hpbw_per_antenna: dict mapping antenna -> HPBW array
                - beam_manager: BeamManager for FITS beam interpolation
                - beam_pattern_per_antenna: dict of analytic beam types
                - beam_pattern_params: dict of beam pattern parameters
                - return_correlations: bool, extract XX/XY/YX/YY (default True)
                - jones_config: dict of Jones term configurations

        Returns
        -------
        dict
            Visibilities for each baseline.
            Keys: (ant1, ant2) tuples matching input baselines
            Values: dict with correlation products:
                - "XX": complex array, shape (N_freq,)
                - "XY": complex array, shape (N_freq,)
                - "YX": complex array, shape (N_freq,)
                - "YY": complex array, shape (N_freq,)
                - "I": complex array, shape (N_freq,) - Stokes I visibility

            If return_correlations=False in kwargs:
                Values: ndarray of shape (N_freq, 2, 2) - raw visibility matrices

        Raises
        ------
        ValueError
            If required parameters are missing or invalid.
        RuntimeError
            If computation fails (e.g., backend error, memory overflow).

        Notes
        -----
        The Radio Interferometer Measurement Equation (RIME) computes:

            V_pq = Σ_s J_p(s) @ C_s @ J_q(s)^H

        Where:
            - V_pq: 2×2 visibility matrix for baseline (p, q)
            - J_p(s): Jones matrix for antenna p, source s
            - C_s: 2×2 coherency matrix for source s (from Stokes params)
            - ^H: Hermitian conjugate

        The full Jones chain is: J = B @ G @ D @ P @ E @ T @ Z @ K
            - K: Geometric phase (fringe rotation)
            - E: Primary beam response
            - G: Electronic gains
            - B: Bandpass
            - D: Polarization leakage
            - P: Parallactic angle
            - Z: Ionosphere (Faraday rotation)
            - T: Troposphere
        """
        pass

    def validate_inputs(
        self,
        antennas: dict[Any, dict],
        baselines: dict[tuple[Any, Any], dict],
        sources: list[dict],
        frequencies: np.ndarray,
        **kwargs,
    ) -> tuple[bool, list[str]]:
        """
        Validate inputs before computation.

        This method checks that all required data is present and correctly
        formatted. Override in subclasses for algorithm-specific validation.

        Parameters
        ----------
        antennas : dict
            Antenna dictionary (see calculate_visibilities).
        baselines : dict
            Baseline dictionary (see calculate_visibilities).
        sources : list
            Source list (see calculate_visibilities).
        frequencies : ndarray
            Frequency array in Hz.
        **kwargs : dict
            Additional parameters to validate.

        Returns
        -------
        tuple
            (is_valid, errors) where:
                - is_valid: bool, True if all inputs are valid
                - errors: list of str, error messages (empty if valid)

        Examples
        --------
        >>> sim = get_simulator("rime")
        >>> valid, errors = sim.validate_inputs(antennas, baselines, sources, freqs)
        >>> if not valid:
        ...     for err in errors:
        ...         print(f"Validation error: {err}")
        """
        errors = []

        # Check antennas
        if not antennas:
            errors.append("Antenna dictionary is empty")
        else:
            for ant_id, ant_data in antennas.items():
                if "Position" not in ant_data:
                    errors.append(f"Antenna {ant_id} missing 'Position' key")

        # Check baselines
        if not baselines:
            errors.append("Baseline dictionary is empty")
        else:
            for bl_key, bl_data in baselines.items():
                if not isinstance(bl_key, tuple) or len(bl_key) != 2:
                    errors.append(f"Invalid baseline key: {bl_key}")
                if "BaselineVector" not in bl_data:
                    errors.append(f"Baseline {bl_key} missing 'BaselineVector' key")

        # Check sources (empty is allowed, just returns zero visibilities)
        if sources:
            for i, src in enumerate(sources):
                if "coords" not in src:
                    errors.append(f"Source {i} missing 'coords' key")
                if "flux" not in src:
                    errors.append(f"Source {i} missing 'flux' key")
                if "spectral_index" not in src:
                    errors.append(f"Source {i} missing 'spectral_index' key")

        # Check frequencies
        if frequencies is None or len(frequencies) == 0:
            errors.append("Frequency array is empty")
        else:
            # Convert to numpy array for validation
            freq_array = np.asarray(frequencies)
            if not np.all(np.isfinite(freq_array)):
                errors.append("Frequency array contains non-finite values")
            elif np.any(freq_array <= 0):
                errors.append("Frequency array contains non-positive values")

        return (len(errors) == 0, errors)

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
        Estimate memory requirements for the simulation.

        Provides rough estimates of memory usage to help users determine
        if the simulation will fit in available memory. Override in
        subclasses for algorithm-specific estimates.

        Parameters
        ----------
        n_antennas : int
            Number of antennas.
        n_baselines : int
            Number of baselines (typically n_antennas * (n_antennas + 1) / 2).
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
            Memory estimates with keys:
                - "output_bytes": int, memory for output visibilities
                - "working_bytes": int, estimated working memory
                - "total_bytes": int, total estimated memory
                - "output_human": str, human-readable output size
                - "total_human": str, human-readable total size
                - "warning": str or None, warning if memory is high

        Examples
        --------
        >>> sim = get_simulator("rime")
        >>> mem = sim.get_memory_estimate(
        ...     n_antennas=350, n_baselines=61425, n_sources=10000, n_frequencies=1024
        ... )
        >>> print(f"Estimated memory: {mem['total_human']}")
        Estimated memory: 4.8 GB
        """
        # Bytes per complex number (complex128 = 16 bytes)
        bytes_per_complex = 16

        # Polarization factor (2×2 matrix vs scalar)
        pol_factor = 4 if polarized else 1

        # Output visibilities: n_baselines × n_freq × n_times × pol_factor
        output_bytes = (
            n_baselines * n_frequencies * n_times * pol_factor * bytes_per_complex
        )

        # Working memory estimate (varies by algorithm)
        # Default: assume we need source arrays, beam patterns, intermediate results
        # This is a rough estimate; subclasses should override for accuracy
        working_bytes = (
            n_sources * n_frequencies * bytes_per_complex * 2  # Source flux arrays
            + n_antennas * n_frequencies * pol_factor * bytes_per_complex  # Beam arrays
            + n_baselines
            * n_frequencies
            * pol_factor
            * bytes_per_complex  # Intermediate
        )

        total_bytes = output_bytes + working_bytes

        # Human-readable formatting
        def format_bytes(b: int) -> str:
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if b < 1024:
                    return f"{b:.1f} {unit}"
                b /= 1024
            return f"{b:.1f} PB"

        # Warning thresholds
        warning = None
        if total_bytes > 16 * 1024**3:  # > 16 GB
            warning = (
                "Very high memory usage. Consider reducing sources or frequencies."
            )
        elif total_bytes > 4 * 1024**3:  # > 4 GB
            warning = "High memory usage. Ensure sufficient RAM available."

        return {
            "output_bytes": output_bytes,
            "working_bytes": working_bytes,
            "total_bytes": total_bytes,
            "output_human": format_bytes(output_bytes),
            "working_human": format_bytes(working_bytes),
            "total_human": format_bytes(total_bytes),
            "warning": warning,
            "details": {
                "n_antennas": n_antennas,
                "n_baselines": n_baselines,
                "n_sources": n_sources,
                "n_frequencies": n_frequencies,
                "n_times": n_times,
                "polarized": polarized,
            },
        }

    def __repr__(self) -> str:
        """String representation of the simulator."""
        return f"<{self.__class__.__name__} name='{self.name}' complexity='{self.complexity}'>"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name}: {self.description}"
