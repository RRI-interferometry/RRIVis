"""Abstract base class for visibility simulators.

This module defines the interface that all visibility simulators must implement,
allowing for different algorithms (direct RIME, FFT-based, matrix-based) to be
swapped without changing the user-facing API.

The design follows the Strategy pattern, enabling runtime selection of simulation
algorithms based on problem characteristics (source count, array density, etc.).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radiosim.backends.base import ArrayBackend
    from radiosim.core.beam import BeamSystem
    from radiosim.core.instrument_adapters import SolverInstrumentView
    from radiosim.core.jones_terms import ResolvedJonesTerms
    from radiosim.core.receptor import ResolvedReceptorSet
    from radiosim.core.sky.containers.model import SourceArrays
    from radiosim.core.time_grid import ObservationTimeGrid

import numpy as np

from radiosim.core.jones_terms import EMPTY_JONES_TERMS


class VisibilitySimulator(ABC):
    """
    Abstract base class for visibility simulators.

    This interface allows swapping visibility calculation algorithms without
    changing the user-facing API. Implementations can optimize for different
    scenarios (accuracy vs speed, few vs many sources, etc.).

    Current Implementations:
        - RIMESimulator: Direct RIME summation, O(N_src × N_bl × N_freq), accurate

    It is the only registered one. Other solver families -- an FFT/NUFFT
    solver, a matrix-based solver -- would be future registrations against this
    same interface; no release promises them and nothing here measures what
    they would cost.

    Examples
    --------
    >>> from radiosim.simulator import get_simulator
    >>> sim = get_simulator("rime")
    >>> print(sim.name, sim.complexity)
    rime O(N_src × N_bl × N_freq)

    The solver inputs come from :meth:`radiosim.api.Simulator.setup`, so the
    call itself is illustrative rather than executed:

    .. code-block:: python

        visibilities = sim.calculate_visibilities(
            instrument=instrument_view,
            beam_system=beam_system,
            source_arrays=source_arrays,
            frequencies=freqs,
            backend=backend,
            location=location,
            time_grid=time_grid,
        )
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
        Whether an end-to-end accelerator run of this simulator has been
        measured.

        This is a claim about evidence, not about whether an accelerator
        library can be imported. RadioSim has measured none: the shipped
        ``RIMESimulator`` overrides this property to return ``False``, the JAX
        declared by every pixi environment is CPU-only, and every measured JAX
        run is slower than NumPy (records: ``output/benchmarks/reference/``,
        register row ``PERF-001``).

        The inherited value here is ``True`` only because it predates that
        finding; it is not a statement that a subclass is accelerated, and no
        shipped simulator relies on it. A subclass may leave it inherited only
        once a measured accelerator record exists for it, and must otherwise
        override it to ``False`` as ``RIMESimulator`` does. Flipping this
        default is a behaviour change, deliberately not made in a
        documentation slice; it is tracked with ``PERF-001``.

        Returns
        -------
        bool
            Whether a measured accelerator run backs this simulator.
        """
        return True

    @abstractmethod
    def calculate_visibilities(
        self,
        instrument: "SolverInstrumentView",
        beam_system: "BeamSystem",
        source_arrays: "SourceArrays",
        frequencies: np.ndarray,
        backend: "ArrayBackend",
        *,
        location: Any,
        time_grid: "ObservationTimeGrid",
        receptors: "ResolvedReceptorSet",
        jones_terms: "ResolvedJonesTerms" = EMPTY_JONES_TERMS,
    ) -> Any:
        """
        Calculate visibilities for all baselines.

        This is the core computation method that each simulator must implement.
        The specific algorithm (direct summation, FFT, matrix multiplication)
        depends on the implementation.

        Parameters
        ----------
        instrument : SolverInstrumentView
            Owned canonical antenna values and selected baseline geometry.

        beam_system : BeamSystem
            Exact canonical per-antenna beam evaluator.

        source_arrays : dict
            Dict of source arrays from ``SkyModel.as_point_source_arrays()``.

        frequencies : ndarray
            Frequency array in Hz. Shape: (N_freq,)

        backend : ArrayBackend
            Computation backend instance from radiosim.backends.
            Provides array operations (numpy-like API) and device management.
            Use get_backend("numpy"), get_backend("jax"), etc.

        location : EarthLocation
            Observer coordinates.

        time_grid : ObservationTimeGrid
            Exact canonical UTC sample-center grid.

        receptors : ResolvedReceptorSet
            Canonical resolved receptor inventory supplying the per-antenna
            receptor term C and reporting-basis transform H.

        jones_terms : ResolvedJonesTerms
            The one canonical Jones-term inventory for the run, resolved once in
            ``Simulator.setup()`` (``Tier7JonesSciencePlan.md`` Section 22).  It
            replaces the untyped dictionary parameter that no caller could ever
            populate (defect D3): a solver receives resolved terms and never
            parses configuration.  The default is the empty optional-term
            inventory.  Always-present beam, receptor, reporting-basis, and
            geometric factors still apply; SCI-006 intentionally changed the
            default polarized linear receptor from the historical identity to
            the east-X permutation.

        Returns
        -------
        backend array
            Receptor cube with shape ``(T, B, F, 2, 2)`` in canonical time,
            selected-baseline, frequency, receptor-row, receptor-column order.

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

        The canonical Jones chain is
        J = H @ G @ B @ Rc @ Kd @ X @ D @ C @ E @ P @ T @ Z
        (``Tier7JonesSciencePlan.md`` Section 12.2), leftmost nearest the
        correlator, with K applied separately as a scalar phase:

        - H: Reporting-basis transform
        - G: Electronic gains
        - B: Bandpass
        - Rc: Cable reflection ripple
        - Kd: Instrumental delay
        - X: Cross-hand phase and delay
        - D: Polarization leakage
        - C: Receptor configuration (basis and static feed rotation)
        - E: Primary beam response
        - P: Parallactic angle / field rotation
        - T: Troposphere
        - Z: Ionosphere (Faraday rotation)
        - K: Geometric phase (fringe rotation)
        """
        pass

    def validate_inputs(
        self,
        instrument: "SolverInstrumentView",
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
        instrument : SolverInstrumentView
            Canonical solver adapter (see calculate_visibilities).
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
        .. code-block:: python

            sim = get_simulator("rime")
            valid, errors = sim.validate_inputs(
                instrument_view, baselines, sources, freqs
            )
            if not valid:
                for err in errors:
                    print(f"Validation error: {err}")
        """
        errors = []

        from radiosim.core.instrument_adapters import SolverInstrumentView

        if type(instrument) is not SolverInstrumentView:
            errors.append("instrument must be a SolverInstrumentView")

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
        >>> from radiosim.simulator import get_simulator
        >>> sim = get_simulator("rime")
        >>> mem = sim.get_memory_estimate(
        ...     n_antennas=350, n_baselines=61425, n_sources=10000, n_frequencies=1024
        ... )
        >>> print(f"Estimated memory: {mem['total_human']}")
        Estimated memory: 3.9 GB
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
