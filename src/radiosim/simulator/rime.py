"""Direct RIME visibility simulator (reference implementation).

This module implements the RIMESimulator, which computes visibilities using
direct summation over sources following the Radio Interferometer Measurement
Equation (RIME). This is the proven, accurate implementation from RadioSim v0.1.x.

The RIME computes:
    V_pq = Σ_sources J_p @ C_source @ J_q^H

Where:
    - V_pq: 2×2 visibility matrix for baseline (p, q)
    - J_p: Jones matrix chain for antenna p
    - C_source: 2×2 coherency matrix from Stokes parameters
    - ^H: Hermitian conjugate
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radiosim.backends.base import ArrayBackend
    from radiosim.core.beam import BeamSystem
    from radiosim.core.instrument_adapters import SolverInstrumentView
    from radiosim.core.jones_terms import ResolvedJonesTerms
    from radiosim.core.receptor import ResolvedReceptorSet
    from radiosim.core.runtime_config import ResolvedSolverExecutionConfig
    from radiosim.core.sky.containers.model import SourceArrays
    from radiosim.core.time_grid import ObservationTimeGrid

import numpy as np

from radiosim.core.jones_terms import EMPTY_JONES_TERMS
from radiosim.core.solver_partition import SERIAL_SOLVER_EXECUTION
from radiosim.simulator.base import VisibilitySimulator


class RIMESimulator(VisibilitySimulator):
    """
    Direct Radio Interferometer Measurement Equation (RIME) simulator.

    This simulator computes visibilities using direct summation over sources
    and baselines, implementing the full polarized RIME [1]_ with Jones
    matrices in the Hamaker-Bregman-Sault formalism [2]_.  It serves as the
    reference implementation for accuracy validation.

    Algorithm
    ---------
    For each baseline (p, q) and frequency ν:

        V_pq(ν) = Σ_s J_p(s, ν) @ C_s(ν) @ J_q(s, ν)^H

    Where the canonical Jones chain is
    J = H @ G @ B @ Rc @ Kd @ X @ D @ C @ E @ P @ T @ Z
    (``Tier7JonesSciencePlan.md`` Section 12.2), leftmost nearest the
    correlator, with K applied separately as a scalar phase:

    - H: Reporting-basis transform (always present)
    - G: Electronic gains (time-variable)
    - B: Bandpass (frequency-dependent)
    - Rc: Cable reflection ripple
    - Kd: Instrumental delay
    - X: Cross-hand phase and delay
    - D: Polarization leakage
    - C: Receptor configuration and static feed rotation (always present)
    - E: Primary beam response (direction-dependent, always present)
    - P: Parallactic angle / field rotation (direction-dependent)
    - T: Tropospheric effects
    - Z: Ionospheric effects (Faraday rotation, TEC)
    - K: Geometric phase delay (fringe rotation), applied separately

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
    - Explicit NumPy, JAX, or Dask backend selection for supported kernels

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
    >>> from radiosim.simulator import get_simulator
    >>> from radiosim.backends import get_backend
    >>>
    >>> # Create RIME simulator
    >>> sim = get_simulator("rime")
    >>> print(sim.name, sim.complexity)
    rime O(N_src × N_bl × N_freq)

    The solver inputs come from :meth:`radiosim.api.Simulator.setup`, so the
    call itself is illustrative rather than executed:

    .. code-block:: python

        # Calculate visibilities with an explicit optional backend
        backend = get_backend("jax")  # or "numpy", "dask"
        visibilities = sim.calculate_visibilities(
            instrument=instrument_view,
            beam_system=beam_system,
            source_arrays=source_arrays,
            frequencies=freqs,
            backend=backend,
            location=location,
            time_grid=time_grid,
        )

        # Backend-native receptor matrix cube: time, baseline, frequency, 2, 2
        assert visibilities.shape == (1, 15, 2, 2, 2)

    See Also
    --------
    radiosim.core.visibility.calculate_visibility : Core implementation
    radiosim.core.jones : Jones matrix framework
    radiosim.backends : Backend abstraction for CPU/GPU

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
        """Whether an end-to-end accelerator run has been measured. It has not.

        Before Tier 6H this returned ``True`` unconditionally, on the strength
        of a JAX backend existing. No GPU or TPU run of this simulator has ever
        been executed or measured: the per-time and per-frequency orchestration
        is host-side Python, coordinate transforms run in astropy, and beam
        interpolation runs in pyuvdata. This will return ``True`` when a
        measured accelerator run exists, which Tier 6 does not produce
        (``Tier6HybridRuntimePlan.md`` Section 14.1, defect D10).
        """
        return False

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
        solver_execution: "ResolvedSolverExecutionConfig" = SERIAL_SOLVER_EXECUTION,
    ) -> Any:
        """
        Calculate visibilities using direct RIME summation.

        Delegates to the core visibility calculation function which contains
        the proven implementation from RadioSim v0.1.x with full polarization
        support and backend abstraction.

        Parameters
        ----------
        instrument : SolverInstrumentView
            Owned canonical antenna values and selected baseline geometry.

        beam_system : BeamSystem
            Exact canonical per-antenna beam evaluator.

        source_arrays : dict
            Dict of source arrays from ``SkyModel.as_point_source_arrays()``.

        frequencies : ndarray
            Frequency array in Hz.

        backend : ArrayBackend
            Computation backend (numpy, jax, or dask).

        location : EarthLocation
            Observer coordinates.

        time_grid : ObservationTimeGrid
            Exact canonical UTC sample-center grid.

        receptors : ResolvedReceptorSet
            Canonical resolved receptor inventory supplying the C and H terms.

        jones_terms : ResolvedJonesTerms
            The run's resolved Jones-term inventory (Section 22).  Defaults to
            the empty inventory, which is the historical forward model.

        solver_execution : ResolvedSolverExecutionConfig, optional
            Resolved solver worker policy (``Tier6HybridRuntimePlan.md``
            Section 11.3).  ``workers=1`` is the exact serial path; larger
            counts spread contiguous time blocks over a thread pool and
            reassemble them in time order, bit-identically.

        Returns
        -------
        backend array
            Receptor cube with shape ``(T, B, F, 2, 2)``.

        Raises
        ------
        ImportError
            If core.visibility module cannot be imported.

        Notes
        -----
        This method is a thin wrapper around
        `radiosim.core.visibility.calculate_visibility()`. All the heavy
        computation is done in the core module, which has been extensively
        tested and validated against other simulators.
        """
        # Import here to avoid circular imports.
        from radiosim.core.visibility import calculate_visibility

        # Delegate to core implementation
        return calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=source_arrays,
            location=location,
            time_grid=time_grid,
            frequencies=frequencies,
            backend=backend,
            receptors=receptors,
            jones_terms=jones_terms,
            solver_execution=solver_execution,
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
