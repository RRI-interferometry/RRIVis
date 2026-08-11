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

from radiosim.core.contraction import (  # pyright: ignore[reportPrivateUsage]
    _TARGET_KERNEL_PAIRS,
)
from radiosim.core.jones_terms import EMPTY_JONES_TERMS
from radiosim.core.solver_partition import SERIAL_SOLVER_EXECUTION
from radiosim.simulator.base import (  # pyright: ignore[reportPrivateUsage]
    VisibilitySimulator,
    _require_kernel_n_sources,
)


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
            The run's resolved optional Jones-term inventory (Section 22).
            Defaults to the current empty inventory; always-present beam,
            receptor, reporting-basis, and geometric factors still apply.

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
        *,
        kernel_n_sources: int | None = None,
    ) -> dict[str, Any]:
        """
        Estimate memory requirements for RIME simulation.

        The estimate distinguishes the complete caller-owned baseline/source
        inputs from the source-dependent working set of one contraction leaf.
        P-a bounds only the latter; output and wrapper assembly storage still
        grow with the logical baseline count.

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
            Whether the logical sky carries full Stokes polarization. The RIME
            output, Jones inputs, and contraction intermediates remain 2x2
            matrices for an I-only sky.
        kernel_n_sources : int, optional
            Source-axis size actually presented to the contraction. ``None``
            uses ``n_sources``. A compiling backend can pass the next
            power-of-two bucket, which remains strictly less than twice the
            positive logical count.

        Returns
        -------
        dict
            Memory estimates with human-readable sizes.
        """
        resolved_kernel_sources = _require_kernel_n_sources(
            n_sources,
            kernel_n_sources,
        )
        bytes_per_complex = 16  # complex128
        bytes_per_real = 8  # float64
        matrix_factor = 4  # every RIME visibility/Jones value is 2x2

        # Output: baselines × freq × time × 2 × 2, even for an I-only sky.
        output_bytes = (
            n_baselines * n_frequencies * n_times * matrix_factor * bytes_per_complex
        )

        # The logical catalog remains alive while one horizon-selected batch is
        # padded. Count the largest fixed-width forms used by either point or
        # HEALPix input: four per-channel Stokes arrays, l/m/n, scalar Stokes,
        # and RA/Dec plus spectral/morphology metadata.
        source_arrays = 4 * n_sources * n_frequencies * bytes_per_real
        direction_cosines = 3 * n_sources * bytes_per_real
        stokes_params = 4 * n_sources * bytes_per_real
        logical_source_metadata = 8 * n_sources * bytes_per_real

        # P-b first creates five padded horizontal/direction arrays, then
        # DirectionBatch owns eight float64 copies. The point path can also
        # retain four Stokes signals, three spectral scalars, three Gaussian
        # morphology arrays, and four per-channel Stokes arrays. Counting all
        # fixed-width optional arrays simultaneously is deliberately
        # conservative and covers the smaller HEALPix payload as well.
        padded_host_directions = 5 * resolved_kernel_sources * bytes_per_real
        direction_batch_host_arrays = 8 * resolved_kernel_sources * bytes_per_real
        fixed_signal_metadata_arrays = 10 + 4 * n_frequencies
        padded_host_signal_metadata = (
            fixed_signal_metadata_arrays * resolved_kernel_sources * bytes_per_real
        )

        # Backend copies coexist with the host batch. l/m/n plus the fixed
        # signal/metadata payload are included; DirectionBatch itself stays
        # host-owned and is counted immediately above.
        backend_source_only_arrays = (
            (3 + fixed_signal_metadata_arrays)
            * resolved_kernel_sources
            * bytes_per_real
        )
        # Point-polarized and both HEALPix paths can materialize an explicit
        # per-source 2x2 coherency. This worst case also safely covers the point
        # I-only specialization's smaller Stokes-I vector.
        source_only_coherency_or_stokes = (
            resolved_kernel_sources * matrix_factor * bytes_per_complex
        )
        # One time/frequency step retains per-antenna Jones evaluations at the
        # actual kernel source count before gathering the selected baselines.
        beam_arrays = (
            n_antennas * resolved_kernel_sources * matrix_factor * bytes_per_complex
        )

        # These complete B*S inputs are constructed before P-a scheduling, so
        # baseline chunking does not bound them. The envelope is a conservative
        # complex array worst case: it can instead be scalar or real depending
        # on the configured morphology and baseline terms.
        caller_jones_inputs = (
            2
            * n_baselines
            * resolved_kernel_sources
            * matrix_factor
            * bytes_per_complex
        )
        caller_phase_input = n_baselines * resolved_kernel_sources * bytes_per_complex
        caller_array_envelope = (
            n_baselines * resolved_kernel_sources * bytes_per_complex
        )

        if n_baselines == 0 or resolved_kernel_sources == 0:
            max_kernel_baselines = 0
            max_kernel_pair_count = 0
        else:
            max_kernel_baselines = max(
                1,
                min(
                    n_baselines,
                    _TARGET_KERNEL_PAIRS // resolved_kernel_sources,
                ),
            )
            max_kernel_pair_count = max_kernel_baselines * resolved_kernel_sources

        # Two matrix-product-sized intermediates, one weighted product, and one
        # scalar weight are the bounded source-dependent leaf component. This
        # is 208 bytes/pair for either polarization mode at complex128.
        contraction_leaf_working = (
            max_kernel_pair_count * bytes_per_complex * (3 * matrix_factor + 1)
        )
        # At concatenation, retained chunk outputs and the assembled output can
        # coexist. This remains O(B) and is deliberately outside the leaf bound.
        contraction_output_assembly = (
            2 * n_baselines * matrix_factor * bytes_per_complex
        )

        breakdown_bytes = {
            "source_arrays": source_arrays,
            "direction_cosines": direction_cosines,
            "stokes_parameters": stokes_params,
            "logical_source_metadata": logical_source_metadata,
            "padded_host_directions": padded_host_directions,
            "direction_batch_host_arrays": direction_batch_host_arrays,
            "padded_host_signal_metadata": padded_host_signal_metadata,
            "backend_source_only_arrays": backend_source_only_arrays,
            "source_only_coherency_or_stokes": source_only_coherency_or_stokes,
            "beam_patterns": beam_arrays,
            "caller_jones_inputs": caller_jones_inputs,
            "caller_phase_input": caller_phase_input,
            "caller_array_envelope": caller_array_envelope,
            "contraction_leaf_working": contraction_leaf_working,
            "contraction_output_assembly": contraction_output_assembly,
        }
        working_bytes = sum(breakdown_bytes.values())
        total_bytes = output_bytes + working_bytes

        def format_bytes(b: int) -> str:
            value = float(b)
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if value < 1024:
                    return f"{value:.1f} {unit}"
                value /= 1024
            return f"{value:.1f} PB"

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
                "logical_n_sources": n_sources,
                "kernel_n_sources": resolved_kernel_sources,
                "n_frequencies": n_frequencies,
                "n_times": n_times,
                "polarized": polarized,
                "matrix_factor": matrix_factor,
                "target_kernel_pairs": _TARGET_KERNEL_PAIRS,
                "max_kernel_baselines": max_kernel_baselines,
                "max_kernel_pair_count": max_kernel_pair_count,
                "caller_array_envelope_assumption": (
                    "conservative optional complex baseline-source array"
                ),
                "estimate_limitations": (
                    "All fixed-width point and HEALPix padded arrays are counted "
                    "concurrently. Additional variable-width spectral "
                    "coefficients are excluded because their coefficient width "
                    "is not an input to this API. Native allocator overhead and "
                    "parallel worker multiplication are also excluded."
                ),
            },
            "breakdown_bytes": breakdown_bytes,
            "breakdown": {
                name: format_bytes(value) for name, value in breakdown_bytes.items()
            },
        }
