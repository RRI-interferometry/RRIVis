Changelog
=========

All notable changes to RadioSim are documented here.

[Unreleased]
------------

Added
^^^^^

- **Every Jones term now implements real physics.** Tier 7 of the remediation
  programme turned the surviving terms from names into mathematics, one slice
  at a time: gains ``G`` and bandpass ``B``; polarization leakage ``D``,
  cross-hand phase and delay ``X``, instrumental delay ``Kd`` and cable
  reflection ``Rc``; parallactic rotation ``P`` with five mount types;
  ionosphere ``Z`` (dispersive phase, thin-shell slant mapping, ionospheric
  Faraday) and troposphere ``T`` (Saastamoinen or explicit zenith delay,
  simple or Niell mapping, opacity); and the two baseline-dependent terms
  ``M`` (closure error) and ``Q`` (time and bandwidth smearing), which apply
  by Hadamard product rather than through the matrix chain. Each carries a
  cited convention, documented units and signs, an analytic invariant test, a
  backend-parity case, and an effect-changes-visibility case. See
  :doc:`user_guide/jones_terms`.
- **A typed** ``jones:`` **configuration section**, one block per enabled
  term, resolved once into a frozen ``ResolvedJonesTerms`` that reaches the
  solvers as a typed parameter. Its absence reproduces the previous forward
  model bit for bit, and its fingerprint enters ``scientific_sha256``. There
  is no ``enabled: false``: delete the block, and a block whose resolved
  parameters make the term exactly the identity is rejected.
- **Direction-batched Jones evaluation.** ``JonesTerm.compute_jones_batch``
  returns ``(n_dir, 2, 2)`` for a direction-dependent term and ``(1, 2, 2)``
  for a direction-independent one, over a ``DirectionBatch``; both solvers
  call one shared ``evaluate_antenna_jones``.
- **Beam pointing offsets and Ruze surface efficiency.** ``beams`` gains
  per-antenna deterministic pointing offsets and a surface-error block, with
  ``ruze_power_efficiency()`` and ``ruze_voltage_factor()`` public in
  ``radiosim.core.beam.runtime``. ``src/radiosim/core/jones/beam/TODO.md``
  became :doc:`development/beam_physics_scope`, which gives every remaining
  beam-physics item a disposition, a citation, and an owning register row.
- **Cross-implementation validation against** ``pyuvsim 1.4.0``, in an
  optional ``crossval`` pixi environment that no gate runs, with the measured
  comparison committed under ``output/crossvalidation/``.

Changed
^^^^^^^

- **The canonical Jones chain order.** ``P`` moved sky-side of ``C``:
  ``J_p = H_p G_p B_p Rc_p Kd_p X_p D_p C_p E_p P_p T_p Z_p``, with ``K``
  applied separately. The previous order was wrong for a circular receptor,
  where it applied a real 2x2 rotation to the ``(R, L)`` pair. This affects
  only runs with ``jones.P`` enabled, which could not exist before it was
  implemented. See :doc:`migration_guide`.
- **Non-**\ ``fixed`` **mount types are accepted.** Receptor resolution used to
  reject every antenna whose ``mount_type`` was not ``fixed``, with a message
  that named a tier rather than a fix. An array whose feeds rotate now
  requires ``jones.P``, and an array whose feeds do not rotate rejects it;
  both messages name the fix.

Removed
^^^^^^^

- **Twenty-six Jones classes**: every exported Jones term that returned the 2x2
  identity for every input, and could therefore be configured without changing
  anything, was removed rather than left as a scaffold. ``GeometricPhaseJones``
  became the ``geometric_phase()`` function (K is per-baseline),
  ``CrosshandPhaseJones`` was renamed ``CrosshandJones``, and the modules
  ``faraday``, ``wterm`` and ``element_beam`` are gone.
  :doc:`migration_guide` names the replacement for each. The nineteen remaining
  exports each declare ``term_status``, and every one of them reads
  ``"implemented"``.
- **The per-direction Jones evaluation methods**: ``JonesTerm.compute_jones``,
  ``JonesTerm.compute_jones_all_sources``,
  ``JonesChain.compute_antenna_jones``,
  ``JonesChain.compute_antenna_jones_all_sources``,
  ``JonesChain.compute_baseline_visibility`` and
  ``JonesBaselineTerm.compute_baseline_term`` are gone, replaced by the
  keyword-only batched ``compute_jones_batch`` and
  ``compute_baseline_factor``. :doc:`migration_guide` gives the per-method
  replacement.
- **jones_config solver parameter**: removed from every solver and simulator
  signature. It was an untyped dictionary hard-coded to ``None`` at the only
  production call site, and every term it could enable was one of the removed
  identity stubs. A typed ``jones:`` configuration section replaces it.
- **visibility.calculation_type**: removed before v1.0. It validated
  ``direct_sum`` and ``spherical_harmonic``, and nothing in the package read
  either value. The solver strategy is selected by ``execution.simulator``,
  whose accepted values are exactly the keys of the simulator registry. A
  document that still sets the key is rejected with removed-field guidance
  naming the replacement. See :doc:`migration_guide`.

[0.2.0] - 2025-12-15
--------------------

Major release with package restructuring.

.. note::

   This entry is kept verbatim as the historical record of what was announced
   at the time. Two of its claims did not survive verification and are
   corrected above rather than edited here. The "Universal GPU acceleration"
   and "run on CPU or GPU" claims were never measured; ``RIMESimulator``
   reports ``supports_gpu = False``, the locked JAX is CPU-only, and the only
   backend performance figures RadioSim publishes are the records under
   ``output/benchmarks/reference/`` — see :doc:`user_guide/backends`. The
   "Complete 8-term Jones chain" listed six terms whose ``compute_jones()``
   returned the 2x2 identity for every input; they became real physics in the
   Unreleased entry above, and the ones RadioSim does not model were deleted
   rather than kept as scaffolding.

Added
^^^^^

- **Package Structure**: Proper Python package installable via ``pip install radiosim``
- **GPU Support**: Universal GPU acceleration via JAX and Numba backends
  - NVIDIA (CUDA 12)
  - AMD (ROCm)
  - Apple Silicon (Metal)
  - Google TPU
- **High-Level API**: New ``Simulator`` class for easy notebook/script usage
- **Backend Abstraction**: Write once, run on CPU or GPU
- **Jones Matrix Framework**: Complete 8-term Jones chain
  - K: Geometric delay
  - E: Primary beam
  - Z: Ionosphere
  - T: Troposphere
  - P: Parallactic angle
  - G: Gain
  - B: Bandpass
  - D: Polarization leakage
- **Precision Control**: Granular control over numerical precision
  - Presets: ``standard`` (float64), ``fast`` (mixed), ``precise`` (float128), ``ultra``
  - Per-component control for coordinates, Jones matrices, accumulation, output
  - float128/complex256 support on NumPy (platform-dependent)
  - Automatic fallback with warnings on unsupported backends
  - YAML configuration support in config files
- **Pydantic Configuration**: Type-safe configuration with validation
- **CLI Commands**: ``radiosim`` and ``radiosim-migrate``
- **Measurement Set I/O**: Export to CASA MS format for QuartiCal/WSClean
  - ``write_ms()``: Write visibilities to MS format
  - ``read_ms()``: Read MS files back into memory
  - ``ms_info()``: Quick metadata summary
  - Install with: ``pip install radiosim[ms]``
- **Test Suite**: 376+ tests covering unit, integration, and performance
- **CI/CD**: GitHub Actions with multi-Python, multi-OS testing
- **Documentation**: Sphinx documentation with autodoc

Changed
^^^^^^^

- Module structure reorganized into subpackages (``core/``, ``backends/``, ``api/``, etc.)
- Import paths changed from ``src.*`` to ``radiosim.*``
- ``file_format`` parameter renamed to ``format_type`` in antenna readers

Fixed
^^^^^

- Numerous bug fixes and performance improvements
- MWA antenna FITS file parsing
- Polarization support edge cases

[0.1.x] - 2024
--------------

Initial development releases.

Added
^^^^^

- Basic visibility calculation
- GLEAM and GSM sky models
- Analytic beam patterns
- Beam FITS file support
- Full polarization RIME
- Bokeh visualization
- HDF5 output

Migration
---------

See :doc:`migration_guide` for upgrading from v0.1.x to v0.2.0.
