Changelog
=========

All notable changes to RadioSim are documented here.

[Unreleased]
------------

Fixed
^^^^^

- **Linear polarization products now honor the declared east-X feed**
  (``SCI-006``, accepted 2026-08-11). Sky brightness remains canonical
  ``(North, East)``, while the receptor term now applies the fixed permutation
  to ``(X=east, Y=north)``. At zero rotation, ``XX=(I-Q)/2``,
  ``XY=(U-iV)/2``, ``YX=(U+iV)/2``, ``YY=(I+Q)/2``, and ``XX-YY=-Q``.
  Ideal/scalar pure-I and ideal circular-output results are unchanged;
  polarized linear results and feed-asymmetric configurations change
  numerically. See
  :doc:`migration_guide` and
  :doc:`development/sci006_polarization_convention`. Exact candidate
  ``f5fa101e`` passed quality, backend parity, and all six compatibility cells
  in CI run ``31434253575``; the retained artifacts and basis transforms were
  independently authenticated before the issue was closed.
- **The linux-64-py311 characterization gate recognizes its second verified
  numerical class** (``CI-001``). A six-draw forced-discrimination experiment
  reproduced both byte-stable classes on one runner, isolated NumPy AVX-512
  dispatch and the OpenBLAS ``SkylakeX`` runtime core as the two contributing
  axes, and measured all changed cubes within the full Tier 6 Section 13.5
  predicate. Digest membership remains the primary gate; ``rtol=1e-12`` plus
  its absolute term remains the failure-path adjudication rule. See
  :doc:`development/ci001_adjudication` for the retained evidence and policy.
- **stokes_to_coherency broadcasts its inputs** (``API-001``). The four
  Stokes inputs now broadcast against each other under the usual NumPy
  rules, so ``stokes_to_coherency(np.ones(5))`` — array ``I`` with the
  scalar Q/U/V defaults — works instead of raising. Genuinely incompatible
  shapes still raise ``ValueError``. Previously valid inputs take the
  identical arithmetic path: broadcasting already-equal shapes is an
  identity view, so no result changes.
- **Console helpers print literal text** (``API-002``). The four
  ``print_success`` / ``print_error`` / ``print_warning`` / ``print_info``
  helpers in ``radiosim.utils.logging`` now escape Rich markup in the
  caller's message, so bracketed text — for example the model-name list in
  ``Simulator.setup()``'s offline pre-flight warning — prints as-is instead
  of being parsed (and silently eaten) as a markup tag. The ``RichHandler``
  installed by ``setup_logging()`` likewise renders logged messages literally
  (``markup=False``). Only the helpers' styled glyph prefixes remain markup.
- **Forced-offline runs stop before cache-tolerant download code.** An
  explicit ``execution.offline: true`` policy now raises the standard
  actionable ``ConnectionError`` even for loaders that normally permit a
  local-cache attempt. This prevents pygdsm or another third-party loader from
  entering download code on a cold cache. A naturally offline host still
  retains the previous warning-and-cache-fallback behavior.

[0.3.0] - 2026-08-02
--------------------

The result of an eight-tier remediation programme. Every section below is a
breaking change against ``0.2.0`` unless it says otherwise: RadioSim is
pre-v1.0, and under the policy in :doc:`contributing` a misleading API is
replaced rather than aliased. :doc:`migration_guide` gives the per-symbol and
per-key replacement for all of it.

Read the "Known limitations" section at the end before reading the capability
list: it names, with register identifiers, everything this release does *not*
do.

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
- **One strict configuration contract.** A single Pydantic ``RadioSimConfig``
  with the sections ``instrument``, ``beams``, ``receptors``,
  ``baseline_selection``, ``sky_model``, ``obs_time``, ``obs_frequency``,
  ``visibility``, ``jones``, ``execution`` and ``workflow``; ``load_config()``,
  ``resolve_config()`` and ``dump_config()`` as the entry points; source-aware
  relative-path resolution; and a pre-flight validator that reports every
  error at once instead of the first.
- **One typed instrument source** with canonical identity, location, antenna
  positions, per-antenna diameters and deterministic provenance, plus typed
  baseline selection (``correlations``, ``length_filter``,
  ``azimuth_ranges_deg``) resolved by ``generate_resolved_baselines()`` and
  ``select_resolved_baselines()``.
- **A canonical beam system.** One ``BeamSystem`` covering analytic apertures,
  illumination tapers and BeamFITS handlers, with shared or per-antenna
  assignment for heterogeneous arrays, and observability as a helper on the
  ``Simulator`` rather than a second engine.
- **An immutable canonical result.** ``Simulator.run()`` returns a frozen
  ``SimulationResult`` carrying the ``(time, baseline, frequency,
  correlation)`` cube, its correlation labels and polarization basis, and two
  digests: ``scientific_sha256`` over the numbers and the physics inputs, and
  ``provenance_sha256`` over the run's environment. HDF5 schema ``4.0.0``,
  summary JSON, Measurement Set and UVFITS export, and atomic run-directory
  publication that leaves the previous run intact on any failure.
- **A receptor and polarization-basis contract.** A ``receptors`` section with
  per-antenna ``basis`` and static ``feed_rotation_deg``, one array-wide
  ``output_basis``, and ``linear_xy``/``circular_rl`` as the two canonical
  correlation coordinate systems.
- **Backend parity across NumPy, JAX and Dask.** Both solvers route their
  Jones chain, geometric phase, coherency construction, contraction and
  accumulation through the selected backend; exactly one kernel is compiled
  (the baseline-batched per-``(time, frequency)`` contraction, under JAX).
  Dask is bit-identical to NumPy and JAX-CPU agrees within ``rtol=1e-12``. The
  benchmark harness, its record schema and the committed reference records
  under ``output/benchmarks/reference/`` landed with it.
- **A hybrid sky model.** One run may carry both point sources and a HEALPix
  map, with first-class sparse HEALPix support, physical-disjointness checks
  when models are combined, and NEST/RING ordering threaded end to end.
- **Executed documentation.** ``pixi run doctest`` runs the package's
  docstring examples, CI executes the shipped example script and the notebook,
  the Sphinx build runs under ``-W --keep-going`` with zero warnings, and
  ``tests/unit/test_tier8_release_acceptance.py`` scans the tracked prose for
  removed symbols, dead paths, unresolvable ``radiosim.`` names, stale
  configuration counts, uncited accelerator claims and non-existent
  ``pixi run`` tasks.
- **Declared network services.** ``get_required_services()`` reports every
  service a configuration will reach, including through a composite recipe, so
  the pre-flight's offline/online line is derived rather than guessed.

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
- **The coherency matrix's Stokes** ``V`` **sign.**
  ``B = (1/2) [[I+Q, U+iV], [U-iV, I-Q]]`` (IAU). The previous form was the
  mirror image, so every circular-hand quantity in a run with non-zero ``V``
  changes sign. This is a numerical change to results, not a rename.
- **Correlation labels, the HDF5 schema and the reported basis** are data
  carried by the result rather than constants: a circular-receptor run reports
  ``("RR", "RL", "LR", "LL")``. HDF5 schema ``4.0.0`` and summary JSON
  ``1.1.0`` are the current formats.
- **Backend** ``auto`` **is a real selection strategy**: it returns JAX only
  when JAX reports a non-CPU device, and NumPy otherwise. It never selects
  Dask. ``RIMESimulator.supports_gpu`` is now ``False``, because no
  accelerator run has been measured (``PERF-001``).
- **Every FITS-beam fingerprint moved** when the beam runtime became
  canonical: peak normalization and interpolation are now applied in one
  place. A stored digest from ``0.2.0`` will not reproduce.
- ``LoaderDefinition.network_service: str | None`` **became**
  ``network_services: tuple[str, ...]``, with no compatibility shim. A
  composite recipe declares the union of the services it dispatches to;
  before the change it could declare neither, and the shipped
  ``configs/realistic_foreground_example.yaml`` reported no required service
  while making two real network calls. See :doc:`migration_guide`.

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
- **The** ``numba`` **backend name and** ``NumbaBackend``. That backend never
  compiled a kernel and its ``mode="gpu"`` path validated a CUDA device and
  then ran NumPy; it is now ``DaskBackend``, selected as ``dask``.
  ``get_backend("numba")`` raises.
- **The four accelerator extras** ``gpu``, ``gpu-cuda``, ``gpu-rocm`` and
  ``tpu``, and the ``gpu`` packaging keyword. ``pip install
  radiosim[gpu-cuda]`` installed ``jax[cuda12]`` and delivered a package that
  has never executed on an accelerator. The JAX stack stays installable as
  ``pip install radiosim[jax]``; a device-named extra returns when
  ``PERF-001`` closes with measurements.
- **The split top-level configuration sections**: ``feeds``, ``compute``, and
  the boolean/parallel-list baseline keys (``use_autocorrelations``,
  ``only_selective_baseline_length``, ``trim_by_angle_ranges`` and their
  companions). Each is rejected with a message naming its typed replacement.
- ``generate_baselines`` **and the low-level beam classes** ``BeamJones``,
  ``AnalyticBeamJones``, ``FITSBeamJones``, ``BeamManager``,
  ``BeamFITSHandler`` and ``AntennaType``, together with the named beam types
  ``airy``, ``cosine``, ``exponential`` and ``short_dipole``.
- ``combine_models``, ``source_format`` **and** ``available_formats`` on the
  sky model, replaced by ``prepare_sky_model()`` and ``SkyModel.formats``, and
  ``run(n_workers=...)``, replaced by the ``execution`` worker policy.

Known limitations
^^^^^^^^^^^^^^^^^

Release notes that list capabilities without listing these would be
incomplete. Each entry is an open row in the project's defect register.

- ``PERF-001`` (roadmap): accelerator performance is undemonstrated. The time
  and frequency axes are host-side Python loops, coordinate transforms and
  beam interpolation are host-side by design, the JAX declared by every pixi
  environment is CPU-only, and every measured JAX-CPU run is slower than NumPy
  (``output/benchmarks/reference/``). RadioSim publishes no GPU or TPU
  performance number.
- ``SCI-004`` (roadmap): there is no spherical-harmonic or m-mode solver.
  ``execution.simulator`` accepts exactly the keys of the simulator registry,
  which currently holds only ``rime``.
- ``SCI-005`` (roadmap): the accepted primary beam is scalar
  (``E = e·I2``). Polarized and cross-polar beams, beam squint, aperture
  blockage, Zernike aberrations and the Ruze error-beam decomposition are out
  of scope, each with its reasoning in
  :doc:`development/beam_physics_scope`.
- ``SCI-006`` (open): RadioSim's local Stokes ``Q`` has the opposite sign to
  ``pyuvsim``/``pyradiosky``'s for the same sky, feed convention and mount.
  The cross-validation characterizes it as a local-basis axis-order swap that
  also flips ``V``; polarized intensity agrees to ``2.1e-3`` relative after
  the swap, but which convention is right is not established.
- ``SCI-007`` (open): after that basis swap, the local linear-polarization
  frame still differs from ``pyuvsim``'s by a fitted ``-0.0576`` degrees,
  unreconciled against the independent frame probes recorded beside it.
- ``CI-001`` (open): one of the eight continuous-integration jobs
  (``linux-64``, Python 3.11) intermittently produces a second byte-stable
  scientific digest for the shipped configurations, with an unidentified
  discriminator. Source regression, package drift, thread counts and test
  ordering are ruled out with evidence. No "CI is green" claim is made while
  it is open.
- ``API-001`` (open, low-priority): ``stokes_to_coherency`` does not
  broadcast its scalar ``Q``/``U``/``V`` keyword defaults against a
  non-scalar ``I`` argument, so the single most basic array-input call
  raises instead of broadcasting. No shipped solver path is affected: both
  production call sites always pass four already-matched-shape arrays.
- ``API-002`` (open, low-priority): ``print_warning`` and its siblings in
  ``utils/logging.py`` leave Rich markup enabled on an interpolated
  message, so a caller-built string containing a ``[...]``-bracketed
  substring (for example a model name list) is silently dropped from the
  rendered output rather than escaped or shown literally.

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
  - Presets: ``standard`` (float64 component settings except float32 HEALPix storage), ``fast`` (mixed), ``precise`` (float128 on critical paths), ``ultra`` (maximum supported precision per path)
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

See :doc:`migration_guide` for upgrading to ``0.3.0``, and for the v0.1.x to
v0.2.0 history it also records.
