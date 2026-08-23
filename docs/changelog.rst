Changelog
=========

All notable changes to RadioSim are documented here.

[Unreleased]
------------

Added
^^^^^

- ``SCI-004`` phase M1: the m-mode full-sidereal harmonic forward model is
  registered as ``execution.simulator: mmode``. It is a second *complete*
  forward model, not a Jones term, a point-source optimization, a map maker, or
  a new name for the direct sum. The simulator registry is now a whole-``SkyModel``
  strategy boundary: one immutable ``SkySolveRequest`` carries the resolved sky
  model, point arrays, instrument view, beam system, location, time and
  frequency coordinates, receptors, Jones inventory, backend and worker policy,
  and ``Simulator.run()`` calls only the selected registered strategy.
  ``RIMESimulator.solve`` is a thin wrapper around the maintained
  point/HEALPix/hybrid path whose arithmetic, component order, source
  reduction, result bytes and fingerprints are unchanged.
- ``Simulator.build_solve_request()`` returns that immutable request directly.
  ``run()`` now calls it rather than assembling the request inline, so an
  evidence generator or an out-of-band re-derivation consumes exactly the
  inputs a run consumed instead of rebuilding a second, divergent definition of
  the strategy boundary.
- ``execution.mmode``, required with ``simulator: mmode`` and rejected with
  ``rime``, carrying three exact convention literals and the four truncation
  dimensions; and the strict ``obs_time`` full-sidereal variant
  (``mode: full_sidereal``, ``sidereal_samples``, ``integration_fraction``).
  The untagged UTC interval is unchanged and remains the only ``rime`` input,
  so every existing document and every serialized ``rime`` snapshot stays
  byte-identical.
- The exact-turn ``CanonicalEraGrid``: sample centres, exposure edges and the
  cell-centred cycle are built in exact rational arithmetic, and each derived
  radian is one round-to-nearest of ``tau`` times its exact rational with no
  intermediate binary64 step. Earth orientation comes from exactly one bundled
  offline table, with no network lookup and no implicit table selection.
- The frozen-CIRS rigid-ERA frame ``radiosim.frozen-cirs-rigid-era.v1``, with
  one normative arcsecond-to-radian polar-motion conversion, a geocentric CIRS
  anchor, the SOFA passive ``[ITRS] = RPOM0 R3(ERA) [CIRS]`` attitude, and a
  Richardson-extrapolated tangent-transport oracle built from public Astropy
  transforms only.
- Orthonormal Condon-Shortley scalar harmonics and the unpadded
  signed-``m``-major packed block table; analytic point-delta sky coefficients
  and the HEALPix **pixel measure**
  ``a_lm = sum_pix(s_pix * Omega_pix * conj(Y_lm))`` -- the same measure the
  private direct oracle sums, so harmonic-versus-direct agreement tests
  truncation and nothing else; and the scalar baseline transfer ``B_lm`` on the
  **iso-Gauss** quadrature grid (``3 * nside`` Gauss-Legendre colatitude rings
  by ``4 * nside`` uniform azimuths) with its rigid ``exp(+i m alpha)`` rotation
  law.
- One shared horizon predicate. Section 6 requires the private direct oracles
  and the harmonic-transfer kernel to invoke the *identical* implementation of
  the strict ``alt > 0`` factor -- one code object, never a re-derivation -- so
  a horizon-application defect cannot differ between the compared models.
- The operational horizon census as a certified-ceiling scan: a uniform
  ``2**-12``-turn exact partition refined at every retained centre, edge and
  frozen root bound, a root-free rule from the design-frozen derivative ceiling,
  sign-change bisection to a fixed ``1e-11 rad`` enclosure, and probe-sign
  orientation with a magnitude floor. It consumes only the public Astropy
  pressure-zero transform, and a cell that reaches the deep-tangency width with
  neither classification rejects the whole certificate.
- ``SimulationResult.solver`` is a strict tagged union. The ``rime`` arm is
  unchanged; an m-mode result carries ``MModeSolverResultProvenance``, whose
  ``stokes_v_basis_bridge`` is always ``radiosim.stokes-ne-theta-phi.v1`` and
  whose ``tangent_polarization_frame`` is the exact literal
  ``not_applicable_scalar_m1`` for a payload with no linear polarization and
  the six-key canonical block otherwise.
- ``SCI-004`` phase M2: the m-mode forward model is **full Stokes**.
  ``MModeSimulator.supports_polarization`` is now ``True``, and a resolved
  payload with any non-zero ``Q``, ``U`` or ``V`` takes the polarized execution
  path and reports ``execution_path: "polarized"``.

  - Spin-weighted harmonics and the four-field packed block table in Section
    5.3's fixed order ``("I", "+2", "-2", "V")``, with the paired spin-reality
    relation and an unpadded signed-``m``-major layout in which invalid
    ``(l, m, s)`` cells do not exist.
  - The RadioSim-to-Shaw basis bridge as one matrix
    ``D = diag(-1, 1)``: the kernel uses ``D P^X D`` and the sky the matching
    ``U_H = -U``. It is not the SCI-006 east-X permutation, which stays inside
    the antenna Jones matrix, and no additional fitted or configurable ``V``
    flip exists.
  - The **resolved receptor matrix** ``M_p = H_p C_p`` now enters the m-mode
    kernel, from the same ``radiosim.core.jones.receptor`` code objects the
    direct chain uses. Stokes-``I`` results are unchanged by this -- for a
    unitary receptor ``M P^I M^H = (1/2) I2``, so every accepted phase-M1
    scalar result, digest and gate value is preserved bit for bit -- and it is
    load-bearing for every polarized component.
  - The polarized transfer ``B^(+2) = integral((K^Q - i K^U) {+2}Y_lm)`` and
    ``B^(-2) = integral((K^Q + i K^U) {-2}Y_lm)``, the two ``1/2`` factors on
    the spin pair in the forward per-``m`` product, and the full-Stokes
    frozen-frame direct oracle that the every-run two-tier gate compares
    against.
  - Analytic full-Stokes point coefficients, the polarized HEALPix pixel
    measure, and hybrid addition in the fixed ``("point", "healpix")`` order.
  - ``TangentPolarizationFrame``: the six-key tangent-basis record a polarized
    payload carries, with the ``north_through_west`` source convention
    converted to IAU north-through-east before storage. A polarized m-mode
    input that declares none is rejected with ``mmode_polarization_frame``,
    before backend allocation, output-path creation, or harmonic work.
  - ``MModeSimulator.get_memory_estimate`` overrides the direct-RIME shape with
    Section 9's seven named components, the logical and scheduled dimensions,
    and a one-block minimum that a smaller budget is rejected against before
    allocation.

Notes
^^^^^

- The constant chain terms act in the **celestial** tangent basis of each
  direction -- the basis the spin expansions use and the basis the direct RIME
  builds its coherency in -- and every mount-dependent tangent rotation belongs
  to the ``P`` term, exactly the identity for the shipped ``fixed`` and
  unspecified mounts. Constant coefficients on spin-weighted fields preserve the
  integrand's spin weight, keeping the spin-``±2`` quadrature spectrally exact.
  A ground-anchored, direction-dependent response would need a measured tangent
  transport; that is outside this scope, and transporting a *constant* matrix
  into the rotating local basis is the identity re-expression of a
  zenith-singular field rather than an alternative convention.
- ``MModeSimulator.supports_gpu`` remains ``False``: no end-to-end accelerator
  run of this solver has been measured, and the recorded
  ``host_harmonics_backend_native_dense_v1`` policy describes where the work
  runs -- Astropy frame work, IERS mapping, beam sampling, HEALPix geometry and
  both harmonic transforms are host-side NumPy for every backend, while only the
  dense per-``m`` contractions and time synthesis are backend-native -- rather
  than claiming an advantage. A polarized capability is not a speed claim.
  Register row ``PERF-001`` governs every performance statement. ``SCI-004``
  remains ``ROADMAP``: no phase of this work closes the row, and none adds a
  fingerprint pin.
- Truncation is gated, not assumed. Every production run executes a two-tier
  gate before any result exists. Tier 1a gates the harmonic pipeline at
  ``1e-8`` on a horizon-free cross-quadrature shell -- the same pipeline with
  the horizon factor removed, whose integrand is smooth and therefore
  spectrally exact. Tier 1b records the with-horizon shell, which no finite
  quadrature can make exact under the strict horizon and which is bounded by a
  reviewed per-fixture budget. Tier 2 reports the truncation deficit against
  the complete frozen-frame direct oracle, gated on strict monotone decrease
  across a quarter and a half of ``lmax`` with a quarter-to-full factor of at
  least two, and discloses it in the result's provenance record. The deficit is
  never called agreement: the forward product reconstructs the band-limited
  transfer kernel, so for a delta sky it is exactly ``S*K_L(n_s)`` and the
  residual is a property of the method.

[0.4.0] - 2026-08-20
--------------------

Added
^^^^^

- ``SCI-005`` Stage 3: full-efield Jones response. A full-efield UVBeam file
  accepted under the ``uvbeam_peak_common_v1`` normalization literal
  (mutually exclusive with ``beams.squint``, which requires an analytic
  beam) composes the generally full ``E = C^dagger J_native`` via the fixed
  real orthogonal chain-tangent conversion
  ``uvbeam_theta_phi_chain_tangent_v1``; Ludwig-3 remains a derived
  diagnostic, never the chain conversion, and IXR is a derived diagnostic
  with no configuration surface. ``BeamFileProvenance`` widens with
  ``None``-default fields so a scalar ``peak`` document stays
  byte-identical. The optional ``crossval`` environment's non-gating
  comparison against ``pyuvsim 1.4.0`` records agreement on the
  mechanism-free total-intensity and Stokes-V classes and a structured,
  mechanism-explained open disagreement on the ``Q``/``U``-carrying
  correlations through a cited ``pyradiosky`` coherency-sign defect; the
  fixture is compared against, never validated against. Strict typed
  rejection throughout; absent-block and scalar-``peak`` results stay
  byte-identical. Independently accepted 2026-08-20; retained evidence and
  acceptance certificates live in ``docs/development/``. With all three
  stages accepted, the ``SCI-005`` register row closed as DONE on
  2026-08-20.
- ``SCI-005`` Stage 2: beam squint. ``beams.squint`` (Cotton/Uson exact
  arcsine frequency law, mechanical-feed ``+pi/2`` squint direction, mount
  field rotation at the resolved boresight) with the two native feeds
  sampling oppositely displaced scalar beams and the beam runtime composing
  the generally full ``E = C^dagger D_b C`` at the resolved beam dtype; for
  a circular receptor ``E`` commutes exactly with every real rotation.
  Strict typed rejection throughout; absent-block results stay
  byte-identical. Independently accepted 2026-08-19; retained evidence and
  acceptance certificates live in ``docs/development/``.
- ``SCI-005`` Stage 1: scalar aperture physics. ``beams.aperture_physics``
  (central blockage, support-leg shadow masks, Noll real unit-RMS Zernike
  surface heights) and the nested ``error_beam_diagnostic`` (separation-domain
  Ruze ensemble-power diagnostic, ``poisson_paired_pupil_separation_v1``)
  with strict typed rejection throughout. Independently accepted 2026-08-18;
  retained evidence and acceptance certificates live in
  ``docs/development/``.

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

Changed
^^^^^^^

- **Automatic backend selection is deterministic and import-free**
  (``PERF-001``). ``get_backend("auto")`` now asks only NumPy to honor the
  requested precision; it never imports or probes JAX and never selects Dask.
  Generic JAX uses the runtime-default device, while named ``cpu``, ``gpu``,
  and ``tpu`` requests and the direct accelerator aliases are strict and never
  fall back to CPU. Generic device-resource reporting no longer uses JAX as a
  fallback. The inherited ``VisibilitySimulator.supports_gpu`` value is now
  ``False``; ``RIMESimulator`` remains explicitly false, and ``True`` requires
  an independently accepted end-to-end accelerator record.
- **Both solvers schedule the compiled contraction in bounded baseline
  chunks, and the JAX path buckets its source axis** (the ``PERF-001`` CPU
  slice, independently accepted 2026-08-14). The single compiled
  per-``(time, frequency)`` contraction leaf is dispatched over bounded
  baseline chunks that never split or reorder the source axis, and the JAX
  path pads the source axis with zero-signal dummy sources to the next power
  of two so far fewer distinct source-axis shapes are compiled. NumPy and
  Dask production paths remain unpadded and byte-identical to their accepted
  values, and no scheduling choice enters ``scientific_sha256`` or public
  configuration. The isolated Linux ``gpu`` pixi environment landed with the
  same slice as readiness infrastructure only: it is not accelerator
  evidence, and every measured JAX-CPU run remains slower than NumPy
  (``output/benchmarks/reference/``). See
  :doc:`development/perf001_runtime_mitigations`.
- **The polarization-frame limitation is quantified and closed as a retained-
  fixture bound** (``SCI-007``, accepted 2026-08-11). RadioSim's operational
  apparent/equinox-of-date frame is a TETE-like ideal spherical construction,
  not an exact Astropy ``TETE`` transform or a full apparent-place model, and
  it omits transport of the catalogue ICRS polarization tangent basis into
  that frame. In the retained HERA-site, three-source, three-time fixture, the
  public source-to-zenith angle spans ``7.64484265255e-4`` to
  ``1.120043332414e-3`` radians. The retained schema-1.2.0 artifact reduces the
  direct linear-polarization residual from ``2.052050642874229e-3`` to
  ``2.400855498837282e-10`` with the exact per-source/time correction. It was
  independently authenticated at evidence successor ``e20f636``; exact-SHA CI
  run ``31461141190`` passed all eight jobs and six compatibility cells.
  Production behavior is unchanged, ``PrecisionConfig.ultra()`` does not add
  frame transport, and no all-sky or cross-platform accuracy claim is made.
  See :doc:`development/sci007_frame_accuracy_bound`.

Known limitations
^^^^^^^^^^^^^^^^^

Release notes that list capabilities without listing these would be
incomplete. Each entry is an open row in the project's defect register. The
other six rows the ``0.3.0`` notes listed here — ``SCI-005``, ``SCI-006``,
``SCI-007``, ``CI-001``, ``API-001``, ``API-002`` — closed during this
release, each through an entry above.

- ``PERF-001`` (roadmap): accelerator performance is undemonstrated. The
  accepted CPU slice covers scheduling, deterministic selection, and
  readiness infrastructure only; the time and frequency axes remain
  host-side Python loops, every gating pixi environment's JAX is CPU-only,
  and every measured JAX-CPU run is slower than NumPy
  (``output/benchmarks/reference/``). The accelerator leg is hardware-gated,
  and RadioSim publishes no GPU or TPU performance number.
- ``SCI-004`` (roadmap): there is no spherical-harmonic or m-mode solver.
  ``execution.simulator`` accepts exactly the keys of the simulator
  registry, which currently holds only ``rime``. A design candidate
  (:doc:`development/sci004_mmode_design`) is drafted and has not been
  independently reviewed.

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

See :doc:`migration_guide` for upgrading to ``0.4.0`` — it records each
breaking change from ``0.2.0`` through ``0.4.0``, and the v0.1.x to v0.2.0
history as well.
