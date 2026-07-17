Configuration Support Matrix
============================

This page distinguishes schema presence, resolved runtime behavior, and
high-level feature support. All public configuration sources share the same
strict Tier 1 resolver.

Unified entry points
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Entry point
     - Input
     - Result
   * - ``load_config(path)``
     - YAML path; paths based at the YAML parent
     - ``ResolvedConfiguration(runtime, workflow, provenance)``
   * - ``resolve_config(config, source=...)``
     - Mapping or ``RadioSimConfig`` plus source context
     - The same resolved bundle
   * - ``Simulator(resolved)``
     - ``ResolvedSimulationConfig`` only
     - Runtime object; no workflow state
   * - ``Simulator.from_yaml(path)``
     - YAML document
     - Simulator built from resolved scientific state
   * - ``Simulator.from_config(model, base_dir=...)``
     - Strict input model
     - Simulator built through the common resolver
   * - ``Simulator.from_mapping(data, base_dir=...)``
     - Python mapping
     - Simulator built through the common resolver
   * - ``Simulator.from_parameters(...)``
     - Complete scientific parameters and explicit Hz channels
     - Simulator built through the common resolver
   * - ``radiosim validate``
     - YAML document
     - Resolved summary without Simulator/runtime/output work

Top-level ownership
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Section
     - Owner
     - Current high-level behavior
   * - ``telescope``
     - Scientific input
     - Name is retained; pyuvdata opt-ins are rejected until Tier 2
   * - ``antenna_layout``
     - Scientific input
     - File, format, and one positive uniform diameter are consumed
   * - ``feeds``
     - Deferred scientific input
     - Non-default top-level receptor settings are rejected until Tier 5
   * - ``beams``
     - Scientific input
     - Analytic aperture/illumination fields are consumed; FITS/mixed/per-antenna controls are rejected until Tier 3
   * - ``baseline_selection``
     - Deferred scientific input
     - Current autos-plus-crosses defaults are accepted; selection changes are rejected until Tier 2
   * - ``location``
     - Required scientific input
     - Finite latitude, longitude, and non-negative height are consumed
   * - ``sky_model``
     - Required scientific input
     - Strict source specs resolve into immutable loader requests
   * - ``obs_time``
     - Required scientific input
     - Parseable start time, positive duration, and cadence are consumed
   * - ``obs_frequency``
     - Required scientific input
     - ``grid`` or exact ``explicit`` mode resolves to an immutable Hz tuple
   * - ``visibility``
     - Scientific input
     - Point-source or HEALPix direct sum is supported; spherical harmonic is rejected
   * - ``execution``
     - Runtime policy
     - Backend, complete precision tree, ``rime`` simulator, and offline policy are resolved once
   * - ``workflow``
     - CLI-only policy
     - Saving, logging, plotting, prompting, skipping, and browser behavior remain outside Simulator runtime state

High-risk settings
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Setting
     - Current behavior
     - Target tier
   * - ``beams.beam_mode: fits|mixed`` and FITS controls
     - Rejected before path/backend/device work; the high-level beam manager is not connected
     - Tier 3
   * - ``beams.per_antenna`` or ``antenna_beam_map``
     - Rejected; per-antenna beam selection is not applied
     - Tier 3
   * - ``antenna_layout.use_different_diameters`` or ``diameters``
     - Rejected; only the uniform diameter reaches setup
     - Tier 2
   * - non-default ``baseline_selection`` values
     - Rejected; all generated baselines are currently used
     - Tier 2
   * - ``telescope.use_pyuvdata_*``
     - Rejected; explicit location and selected layout file remain required
     - Tier 2
   * - non-default top-level ``feeds`` values
     - Rejected; receptor physics is not connected
     - Tier 5
   * - ``visibility.calculation_type: spherical_harmonic``
     - Rejected; no high-level m-mode solver exists
     - Tier 7
   * - ``workflow.result_format: uvfits``
     - Rejected; ``Simulator.save`` currently dispatches HDF5, JSON summary, and optional MS
     - Tier 4
   * - ``Simulator.run(n_workers=...)``
     - Rejected because solver-worker control is not implemented
     - Tier 6

Supported analytic beam controls
--------------------------------

``beams.aperture_shape``, ``taper``, ``edge_taper_dB``, ``feed_model``,
``feed_computation``, ``feed_params``, ``reflector_type``, ``magnification``,
and ``aperture_params`` configure the current analytic beam. These describe
aperture illumination, not the rejected top-level receptor/feed section.

Backend boundary
----------------

The resolver honors ``execution.backend`` and complete precision overrides.
NumPy is the deterministic default and ``auto`` is a real strategy. This does
not establish complete acceleration of every high-level calculation. Host-side
loops, Astropy work, transfers, and incomplete backend coverage remain; no
blanket GPU or speedup claim is supported by the current test suite.

Output boundary
---------------

The CLI consumes ``workflow`` after a successful run. Python construction and
``run()`` do not implicitly save or plot. HDF5 and JSON remain pre-Tier-4
formats with known result-contract limitations. Measurement Set output depends
on optional runtime support and lacks a complete round-trip guarantee here.
UVFITS is rejected.

Observability boundary
----------------------

``Simulator.plot_observability()`` builds a visualization from the Simulator's
resolved location, time, frequency, analytic beam, and optional prepared sky.
It is a helper, not a separate product, backend, or visibility solver.
