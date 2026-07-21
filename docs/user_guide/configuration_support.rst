Configuration Support Matrix
============================

All public configuration sources use the same strict resolver.

Entry points
------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Entry point
     - Input
     - Result
   * - ``load_config(path)``
     - YAML with paths based at its parent
     - Resolved runtime, workflow, and provenance
   * - ``resolve_config(config, source=...)``
     - Mapping or ``RadioSimConfig`` with source context
     - The same resolved bundle
   * - ``Simulator(resolved)``
     - ``ResolvedSimulationConfig`` only
     - Runtime object without workflow state
   * - ``Simulator.from_yaml(path)``
     - YAML document
     - Simulator through the common resolver
   * - ``Simulator.from_config(model, base_dir=...)``
     - Strict input model
     - Simulator through the common resolver
   * - ``Simulator.from_mapping(data, base_dir=...)``
     - Python mapping
     - Simulator through the common resolver
   * - ``Simulator.from_parameters(...)``
     - Typed instrument, typed baseline selection, and scientific values
     - Simulator with explicit-Hz frequency input
   * - ``radiosim validate``
     - YAML document
     - Resolved summary without runtime/output work

Scientific ownership
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Section
     - Active behavior
   * - ``instrument``
     - One discriminated source resolves canonical identities, positions,
       location, diameters, and deterministic provenance
   * - ``baseline_selection``
     - Typed correlation, length-target/range, and axial-azimuth filtering
   * - ``beams``
     - Four strict modes parse and resolve. Only direct-circular ``analytic`` is
       Simulator-active; FITS-backed modes and other analytic variants fail at
       the runtime boundary
   * - ``sky_model``
     - Strict source requests for point or HEALPix preparation
   * - ``obs_time`` / ``obs_frequency``
     - Validated cadence and exact frequency samples
   * - ``visibility``
     - Point-source or HEALPix direct sum
   * - ``execution``
     - Backend, precision, RIME simulator, and offline policy
   * - ``workflow``
     - CLI-only saving, logging, plotting, prompting, and browser policy

Instrument source support
-------------------------

``layout_file`` supports ``radiosim``, ``casa_loc``, ``measurement_set``,
``uvfits``, and ``mwa_metafits``. ``known_telescope`` uses a named registry
source with an explicit offline/network policy. It is a source kind, not a
file format. See :doc:`instrument_resolution`.

Feature boundaries
------------------

Heterogeneous positive antenna diameters are used by both point and HEALPix
analytic visibility paths. Observability plotting deliberately accepts only a
uniform resolved diameter and raises before sky preparation for heterogeneous
arrays. Feed/receptor physics, FITS beams, mixed beam modes, explicit
Measurement Set phase centres, UVFITS writing, spherical-harmonic simulation,
and worker control are not implemented.

Beam support by stage
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Beam declaration
     - Schema
     - Path resolution
     - Simulator runtime
   * - Direct circular analytic
     - Supported
     - Supported
     - Supported
   * - Other analytic variants
     - Supported
     - Supported
     - Explicitly pending
   * - Shared FITS
     - Supported
     - Supported
     - Explicitly pending
   * - Per-antenna FITS
     - Supported
     - Supported
     - Explicitly pending
   * - Mixed analytic/FITS
     - Supported
     - Supported
     - Explicitly pending

FITS path resolution checks and records sources but does not read BeamFITS
content, resolve antenna references, or imply UVBeam, solver, GPU interpolation,
or FITS observability support.

NumPy is the deterministic backend default. Selecting JAX, Numba, or ``auto``
does not establish complete accelerator coverage for the high-level workflow.

Output boundary
---------------

``Simulator.save`` dispatches HDF5, JSON summary, and optional Measurement Set
output. HDF5 and JSON retain the deterministic instrument provenance snapshot.
Measurement Set construction uses canonical names, numbers, diameters,
location, and selected baselines; it does not promise a general round-trip or
explicit phase-centre API. UVFITS output is rejected.
