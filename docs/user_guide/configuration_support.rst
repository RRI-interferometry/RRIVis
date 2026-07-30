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
     - ``analytic``, ``shared_fits``, ``per_antenna_fits``, and ``mixed`` all
       resolve and run through one canonical Simulator beam system
   * - ``receptors``
     - Linear or circular two-feed receptors per antenna, static
       ``feed_rotation_deg``, and one resolved array-wide ``output_basis`` that
       names the reported correlation labels
   * - ``sky_model``
     - Strict source requests for point or HEALPix preparation
   * - ``obs_time`` / ``obs_frequency``
     - Canonical UTC sample centers and exposures, exact frequency centers,
       and required positive channel widths
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
visibility paths. Observability selects the same canonical beam evaluator and
requires an explicit reference antenna for scientifically heterogeneous
assignments.

Receptor and polarization-basis physics is implemented for ideal orthogonal
two-feed receptors: the receptor-configuration term ``C`` and the
basis-transform term ``H`` are substantive, both bases run end to end, and the
resolved basis names the correlation labels in memory, in HDF5, in the summary
JSON, in Measurement Set and UVFITS exports, and in every renderer. Polarization
leakage (``D``), parallactic rotation (``P``), gains (``G``), bandpass (``B``),
elliptical or non-orthogonal feed pairs, single-feed and multi-feed antennas,
and a frequency- or time-dependent receptor basis are not implemented. Arbitrary
BeamFITS variants, explicit Measurement Set phase centres, spherical-harmonic
simulation, and worker control are also not implemented.

Receptor support by mode
------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Declaration
     - Resolved ``output_basis``
     - Reported correlations
   * - omitted section, or ``basis: linear`` with ``output_basis: auto``
     - ``linear_xy``
     - ``XX, XY, YX, YY``
   * - ``basis: circular`` with ``output_basis: auto``
     - ``circular_rl``
     - ``RR, RL, LR, LL``
   * - any array with ``output_basis: linear``
     - ``linear_xy``
     - ``XX, XY, YX, YY``
   * - any array with ``output_basis: circular``
     - ``circular_rl``
     - ``RR, RL, LR, LL``
   * - mixed bases with ``output_basis: auto``
     - rejected
     - ``AmbiguousOutputBasisError`` naming both antenna counts

A non-zero ``feed_rotation_deg`` is a static topocentric rotation for the whole
observation, because the parallactic-angle term is not implemented. Combining it
with an enabled parallactic term is rejected rather than silently dropping the
time-dependent part.

Beam support by stage
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Beam declaration
     - Schema
     - Path resolution
     - Simulator runtime
   * - ``analytic``: ``circular_aperture``
     - Supported
     - Supported
     - Supported
   * - ``analytic``: ``rectangular_aperture``,
       ``elliptical_aperture``, ``analytical_illumination``, or
       ``numerical_illumination``
     - Supported
     - Supported
     - Supported
   * - ``shared_fits``
     - Supported
     - Supported
     - Supported within the accepted scalar subset
   * - ``per_antenna_fits``
     - Supported
     - Supported
     - Supported within the accepted scalar subset
   * - ``mixed``
     - Supported
     - Supported
     - Supported within the accepted scalar subset

FITS path validation checks and records sources but does not read BeamFITS
content. ``Simulator.setup`` resolves canonical antenna references, loads and
validates the accepted scalar subset, and publishes state atomically. This does
not imply arbitrary full-polarization BeamFITS, GPU interpolation, automatic
NSIDE mutation, or resampling support.

NumPy is the deterministic backend default. Selecting JAX, Numba, or ``auto``
does not establish complete accelerator coverage for the high-level workflow.

Output boundary
---------------

``Simulator.save`` accepts an exact final artifact path and a typed
``ResultFormat`` for HDF5, summary JSON, Measurement Set, or UVFITS. Direct
Python and ``simulate`` calls never prompt or suffix. Config mode preflights
``collision_policy`` and manifest ownership before runtime, builds the complete
run in sibling staging, and publishes atomically. Summary JSON is explicitly
incomplete metadata; HDF5 is the complete reconstructable result.
``Simulator.plot`` renders the published result into one explicit directory
from the canonical coordinate arrays.  Configured workflow plotting is
preflighted before runtime — only ``plotting_backend: bokeh`` is implemented —
then staged with the run and opened from published paths last.

Retained visualization controls are ``plot_results``, ``open_plots_in_browser``,
``plotting_backend``, and ``visibility_phase_unit`` (exactly ``radians`` or
``degrees``).  Every other visualization input was removed and is rejected with
exact migration text.
