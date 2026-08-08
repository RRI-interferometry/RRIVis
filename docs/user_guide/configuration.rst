Configuration Guide
===================

YAML, Python mappings, typed models, and parameter construction share one
strict resolver. Unknown fields are rejected, and the pre-v1 API does not keep
compatibility aliases for removed input shapes.

Complete document
-----------------

``instrument`` owns exactly one source of antenna positions and identities.
Local formats also require explicit identity and geodetic location:

.. code-block:: yaml

   instrument:
     source:
       kind: layout_file
       path: ../antenna_layout_examples/hera_5.txt
       format: radiosim
       telescope_name: HERA
     location:
       longitude_deg: 21.4283
       latitude_deg: -30.72152
       height_m: 1073.0
     default_diameter_m: 14.0
     diameter_overrides: []

   baseline_selection:
     correlations: all
     length_filter: null
     azimuth_ranges_deg: []

   beams:
     mode: analytic
     model:
       kind: circular_aperture
       taper:
         kind: gaussian
         edge_taper_db: 10.0

   receptors:
     default:
       basis: linear
       feed_rotation_deg: 0.0
     overrides: []
     output_basis: auto

   jones:
     G:
       amplitude_error: 0.02

   obs_time:
     start_time: "2025-01-01T00:00:00"
     duration_seconds: 60.0
     time_step_seconds: 10.0

   obs_frequency:
     mode: explicit
     channel_frequencies_hz: [100000000.0, 101500000.0, 108000000.0]
     channel_widths_hz: [1000000.0, 1000000.0, 1000000.0]

   sky_model:
     sources:
       - kind: test_sources
         num_sources: 3
         seed: 7

   visibility:
     sky_representation: point_sources
     allow_lossy_point_rasterization: false

   execution:
     backend: numpy
     precision:
       preset: standard
     simulator: rime
     offline: true
     sky_loading:
       max_workers: null
       executor: auto
     solver:
       workers: 1
       executor: thread

   workflow:
     output_dir: output
     result_filename: visibilities
     result_format: hdf5
     collision_policy: error
     save_results: false
     plot_results: false
     open_plots_in_browser: false
     plotting_backend: bokeh
     visibility_phase_unit: radians
     save_log: false

See :doc:`instrument_resolution` for source formats, precedence, baseline
criteria, lifecycle, and output provenance.

Frequency modes
---------------

``obs_frequency`` is a discriminated union. ``mode: explicit`` accepts one or
more strictly increasing values in Hz. ``mode: grid`` accepts a start,
interval, inclusive bandwidth, and ``Hz``, ``kHz``, ``MHz``, or ``GHz`` unit.
The bandwidth-to-interval ratio must be integral; resolution preserves the
requested spacing.

Execution and workflow
----------------------

``execution`` selects the backend, complete precision policy, simulator,
global offline policy, and the two worker blocks below. ``workflow`` is CLI-only
post-run orchestration. ``ResolvedSimulationConfig`` excludes workflow state, and
Python constructors never save, plot, prompt, skip, log, or open a browser
implicitly.

``execution.backend`` is exactly ``numpy``, ``jax``, ``dask``, or ``auto``. See
:doc:`backends` for what each one executes, for ``auto``'s precedence, and for
the measured comparison between them.

Worker policy
-------------

Loader concurrency and solver concurrency are separate policies with separate
failure modes, so they are separate blocks. Neither is a single global
``n_workers``; the removed ``execution.n_workers`` field is rejected with a
message naming both replacements.

.. code-block:: yaml

   execution:
     sky_loading:
       max_workers: null   # null => auto: min(requested loads, cpu_count, 8)
       executor: auto      # auto | thread | process
     solver:
       workers: 1          # clamped to the number of time samples
       executor: thread    # the only supported value

``sky_loading`` governs how many sky models are loaded concurrently.
``executor: auto`` uses processes when the loader arguments can be pickled and
falls back to threads when they cannot; ``thread`` forces threads; ``process``
demands processes and fails loudly if the arguments cannot cross the boundary.
Global offline policy is installed in every worker, so an offline run cannot
reach the network from inside one.

``solver.workers`` parallelizes the **time axis only**: each worker computes a
contiguous block of time samples and the blocks are reassembled in time order,
so every ``workers`` value produces a bit-identical result. The pre-clamp
request is recorded even when the clamp lowers it. ``executor: process`` is
rejected — the solver closure holds beam handlers and astropy objects that
cannot cross a process boundary.

Both resolved values reach ``result.resolved_config``, and therefore
``provenance_sha256``, the HDF5 ``resolved_config_json``, and the summary JSON.

Rejections, verbatim:

.. code-block:: text

   execution.n_workers: not a field; use execution.sky_loading.max_workers for
   sky-loader concurrency or execution.solver.workers for solver concurrency.

   execution.sky_loading.max_workers must be a positive integer or null (null
   means auto).

   execution.solver.workers must be a positive integer.

   execution.solver.executor=process: unsupported; the solver closure holds beam
   handlers and astropy objects that cannot cross a process boundary. Use
   execution.solver.executor=thread.

Sky representation and the ``hybrid`` mode
------------------------------------------

``visibility.sky_representation`` is ``point_sources``, ``healpix_map``, or
``hybrid``. ``hybrid`` solves a point component and a HEALPix component on one
shared instrument, beam system, receptor set, time grid, and backend, then sums
them: ``V_total = V_point + V_healpix``. It is not an approximation and not a
conversion — neither payload is materialized into the other.

.. code-block:: yaml

   visibility:
     sky_representation: hybrid

The ``sky_model`` section must then contribute both kinds of payload — at least
one source that resolves to point sources and at least one that resolves to a
HEALPix map. ``configs/hybrid_sky_example.yaml`` is a complete, offline,
runnable document that does exactly that. The result records which components it
solved, their element counts, and a separate timing for each, in
``result.solver`` and ``result.performance``.

``allow_lossy_point_rasterization`` (default ``false``) gates the one remaining
lossy path: rasterizing point sources into a HEALPix grid, which quantizes
positions to pixel centers.

Three runtime rejections replace conversions that used to happen silently. Each
is raised before any beam load, backend allocation, or output path is created:

.. code-block:: text

   visibility.sky_representation=hybrid requires a sky model with both a
   point-source payload and a HEALPix payload; the resolved model carries only
   {formats}. Request point_sources or healpix_map, or add a source of the missing
   kind.

   visibility.sky_representation=point_sources would discard the HEALPix payload
   carried by the resolved sky model. Request hybrid to sum both components, or set
   visibility.allow_lossy_point_materialization=true to convert the HEALPix payload
   to point sources.

   visibility.sky_representation=healpix_map would rasterize {n} point source(s)
   into the HEALPix grid, which quantizes positions to pixel centers. Request
   hybrid to sum both components, or set
   visibility.allow_lossy_point_rasterization=true to opt in.

``visibility`` has no ``calculation_type`` field. It was removed before v1.0:
it validated two values and no module read either of them. The solver strategy
is selected by ``execution.simulator``, whose accepted values are exactly the
keys of the simulator registry — currently the single value ``rime``. A document
that still sets ``visibility.calculation_type`` is rejected with removed-field
guidance; see :doc:`../migration_guide`.

Output and result formats
-------------------------

``result_format`` accepts exactly ``hdf5``, ``summary_json``, ``ms``, or
``uvfits``. HDF5 is complete; summary JSON is metadata-only. The four
``collision_policy`` values are ``error``, ``replace``, ``suffix``, and
``prompt``. Only ``prompt`` may ask a question, only for a valid owned run on
an actual TTY. A manifest safely defines run-directory ownership.

The visualization controls are ``plot_results``, ``open_plots_in_browser``,
``plotting_backend`` (only ``bokeh`` is implemented), and
``visibility_phase_unit``, which is exactly ``radians`` or ``degrees`` and
changes only the displayed phase axis. Configured plots render into the staged
run directory and their published paths are opened last. Removed workflow
fields are rejected with exact migration text; see :doc:`../migration_guide`.

Loading and serialization
-------------------------

.. code-block:: python

   from radiosim.io import dump_config, load_config

   bundle = load_config("configs/config.yaml")
   runtime = bundle.runtime
   print(runtime.instrument.source)

   dump_config(input_model, "copied-config.yaml")

``load_config`` returns ``ResolvedConfiguration(runtime, workflow,
provenance)``. ``dump_config`` accepts only a strict ``RadioSimConfig`` input
model. Resolved paths and result provenance are not serialized as input YAML.

Path and override rules
-----------------------

- YAML-relative paths use the YAML file's parent.
- Mapping/model relative paths require ``base_dir``.
- Explicit path overrides use the captured invocation directory.
- ``~`` is expanded and environment-variable syntax is rejected.
- Input paths are checked before backend or scientific loading.
- An instrument path override is valid only for ``layout_file`` sources.
- Location and frequency overrides replace complete typed values.
- Precedence is ``explicit override > document value > declared default``.

Nested FITS beam paths follow the same source bases. Shared sources are recorded
as ``beams.beam.path``; assignment sources use indexed keys such as
``beams.assignments[0].beam.path``. Checked inputs must be readable regular
files. Resolution normalizes and fingerprints these declarations without
reading BeamFITS content.

Beam declarations
-----------------

``beams.mode`` is one of ``analytic``, ``shared_fits``,
``per_antenna_fits``, or ``mixed``. See :doc:`beam_models` for the complete
tagged shapes and all five analytic model variants. Schema validation and path
resolution accept all four modes, and ``Simulator.setup`` resolves and loads
all four through one canonical ``BeamSystem``. Path validation alone does not
read FITS content; setup performs canonical antenna assignment, validates the
accepted scalar FITS subset, and loads every required handler atomically.
Point-source, HEALPix, observability, sampling-advice, and result-provenance
paths consume the same loaded state. FITS failures never fall back to analytic
evaluation.

An ``allow_network`` known-telescope source conflicts with global offline mode.
Validation and offline tests never enumerate the live registry.

Two optional blocks describe the *mount and the dish* rather than the beam
model, so they are accepted in all four modes:

.. code-block:: yaml

   beams:
     mode: analytic
     model:
       kind: circular_aperture

     pointing:                              # optional; absent = no offset
       default:                             # optional array-wide default
         azimuth_offset_deg: 0.0
         elevation_offset_deg: 0.0
       per_antenna:                         # optional; overrides the default
         - antenna: {kind: number, number: 1}
           azimuth_offset_deg: 90.0
           elevation_offset_deg: 0.25

     surface_error:                         # optional; absent = no Ruze factor
       default:
         rms_surface_error_m: 0.001
       per_antenna:
         - antenna: {kind: name, name: ANT0}
           rms_surface_error_m: 0.004

``pointing`` is a deterministic mount mispointing and ``surface_error`` is the
Ruze random-surface RMS; :doc:`beam_models` gives the exact geometry and the
exact factor. Both blocks follow the same rules as the rest of the strict
schema: ``per_antenna`` entries are keyed by the tagged Tier 2 antenna
reference and reject an unknown or repeated antenna, and every angle carries
``_deg`` while every length carries ``_m``.

Two rules are worth stating because they are what keeps an absent block honest:

- A block **every one of whose authored numbers is zero** is rejected. A
  present block with no effect is the configuration surface that accepts a
  value and discards it, which is exactly what ``jones``'s identity rejection
  exists to prevent. A zero *entry* alongside a non-zero sibling is accepted,
  and is the honest way to say that one antenna is perfectly pointed.
- An offset of exactly ``(0, 0)`` and an RMS of exactly ``0.0`` resolve to
  *absence*, not to a stored zero. A document authoring them is therefore
  bit-identical to a document authoring nothing — same cube, same beam
  fingerprints, same ``scientific_sha256``.

Jones-term declarations
-----------------------

``jones`` selects which instrumental Jones terms corrupt the simulated
visibilities. The whole section is optional, every term inside it is optional,
and omitting it produces exactly the visibilities — and exactly the
``scientific_sha256`` — that a document without the section produced before it
existed.

RadioSim implements eleven configurable terms today, and every letter this
section accepts names a term that runs. Nine are per-antenna chain terms: ``G``
(complex electronic gain), ``B`` (bandpass), ``Rc`` (cable reflection), ``Kd``
(instrumental delay), ``X`` (cross-hand phase and delay), ``D`` (polarization
leakage), ``P`` (parallactic angle), ``T`` (troposphere) and ``Z``
(ionosphere). The first five are diagonal, and so is ``T``; ``D``, ``P`` and a
``Z`` with Faraday rotation are not. ``Kd``, ``X``, ``P``, ``Z`` and a ``T``
with no opacity are unitary. ``P``, ``T`` and ``Z`` are direction-dependent.

The other two — ``M`` (per-baseline closure error) and ``Q`` (time and bandwidth
smearing) — are **not** chain terms. They are baseline-dependent and apply by
Hadamard product to the visibilities rather than as a factor of any antenna's
Jones matrix, which is why ``M`` is the one term that breaks closure phase and
why ``Q`` is configured by two switches and nothing else: its channel width and
its integration time come from the observation, not from the term.

.. code-block:: yaml

   jones:
     G:
       amplitude_error: 0.02
       phase_error_rad: 0.1
       per_antenna:
         - antenna: 12
           feed: 0
           amplitude_error: 0.05
       time_model:
         kind: linear_drift
         rate_per_hour: 0.01
     B:
       model:
         kind: polynomial
         coefficients: [1.0, 0.0, -0.05]
     Rc:
       amplitude: 0.01           # 0 < |A| < 1, rejected outside
       cable_delay_s: 1.5e-7
     Kd:
       delay_s: 1.0e-9
     X:
       phase_rad: 0.1
       delay_s: 0.0
     D:
       d_terms:
         kind: explicit          # explicit | ixr | frequency_polynomial
         d0: [0.02, 0.0]
         d1: [0.0, 0.02]
     P:
       enabled: true             # the whole block; P has no free parameter
     T:
       zenith_delay:
         kind: saastamoinen      # explicit | saastamoinen
         surface_pressure_hpa: 1013.25
         zenith_wet_delay_m: 0.05
       mapping_function: niell   # simple | niell
       minimum_elevation_deg: 5.0
       opacity:                  # optional; absent means transparent
         zenith_opacity: 0.02
     Z:
       tec:
         kind: constant          # constant | gradient
         vertical_tec_tecu: 10.0
       shell_height_km: 350.0
       minimum_elevation_deg: 5.0
       faraday:                  # optional; absent means phase only
         rotation_measure_rad_m2: 0.5
     M:
       per_baseline:             # keyed by the ordered antenna-number pair
         - antennas: [0, 1]
           matrix: [[[1.02, 0.03], [0.98, -0.01]],
                    [[0.97, 0.01], [1.04, 0.02]]]
     Q:
       bandwidth_smearing: true  # both switches required; no default
       time_smearing: true

Terms are applied in the canonical chain order regardless of the order the keys
appear in the document, so two files that enable the same terms produce the same
visibilities and the same fingerprint.

There is no ``enabled: false``. To disable a term, delete its block — and a
block whose resolved parameters make the term exactly the identity is
*rejected*, because a term that cannot change the visibilities is
indistinguishable from no term at all. A ``jones:`` key present with nothing
under it is rejected for the same reason. ``P`` is the one block with an
``enabled`` key at all, because the parallactic angle has no other parameter,
and writing ``false`` there reaches the same rejection.

``jones.P`` is also the one term paired with the *instrument*: which antennas'
feeds rotate is decided by each antenna's ``mount_type``. An antenna on a
rotating mount (``alt-az`` or either Nasmyth variant) with no ``jones.P`` is
rejected, an array on which nothing rotates cannot configure ``jones.P``, and a
mount type outside the five ``P`` models is rejected either way. See
:doc:`jones_terms` for the three messages.

``jones.M`` is keyed by *baseline* rather than by antenna: an ordered pair the
resolved baseline selection does not contain is rejected, and so is a repeated
pair. Its neutral value is ``1`` in every entry rather than the identity matrix,
because the product is elementwise — see :doc:`jones_terms`.

``per_antenna`` entries are keyed by antenna **number** and validated against
the resolved instrument, so an unknown number, a repeated ``(antenna, feed)``
pair, or a feed index outside ``{0, 1}`` each fail with a message naming the
term. ``X`` is the one exception to the ``feed`` key: its parameter is the
relative phase *between* an antenna's two feeds, so its entries are keyed by
antenna alone. Every ``jones`` rejection is raised before any beam is loaded,
any sky model is fetched, or any network access happens.

Write floats in scientific notation with a signed exponent — ``1.0e+8``, not
``1.0e8`` — because YAML 1.1 parses the unsigned form as a string.

See :doc:`jones_terms` for each term's mathematics, units, citation, and full
field list, and for where the enabled terms are recorded in the outputs.

Receptor declarations
---------------------

``receptors`` describes the *receiving* receptors: which pair of orthogonal
feeds each antenna carries and which single basis the whole array is reported
in. It is a separate concern from ``beams``, which describes how each reflector
aperture is *illuminated*. The two vocabularies do not overlap: ``receptors``
owns ``basis``, ``feed_rotation_deg``, and ``output_basis``, while ``beams``
owns ``illumination``, ``taper``, and edge angles.

The whole section is optional. Omitting it is exactly equivalent to::

   receptors:
     default:
       basis: linear
       feed_rotation_deg: 0.0
     overrides: []
     output_basis: auto

``default.basis`` is exactly ``linear`` (feeds ``x``, ``y``) or ``circular``
(feeds ``r``, ``l``, IAU sense). Every antenna carries exactly two ideal
orthogonal feeds. ``default.feed_rotation_deg`` is a finite static offset from
the nominal orientation of the selected basis, in degrees.
``output_basis`` is ``auto``, ``linear``, or ``circular``, and selects the one
basis every reported visibility is expressed in.

Homogeneous circular, resolved automatically:

.. code-block:: yaml

   receptors:
     default:
       basis: circular
     output_basis: auto      # resolves to circular_rl

Homogeneous linear rotated 45 degrees:

.. code-block:: yaml

   receptors:
     default:
       basis: linear
       feed_rotation_deg: 45.0

Heterogeneous rotations within one basis, per antenna:

.. code-block:: yaml

   receptors:
     default:
       basis: linear
     overrides:
       - antenna: {kind: number, number: 3}
         feed_rotation_deg: 30.0
       - antenna: {kind: name, name: HERA-11}
         feed_rotation_deg: -15.0

Heterogeneous bases, with the common output basis named explicitly:

.. code-block:: yaml

   receptors:
     default:
       basis: linear
     overrides:
       - antenna: {kind: number, number: 7}
         basis: circular
     output_basis: circular    # every antenna transformed into R/L

Circular-native receptors reported in a linear basis:

.. code-block:: yaml

   receptors:
     default:
       basis: circular
     output_basis: linear

``overrides`` entries use the same tagged ``antenna`` references as
``instrument.diameter_overrides`` and ``beams.assignments``, so
``{kind: number, number: N}`` and ``{kind: name, name: NAME}`` are both
accepted. Each entry must set at least one of ``basis`` or
``feed_rotation_deg``, and no two entries may resolve to the same canonical
antenna.

Correlation labels follow the resolved output basis. ``linear_xy`` reports
``XX, XY, YX, YY`` and ``circular_rl`` reports ``RR, RL, LR, LL``; the
correlation axis is the row-major flattening of the 2x2 visibility matrix in
both cases, so indices ``0`` and ``3`` are always the parallel hands. Read
``result.correlations`` and ``result.polarization_basis`` rather than assuming
the linear labels. See :doc:`jones_matrices` for the receptor mathematics and
the boundaries Tier 5 does not cross.

The canonical sky brightness axes are ``(North, East)``. At zero feed rotation,
linear output is physically ``(X=east, Y=north)`` with feed angles
``(pi/2, 0)``. Consequently
``(XX,XY,YX,YY)=((I-Q)/2,(U-iV)/2,(U+iV)/2,(I+Q)/2)`` and
``XX-YY=-Q``. ``feed_rotation_deg`` rotates the physical pair; it does not
change which configured linear feed index means X/east (0) or Y/north (1).

Boundaries. ``output_basis: auto`` cannot resolve a mixed array and is rejected
with a count of linear and circular antennas; name the basis instead. A
``basis`` value other than ``linear`` or ``circular``, a single-feed or
multi-feed antenna, elliptical or non-orthogonal feed pairs, independent
per-feed angles, and a frequency- or time-dependent basis are all rejected.
``feed_rotation_deg`` is the **static** part of the orientation; the
time-dependent part is ``jones.P``, and the two compose into a rotation by
:math:`\chi + \psi(t)` rather than conflicting. Receptor resolution does not
look at ``mount_type`` at all — that is ``jones.P``'s pairing, above. The
removed top-level ``feeds`` key is rejected with a pointer at this section; see
:doc:`../migration_guide`.
