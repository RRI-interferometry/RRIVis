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
     calculation_type: direct_sum
     sky_representation: point_sources

   execution:
     backend: numpy
     precision:
       preset: standard
     simulator: rime
     offline: true

   workflow:
     output_dir: output
     result_filename: visibilities
     result_format: hdf5
     save_results: false
     plot_results: false
     open_plots_in_browser: false
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

``execution`` selects the backend, complete precision policy, simulator, and
global offline policy. ``workflow`` is CLI-only post-run orchestration.
``ResolvedSimulationConfig`` excludes workflow state, and Python constructors
never save, plot, prompt, skip, log, or open a browser implicitly.

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
